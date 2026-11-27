"""Native multimodal Scene-to-MJCF Agent loop.

Assumes the LLM is capable: no welded workflow state machines, no category
whitelists, and no hardcoded bad-case interceptions. All optional guidance or
guardrails live outside this loop as reversible Cordis plugins. Every run
produces a post-training-ready multimodal trace.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..cordis.kernel import CordisContext
from .assets import AssetCatalog
from .llm import ChatModel, image_to_data_url
from .mjcf import write_xml_atomic
from .randomizer import RandomizedVariant, randomize_xml
from .tools import SceneToolbox, SceneWorkspace
from .trace import AgentTrace, describe_image_file
from .verifier import VerificationReport, verify_mujoco_xml


_BASE_SYSTEM_PROMPT = """\
You are SceneFactory, an autonomous 3D physical scene authoring agent whose goal is to produce \
a clean, high-fidelity, physically stable MuJoCo MJCF XML scene matching the user's text and \
optional reference image(s).

You have full agency over how to construct the scene:
- Use `search_assets` and `insert_asset` whenever local multi-part assets fit well.
- Use `write_xml` or `edit_xml` freely to author custom geometry, layouts, materials, lighting, and cameras.
- Call `verify_xml` to test compilation, settling stability, and visually inspect the rendered views.
- If MuJoCo reports compile errors, penetration, falls, or visual mismatch with the request/image, \
diagnose the cause and refine the XML yourself before finishing.
- When you are satisfied with the verified XML, reply concisely summarizing the final scene."""


@dataclass
class AgentResult:
    xml: str
    xml_sha256: str
    verification: VerificationReport
    trace: AgentTrace
    variants: list[RandomizedVariant] = field(default_factory=list)
    final_message: str = ""
    output_dir: Path | None = None


class SceneAgent:
    """Thin, transparent multi-turn VLM/LLM loop over MuJoCo XML tools."""

    def __init__(
        self,
        model: ChatModel,
        *,
        catalog: AssetCatalog | None = None,
        cordis: CordisContext | None = None,
        max_turns: int = 12,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.model = model
        self.catalog = catalog or AssetCatalog()
        self.cordis = cordis or CordisContext()
        self.max_turns = int(max_turns)

    def run(
        self,
        prompt: str,
        *,
        images: Sequence[str | Path] = (),
        output_dir: str | Path | None = None,
        randomize_count: int = 0,
        seed: int = 42,
        verify_final: bool = True,
    ) -> AgentResult:
        out_dir = Path(output_dir).resolve() if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        workspace = SceneWorkspace(render_root=(out_dir / "renders") if out_dir else None)
        toolbox = SceneToolbox(catalog=self.catalog, workspace=workspace)

        state: dict[str, Any] = {
            "prompt": prompt,
            "has_images": bool(images),
            "turn": 0,
        }
        prompt_parts = [_BASE_SYSTEM_PROMPT, *self.cordis.collect_prompt_sections(state)]
        system_prompt = "\n\n".join(prompt_parts)

        # Build live multimodal user message AND lightweight trace-safe record.
        live_user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        trace_user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        input_image_meta: list[dict[str, Any]] = []
        for img_path in images:
            data_url, mime = image_to_data_url(img_path)
            meta = describe_image_file(img_path, mime)
            input_image_meta.append(meta)
            live_user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            trace_user_content.append({"type": "image_ref", **meta})

        live_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": live_user_content if images else prompt},
        ]
        trace = AgentTrace(
            prompt=prompt,
            model=self.model.model,
            input_images=input_image_meta,
            cordis_plugins=self.cordis.snapshot(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": trace_user_content if images else prompt},
            ],
        )

        final_message = ""
        for turn in range(1, self.max_turns + 1):
            state["turn"] = turn
            trace.turns = turn
            response = self.model.chat(live_messages, tools=toolbox.schemas)
            if response.model:
                trace.model = response.model
            trace.add_usage(response.usage)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.content}
            if response.reasoning:
                assistant_msg["reasoning"] = response.reasoning
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
            live_messages.append(assistant_msg)
            trace.messages.append(assistant_msg)

            if not response.tool_calls:
                final_message = response.content
                break

            rendered_for_turn: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                args, hints, block_reason = self.cordis.run_before_tool(tc.name, tc.arguments, state)
                if block_reason is not None:
                    result: dict[str, Any] = {"ok": False, "blocked": True, "error": block_reason}
                    imgs: list[dict[str, Any]] = []
                else:
                    result, imgs = toolbox.execute(tc.name, args)
                    if hints:
                        result["advisory_hints"] = hints
                    if tc.name == "verify_xml" and "verification" in result:
                        result["verification"] = self.cordis.run_after_verify(
                            workspace.xml, result["verification"]
                        )
                    result = self.cordis.run_after_tool(tc.name, args, result, state)

                serialized = json.dumps(result, ensure_ascii=False)
                tool_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": serialized}
                live_messages.append(tool_msg)
                trace.messages.append(tool_msg)
                rendered_for_turn.extend(imgs)

            # Feed rendered MuJoCo views back to the VLM for genuine visual self-correction.
            if rendered_for_turn:
                live_obs: list[dict[str, Any]] = [
                    {"type": "text", "text": f"Rendered MuJoCo views from verify_xml (turn {turn}):"}
                ]
                trace_obs: list[dict[str, Any]] = [
                    {"type": "text", "text": f"Rendered MuJoCo views from verify_xml (turn {turn}):"}
                ]
                for img in rendered_for_turn:
                    live_obs.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img['png_base64']}"},
                        }
                    )
                    trace_obs.append(
                        {
                            "type": "image_ref",
                            "camera": img["camera"],
                            "path": img.get("path"),
                            "sha256": img["sha256"],
                            "mime_type": "image/png",
                        }
                    )
                live_messages.append({"role": "user", "content": live_obs})
                trace.messages.append({"role": "user", "content": trace_obs})

        xml = workspace.xml
        xml_sha = hashlib.sha256(xml.encode("utf-8")).hexdigest()
        verification = workspace.last_verification
        if verify_final and (verification is None or verification.xml_sha256 != xml_sha):
            verification = verify_mujoco_xml(
                xml,
                render=True,
                render_dir=(out_dir / "renders" / "final") if out_dir else None,
            )
        if verification is None:
            verification = VerificationReport(compiled=False, xml_sha256=xml_sha)

        variants = list(workspace.variants)
        if randomize_count > 0 and not variants and verification.compiled:
            variants, _ = randomize_xml(xml, count=randomize_count, seed=seed)

        trace.final_xml_sha256 = xml_sha
        trace.verification = verification.to_dict(include_base64=False)
        trace.verifier_score = verification.score
        trace.variant_count = len(variants)
        trace.fragment_cache = toolbox.fragments.stats()

        if out_dir:
            xml_path = write_xml_atomic(out_dir / "scene.xml", xml)
            trace.final_xml_path = str(xml_path.resolve())
            if variants:
                vdir = out_dir / "variants"
                vdir.mkdir(parents=True, exist_ok=True)
                for var in variants:
                    write_xml_atomic(vdir / f"variant_{var.index:03d}_seed{var.seed}.xml", var.xml)
            trace.save(out_dir / "trace.json")

        return AgentResult(
            xml=xml,
            xml_sha256=xml_sha,
            verification=verification,
            trace=trace,
            variants=variants,
            final_message=final_message,
            output_dir=out_dir,
        )
