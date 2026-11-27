"""Core Scene-to-MJCF v2 engine: multimodal Agent, MuJoCo verifier, randomizer, and trace."""

from .agent import AgentResult, SceneAgent
from .assets import AssetCatalog, AssetEntry
from .llm import LLMClient, LLMConfig, LLMResponse, ToolCall, load_llm_config
from .mjcf import FragmentCache, create_base_xml, inspect_xml_summary, merge_fragment
from .randomizer import RandomizedVariant, randomize_xml
from .tools import SceneToolbox, SceneWorkspace
from .trace import AgentTrace, export_sft_jsonl
from .verifier import VerificationReport, encode_png, verify_mujoco_xml

__all__ = [
    "AgentResult",
    "AgentTrace",
    "AssetCatalog",
    "AssetEntry",
    "FragmentCache",
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "RandomizedVariant",
    "SceneAgent",
    "SceneToolbox",
    "SceneWorkspace",
    "ToolCall",
    "VerificationReport",
    "create_base_xml",
    "encode_png",
    "export_sft_jsonl",
    "inspect_xml_summary",
    "load_llm_config",
    "merge_fragment",
    "randomize_xml",
    "verify_mujoco_xml",
]
