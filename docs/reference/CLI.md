# CLI Reference

The installed command is `scene-factory`. Every command supports `--help`.

## Commands

```text
scene-factory list-recipes
scene-factory doctor [--json]
scene-factory build --recipe NAME --output PATH --blender
scene-factory blender render PATH/to/blender_manifest.json [--blender-exe PATH] [--no-png]
scene-factory build --recipe NAME --seed INT --output PATH
scene-factory build --intent PATH --seed INT --output PATH
scene-factory build --recipe NAME --seed INT --output PATH --no-mjcf
scene-factory batch --recipe NAME --count INT --seed-start INT --output PATH
scene-factory batch --intent PATH --count INT --seed-start INT --output PATH
scene-factory-web --host 127.0.0.1 --port 8765 --output PATH
scene-factory intent validate PATH
scene-factory intent inspect PATH
scene-factory intent schema
scene-factory dataset inspect PATH
scene-factory dataset validate PATH
scene-factory dataset reproduce PATH
scene-factory task plan --scene PATH --object ID --state NAME --output PATH
scene-factory task validate --scene PATH --plan PATH
scene-factory task replay --scene PATH --plan PATH
scene-factory task execute --scene PATH --plan PATH --executor dry-run --output PATH
scene-factory task execution-validate --scene PATH --plan PATH --trace PATH
scene-factory executor inspect --executor dry-run
scene-factory executor conformance --executor dry-run --output PATH
scene-factory executor validate-report PATH
scene-factory asset inspect --asset-id ID
scene-factory asset normalize SOURCE --output PATH --asset-id ID --category NAME
scene-factory asset collision --collision-path PATH --status STATUS --enabled
scene-factory llm-status
scene-factory llm-test
```

`build` and `batch` export MJCF by default; use `--no-mjcf` to omit it. MJCF
export does not import MuJoCo. `build --usd`, `batch --usd`, and the Isaac asset
commands require an Isaac/USD environment. The command parser itself remains
importable in ordinary Python. `build --blender` prepares a portable visual case
(manifest, renderer script, GLB copies) without Blender or `bpy`; `blender render`
requires an installed Blender executable and produces `scene.blend` and,
unless `--no-png` is used, `render.png`. This is a static view, not robot
physics. `doctor` reports the core setup and optional Isaac, CUDA, MuJoCo, uv,
Blender, and an actual USD Stage creation probe. The core-ready status is
separate from optional runtime readiness.

When USD is requested, the scene ZIP stores a self-contained `scene/scene.usdz`
with referenced layers and textures. Open the USDZ after extracting the outer
ZIP; do not unpack its internal USD files. The separate `scene.usd` output
can still reference local assets. Missing USD dependencies fail bundle creation.
Installed USD export and editor launch use `scene_factory.isaac_export` and
`scene_factory.isaac_preview`, not scripts under the checkout's `tools` directory.

## Physics sandbox and HTTP sessions

The following commands belong to the current development branch. The three build
entry points have different inputs:

| Command | Input and result |
| --- | --- |
| `build` | A recipe or SceneIntent produces a scene layout and exports. |
| `sandbox build` | A layout produces a frozen local environment package. |
| `sandbox submit-build` | A JSON request submits an asynchronous HTTP build. |

```sh
scene-factory sandbox capabilities --backend mujoco
scene-factory sandbox schema
scene-factory sandbox build outputs/basic-scene/layout.json --output outputs/environment
scene-factory sandbox verify outputs/environment
scene-factory sandbox archive outputs/environment --output outputs/environment.zip
scene-factory sandbox serve --root outputs/service --scene-root outputs --max-sessions 1 --mujoco-worker-pool
```

Run the server in one terminal and use another terminal for requests:

```sh
scene-factory sandbox create create.json --server-url http://127.0.0.1:8765
scene-factory sandbox reset SESSION_ID reset.json --server-url http://127.0.0.1:8765
scene-factory sandbox step SESSION_ID step.json --server-url http://127.0.0.1:8765
scene-factory sandbox observe SESSION_ID --modalities state --server-url http://127.0.0.1:8765
scene-factory sandbox close SESSION_ID --server-url http://127.0.0.1:8765
scene-factory sandbox submit-build build-request.json --server-url http://127.0.0.1:8765
scene-factory sandbox build-status BUILD_ID --server-url http://127.0.0.1:8765
```

Replace `SESSION_ID` and `BUILD_ID` with returned IDs. JSON request examples and
action/observation contracts are in [SANDBOX_SERVICE.md](../sandbox/SANDBOX_SERVICE.md).
The server reads input paths within `--scene-root`; container requests must use
paths visible inside the container. `create.json` may specify `history_limit`
from 1 to 100000 (default 4096), for example 10000 for long action traces.
`service serve/create/reset/step/observe/close` and the other session operations
are aliases of their `sandbox` equivalents. Local environment commands remain
under `sandbox`.

MuJoCo supports runtime-state snapshots; Isaac currently requires a new session
for each episode. Images and action modes depend on backend capabilities.
Isaac frozen builds require `--backend isaac --usd PATH --robot-asset-root PATH`;
see [ISAAC_BUILD.md](../isaac/ISAAC_BUILD.md). Server deployment and container validation
status are documented in [SANDBOX_DEPLOYMENT.md](../sandbox/SANDBOX_DEPLOYMENT.md).

## Exit codes

- `0`: command completed successfully;
- `1`: command, configuration, runtime, or file-system error;
- `2`: validation or acceptance failure.

Machine-readable commands print JSON on successful execution and validation
failures. Human-readable `list-recipes` prints one recipe name per line.

## Resource overrides

`--registry` and `--recipes` override the default resource locations. Installed
packages resolve their resources from the platform data directory. The optional
`SCENE_FACTORY_HOME` environment variable can override the complete resource
root for controlled deployments.
