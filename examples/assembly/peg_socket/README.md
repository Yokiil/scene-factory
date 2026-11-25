# Original peg/socket fixture

This directory contains SceneFactory's original, MIT-licensed rigid-body assembly fixture. The four fixture files are copied byte-for-byte from the authored experiment inputs; their geometry and physical parameters have not been adjusted for delivery.

- `scene.usda`: self-contained OpenUSD geometry, collision shapes and rigid-body properties. It has no external USD references. A Franka robot is loaded separately by the Isaac runtime and is not bundled in this fixture.
- `layout.json`: object identities, initial poses, robot mounting pose and assembly task reference.
- `assembly_spec.json`: insertion depth, pose/force thresholds, holding time and release requirements.
- `SOURCE.json`: original asset provenance, MIT licensing, geometry parameters and the USD SHA-256.

The cylindrical peg has a 10 mm radius and 100 mm length. The square socket has a 10.75 mm half-width, giving a minimum radial clearance of 0.75 mm. Units are metres, kilograms, seconds and radians unless a field states otherwise; quaternions use `wxyz` order.

Asset authoring and a readable USD file do not establish successful robotic assembly. `SOURCE.json` records the asset-authoring stage only. For the actual trial outcome, success/failure classification, force evidence and limitations, consult the final delivery report and the corresponding recorded assembly-trial evidence. Synthetic trace tests are not physical acceptance. Linux GPU deployment remains unverified until it is run on a target server.

The fixture can be regenerated with `tools/build_assembly_fixture.py` using a Python interpreter with OpenUSD (`pxr`). Real robot trials additionally require the compatible Isaac runtime and separately available Franka assets described in the deployment documentation.
