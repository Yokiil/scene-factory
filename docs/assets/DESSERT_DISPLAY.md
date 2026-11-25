# Three-tier dessert display

The `three_tier_dessert_display` recipe recreates the reference composition as
43 separate scene objects: one wooden stand, 24 strawberry petit fours on the
lower tray, 12 cream cupcakes on the middle tray, and six pink mini cakes on
the upper tray. A rounded inverted-U frame has two vertical uprights passing
through the middle of the trays, with space between the cakes for both rods.
The assets are original procedural geometry with matching USD
and GLB visuals. The pleated wrappers, piped cream, berries, and petals are modelled shapes, not a
photogrammetric copy of the photo.

The stand registers `lower`, `middle`, and `upper` support surfaces. Every cake
has a fixed pose and support reference, so the arrangement is deterministic and
can be edited one item at a time. Collision layers use simple convex parts;
frosting and fruit detail is visual only. Robot pickup and assembly have not
been validated.

Generate and inspect the scene:

```powershell
python -m scene_factory build --recipe three_tier_dessert_display --seed 42 `
  --output <ascii-dir>\dessert_scene --usd
```

Use an ASCII-only output directory on Windows for OpenUSD. The webpage also
has a **三层甜点架** example for interactive GLB preview.

The authored assets can be regenerated with
`python tools/build_dessert_display.py`. The generator resets changed assets
to `normalized`; they become selectable again only after the real Isaac/PhysX
acceptance command succeeds:

```powershell
<isaac-python> tools/validate_kitchen_dining_assets.py `
  --pack-tag dessert_display_pack --output <ascii-dir>\dessert_qa --promote
```

To check the assembled scene after USD export, run
`<isaac-python> tools/validate_dessert_scene.py <ascii-dir>\dessert_scene\scene.usd
--report <ascii-dir>\dessert_scene\settling_report.json`. This separate
PhysX check measures cake drift and speed after 240 steps; it is not a robot
assembly task.
