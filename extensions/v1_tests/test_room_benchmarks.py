from scene_factory.room_benchmarks import generate_room_scene


def test_scale_fixtures_preserve_requested_counts_and_workbench_clearance():
    for count in (20, 50, 100):
        scene = generate_room_scene(count)
        assert len(scene["objects"]) == count
        assert len({item["object_id"] for item in scene["objects"]}) == count
        assert scene["benchmark_scope"]["robot"] == "not_instantiated"
        assert scene["task"] == {}
        for item in scene["objects"]:
            if item["object_id"].startswith("background_"):
                x, y, _ = item["pose"]["position"]
                assert abs(x) > 2 or not -6 <= y <= -2
