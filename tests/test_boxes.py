from camsvc.camera import boxes_timeline


def test_boxes_timeline_relative_to_clip_start():
    box = ["car", "vehicle", 0.9, 0.1, 0.2, 0.3, 0.4]
    entries = [(99.0, [box]),          # long before the clip: dropped
               (100.2, [box]),         # in the pre-buffer
               (105.0, []),            # gone
               (200.0, [box])]         # after the clip: dropped
    tl = boxes_timeline(entries, clip_start=100.0, duration=30.0)
    assert tl == {"version": 1, "frames": [[0.2, [box]], [5.0, []]]}
