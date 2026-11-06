from camsvc.camera import boxes_timeline, frac_box


def test_boxes_timeline_relative_to_clip_start():
    box = ["car", "vehicle", 0.9, 0.1, 0.2, 0.3, 0.4, 1]       # parked car
    entries = [(99.0, [box]),          # long before the clip: dropped
               (100.2, [box]),         # in the pre-buffer
               (105.0, []),            # gone
               (200.0, [box])]         # after the clip: dropped
    marks = [(101.0, "plate", "JBX29N", [0.1, 0.2, 0.3, 0.4]), (300.0, "face", "Sam", [0, 0, 1, 1])]
    tl = boxes_timeline(entries, clip_start=100.0, duration=30.0, marks=marks)
    assert tl["frames"] == [[0.2, [box]], [5.0, []]]
    assert tl["marks"] == [[1.0, "plate", "JBX29N", [0.1, 0.2, 0.3, 0.4]]]
    assert "marks" not in boxes_timeline(entries, 100.0, 30.0)


def test_frac_box():
    assert frac_box((64, 36, 320, 180), (360, 640)) == [0.1, 0.1, 0.5, 0.5]
