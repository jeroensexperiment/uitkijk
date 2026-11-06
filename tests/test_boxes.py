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


def test_head_region_is_a_square_around_the_head():
    from camsvc.camera import head_region
    # small, far-away person on a 640x360 sub stream, main stream 1920x1080 (scale 3)
    x1, y1, x2, y2 = head_region((300, 100, 320, 150), 3.0, 3.0, (1080, 1920))
    assert x2 - x1 == 400 and y2 - y1 == 400               # at least 400 px, not 60x105
    assert x1 < 930 < x2 and y1 < 300 + 22 < y2            # head: top centre of the person
    # at the edge of the image: shifted inside, not cut off
    x1, y1, x2, y2 = head_region((0, 0, 30, 60), 3.0, 3.0, (1080, 1920))
    assert (x1, y1) == (0, 0) and x2 - x1 == 400 and y2 - y1 == 400
    # big person close by: at most 640 px, so the face detector sees it unscaled
    x1, y1, x2, y2 = head_region((100, 0, 400, 360), 3.0, 3.0, (1080, 1920))
    assert x2 - x1 == 640 and y2 - y1 == 640
