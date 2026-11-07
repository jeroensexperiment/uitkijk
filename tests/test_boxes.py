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


def test_event_writes_boxes_next_to_the_clip(tmp_path):
    """The whole path: frames with a person -> event -> clip saved with <clip>.boxes."""
    import json

    import numpy as np

    from camsvc.camera import Camera
    from camsvc.config import load_config
    from camsvc.detector import Detection
    from camsvc.events import EventStore

    class FakeDetector:
        accelerated = False
        device = backend = "test"

        def detect(self, bgr, labels, min_score):
            return [Detection("person", 0.9, (100, 50, 160, 300), "person")] if "person" in labels else []

    clips, state = tmp_path / "clips", tmp_path / "state"
    state.mkdir()
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"storage: {{clips_dir: {clips}, state_dir: {state}}}\n"
                        "cameras:\n  - name: oprit\n    main_url: rtsp://x/1\n")
    cfg = load_config(cfg_file)
    cam = Camera(cfg.cameras[0], clips, state, FakeDetector(), EventStore(clips, ""), None, {}, lambda: None)
    bgr = np.zeros((360, 640, 3), np.uint8)
    t0 = 1000.0
    for i in range(12):                                  # 6 s of frames with motion and a person
        cam._on_frame(t0 + i * 0.5, True, bgr)
    ev = cam.event
    assert ev is not None and ev.boxes, "the event has no boxes"
    cam._end_event()
    cam.recorder.clip_starts[ev.id] = t0 - 5.0          # the recorder started with 5 s pre-buffer
    part = tmp_path / "part.mp4.part"
    part.write_bytes(b"x" * 100)
    cam._on_clip_done(ev.id, part, 30.0)
    saved = list(clips.rglob("*.boxes"))
    assert len(saved) == 1, list(clips.rglob("*"))
    data = json.loads(saved[0].read_text())
    first_t, first_boxes = data["frames"][0]
    assert 4.9 <= first_t <= 11.1
    assert first_boxes[0][:3] == ["person", "person", 0.9]
    assert first_boxes[0][3:7] == [round(100 / 640, 3), round(50 / 360, 3), 0.25, round(300 / 360, 3)]


def test_face_size_estimate():
    from camsvc.camera import face_size_estimate
    # far away: 20x60 px on a 640x360 sub stream, 1920x1080 main -> face ~26 px: not yet worth a try
    assert face_size_estimate((300, 100, 320, 160), (360, 640), (1080, 1920)) < 30
    # close by, upper body behind a hedge: wide box -> face well over 36 px
    assert face_size_estimate((200, 50, 290, 200), (360, 640), (1080, 1920)) >= 80
