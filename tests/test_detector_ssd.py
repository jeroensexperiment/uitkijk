import numpy as np

from camsvc.detector import ObjectDetector


def _ssd_detector(boxes, classes, scores):
    det = ObjectDetector.__new__(ObjectDetector)
    det.input_size = det._ssd_size = 320
    import threading
    det._lock = threading.Lock()
    det._ssd = lambda rgb: (np.array(boxes, np.float32), np.array(classes), np.array(scores, np.float32))
    return det


def test_ssd_classes_and_boxes():
    # 640x320 frame: letterboxed into 320x160 with 80 px padding top and bottom
    det = _ssd_detector(boxes=[[0.25, 0.0, 0.75, 0.5],    # person, left half
                               [0.25, 0.5, 0.75, 1.0],    # dog (SSD id 17 = COCO-80 16)
                               [0.25, 0.5, 0.75, 1.0]],   # traffic light: not used
                        classes=[0, 17, 9], scores=[0.8, 0.7, 0.9])
    found = det.detect(np.zeros((320, 640, 3), np.uint8), ["person", "animal", "vehicle"], 0.5)
    assert [(d.label, round(d.score, 1)) for d in found] == [("person", 0.8), ("animal", 0.7)]
    assert found[0].box == (0, 0, 320, 319)
    assert found[1].box == (320, 0, 639, 319)


def test_ssd_min_score_and_labels():
    det = _ssd_detector(boxes=[[0.3, 0.3, 0.6, 0.6]] * 2, classes=[0, 2], scores=[0.4, 0.9])
    found = det.detect(np.zeros((320, 320, 3), np.uint8), ["person"], 0.5)
    assert found == []
