from camsvc.logredact import redact


def test_password_hidden():
    line = ("[car] sub stream error: Server returned 404 Not Found: "
            "'rtsp://stream:P%40ss@10.1.1.66:554/cam/realmonitor?channel=1&subtype=1'")
    out = redact(line)
    assert "P%40ss" not in out
    assert "'rtsp://stream:***@10.1.1.66:554/cam/realmonitor?channel=1&subtype=1'" in out


def test_unencoded_at_and_plain_text_untouched():
    assert redact("rtsp://admin:a@b@cam:554/x") == "rtsp://admin:***@cam:554/x"
    assert redact("http://localhost:1984 ok, 10:30 user@example") == "http://localhost:1984 ok, 10:30 user@example"
