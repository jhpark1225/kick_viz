"""Cloud의 네이티브 로더 실패가 UI용 오류가 되고 리소스가 정리되는지 확인."""

import builtins
import tempfile

import numpy as np
import pytest

from video.pose_extract import (PoseRuntimeError, VideoError, extract_from_bytes,
                                find_model, info_from_bytes)


@pytest.fixture
def blank_video(tmp_path):
    import cv2
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    assert writer.isOpened(), "Test video encoder unavailable"
    try:
        for _ in range(3):
            writer.write(np.full((120, 160, 3), 160, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


@pytest.mark.parametrize("error, visible", [
    (OSError("libEGL.so.1: cannot open shared object file: No such file or directory"), "libegl1"),
    (OSError("libGLESv2.so.2: cannot open shared object file: No such file or directory"), "libgles2"),
    (OSError("libGL.so.1: cannot open shared object file: No such file or directory"), "libgl1"),
    (OSError("/private/server/libc.so.6: version `GLIBC_2.38' not found"), "Manage app"),
    (RuntimeError("model initialization failed"), "Manage app"),
    (ValueError("invalid model buffer"), "Manage app"),
])
def test_native_init_failure_is_handled_and_cleaned(blank_video, tmp_path, monkeypatch,
                                                   caplog, error, visible):
    import cv2
    from mediapipe.tasks.python import BaseOptions, vision

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    captures = []
    original_capture = cv2.VideoCapture

    def capture(*args, **kwargs):
        instance = original_capture(*args, **kwargs)
        captures.append(instance)
        return instance

    def fail(options):
        assert options.base_options.delegate == BaseOptions.Delegate.CPU
        assert options.base_options.model_asset_buffer
        raise error

    monkeypatch.setattr(cv2, "VideoCapture", capture)
    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", fail)
    before = set(tmp_path.iterdir())
    with pytest.raises(PoseRuntimeError, match=visible) as caught:
        extract_from_bytes(blank_video, "내 킥.mp4", find_model())
    assert isinstance(caught.value, VideoError)  # 기존 Streamlit 예외 처리와 호환
    assert caught.value.__cause__ is error
    assert "/private/server" not in str(caught.value)
    assert str(error) in caplog.text  # 관리자 로그에는 실제 진단 원인 보존
    assert captures and all(not cap.isOpened() for cap in captures)
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("module, action", [("cv2", "metadata"), ("mediapipe", "analysis")])
def test_missing_backend_import_is_handled(blank_video, tmp_path, monkeypatch, module, action):
    original_import = builtins.__import__

    def fail_import(name, *args, **kwargs):
        if name == module:
            raise ImportError(f"No module named {module}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(builtins, "__import__", fail_import)
    before = set(tmp_path.iterdir())
    with pytest.raises(PoseRuntimeError, match="서버의 영상 분석 엔진"):
        if action == "metadata":
            info_from_bytes(blank_video, ".mp4")
        else:
            extract_from_bytes(blank_video, "kick.mp4", find_model())
    assert set(tmp_path.iterdir()) == before


def test_missing_model_is_handled(blank_video, tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    before = set(tmp_path.iterdir())
    with pytest.raises(PoseRuntimeError, match="포즈 모델 파일"):
        extract_from_bytes(blank_video, "kick.mp4", tmp_path / "missing.task")
    assert set(tmp_path.iterdir()) == before


def test_unrelated_programming_error_is_not_hidden(blank_video, monkeypatch):
    from mediapipe.tasks.python import vision

    def fail(options):
        raise TypeError("unrelated programming bug")

    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", fail)
    with pytest.raises(TypeError, match="unrelated programming bug"):
        extract_from_bytes(blank_video, "kick.mp4", find_model())
