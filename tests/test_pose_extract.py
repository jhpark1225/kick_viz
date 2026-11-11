import glob
import os
import tempfile

import numpy as np
import pytest

from video.pose_extract import TempVideo, VideoError, check_upload, extract_from_bytes, find_model


def test_check_upload():
    check_upload("kick.MP4", 10)
    with pytest.raises(VideoError):
        check_upload("kick.mkv", 10)
    with pytest.raises(VideoError):
        check_upload("kick.mp4", 201 * 1024 * 1024)


def test_temp_video_deleted():
    with TempVideo(b"abc", ".mp4") as path:
        assert os.path.exists(path)
    assert not os.path.exists(path)


@pytest.mark.skipif(find_model() is None, reason="models/ 에 포즈 모델 없음")
def test_no_person_video_raises_and_cleans_up(tmp_path):
    """사람이 없는 영상 → 친절한 오류, 임시 파일은 남지 않는다 (MediaPipe VIDEO 모드 연결 확인)."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    path = str(tmp_path / "blank.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (320, 180))
    for i in range(20):
        f = np.full((180, 320, 3), 200, np.uint8)
        cv2.circle(f, (50 + i * 10, 90), 15, (0, 0, 255), -1)
        vw.write(f)
    vw.release()
    data = open(path, "rb").read()
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.mp4")))
    with pytest.raises(VideoError, match="검출되지"):
        extract_from_bytes(data, "blank.mp4", find_model())
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "*.mp4"))) == before
