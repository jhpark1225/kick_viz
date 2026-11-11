"""영상 → 프레임별 랜드마크 (MediaPipe Tasks PoseLandmarker, VIDEO 모드).

영상은 임시 파일로만 처리하고 분석 직후 삭제한다. 디스크에 영구 저장하거나 앱 서버 밖으로 전송하지 않는다.
"""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from video.angles_from_pose import LANDMARKS
from video.pose_series import PoseSeries

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILES = ["pose_landmarker_full.task", "pose_landmarker_heavy.task"]
MODEL_URLS = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                 "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
ALLOWED_EXT = {".mp4", ".mov", ".avi"}
MAX_BYTES = 200 * 1024 * 1024
MAX_SECONDS = 30.0
INFER_MAX_WIDTH = 1280      # 추론용 축소 폭 (랜드마크는 정규화 좌표라 원본 해상도로 복원)
THUMB_WIDTH = 360           # 이벤트 썸네일 폭 (메모리에만 JPEG로 보관)


class VideoError(ValueError):
    """사용자에게 보여줄 영상 입력 오류."""


def find_model() -> Path | None:
    """models/ 폴더의 포즈 모델 파일 (없으면 None)."""
    for name in MODEL_FILES:
        path = MODEL_DIR / name
        if path.exists():
            return path
    return None


def model_help() -> str:
    """모델 파일이 없을 때 안내 문구."""
    return ("포즈 모델 파일이 없습니다. 아래 공식 경로에서 `pose_landmarker_full.task`를 내려받아 "
            f"`{MODEL_DIR.name}/` 폴더에 넣어주세요.\n\n{MODEL_URLS['pose_landmarker_full.task']}")


def check_upload(name: str, size: int) -> None:
    """확장자·크기 검사."""
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise VideoError("mp4, mov, avi 영상만 올릴 수 있습니다.")
    if size > MAX_BYTES:
        raise VideoError("영상은 200 MB 이하만 올릴 수 있습니다.")


class TempVideo:
    """업로드 바이트를 임시 파일로 쓰고, with 블록이 끝나면 반드시 삭제한다."""

    def __init__(self, data: bytes, suffix: str):
        self.data, self.suffix, self.path = data, suffix, None

    def __enter__(self) -> str:
        fd, self.path = tempfile.mkstemp(suffix=self.suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(self.data)
        return self.path

    def __exit__(self, *exc) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


def video_info(path: str) -> dict:
    """영상 메타데이터: fps, 프레임 수, 크기, 길이(s)."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError("영상을 열 수 없습니다. 다른 형식(mp4)으로 저장해 다시 올려주세요.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    fps = fps if 1.0 <= fps <= 1000.0 else 30.0
    return {"fps": float(fps), "frames": n, "width": w, "height": h, "duration": n / fps if fps else 0.0}


def info_from_bytes(data: bytes, suffix: str) -> dict:
    """업로드 바이트의 메타데이터 (임시 파일은 바로 삭제)."""
    with TempVideo(data, suffix) as path:
        return video_info(path)


def extract_pose(path: str, model_path: Path, progress: Callable[[float], None] | None = None,
                 thumb_width: int = THUMB_WIDTH) -> tuple[PoseSeries, list[bytes]]:
    """영상 → 33개 중 사용할 17개 랜드마크(픽셀 좌표·visibility) + 프레임별 JPEG 썸네일."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    info = video_info(path)
    if info["duration"] > MAX_SECONDS + 0.5:
        raise VideoError(f"영상 길이가 {info['duration']:.1f} s입니다. {MAX_SECONDS:.0f} s 이하로 잘라서 올려주세요.")
    W, Hh, fps = info["width"], info["height"], info["fps"]
    # 경로 대신 바이트로 넘긴다: MediaPipe(C++)는 Windows에서 한글이 든 경로를 열지 못한다
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    xy = {k: [] for k in LANDMARKS}
    vis = {k: [] for k in LANDMARKS}
    thumbs: list[bytes] = []
    cap = cv2.VideoCapture(path)
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            i, last_ts = 0, -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = max(int(round(i * 1000.0 / fps)), last_ts + 1)   # VIDEO 모드: 증가하는 ms 타임스탬프
                last_ts = ts
                small = frame
                if frame.shape[1] > INFER_MAX_WIDTH:
                    s = INFER_MAX_WIDTH / frame.shape[1]
                    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                if res.pose_landmarks:
                    lms = res.pose_landmarks[0]
                    for k, idx in LANDMARKS.items():
                        lm = lms[idx]
                        xy[k].append((lm.x * W, lm.y * Hh))
                        vis[k].append(float(lm.visibility if lm.visibility is not None else 1.0))
                else:
                    for k in LANDMARKS:
                        xy[k].append((np.nan, np.nan))
                        vis[k].append(0.0)
                ts_scale = thumb_width / frame.shape[1]
                thumb = cv2.resize(frame, None, fx=ts_scale, fy=ts_scale, interpolation=cv2.INTER_AREA)
                thumbs.append(cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes())
                i += 1
                if progress and info["frames"]:
                    progress(min(i / info["frames"], 1.0))
    finally:
        cap.release()
    if not thumbs:
        raise VideoError("영상에서 프레임을 읽지 못했습니다.")
    raw = PoseSeries(xy={k: np.array(v, dtype=float) for k, v in xy.items()},
                     vis={k: np.array(v, dtype=float) for k, v in vis.items()},
                     width=W, height=Hh, meta_fps=fps)
    detected = float(np.mean(raw.vis["l_hip"] > 0))
    if detected < 0.3:
        raise VideoError("사람이 거의 검출되지 않았습니다. 전신이 보이도록 측면에서 다시 촬영해 주세요.")
    return raw, thumbs


def extract_from_bytes(data: bytes, name: str, model_path: Path,
                       progress: Callable[[float], None] | None = None) -> tuple[PoseSeries, list[bytes]]:
    """업로드 바이트 → 랜드마크. 임시 파일은 분석 후 즉시 삭제된다."""
    check_upload(name, len(data))
    with TempVideo(data, Path(name).suffix.lower()) as path:
        return extract_pose(path, model_path, progress)
