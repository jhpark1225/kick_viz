"""영상 분석 파이프라인에서 주고받는 데이터 구조."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseSeries:
    """영상(또는 합성)에서 얻은 원시 랜드마크. 이미지 픽셀 좌표(y 아래로 증가).

    xy[name]: (n, 2) 픽셀 좌표, vis[name]: (n,) visibility (0~1).
    """
    xy: dict[str, np.ndarray]
    vis: dict[str, np.ndarray]
    width: int
    height: int
    meta_fps: float
    source: str = "video"

    @property
    def n(self) -> int:
        return len(next(iter(self.xy.values())))


@dataclass
class Processed:
    """전처리된 랜드마크. 몸 기준 좌표계(m, 위 = +y, 차는 방향 = +x, 지면 ≈ 0, r_ = 차는 다리)."""
    t: np.ndarray
    xy: dict[str, np.ndarray]
    vis: dict[str, np.ndarray]
    angles: dict[str, np.ndarray]
    fps: float
    H: float
    kick_side: str                      # "R" 또는 "L" (원래 영상에서 차는 발)
    flipped_x: bool                     # 왼쪽으로 차는 영상이라 x를 반전했는지
    px_per_m: float
    cutoff_hz: float | None             # 적용한 저역통과 차단 주파수 (None = 필터 생략)
    xy_px: dict[str, np.ndarray]        # 필터된 픽셀 좌표 (좌우 교환 후, 오버레이용)
    xy_unfiltered: dict[str, np.ndarray]  # 보간만 한 몸 기준 좌표 (임팩트 검출용, 필터 번짐 방지)
    warnings: list[tuple[str, str]] = field(default_factory=list)   # (수준, 문구)
    view: str = "side"                  # "side"(측면, 시상면) 또는 "rear"(후면, 관상면)

    @property
    def n(self) -> int:
        return len(self.t)
