from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np
import cv2


def _compute_mpp_2d(roi_poly: np.ndarray,
                    world_height: float,
                    world_width: float,
                    scale: float = 1.0) -> Tuple[float, float]:
    """
    Meniru logika compute_average_mpp pada estimasi_kiri:
    - Ambil 4 titik A,B,C,D dari ROI
    - Hitung panjang sisi-sisi AB, BC, CD, DA
    - Hitung mpp_w dari sisi horizontal (BC & DA) dengan world_width
    - Hitung mpp_h dari sisi vertikal (AB & CD) dengan world_height
    - Ambil rata-rata mpp_w & mpp_h -> mpp_avg
    - Kembalikan mpp_x, mpp_y sama (isotropik) seperti meters_per_pixel di estimasi_kiri
    """
    pts = roi_poly[0].astype(float)

    if pts.shape[0] >= 4:
        A, B, C, D = pts[0], pts[1], pts[2], pts[3]

        dist_AB = float(np.hypot(B[0] - A[0], B[1] - A[1]))
        dist_BC = float(np.hypot(C[0] - B[0], C[1] - B[1]))
        dist_CD = float(np.hypot(D[0] - C[0], D[1] - C[1]))
        dist_DA = float(np.hypot(A[0] - D[0], A[1] - D[1]))

        mpp_w = world_width  / max(1e-6, (dist_BC + dist_DA) / 2.0)
        mpp_h = world_height / max(1e-6, (dist_AB + dist_CD) / 2.0)
        mpp_avg = (mpp_w + mpp_h) / 2.0
    else:
        mpp_avg = 0.0

    if (not np.isfinite(mpp_avg)) or mpp_avg <= 0:
        # fallback seperti di estimasi_kiri
        mpp_avg = 0.027

    mpp_scaled = mpp_avg * scale
    # Isotropik: x dan y sama, seperti meters_per_pixel tunggal di estimasi_kiri
    return mpp_scaled, mpp_scaled

@dataclass
class SpeedEstimator:
    ROI1: np.ndarray
    ROI2: np.ndarray
    world_height: float
    world_width: float
    scale: float
    use_union_mask: bool = True

    mpp1_x: float = field(init=False)
    mpp1_y: float = field(init=False)
    mpp2_x: float = field(init=False)
    mpp2_y: float = field(init=False)

    roi2_stats: Dict[int, Dict[str, float | int | bool | None]] = field(default_factory=dict)

    def __post_init__(self):
        self.mpp1_x, self.mpp1_y = _compute_mpp_2d(
            self.ROI1, self.world_height, self.world_width, self.scale
        )
        self.mpp2_x, self.mpp2_y = _compute_mpp_2d(
            self.ROI2, self.world_height, self.world_width, self.scale
        )

    def make_roi_mask(self, shape_hw):
        mask = np.zeros(shape_hw, np.uint8)
        cv2.fillPoly(mask, self.ROI1, 255)
        if self.use_union_mask:
            cv2.fillPoly(mask, self.ROI2, 255)
        return mask

    def is_inside_roi1(self, pt):
        return cv2.pointPolygonTest(self.ROI1[0], pt, False) >= 0

    def is_inside_roi2(self, pt):
        return cv2.pointPolygonTest(self.ROI2[0], pt, False) >= 0

    def is_inside_any(self, pt):
        return self.is_inside_roi1(pt) or self.is_inside_roi2(pt)

    def active_mpp(self, pt_xy):
        x, y = pt_xy
        if self.use_union_mask:
            if self.is_inside_roi1((x, y)):
                return self.mpp1_x, self.mpp1_y
            if self.is_inside_roi2((x, y)):
                return self.mpp2_x, self.mpp2_y
        return self.mpp1_x, self.mpp1_y

    def speed_2d(self, prev_xy, now_xy, dt, pt_region):
        """
        Secara konsep sama dengan:
        - disp_px = hypot(dx, dy)
        - v_m_s   = (disp_px / dt) * meters_per_pixel
        - v_kmh   = v_m_s * 3.6
        Di sini meters_per_pixel digantikan mpp_x == mpp_y (isotropik).
        """
        mpp_x, mpp_y = self.active_mpp(pt_region)

        dx_px = now_xy[0] - prev_xy[0]
        dy_px = now_xy[1] - prev_xy[1]

        dx_m = dx_px * mpp_x
        dy_m = dy_px * mpp_y

        dist_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
        v_mps = dist_m / max(1e-6, dt)
        return v_mps * 3.6

    def _update_roi2_stats(self, tid, in_roi2, last_speed):
        if in_roi2 and last_speed is not None:
            s = self.roi2_stats.get(
                tid,
                {"sum": 0.0, "n": 0, "avg": None, "finalized": False}
            )
            s["sum"] += float(last_speed)
            s["n"] += 1
            s["avg"] = s["sum"] / s["n"]
            self.roi2_stats[tid] = s

        elif (not in_roi2) and tid in self.roi2_stats and not self.roi2_stats[tid].get("finalized", False):
            self.roi2_stats[tid]["finalized"] = True

    def get_roi2_avg(self, tid):
        v = self.roi2_stats.get(tid, {}).get("avg")
        return float(v) if v is not None else None

    def label_for(self, tid, pt_xy, last_speed_kmh):
        in_roi2 = self.is_inside_roi2(pt_xy)
        in_roi1 = self.is_inside_roi1(pt_xy)

        self._update_roi2_stats(tid, in_roi2, last_speed_kmh)

        label = f"ID {tid+1}"
        if in_roi2 and last_speed_kmh is not None:
            label += f" | {last_speed_kmh:.1f} km/h"
        elif in_roi1:
            avg = self.get_roi2_avg(tid)
            label += f" | Avg {avg:.1f} km/h" if avg else " | Avg -"
        return label

    def exit_log(self, tr):
        tid = tr.id

        if tid in self.roi2_stats and self.roi2_stats[tid].get("avg") is not None:
            s = self.roi2_stats[tid]
            return f"[EXIT] ID {tid+1:02d} | ROI2_avg={s['avg']:6.1f} km/h | nROI2={s['n']}"

        elif getattr(tr, "speed_history", None):
            avg_all = float(np.mean(tr.speed_history))
            return f"[EXIT] ID {tid+1:02d} | avg_all_zone={avg_all:6.1f} km/h | n={len(tr.speed_history)} sampel"

        else:
            return f"[EXIT] ID {tid+1:02d} | tidak ada sampel kecepatan valid"
