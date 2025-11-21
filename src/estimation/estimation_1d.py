from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

def _compute_vertical_mpp(roi_poly: np.ndarray, world_distance_m: float, scale: float) -> float:
    pts = roi_poly[0]
    bottom_pts = pts[np.argsort(pts[:, 1])[-2:]]
    top_pts    = pts[np.argsort(pts[:, 1])[:2]]
    y_near = float(np.mean(bottom_pts[:, 1]))
    y_far  = float(np.mean(top_pts[:, 1]))
    mpp = world_distance_m / max(1e-6, abs(y_near - y_far))
    return mpp * scale

@dataclass
class SpeedEstimator:
    ROI1: np.ndarray
    ROI2: np.ndarray
    world_distance: float        # pakai cfg.height
    scale: float                 # pakai cfg.scale
    use_union_mask: bool = True

    mpp1: float = field(init=False)
    mpp2: float = field(init=False)
    roi2_stats: Dict[int, Dict[str, float | int | bool | None]] = field(default_factory=dict)

    def __post_init__(self):
        self.mpp1 = _compute_vertical_mpp(self.ROI1, self.world_distance, self.scale)
        self.mpp2 = _compute_vertical_mpp(self.ROI2, self.world_distance, self.scale)

    def is_inside_roi1(self, pt: Tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.ROI1[0], (float(pt[0]), float(pt[1])), False) >= 0

    def is_inside_roi2(self, pt: Tuple[float, float]) -> bool:
        return cv2.pointPolygonTest(self.ROI2[0], (float(pt[0]), float(pt[1])), False) >= 0

    def is_inside_any(self, pt: Tuple[float, float]) -> bool:
        return self.is_inside_roi1(pt) or self.is_inside_roi2(pt)

    def make_roi_mask(self, shape_hw: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape_hw, np.uint8)
        cv2.fillPoly(mask, self.ROI1, 255)
        if self.use_union_mask:
            cv2.fillPoly(mask, self.ROI2, 255)
        return mask

    def region_mpp(self, pt: Tuple[float, float]) -> float:
        if self.use_union_mask:
            if self.is_inside_roi1(pt):
                return self.mpp1
            if self.is_inside_roi2(pt):
                return self.mpp2
        return self.mpp1

    def active_mpp_index(self, pt: Tuple[float, float]) -> int:
        return 2 if self.is_inside_roi2(pt) else 1

    # ------------- ROI2 averaging -------------
    def _update_roi2_stats(self, tid: int, in_roi2: bool, last_speed_kmh: Optional[float]) -> None:
        if in_roi2 and last_speed_kmh is not None:
            s = self.roi2_stats.get(tid, {"sum": 0.0, "n": 0, "avg": None, "finalized": False})
            s["sum"] += float(last_speed_kmh)
            s["n"]   += 1
            s["avg"]  = s["sum"] / s["n"]
            self.roi2_stats[tid] = s
        elif (not in_roi2) and (tid in self.roi2_stats) and (not self.roi2_stats[tid].get("finalized", False)):
            self.roi2_stats[tid]["finalized"] = True

    def get_roi2_avg(self, tid: int) -> Optional[float]:
        v = self.roi2_stats.get(tid, {}).get("avg", None)
        return float(v) if v is not None else None

    def label_for(self, tid: int, pt_xy: Tuple[int, int], last_speed_kmh: Optional[float]) -> str:
        in_roi2 = self.is_inside_roi2(pt_xy)
        in_roi1 = self.is_inside_roi1(pt_xy)

        # update statistik ROI2
        self._update_roi2_stats(tid, in_roi2, last_speed_kmh)

        label = f"ID {tid+1}"
        if in_roi2:
            if last_speed_kmh is not None:
                label += f" | {last_speed_kmh:.1f} km/h"
        elif in_roi1:
            avg = self.get_roi2_avg(tid)
            label += f" | Avg {avg:.1f} km/h" if avg is not None else " | Avg -"
        return label

    def exit_log(self, tr) -> str:
        tid = tr.id
        if tid in self.roi2_stats and self.roi2_stats[tid].get("avg") is not None:
            s = self.roi2_stats[tid]
            return f"[EXIT] ID {tid+1:02d} | ROI2_avg={s['avg']:6.1f} km/h | nROI2={s['n']}"
        elif getattr(tr, "speed_history", None):
            avg_all = float(np.mean(tr.speed_history))
            return f"[EXIT] ID {tid+1:02d} | avg_all_zone={avg_all:6.1f} km/h | n={len(tr.speed_history)} sampel"
        else:
            return f"[EXIT] ID {tid+1:02d} | tidak ada sampel kecepatan yang valid"
