import cv2
import numpy as np
from typing import Callable

feature_params = dict(
    maxCorners=20, 
    qualityLevel=0.5,
    minDistance=5, 
    blockSize=10
)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
)

def create_kalman(dt: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], np.float32
    )
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)

    q = 1e-2
    dt2 = dt * dt              
    dt3 = dt * dt * dt         
    dt4 = dt2 * dt2           
    Q = np.array(
        [[dt4/4,   0,       dt3/2,   0    ],
         [0,       dt4/4,   0,       dt3/2],
         [dt3/2,   0,       dt2,     0    ],
         [0,       dt3/2,   0,       dt2  ]], np.float32
    ) * q

    kf.processNoiseCov = Q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1e2
    return kf

class Track:
    _next_id = 0

    def __init__(self, 
        centroid: tuple[int, int], dt: float,
        bbox: tuple[int, int, int, int] | None = None):
        
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = create_kalman(dt)
        self.kf.statePost = np.array(
            [[float(centroid[0])],[float(centroid[1])],[0], [0]], np.float32
        ) 
        self.frames_lost = 0
        self.bbox = bbox
        self.prev_points: np.ndarray | None = None
        self.last_speed_kmh: float | None = None
        self.speed_history: list[float] = []
        self.alpha = 0.3

    def predict(self) -> np.ndarray:
        s = self.kf.predict()
        return np.array([s[0, 0], s[1, 0]], np.float32)

    def correct(self, meas_xy: tuple[int, int]) -> None:
        self.kf.correct(
            np.array([[float(meas_xy[0])], [float(meas_xy[1])]], np.float32)
        )
        self.frames_lost = 0

    def get_position(self) -> np.ndarray:
        s = self.kf.statePost
        return np.array([s[0, 0], s[1, 0]], np.float32)

    def set_bbox(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox

    def _bbox_mask(self, img_shape: tuple[int, int], pad: int = 6) -> np.ndarray:
        mask = np.zeros(img_shape, np.uint8)
        if self.bbox is None:
            return mask
        x, y, w, h = self.bbox
        cv2.rectangle(mask, (x - pad, y - pad), (x + w + pad, y + h + pad), 255, -1)
        return mask

    def seed_features(self, gray_img: np.ndarray) -> None:
        mask = self._bbox_mask(gray_img.shape)
        if mask.sum() == 0:
            self.prev_points = None
            return
        p = cv2.goodFeaturesToTrack(gray_img, mask=mask, **feature_params)
        self.prev_points = p.astype(np.float32) if p is not None else None

    def update_flow_and_speed(self, prev_gray: np.ndarray, curr_gray: np.ndarray, dt: float,
                              mpp_vertical: float, min_pts: int = 6, vmax_kmh: float = 200.0) -> None:
        if self.prev_points is None or len(self.prev_points) < min_pts:
            self.seed_features(prev_gray)
            return

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, self.prev_points, None, **lk_params
        )
        if p1 is None:
            self.prev_points = None
            return

        good_new = p1[st == 1]
        good_old = self.prev_points[st == 1]
        if good_new is None or len(good_new) < min_pts:
            self.prev_points = None
            return

        dy = good_old[:, 1] - good_new[:, 1]
        v_m_s = (np.abs(dy) / dt) * mpp_vertical
        v_kmh = v_m_s * 3.6

        if len(v_kmh) >= 3:
            q1, q3 = np.percentile(v_kmh, [25, 75])
            iqr = q3 - q1
            v_kmh = v_kmh[(v_kmh > q1 - 1.5 * iqr) & (v_kmh < q3 + 1.5 * iqr)]

        v_med = float(np.median(v_kmh)) if len(v_kmh) > 0 else 0.0
        self.last_speed_kmh = v_med if self.last_speed_kmh is None \
            else (1 - self.alpha) * self.last_speed_kmh + self.alpha * v_med

        self.speed_history.append(self.last_speed_kmh)
        self.prev_points = good_new.reshape(-1, 1, 2).astype(np.float32)

class MultiObjectTracker:
    def __init__(self, dt: float, match_distance_px: float = 100.0, max_track_lost: int = 5,
                 inside_region: Callable[[tuple[float, float]], bool] | None = None):
        self.tracks: list[Track] = []
        self.dt = dt
        self.match_distance_px = match_distance_px
        self.max_track_lost = max_track_lost
        self.inside_region = inside_region

    def update(self, detections: list[dict]) -> tuple[list[int], list[Track]]:
        exited: list[Track] = []
        if not self.tracks and not detections:
            return [], exited

        predictions = [tr.predict() for tr in self.tracks]
        assigned_t, assigned_d = set(), set()

        for d_idx, det in enumerate(detections):
            det_xy = np.array(det["centroid"], np.float32)
            best_t, best_dist = None, self.match_distance_px
            for t_idx, pred in enumerate(predictions):
                if t_idx in assigned_t:
                    continue
                dist = np.linalg.norm(pred - det_xy)
                if dist < best_dist:
                    best_dist = dist
                    best_t = t_idx
            if best_t is not None:
                tr = self.tracks[best_t]
                tr.correct(det["centroid"])
                tr.set_bbox(det["bbox"])
                assigned_t.add(best_t)
                assigned_d.add(d_idx)

        for d_idx, det in enumerate(detections):
            if d_idx not in assigned_d:
                self.tracks.append(Track(det["centroid"], self.dt, bbox=det["bbox"]))

        survivors: list[Track] = []
        for t_idx, tr in enumerate(self.tracks):
            if t_idx not in assigned_t and t_idx < len(predictions):
                tr.frames_lost += 1

            x, y = tr.get_position()
            if self.inside_region is not None and not self.inside_region((x, y)):
                tr.frames_lost = self.max_track_lost + 1

            if tr.frames_lost <= self.max_track_lost:
                survivors.append(tr)
            else:
                exited.append(tr)

        self.tracks = survivors
        return [tr.id for tr in self.tracks], exited