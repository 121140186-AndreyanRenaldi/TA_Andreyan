import cv2
import numpy as np

from vision.roi import draw_roi
from vision.tracking import MultiObjectTracker
from config.chooser import choose_scheme

WINDOW_NAME = "Estimasi Kecepatan Lucas-Kanade"


def main():
    cfg = choose_scheme()

    # -------------------------
    # Pilih estimator
    # -------------------------
    if cfg.scheme_type == "1d":
        from estimation.estimation_1d import SpeedEstimator
        estimator = SpeedEstimator(
            ROI1=cfg.roi1,
            ROI2=cfg.roi2,
            world_distance=cfg.height,   # Jarak vertikal
            scale=cfg.scale,
            use_union_mask=cfg.use_union_mask
        )
    elif cfg.scheme_type == "2d":
        from estimation.estimation_2d import SpeedEstimator
        estimator = SpeedEstimator(
            ROI1=cfg.roi1,
            ROI2=cfg.roi2,
            world_height=cfg.height,
            world_width=cfg.width,
            scale=cfg.scale,
            use_union_mask=cfg.use_union_mask
        )
    else:
        raise ValueError(f"Tipe estimasi tidak dikenal: {cfg.scheme_type}")

    # -------------------------
    # Open video
    # -------------------------
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise SystemExit(f"Gagal membuka video: {cfg.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    dt = 1.0 / fps

    print(f"[INFO] Skema: {cfg.name}")

    # -------------------------
    # First frame
    # -------------------------
    ret, first_frame = cap.read()
    if not ret:
        raise SystemExit("Video kosong")

    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    roi_mask = estimator.make_roi_mask(first_frame.shape[:2])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    inside_fn = estimator.is_inside_any if cfg.use_union_mask else estimator.is_inside_roi1

    # -------------------------
    # Tracker
    # -------------------------
    mot = MultiObjectTracker(
        dt=dt,
        match_distance_px=cfg.match_distance,
        max_track_lost=cfg.max_lost,
        inside_region=inside_fn
    )

    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # -------------------------
    # Loop
    # -------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Foreground mask
        fg = bg.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=2)
        fg = cv2.bitwise_and(fg, roi_mask)

        # Contour detection
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for c in contours:
            if cv2.contourArea(c) < cfg.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if inside_fn((cx, cy)):
                detections.append({"centroid": (cx, cy), "bbox": (x, y, w, h)})

        active_ids, exited_tracks = mot.update(detections)

        for tr in exited_tracks:
            print(estimator.exit_log(tr))

        vis = draw_roi(frame.copy(), cfg.roi1, cfg.roi2)

        # -------------------------
        # Speed estimation
        # -------------------------
        for tr in mot.tracks:
            x, y = tr.get_position().astype(float)

            if cfg.scheme_type == "1d":
                # -------------------------
                # Mode 1D: gunakan update_flow_and_speed biasa
                # -------------------------
                region_mpp = estimator.region_mpp((x, y))
                tr.update_flow_and_speed(
                    gray_prev, gray_curr,
                    dt=dt,
                    mpp_vertical=region_mpp,
                    min_pts=6,
                    vmax_kmh=200.0
                )
                last_speed = tr.last_speed_kmh

            else:
                # -------------------------
                # Mode 2D: hitung dx+dy
                # -------------------------
                if tr.prev_points is not None and len(tr.prev_points) > 0:
                    old = tr.prev_points.reshape(-1, 2).mean(axis=0)
                    new = np.array([x, y], dtype=float)
                    last_speed = estimator.speed_2d(old, new, dt, (x, y))
                else:
                    tr.seed_features(gray_prev)
                    last_speed = None

                tr.last_speed_kmh = last_speed

            # Draw keypoints
            if estimator.is_inside_roi2((x, y)):
                if tr.prev_points is not None:
                    for p in tr.prev_points.reshape(-1, 2):
                        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

            # Draw label
            xi, yi = int(x), int(y)
            label = estimator.label_for(tr.id, (xi, yi), last_speed)

            cv2.circle(vis, (xi, yi), 5, (0, 255, 255), -1)
            cv2.putText(vis, label, (xi - 10, yi - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow(WINDOW_NAME, vis)
        gray_prev = gray_curr

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
