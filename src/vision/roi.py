import cv2
import numpy as np

def fill_roi(overlay, roi_points, color=(0, 255, 0), spacing=10):
    mask = np.zeros_like(overlay)
    cv2.fillPoly(mask, [roi_points], (255, 255, 255))

    stripes = np.zeros_like(overlay, dtype=np.uint8)
    for x in range(-overlay.shape[0], overlay.shape[1], spacing):
        cv2.line(stripes, (x, 0), (x + overlay.shape[0], overlay.shape[0]), color, 1)

    striped_area = cv2.bitwise_and(stripes, mask)
    blended = cv2.addWeighted(overlay, 1.0, striped_area, 0.4, 0)

    cv2.polylines(blended, [roi_points], True, color, 1)
    return blended

def label_points(frame, roi_points, labels, color=(255, 255, 255), offsets=None):
    if offsets is None:
        offsets = [(10, -10)] * len(roi_points)

    for (x, y), label, (dx, dy) in zip(roi_points, labels, offsets):
        cv2.rectangle(frame, (x-3, y-3), (x+3, y+3), color, -1)
        cv2.putText(frame, label, (x + dx, y + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def draw_roi(overlay, roi1, roi2):
    overlay = fill_roi(overlay, roi1[0], color=(0, 255, 0), spacing=12)
    overlay = fill_roi(overlay, roi2[0], color=(255, 0, 0), spacing=10)

    labels_roi1 = ['A', 'B', 'E', 'F']
    labels_roi2 = ['', 'C', 'D', '']

    offsets_roi1 = [(-30, 10), (-30, 10), (30, 10), (30, 10)]
    offsets_roi2 = [(-30, -10), (-30, -10), (30, -10), (30, -10)]

    overlay = label_points(overlay, roi1[0], labels_roi1, color=(255, 255, 0), offsets=offsets_roi1)
    overlay = label_points(overlay, roi2[0], labels_roi2, color=(255, 255, 0), offsets=offsets_roi2)

    return overlay