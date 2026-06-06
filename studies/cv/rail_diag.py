"""Standalone rail-fit diagnostic: dump the white-line candidates + chosen rails
on the mid start-straight frame so we can SEE what fit_rails is doing."""
import os
import cv2
import numpy as np

MAIN = "/home/ytc/Projects/hkjc-tools"
VIDEO = os.path.join(MAIN, "data_cache/video/drone_20260603_01.mp4")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def turf_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 0] > 30) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).astype(np.uint8)


def main():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(8.5 * 25))   # mid start straight
    ok, img = cap.read()
    cap.release()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    turf = turf_mask(img)
    near = cv2.dilate(turf, np.ones((25, 25), np.uint8))
    white = ((gray > 170) & (near > 0)).astype(np.uint8) * 255
    cv2.imwrite(f"{OUT}/diag_white.png", white)
    cv2.imwrite(f"{OUT}/diag_turf.png", turf * 255)
    edges = cv2.Canny(white, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=70,
                            minLineLength=int(h * 0.25), maxLineGap=60)
    vis = img.copy()
    print("n lines", 0 if lines is None else len(lines))
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = abs(abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) - 90)
            if ang < 35:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, f"{int((x1+x2)/2)}", (int((x1+x2)/2), int((y1+y2)/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(f"{OUT}/diag_lines.png", vis)
    # column-wise white density profile (rails = tall narrow peaks at turf edges)
    col = (white > 0).sum(0)
    xs = np.where(col > h * 0.2)[0]
    print("cols with strong white:", xs[:50])
    print("frame", w, "x", h)


if __name__ == "__main__":
    main()
