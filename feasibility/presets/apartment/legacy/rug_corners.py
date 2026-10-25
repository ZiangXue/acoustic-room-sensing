"""Zoomed view of the rug with the chosen block corners P1..P8, for the measuring session."""
import cv2
import sys

sys.path.insert(0, "tools")

PTS = [(928, 736), (830, 712), (880, 748), (761, 822), (716, 874), (500, 771), (600, 733), (692, 785)]
img = cv2.imread("data/sample_frame.jpg")
x0, y0, x1, y1 = 470, 690, 960, 900
s = 3.3
c = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
for i, (x, y) in enumerate(PTS, 1):
    X, Y = int((x - x0) * s), int((y - y0) * s)
    cv2.drawMarker(c, (X, Y), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
    cv2.putText(c, f"P{i}", (X + 12, Y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
import csv
for r in csv.DictReader(open("measurements/rug_distances.csv")):
    a, b = PTS[int(r["a"][1:]) - 1], PTS[int(r["b"][1:]) - 1]
    pa, pb = [(int((p[0] - x0) * s), int((p[1] - y0) * s)) for p in (a, b)]
    cv2.line(c, pa, pb, (0, 128, 255), 2)
    cv2.putText(c, r["inches"] + " in", ((pa[0] + pb[0]) // 2 + 8, (pa[1] + pb[1]) // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 255), 2)
cv2.imwrite("data/rug_corners.jpg", c)
