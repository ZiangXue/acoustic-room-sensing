"""Draw the candidate walking area and measurement landmarks on the sample frame.

Pixel positions are eyeballed from the frame, only meant to show the user what to measure.
"""
import cv2
import numpy as np

import csv


def draw_distances(img, pts, scale=1.0, offset=(0, 0), thick=2):
    """Draw each measured pair from measurements/rug_distances.csv as a line with its length."""
    with open("measurements/rug_distances.csv") as f:
        for r in csv.DictReader(f):
            a = pts[int(r["a"][1:]) - 1]
            b = pts[int(r["b"][1:]) - 1]
            pa = (int((a[0] - offset[0]) * scale), int((a[1] - offset[1]) * scale))
            pb = (int((b[0] - offset[0]) * scale), int((b[1] - offset[1]) * scale))
            cv2.line(img, pa, pb, (0, 128, 255), thick)
            mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            label = f'{r["inches"]} in'
            cv2.putText(img, label, (mid[0] + 6, mid[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * max(scale, 1) ** 0.5, (0, 128, 255), 2)

img = cv2.imread("data/sample_frame.jpg")
over = img.copy()

# candidate walking area: all open floor incl. the rug, up to the desk edge and the right wall
area = np.array([(630, 565), (880, 578), (1030, 612), (1330, 628), (1520, 662), (1600, 760),
                 (1560, 900), (1490, 1050), (1230, 1075), (1000, 1055), (760, 900),
                 (500, 940), (335, 835)])
cv2.fillPoly(over, [area], (0, 200, 0))
img = cv2.addWeighted(over, 0.28, img, 0.72, 0)
cv2.polylines(img, [area], True, (0, 255, 0), 3)
cv2.putText(img, "walking area", (1150, 1040), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

# O = suggested origin, x along the back wall, y toward the camera
O = (877, 572)
cv2.arrowedLine(img, O, (1020, 604), (0, 0, 255), 4, tipLength=0.15)
cv2.arrowedLine(img, O, (905, 700), (255, 0, 0), 4, tipLength=0.15)
cv2.putText(img, "x", (1030, 625), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
cv2.putText(img, "y", (870, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

# fixed landmarks (yellow): origin and two rug corners
marks = {
    "O": (877, 572),   # bottom-left of the centre door frame, on the floor
    "R1": (688, 623),  # rug corner, back
    "R2": (962, 735),  # rug corner, right
}
for k, (x, y) in marks.items():
    cv2.circle(img, (x, y), 11, (0, 255, 255), 3)
    cv2.putText(img, k, (x + 12, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

# rug pattern points (cyan): distinctive corners of the rug's blocks, no sticker needed
# exact corners of white blocks, found by contour detection (tools/rug_corners.py)
rug_pts = [(928, 736), (830, 712), (880, 748), (761, 822), (716, 874), (500, 771), (600, 733), (692, 785)]
for i, (x, y) in enumerate(rug_pts, 1):
    cv2.circle(img, (x, y), 9, (255, 255, 0), -1)
    cv2.putText(img, f"P{i}", (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

draw_distances(img, rug_pts, thick=3)

# sticker positions (magenta), on bare floor only
stickers = [(700, 585), (860, 605), (1000, 700), (1200, 690), (1400, 665), (1520, 700),
            (900, 800), (1100, 790), (1300, 780), (1480, 830), (930, 940), (1150, 930),
            (1380, 960), (1500, 1000)]
for i, (x, y) in enumerate(stickers, 1):
    cv2.circle(img, (x, y), 9, (255, 0, 255), -1)
    cv2.putText(img, f"S{i}", (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

cv2.imwrite("data/sample_annotated.jpg", img)
print("wrote data/sample_annotated.jpg")
