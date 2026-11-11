"""원본 대조용 확대 그림 만들기 (yolo_service.py 실행 후 사용)

    .venv\\Scripts\\python.exe make_figures.py
좌표는 버스 예제 사진(input/bus.jpg, 810x1080) 기준이다.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import yolo_service as ys

OUT = ys.ROOT / "outputs" / "yolo" / "bus"
g = ys.imread(ys.ROOT / "input" / "bus.jpg")
res = json.loads((OUT / "conf0.25" / "result.json").read_text(encoding="utf-8"))
lab = ys.imread(OUT / "conf0.25" / "mask_label.png")[:, :, 0]
bus_v = next(i["mask_value"] for i in res["segmentation"]["instances"] if i["class_name"] == "bus")

# 1) 왼쪽 사람들: 원본 / 박스 / 마스크
o = ys.imread(OUT / "conf0.25" / "masks_overlay.jpg")
b = ys.imread(OUT / "conf0.25" / "boxes.jpg")
crop = lambda im: im[380:910, 0:360]
ys.imwrite(OUT / "zoom_left_people.jpg",
           ys.side_by_side([crop(g), crop(b), crop(o)], ["original", "boxes", "masks"], height=520))

# 2) 두 사람 사이로 보이는 버스 부분: 저장된 mask_label.png 로 다시 칠함
y0, y1, x0, x1 = 520, 740, 150, 330
col = g.copy()
for inst in res["segmentation"]["instances"]:
    sel = lab == inst["mask_value"]
    col[sel] = (0.5 * g[sel] + 0.5 * np.array(ys.color_for(inst["class_id"]))).astype(np.uint8)
bus = cv2.cvtColor(((lab == bus_v) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
ys.imwrite(OUT / "zoom_between_people.jpg",
           ys.side_by_side([g[y0:y1, x0:x1], col[y0:y1, x0:x1], bus[y0:y1, x0:x1]],
                           ["original", "overlay (saved mask)", "bus mask only"], height=400))

# 3) 놓친/걸러진 대상
ys.imwrite(OUT / "zoom_missed.jpg",
           ys.side_by_side([g[220:360, 0:120], g[640:760, 200:320]],
                           ["sign: 'stop sign' 0.13", "phone: no candidate"], height=300))
print("ok")
