"""
작은 YOLO 서비스 (로컬 CPU 실행)

- yolo26n.pt      : detection  -> 박스 / 클래스 / 점수
- yolo26n-seg.pt  : segmentation -> 마스크(윤곽)
- 같은 사진을 conf=0.25, conf=0.50 두 임계값으로 돌려서 무엇이 사라지는지 비교
- outputs/yolo/ 에 결과 이미지, 박스 JSON, 마스크 PNG, 실행 코드, manifest.json 저장

실행 예:
    .venv\\Scripts\\python.exe yolo_service.py --images input\\bus.jpg input\\my_photo.jpg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DET_WEIGHTS = MODELS_DIR / "yolo26n.pt"
SEG_WEIGHTS = MODELS_DIR / "yolo26n-seg.pt"

# 관찰 대상: COCO 80개 클래스 중 "거리 사진"에서 의미 있는 것만 고름
WATCH_CLASSES = ["person", "bus", "car", "bicycle", "motorcycle", "truck",
                 "traffic light", "stop sign", "handbag", "backpack"]

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
           (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
           (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
           (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132)]


# ---------- 한글 경로에서도 안전한 이미지 읽기/쓰기 (cv2.imread 는 한글 경로를 못 읽음) ----------
def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 열 수 없습니다: {path}")
    return img


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise ValueError(f"이미지를 저장할 수 없습니다: {path}")
    buf.tofile(str(path))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def color_for(i: int):
    return PALETTE[i % len(PALETTE)]


# ---------- 그리기 ----------
def draw_label(img, text, x1, y1, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y_top = max(0, y1 - th - base - 4)
    cv2.rectangle(img, (x1, y_top), (x1 + tw + 6, y_top + th + base + 4), color, -1)
    cv2.putText(img, text, (x1 + 3, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def draw_boxes(img, dets, dashed_ids=()):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
        color = color_for(d["class_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        draw_label(out, f'#{d["id"]} {d["class_name"]} {d["score"]:.2f}', x1, y1, color)
    return out


def title_bar(img, text):
    bar = np.full((36, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def side_by_side(images, titles, height=480):
    tiles = []
    for im, t in zip(images, titles):
        s = height / im.shape[0]
        r = cv2.resize(im, (int(im.shape[1] * s), height))
        tiles.append(title_bar(r, t))
    return np.hstack(tiles)


# ---------- 추론 ----------
def run_detection(model, img, conf, imgsz, device):
    r = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    dets = []
    for i, b in enumerate(r.boxes):
        cid = int(b.cls.item())
        dets.append({
            "id": i,
            "class_id": cid,
            "class_name": r.names[cid],
            "score": round(float(b.conf.item()), 4),
            "xyxy": [round(float(v), 1) for v in b.xyxy[0].tolist()],
        })
    return dets, r.speed


def run_segmentation(model, img, conf, imgsz, device):
    """retina_masks=True 로 원본 크기 픽셀 마스크(masks.data)를 그대로 사용한다.
    (처음 버전은 masks.xy 다각형을 다시 칠했는데, 조각난 마스크를 한 다각형으로 이으면서
     1px 짜리 가는 선이 생겨 원본과 어긋났다 -> README 수정 내역 참고)"""
    r = model.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False, retina_masks=True)[0]
    h, w = img.shape[:2]
    label_mask = np.zeros((h, w), np.uint8)       # 0 = 배경, k = k번째 객체(1부터)
    insts = []
    if r.masks is not None:
        data = r.masks.data.cpu().numpy()
        # 점수 낮은 것부터 칠해서, 겹치는 곳은 점수 높은 객체가 차지하게 한다
        order = sorted(range(len(r.boxes)), key=lambda k: float(r.boxes[k].conf.item()))
        for i in order:
            m = data[i]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            label_mask[m > 0.5] = i + 1
        for i, b in enumerate(r.boxes):
            cid = int(b.cls.item())
            m = (label_mask == i + 1)
            box = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            area = int(m.sum())
            insts.append({
                "id": i,
                "mask_value": i + 1,
                "class_id": cid,
                "class_name": r.names[cid],
                "score": round(float(b.conf.item()), 4),
                "xyxy": box,
                "mask_area_px": area,
                # 박스 안에서 마스크가 차지하는 비율 = 1 - (박스 안에 들어 있던 배경 비율)
                "mask_fill_ratio_in_box": round(area / box_area, 3),
            })
    return insts, label_mask, r.speed


def overlay_masks(img, label_mask, insts, alpha=0.5):
    out = img.copy()
    color_layer = np.zeros_like(img)
    for d in insts:
        color_layer[label_mask == d["mask_value"]] = color_for(d["class_id"])
    sel = label_mask > 0
    out[sel] = (img[sel] * (1 - alpha) + color_layer[sel] * alpha).astype(np.uint8)
    for d in insts:
        m = (label_mask == d["mask_value"]).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color_for(d["class_id"]), 2)
        x1, y1 = int(d["xyxy"][0]), int(d["xyxy"][1])
        draw_label(out, f'#{d["id"]} {d["class_name"]} {d["score"]:.2f}', x1, y1, color_for(d["class_id"]))
    return out


def color_mask(label_mask, insts):
    out = np.zeros((*label_mask.shape, 3), np.uint8)
    for d in insts:
        out[label_mask == d["mask_value"]] = color_for(d["class_id"])
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def compare(low, high, low_conf, high_conf):
    """낮은 임계값 결과 중 높은 임계값에서 사라진 것 / 남은 것."""
    kept, removed = [], []
    for d in low:
        match = any(h["class_id"] == d["class_id"] and iou(h["xyxy"], d["xyxy"]) > 0.7 for h in high)
        (kept if match else removed).append(d)
    return {
        "low_conf": low_conf, "high_conf": high_conf,
        "count_low": len(low), "count_high": len(high),
        "removed_when_raising_threshold": removed,
        "candidates_between_thresholds": [d for d in low if low_conf <= d["score"] < high_conf],
        "kept": [d["id"] for d in kept],
    }


# ---------- 한 장 처리 ----------
def process_image(img_path: Path, det_model, seg_model, confs, imgsz, device, out_root: Path):
    img = imread(img_path)
    stem = img_path.stem
    out = out_root / stem
    out.mkdir(parents=True, exist_ok=True)
    imwrite(out / "original.jpg", img)

    summary = {"input": rel(img_path), "width": img.shape[1], "height": img.shape[0], "runs": {}}
    det_by_conf = {}
    for conf in confs:
        tag = f"conf{conf:.2f}"
        d = out / tag
        dets, det_speed = run_detection(det_model, img, conf, imgsz, device)
        insts, label_mask, seg_speed = run_segmentation(seg_model, img, conf, imgsz, device)
        det_by_conf[conf] = dets

        box_img = draw_boxes(img, dets)
        seg_img = overlay_masks(img, label_mask, insts)
        imwrite(d / "boxes.jpg", box_img)
        imwrite(d / "masks_overlay.jpg", seg_img)
        imwrite(d / "mask_label.png", label_mask)          # 픽셀값 = mask_value (다른 프로그램용)
        imwrite(d / "mask_color.png", color_mask(label_mask, insts))  # 사람이 보기 좋은 색 마스크
        imwrite(d / "panel.jpg", side_by_side([img, box_img, seg_img],
                                              ["original", f"detection {tag}", f"segmentation {tag}"]))

        watched = lambda xs: [x for x in xs if x["class_name"] in WATCH_CLASSES]
        payload = {
            "image": rel(img_path), "conf": conf, "imgsz": imgsz, "device": device,
            "coordinate_format": "xyxy = [x1, y1, x2, y2] (원본 이미지 픽셀, 왼쪽 위가 0,0)",
            "detection": {"model": DET_WEIGHTS.name, "speed_ms": det_speed, "boxes": dets,
                          "watched_classes_only": watched(dets)},
            "segmentation": {"model": SEG_WEIGHTS.name, "speed_ms": seg_speed, "instances": insts,
                             "mask_file": "mask_label.png",
                             "mask_note": "mask_label.png 의 픽셀값 k 는 instances 의 mask_value=k 객체"},
        }
        (d / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["runs"][tag] = {"det_count": len(dets), "seg_count": len(insts),
                                "classes": sorted({x["class_name"] for x in dets})}
        print(f"  [{stem}] {tag}: 박스 {len(dets)}개, 마스크 {len(insts)}개")

    # 임계값 비교 (가장 낮은 값 vs 가장 높은 값)
    lo, hi = min(confs), max(confs)
    cmp = compare(det_by_conf[lo], det_by_conf[hi], lo, hi)
    (out / "compare_thresholds.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")
    # 비교 그림: 사라진 박스를 빨간 점선 느낌(두꺼운 빨강)으로 강조
    lo_img = draw_boxes(img, det_by_conf[lo])
    for r in cmp["removed_when_raising_threshold"]:
        x1, y1, x2, y2 = [int(round(v)) for v in r["xyxy"]]
        cv2.rectangle(lo_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
    hi_img = draw_boxes(img, det_by_conf[hi])
    imwrite(out / "compare_thresholds.jpg",
            side_by_side([lo_img, hi_img], [f"conf {lo:.2f} (red = removed at {hi:.2f})", f"conf {hi:.2f}"]))
    summary["removed_when_raising_threshold"] = [
        f'#{r["id"]} {r["class_name"]} {r["score"]:.2f}' for r in cmp["removed_when_raising_threshold"]]
    return summary


def package_versions():
    from importlib.metadata import version, PackageNotFoundError
    vers = {"python": platform.python_version()}
    for p in ["ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pillow"]:
        try:
            vers[p] = version(p)
        except PackageNotFoundError:
            vers[p] = None
    return vers


def main():
    ap = argparse.ArgumentParser(description="로컬 YOLO 서비스 (detection + segmentation)")
    ap.add_argument("--images", nargs="+", default=[str(ROOT / "input" / "bus.jpg")])
    ap.add_argument("--confs", nargs="+", type=float, default=[0.25, 0.50])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "yolo"))
    args = ap.parse_args()

    from ultralytics import YOLO
    det_model = YOLO(str(DET_WEIGHTS))
    seg_model = YOLO(str(SEG_WEIGHTS))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    results, inputs = [], []
    for p in args.images:
        path = Path(p)
        if not path.is_absolute():
            path = (Path.cwd() / path)
        if not path.exists():
            print(f"[건너뜀] 파일이 없습니다: {p}")
            inputs.append({"file": p, "status": "missing - 처리하지 않음"})
            continue
        print(f"처리 중: {path.name}")
        results.append(process_image(path, det_model, seg_model, args.confs, args.imgsz, args.device, out_root))
        src_note = ROOT / "input" / "SOURCES.json"
        sources = json.loads(src_note.read_text(encoding="utf-8")) if src_note.exists() else {}
        inputs.append({"file": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size,
                       "source": sources.get(path.name, "출처 미기재")})

    # 실행 코드 사본
    code_dir = out_root / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(__file__, code_dir / Path(__file__).name)

    manifest = {
        "inputs": inputs,
        "models": {
            "detection": {"file": rel(DET_WEIGHTS), "sha256": sha256(DET_WEIGHTS),
                          "task": det_model.task, "num_classes": len(det_model.names)},
            "segmentation": {"file": rel(SEG_WEIGHTS), "sha256": sha256(SEG_WEIGHTS),
                             "task": seg_model.task, "num_classes": len(seg_model.names)},
        },
        "watched_classes": WATCH_CLASSES,
        "packages": package_versions(),
        "execution": {
            "where": "로컬 PC (CPU, 인터넷 서비스 아님)",
            # 개인정보 최소화: 운영체제 세부 버전·컴퓨터 종류·실행 시각은 기록하지 않음
            "python_executable": "YOLO과제/.venv (가상환경)",
            "working_dir": "YOLO과제/",
            "command": "python " + " ".join([Path(sys.argv[0]).name] + [rel(Path(a)) if Path(a).exists() else a
                                                                         for a in sys.argv[1:]]),
        },
        "settings": {"imgsz": args.imgsz, "device": args.device, "confs": args.confs,
                     "iou_for_threshold_compare": 0.7, "retina_masks": True},
        "results": results,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {rel(out_root)}/manifest.json")


if __name__ == "__main__":
    main()
