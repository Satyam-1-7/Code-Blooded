import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def compute_box_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / union

def apply_nms(detections, iou_threshold=0.40):
    """Sorts detections by confidence descending and suppresses overlapping duplicates."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in sorted_dets:
        box_d = d["xyxy"]
        overlap = False
        for k in kept:
            if compute_box_iou(box_d, k["xyxy"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(d)
    return kept

def main():
    model_path = os.path.abspath("backend/models/best.pt")
    model = YOLO(model_path)
    
    samples = sorted(glob.glob("backend/samples/*.png"))
    for s in samples:
        img = cv2.imread(s)
        res = model.predict(img, conf=0.15, iou=0.45, verbose=False)
        raw_list = []
        for r in res:
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    xyxy = b.xyxy[0].tolist()
                    raw_list.append({
                        "class_id": cls_id,
                        "class_name": model.names.get(cls_id, str(cls_id)),
                        "confidence": conf,
                        "xyxy": xyxy
                    })
        print(f"\n--- Sample: {os.path.basename(s)} ---")
        print(f"Raw YOLO detections ({len(raw_list)}):")
        for r in raw_list:
            print(f"  {r['class_name']} (cls {r['class_id']}): conf={r['confidence']:.4f}, box={r['xyxy']}")
        
        filtered = apply_nms(raw_list, iou_threshold=0.40)
        print(f"After NMS Overlap Suppression ({len(filtered)}):")
        for f in filtered:
            print(f"  * KEPT: {f['class_name']} (cls {f['class_id']}): conf={f['confidence']:.4f}")

if __name__ == "__main__":
    main()
