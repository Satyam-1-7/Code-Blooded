import os
import cv2
import numpy as np
from ultralytics import YOLO

def test_detections():
    model_path = os.path.abspath("backend/models/best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.abspath("backend/best.pt")
    
    print("=" * 60)
    print(f"1. Model Verification:")
    print(f"   Model Path: {model_path}")
    model = YOLO(model_path)
    print(f"   Class Names: {model.names}")
    print(f"   Number of Classes: {len(model.names)}")
    print(f"   Task: {getattr(model, 'task', 'detect')}")
    print("=" * 60)

    # Test 1: Blank Image (512x640)
    blank_img = np.zeros((512, 640, 3), dtype=np.uint8)
    res_blank = model.predict(blank_img, conf=0.15, iou=0.45, verbose=False)
    boxes_blank = len(res_blank[0].boxes) if res_blank[0].boxes is not None else 0
    obb_blank = len(res_blank[0].obb) if hasattr(res_blank[0], 'obb') and res_blank[0].obb is not None else 0
    print(f"\n2. Blank Image Test:")
    print(f"   Total Detections: {boxes_blank + obb_blank} (Expected: 0)")

    # Test 2: Wallpaper Image (Simulated photo with high color variance)
    np.random.seed(123)
    wallpaper_img = np.random.randint(60, 220, (512, 640, 3), dtype=np.uint8)
    res_wall = model.predict(wallpaper_img, conf=0.15, iou=0.45, verbose=False)
    boxes_wall = len(res_wall[0].boxes) if res_wall[0].boxes is not None else 0
    obb_wall = len(res_wall[0].obb) if hasattr(res_wall[0], 'obb') and res_wall[0].obb is not None else 0
    print(f"\n3. Wallpaper / Non-Sonar Image Test:")
    print(f"   Total Detections: {boxes_wall + obb_wall} (Expected: 0)")

    # Test 3: Real Shipwreck Sonar Image
    shipwreck_path = "backend/samples/KLSG_Shipwreck_445kHz.png"
    if os.path.exists(shipwreck_path):
        ship_img = cv2.imread(shipwreck_path)
        res_ship = model.predict(ship_img, conf=0.20, iou=0.45, verbose=False)
        print(f"\n4. Real Shipwreck Sonar Image Test ({shipwreck_path}):")
        raw_results = []
        if res_ship[0].boxes is not None and len(res_ship[0].boxes) > 0:
            for b in res_ship[0].boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                cls_name = model.names.get(cls_id, str(cls_id))
                xyxy = b.xyxy[0].tolist()
                raw_results.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "coords": xyxy
                })
        print(f"   Raw YOLO Inference Output: {raw_results}")
        for item in raw_results:
            print(f"   -> Detected: {item['class_name']} with confidence {item['confidence']:.4f} at {item['coords']}")

if __name__ == "__main__":
    test_detections()
