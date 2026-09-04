"""
Task 2: AI Training & Inference Pipeline
1. YOLOv8-OBB (Oriented Bounding Boxes): Detects rigid objects (pipes, shipwrecks, mines, containers) with orientation angle theta.
2. U-Net Segmentation Network: Segmenting amorphous ghost fishing nets and acoustic shadows from the WATERS dataset.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================================================================
# 1. YOLOv8-OBB CONFIGURATION & EXPORT UTILS
# ==============================================================================
YOLO_OBB_CLASSES = {
    0: "shipwreck",
    1: "pipeline",
    2: "naval_mine_uxo",
    3: "cargo_container",
    4: "debris_field"
}

def generate_yolov8_obb_yaml(dataset_root: str = "./data/SeabedObjects-KLSG", output_path: str = "./yolov8_obb_sonar.yaml"):
    """
    Creates dataset YAML for Ultralytics YOLOv8-OBB training on side-scan sonar.
    """
    yaml_content = f"""
# Ultralytics YOLOv8-OBB Dataset Configuration for SeabedObjects-KLSG
path: {os.path.abspath(dataset_root)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: shipwreck
  1: pipeline
  2: naval_mine_uxo
  3: cargo_container
  4: debris_field
"""
    with open(output_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"[+] YOLOv8-OBB dataset YAML generated at: {output_path}")


def train_yolov8_obb(data_yaml: str = "./yolov8_obb_sonar.yaml", epochs: int = 50, imgsz: int = 640):
    """
    Script to train YOLOv8n-obb model on pre-processed sonar images.
    """
    try:
        from ultralytics import YOLO
        print(f"[*] Initializing YOLOv8n-OBB for rigid seabed targets...")
        model = YOLO("yolov8n-obb.pt")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            degrees=180.0,      # Sonar images are orientation invariant along track
            fliplr=0.5,
            flipud=0.5,
            name="sonar_yolov8_obb_run"
        )
        print("[✓] YOLOv8-OBB training finished.")
        return model
    except ImportError:
        print("[!] Ultralytics not installed in this environment. Script provides training architecture for production CLI execution:")
        print(f"    yolo task=obb mode=train model=yolov8n-obb.pt data={data_yaml} epochs={epochs} imgsz={imgsz}")
        return None


# ==============================================================================
# 2. PYTORCH U-NET FOR GHOST NET (WATERS DATASET) SEGMENTATION
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SonarGhostNetUNet(nn.Module):
    """
    Lightweight 4-level U-Net for segmenting amorphous ghost fishing nets and acoustic shadows.
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = DoubleConv(256, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottleneck(self.pool(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out_conv(d1)


def simulate_model_inference(preprocessed_image_np: np.ndarray) -> dict:
    """
    Mock/Live inference wrapper: Produces YOLOv8-OBB oriented boxes and U-Net ghost net masks.
    """
    h, w = preprocessed_image_np.shape[:2]
    
    # 1. YOLOv8-OBB detections [cx, cy, width, height, angle_rad, class_id, conf]
    obb_detections = [
        {
            "class_id": 2,
            "class_name": "naval_mine_uxo",
            "confidence": 0.945,
            "bbox_obb": {
                "cx": int(w * 0.28),
                "cy": int(h * 0.28),
                "w": 90,
                "h": 65,
                "angle_deg": 24.5  # Orientation angle relative to towfish heading
            },
            "severity": "Red",
            "risk_level": "High Risk"
        },
        {
            "class_id": 0,
            "class_name": "shipwreck_wreckage",
            "confidence": 0.882,
            "bbox_obb": {
                "cx": int(w * 0.72),
                "cy": int(h * 0.44),
                "w": 180,
                "h": 95,
                "angle_deg": -18.2
            },
            "severity": "Yellow",
            "risk_level": "Medium Risk"
        },
        {
            "class_id": 1,
            "class_name": "subsea_pipeline",
            "confidence": 0.918,
            "bbox_obb": {
                "cx": int(w * 0.42),
                "cy": int(h * 0.76),
                "w": 190,
                "h": 50,
                "angle_deg": 65.0
            },
            "severity": "Green",
            "risk_level": "Low Risk"
        }
    ]
    
    # 2. U-Net ghost net segmentation polygon
    ghost_net_polygon = [
        {"x": int(w * 0.55), "y": int(h * 0.20)},
        {"x": int(w * 0.68), "y": int(h * 0.22)},
        {"x": int(w * 0.72), "y": int(h * 0.35)},
        {"x": int(w * 0.62), "y": int(h * 0.40)},
        {"x": int(w * 0.52), "y": int(h * 0.32)}
    ]
    
    return {
        "obb_detections": obb_detections,
        "unet_ghost_net": {
            "name": "Ghost Fishing Net Entanglement",
            "confidence": 0.897,
            "severity": "Yellow",
            "polygon": ghost_net_polygon,
            "area_sq_meters": 48.5
        }
    }

if __name__ == "__main__":
    print("[+] Initializing U-Net model verification...")
    model = SonarGhostNetUNet(in_channels=1, num_classes=2)
    dummy_input = torch.randn(1, 1, 256, 256)
    out = model(dummy_input)
    print(f"[✓] U-Net Forward Pass Successful! Output shape: {out.shape}")
