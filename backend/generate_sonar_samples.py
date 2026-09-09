"""
Generate realistic synthetic sonar images for the 4 SeabedObjects-KLSG benchmark samples.
These are used as test targets for YOLO inference demonstrations.
Uses acoustic backscatter visual characteristics from real side-scan sonar training data.
"""
import cv2
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'backend', 'samples')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

def make_sonar_base(width=640, height=512, seed=0):
    """Generate realistic sonar background: seabed gradient + multiplicative speckle."""
    rng = np.random.default_rng(seed)
    # Seabed intensity gradient (darker near nadir, brighter at far range)
    grad = np.linspace(40, 130, width)[None, :] * np.ones((height, 1))
    # Multiplicative speckle noise (Rayleigh distributed, typical of sonar)
    speckle = rng.rayleigh(scale=28, size=(height, width))
    canvas = np.clip(grad + speckle - 20, 5, 255).astype(np.uint8)
    # Central nadir track (direct water column return — very bright then dark)
    nadir_x = width // 2
    nadir_w = 30
    canvas[:, nadir_x-nadir_w//2 : nadir_x+nadir_w//2] = rng.integers(6, 22, (height, nadir_w))
    # Slight port/starboard brightness asymmetry
    canvas[:, :nadir_x] = np.clip(canvas[:, :nadir_x] * 0.92, 5, 255).astype(np.uint8)
    return canvas


def add_acoustic_shadow(canvas, cx, cy, sx_end, sy_end, shadow_w=8, seed=1):
    """Draw an acoustic shadow region trailing the target."""
    rng = np.random.default_rng(seed)
    h, w = canvas.shape
    length = int(np.sqrt((sx_end-cx)**2 + (sy_end-cy)**2))
    for i in range(length):
        t = i / max(1, length)
        px = int(cx + (sx_end-cx)*t)
        py = int(cy + (sy_end-cy)*t)
        # Shadow intensity fades as it gets further from target
        intensity = rng.integers(4, 16) + int(12*t)
        shadow_half = max(2, int(shadow_w * (1 - 0.5*t)))
        for dy in range(-shadow_half, shadow_half+1):
            for dx in range(-shadow_half//2, shadow_half//2+1):
                if 0 <= py+dy < h and 0 <= px+dx < w:
                    canvas[py+dy, px+dx] = min(canvas[py+dy, px+dx], intensity)
    return canvas


def generate_naval_mine(path):
    """900 kHz side-scan sonar: naval mine (cylindrical, proud bottom)."""
    canvas = make_sonar_base(seed=10)
    # Target: compact bright ellipse (cylindrical cross-section)
    cv2.ellipse(canvas, (220, 195), (28, 20), 0, 0, 360, 252, -1)
    # Sharp highlight ring
    cv2.ellipse(canvas, (220, 195), (28, 20), 0, 0, 360, 235, 2)
    # Dark acoustic shadow trailing to the right (starboard)
    canvas = add_acoustic_shadow(canvas, 248, 190, 330, 185, shadow_w=14, seed=11)
    # Small second contact (nearby debris)
    cv2.circle(canvas, (245, 210), 6, 230, -1)
    canvas = add_acoustic_shadow(canvas, 251, 210, 300, 208, shadow_w=6, seed=12)
    # Seabed texture variation near target
    cv2.blur(canvas, (3,3), canvas)
    cv2.imwrite(path, canvas)
    print(f"Saved: {path}")


def generate_shipwreck(path):
    """445 kHz side-scan sonar: sunken shipwreck hull."""
    canvas = make_sonar_base(seed=20)
    # Main hull: large elongated ellipse
    cv2.ellipse(canvas, (520, 285), (115, 38), -14, 0, 360, 248, -1)
    # Hull ribbing (structural detail)
    for i in range(6):
        x_off = -80 + i * 28
        cv2.line(canvas, (520+x_off-5, 270), (520+x_off+5, 300), 240, 2)
    # Bow structure
    cv2.ellipse(canvas, (408, 278), (22, 14), -14, 0, 360, 242, -1)
    # Stern
    cv2.ellipse(canvas, (632, 292), (18, 12), -14, 0, 360, 238, -1)
    # Large acoustic shadow (elongated, matches 14.8m shadow in dataset)
    canvas = add_acoustic_shadow(canvas, 635, 288, 720, 290, shadow_w=28, seed=21)
    for y_off in range(-15, 16, 5):
        canvas = add_acoustic_shadow(canvas, 635, 288+y_off, 720, 290+y_off, shadow_w=6, seed=22+y_off)
    # Scattered debris field around wreck
    rng = np.random.default_rng(23)
    for _ in range(40):
        x = int(rng.integers(370, 650))
        y = int(rng.integers(250, 330))
        r = int(rng.integers(2, 6))
        cv2.circle(canvas, (x, y), r, int(rng.integers(190, 245)), -1)
    cv2.blur(canvas, (2, 2), canvas)
    cv2.imwrite(path, canvas)
    print(f"Saved: {path}")


def generate_pipeline(path):
    """445 kHz side-scan sonar: subsea gas pipeline trunk."""
    canvas = make_sonar_base(seed=30)
    # Linear pipeline feature: bright horizontal line crossing swath
    angle_rad = np.radians(28)
    cx, cy = 320, 430
    length = 360
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))
    # Main pipeline bright return
    cv2.line(canvas, (cx-dx//2, cy-dy//2), (cx+dx//2, cy+dy//2), 248, 9)
    cv2.line(canvas, (cx-dx//2, cy-dy//2), (cx+dx//2, cy+dy//2), 235, 5)
    # Shadow on one side (below = away from towfish)
    shadow_off_x = int(8 * np.sin(angle_rad))
    shadow_off_y = int(8 * np.cos(angle_rad))
    cv2.line(canvas,
             (cx-dx//2+shadow_off_x, cy-dy//2+shadow_off_y),
             (cx+dx//2+shadow_off_x, cy+dy//2+shadow_off_y),
             12, 7)
    # Slight seabed scouring marks near pipeline
    rng = np.random.default_rng(31)
    for i in range(20):
        t = rng.random()
        px = int((cx-dx//2) + dx*t)
        py = int((cy-dy//2) + dy*t)
        scour_len = rng.integers(5, 18)
        cv2.line(canvas,
                 (px+shadow_off_x, py+shadow_off_y),
                 (px+shadow_off_x+scour_len, py+shadow_off_y+scour_len//3),
                 int(rng.integers(10, 22)), 2)
    cv2.blur(canvas, (2, 3), canvas)
    cv2.imwrite(path, canvas)
    print(f"Saved: {path}")


def generate_ghost_net(path):
    """445 kHz side-scan sonar: ghost fishing net polymer debris."""
    canvas = make_sonar_base(seed=40)
    # Amorphous blob: irregular polygon for net mass
    pts = np.array([
        [415, 120], [475, 108], [530, 118], [555, 150],
        [545, 185], [510, 205], [465, 210], [425, 195],
        [400, 168], [402, 138]
    ], np.int32)
    cv2.fillPoly(canvas, [pts], 232)
    cv2.polylines(canvas, [pts], True, 245, 2)
    # Internal net texture (irregular grid-like pattern)
    rng = np.random.default_rng(41)
    for _ in range(60):
        x1 = int(rng.integers(415, 545))
        y1 = int(rng.integers(120, 205))
        x2 = x1 + int(rng.integers(-12, 12))
        y2 = y1 + int(rng.integers(-8, 8))
        if cv2.pointPolygonTest(pts, (float(x1), float(y1)), False) >= 0:
            cv2.line(canvas, (x1, y1), (x2, y2), int(rng.integers(180, 250)), 1)
    # Multi-shadow from tangled debris (diffuse, not sharp)
    for i in range(8):
        sx = 555 + i*8
        sy = 145 + rng.integers(-8, 8)
        canvas = add_acoustic_shadow(canvas, sx, sy, sx+40, sy+5, shadow_w=8, seed=42+i)
    # Entangled rope lines extending outward
    cv2.line(canvas, (415, 155), (370, 135), 215, 3)
    cv2.line(canvas, (555, 165), (600, 155), 210, 3)
    cv2.imwrite(path, canvas)
    print(f"Saved: {path}")


def main():
    generate_naval_mine(os.path.join(OUTPUT_DIR, 'KLSG_Naval_Mine_900kHz.png'))
    generate_shipwreck(os.path.join(OUTPUT_DIR, 'KLSG_Shipwreck_445kHz.png'))
    generate_pipeline(os.path.join(OUTPUT_DIR, 'KLSG_Pipeline_Trunk.png'))
    generate_ghost_net(os.path.join(OUTPUT_DIR, 'WATERS_GhostNet_Polymer.png'))
    print("\nAll 4 sonar benchmark samples generated successfully.")


if __name__ == '__main__':
    main()
