import { SonarTarget, PolygonPoint } from '../types';

export type SonarColorPalette = 'amber' | 'grayscale' | 'cyan' | 'green';
export type PipelineVisualMode = 'all' | 'raw_only' | 'lee_filtered' | 'clahe_enhanced' | 'yolo_obb' | 'unet_masks' | 'shadow_rays';

/**
 * Renders realistic synthetic side-scan sonar image with customizable speckle & contrast
 */
export function renderSyntheticSonarCanvas(
  canvas: HTMLCanvasElement,
  palette: SonarColorPalette = 'amber',
  applyLeeFilter: boolean = false,
  applyClahe: boolean = false,
  customSeed: number = 42
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;

  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  // Generate acoustic seafloor reverberation texture
  const nadirCenter = width / 2;
  const nadirWidth = 34;

  for (let y = 0; y < height; y++) {
    const yNorm = y / height;
    const baseGradient = 42 + Math.sin(yNorm * Math.PI * 4) * 12 + yNorm * 30;

    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const distToNadir = Math.abs(x - nadirCenter);

      // Rayleigh speckle distribution simulation
      const noise = (Math.random() - 0.5) * (applyLeeFilter ? 12 : 45);
      let intensity = baseGradient + noise;

      // Water column nadir zone (low acoustic return)
      if (distToNadir < nadirWidth / 2) {
        intensity = 14 + (Math.random() - 0.5) * (applyLeeFilter ? 4 : 12);
      } else if (distToNadir < nadirWidth / 2 + 12) {
        // First bottom echo (high specular reflection)
        intensity += 70 + (Math.random() - 0.5) * (applyLeeFilter ? 8 : 28);
      }

      // Sand ripple acoustic diffraction patterns
      intensity += Math.sin((x + y * 0.4) * 0.08) * 16;

      // CLAHE-style adaptive dynamic range expansion
      if (applyClahe) {
        intensity = Math.pow(intensity / 255.0, 0.82) * 255.0;
        intensity = Math.min(255, intensity * 1.25);
      }

      intensity = Math.max(0, Math.min(255, intensity));

      // Apply Sonar Color Palette
      if (palette === 'amber') {
        data[idx] = Math.min(255, intensity * 1.15);     // Red
        data[idx + 1] = Math.min(255, intensity * 0.72); // Green
        data[idx + 2] = Math.min(255, intensity * 0.22); // Blue
      } else if (palette === 'cyan') {
        data[idx] = Math.min(255, intensity * 0.2);
        data[idx + 1] = Math.min(255, intensity * 0.85);
        data[idx + 2] = Math.min(255, intensity * 1.1);
      } else if (palette === 'green') {
        data[idx] = Math.min(255, intensity * 0.25);
        data[idx + 1] = Math.min(255, intensity * 1.1);
        data[idx + 2] = Math.min(255, intensity * 0.35);
      } else {
        // Grayscale
        data[idx] = intensity;
        data[idx + 1] = intensity;
        data[idx + 2] = intensity;
      }
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // Overlay synthetic target anomalies with bright acoustic highlight + long acoustic shadow
  // Target 1: Naval mine (High Risk)
  drawAcousticTarget(ctx, 140, 110, 110, 85, palette);
  // Target 2: Shipwreck Keel (Medium Risk)
  drawAcousticTarget(ctx, 390, 190, 180, 110, palette);
  // Target 3: Subsea Pipeline (Low Risk)
  drawAcousticTarget(ctx, 210, 360, 240, 70, palette);
  // Target 4: WATERS Ghost Net
  drawAcousticGhostNet(ctx, 320, 90, 140, 100, palette);
}

function drawAcousticTarget(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  palette: SonarColorPalette
) {
  // Acoustic Highlight (specular hard reflection)
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(x + 25, y + 25, w * 0.22, h * 0.22, 0, 0, Math.PI * 2);
  if (palette === 'amber') {
    ctx.fillStyle = '#fffae0';
  } else if (palette === 'cyan') {
    ctx.fillStyle = '#e0faff';
  } else if (palette === 'green') {
    ctx.fillStyle = '#eaffea';
  } else {
    ctx.fillStyle = '#ffffff';
  }
  ctx.shadowColor = '#ffffff';
  ctx.shadowBlur = 10;
  ctx.fill();
  ctx.restore();

  // Acoustic Shadow (dark acoustic blockage behind target)
  ctx.save();
  ctx.fillStyle = 'rgba(8, 12, 18, 0.95)';
  ctx.fillRect(x + w * 0.35, y + 10, w * 0.58, h * 0.68);
  ctx.restore();
}

function drawAcousticGhostNet(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  palette: SonarColorPalette
) {
  ctx.save();
  // Irregular mesh highlight
  ctx.fillStyle = palette === 'amber' ? 'rgba(255, 230, 180, 0.75)' : 'rgba(200, 245, 255, 0.75)';
  ctx.beginPath();
  ctx.moveTo(x + 10, y + 10);
  ctx.bezierCurveTo(x + 50, y - 5, x + 90, y + 20, x + 110, y + 40);
  ctx.bezierCurveTo(x + 80, y + 80, x + 40, y + 60, x + 10, y + 50);
  ctx.closePath();
  ctx.fill();

  // Diffuse shadow zones
  ctx.fillStyle = 'rgba(10, 14, 20, 0.90)';
  ctx.beginPath();
  ctx.ellipse(x + 75, y + 55, w * 0.35, h * 0.28, 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/**
 * Draw OpenCV-style YOLOv8-OBB bounding boxes, U-Net polygon segmentation masks, and shadow math rays
 */
export function drawOpenCVAnnotations(
  ctx: CanvasRenderingContext2D,
  targets: SonarTarget[],
  canvasWidth: number,
  canvasHeight: number,
  selectedTargetId?: string | null,
  showOBB: boolean = true,
  showUNetMasks: boolean = true,
  showShadowGeometry: boolean = true
) {
  const scaleX = canvasWidth / 640.0;
  const scaleY = canvasHeight / 512.0;

  targets.forEach((target) => {
    const isSelected = selectedTargetId === target.id;
    const color = target.color_hex;

    // 1. Draw U-Net Segmentation Polygon if available (WATERS Ghost Nets)
    if (showUNetMasks && target.unet_segmentation_polygon && target.unet_segmentation_polygon.length > 2) {
      ctx.save();
      ctx.beginPath();
      target.unet_segmentation_polygon.forEach((pt, idx) => {
        const px = pt.x * scaleX;
        const py = pt.y * scaleY;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(245, 158, 11, 0.32)';
      ctx.fill();
      ctx.strokeStyle = '#F59E0B';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.setLineDash([4, 2]);
      ctx.stroke();
      ctx.restore();
    }

    // 2. Draw YOLOv8-OBB Oriented Bounding Box
    if (showOBB && target.bbox_obb) {
      const { cx, cy, w, h, angle_deg } = target.bbox_obb;
      const scaledCx = cx * scaleX;
      const scaledCy = cy * scaleY;
      const scaledW = w * scaleX;
      const scaledH = h * scaleY;
      const angleRad = (angle_deg * Math.PI) / 180;

      ctx.save();
      ctx.translate(scaledCx, scaledCy);
      ctx.rotate(angleRad);

      // Oriented Rectangle
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 3.5 : 2;
      if (isSelected) {
        ctx.setLineDash([6, 3]);
      }
      ctx.strokeRect(-scaledW / 2, -scaledH / 2, scaledW, scaledH);
      ctx.setLineDash([]);

      // Heading / Orientation Axis Vector
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(scaledW * 0.45, 0);
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Corner Crosshairs
      const cornerLen = 8;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;

      // Top-Left corner
      ctx.beginPath();
      ctx.moveTo(-scaledW / 2, -scaledH / 2 + cornerLen);
      ctx.lineTo(-scaledW / 2, -scaledH / 2);
      ctx.lineTo(-scaledW / 2 + cornerLen, -scaledH / 2);
      ctx.stroke();

      // Top-Right corner
      ctx.beginPath();
      ctx.moveTo(scaledW / 2 - cornerLen, -scaledH / 2);
      ctx.lineTo(scaledW / 2, -scaledH / 2);
      ctx.lineTo(scaledW / 2, -scaledH / 2 + cornerLen);
      ctx.stroke();

      ctx.restore();
    } else {
      // Standard AABB fallback
      const [origX, origY, origW, origH] = target.bbox;
      const x = origX * scaleX;
      const y = origY * scaleY;
      const w = origW * scaleX;
      const h = origH * scaleY;

      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 3.5 : 2;
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
    }

    // 3. Label Badge (OpenCV style)
    const [origX, origY] = target.bbox;
    const x = origX * scaleX;
    const y = origY * scaleY;

    const angleTag = target.bbox_obb ? ` [${target.bbox_obb.angle_deg > 0 ? '+' : ''}${target.bbox_obb.angle_deg}°]` : '';
    const label = `${target.id}: ${target.name}${angleTag} [${(target.confidence * 100).toFixed(1)}%]`;
    
    ctx.save();
    ctx.font = 'bold 11px ui-monospace, SFMono-Regular, monospace';
    const textMetrics = ctx.measureText(label);
    const badgeWidth = textMetrics.width + 12;
    const badgeHeight = 20;
    const badgeY = Math.max(0, y - badgeHeight);

    // Badge Background
    ctx.fillStyle = color;
    ctx.fillRect(x, badgeY, badgeWidth, badgeHeight);

    // Label Text
    ctx.fillStyle = target.severity === 'Yellow' ? '#18181b' : '#ffffff';
    ctx.fillText(label, x + 6, badgeY + 14);

    // Shadow Verification / Telemetry Pill below box
    if (showShadowGeometry && target.shadow_metrics) {
      const shadowInfo = `3D Relief: ${target.shadow_metrics.estimated_target_height_m}m | Shadow Conf: ${target.shadow_metrics.shadow_confidence_pct}%`;
      ctx.font = '10px sans-serif';
      ctx.fillStyle = 'rgba(15, 23, 42, 0.90)';
      const infoMetrics = ctx.measureText(shadowInfo);
      ctx.fillRect(x, y + target.bbox[3] * scaleY + 2, infoMetrics.width + 10, 17);
      ctx.fillStyle = '#38bdf8';
      ctx.fillText(shadowInfo, x + 5, y + target.bbox[3] * scaleY + 14);
    }
    ctx.restore();
  });
}
