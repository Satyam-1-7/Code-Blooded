import React, { useRef, useEffect, useState } from 'react';
import { Eye, ShieldAlert, Sparkles, Crosshair, ZoomIn, ZoomOut, RotateCcw, Sliders, Layers, Compass } from 'lucide-react';
import { SonarTarget } from '../types';
import { SonarColorPalette, renderSyntheticSonarCanvas, drawOpenCVAnnotations } from '../utils/sonarSynthetic';

interface DualViewSonarProps {
  customImageSrc: string | null;
  palette: SonarColorPalette;
  activeTargets: SonarTarget[];
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
  fileName: string | null;
  isProcessing?: boolean;
  leeWindowSize?: number;
  claheClipLimit?: number;
}

export const DualViewSonar: React.FC<DualViewSonarProps> = ({
  customImageSrc,
  palette,
  activeTargets,
  selectedTargetId,
  onSelectTarget,
  fileName,
  isProcessing = false,
  leeWindowSize = 7,
  claheClipLimit = 3.0,
}) => {
  const rawCanvasRef = useRef<HTMLCanvasElement>(null);
  const cvCanvasRef = useRef<HTMLCanvasElement>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [hoveredTarget, setHoveredTarget] = useState<SonarTarget | null>(null);

  // Layer Visibility Controls
  const [showOBB, setShowOBB] = useState<boolean>(true);
  const [showUNet, setShowUNet] = useState<boolean>(true);
  const [showShadowGeo, setShowShadowGeo] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<'both' | 'lee' | 'clahe'>('both');

  // Render raw and OpenCV annotated canvases
  useEffect(() => {
    const rawCanvas = rawCanvasRef.current;
    const cvCanvas = cvCanvasRef.current;
    if (!rawCanvas || !cvCanvas) return;

    const width = 640;
    const height = 512;
    rawCanvas.width = width;
    rawCanvas.height = height;
    cvCanvas.width = width;
    cvCanvas.height = height;

    const rawCtx = rawCanvas.getContext('2d');
    const cvCtx = cvCanvas.getContext('2d');
    if (!rawCtx || !cvCtx) return;

    if (customImageSrc) {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        rawCtx.drawImage(img, 0, 0, width, height);
        cvCtx.drawImage(img, 0, 0, width, height);
        drawOpenCVAnnotations(cvCtx, activeTargets, width, height, selectedTargetId, showOBB, showUNet, showShadowGeo);
      };
      img.src = customImageSrc;
    } else {
      // Render raw side-scan sonar image
      renderSyntheticSonarCanvas(rawCanvas, palette, viewMode === 'lee' || viewMode === 'clahe', viewMode === 'clahe');

      // Copy to detection canvas and draw OpenCV + YOLO-OBB + U-Net overlays
      cvCtx.drawImage(rawCanvas, 0, 0);
      drawOpenCVAnnotations(cvCtx, activeTargets, width, height, selectedTargetId, showOBB, showUNet, showShadowGeo);
    }
  }, [customImageSrc, palette, activeTargets, selectedTargetId, showOBB, showUNet, showShadowGeo, viewMode]);

  // Handle canvas clicks to select targets
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = cvCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = ((e.clientX - rect.left) / rect.width) * 640;
    const clickY = ((e.clientY - rect.top) / rect.height) * 512;

    const matched = activeTargets.find((t) => {
      const [x, y, w, h] = t.bbox;
      return clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h;
    });

    onSelectTarget(matched ? matched.id : null);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = cvCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const posX = ((e.clientX - rect.left) / rect.width) * 640;
    const posY = ((e.clientY - rect.top) / rect.height) * 512;

    const matched = activeTargets.find((t) => {
      const [x, y, w, h] = t.bbox;
      return posX >= x && posX <= x + w && posY >= y && posY <= y + h;
    });

    setHoveredTarget(matched || null);
  };

  const activeFocusTarget = activeTargets.find((t) => t.id === selectedTargetId);

  return (
    <div id="dual-view-section" className="space-y-4">
      {/* Header with Title and Zoom Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 bg-slate-900/90 border border-slate-800 p-4 rounded-xl">
        <div>
          <div className="flex items-center gap-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-slate-100">
              Dual-View Acoustic Inspection & AI Overlay
            </h3>
            {isProcessing && (
              <span className="text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded animate-pulse">
                Running OpenCV & AI Inference...
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Synchronized comparative inspection: Raw Sonar Backscatter vs. YOLOv8-OBB & U-Net Segmentations
          </p>
        </div>

        {/* Layer Toggles & Zoom Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Layer toggles */}
          <div className="flex items-center gap-1 bg-slate-950/60 border border-slate-800 p-1 rounded-lg text-xs">
            <button
              onClick={() => setShowOBB(!showOBB)}
              className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                showOBB ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle YOLOv8-OBB Oriented Bounding Boxes"
            >
              YOLO-OBB (θ)
            </button>
            <button
              onClick={() => setShowUNet(!showUNet)}
              className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                showUNet ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle U-Net Ghost Net Polygons"
            >
              U-Net Nets
            </button>
            <button
              onClick={() => setShowShadowGeo(!showShadowGeo)}
              className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                showShadowGeo ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle Acoustic Shadow Height Math"
            >
              Shadow (h)
            </button>
          </div>

          {/* Zoom and Inspector Tools */}
          <div className="flex items-center gap-1 bg-slate-950/60 border border-slate-800 p-1 rounded-lg">
            <button
              id="zoom-out-btn"
              onClick={() => setZoomLevel((prev) => Math.max(0.8, prev - 0.1))}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-xs font-mono text-cyan-400 font-semibold px-1.5">
              {(zoomLevel * 100).toFixed(0)}%
            </span>
            <button
              id="zoom-in-btn"
              onClick={() => setZoomLevel((prev) => Math.min(1.6, prev + 0.1))}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              id="zoom-reset-btn"
              onClick={() => setZoomLevel(1)}
              className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Reset Zoom"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* 2-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Column 1: Raw Sonar Image */}
        <div id="col-raw-sonar" className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-col">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
              <h4 className="text-sm font-semibold text-slate-200">
                Column 1: Raw Sonar Backscatter
              </h4>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-slate-400 bg-slate-800/70 px-2 py-0.5 rounded">
                {fileName || 'KLSG_Survey_Track_042.png'}
              </span>
            </div>
          </div>

          {/* Canvas Wrapper with Zoom */}
          <div className="relative overflow-hidden rounded-lg bg-slate-950 border border-slate-800/80 aspect-[640/512] flex items-center justify-center">
            <canvas
              ref={rawCanvasRef}
              id="raw-sonar-canvas"
              style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'center center' }}
              className="w-full h-full object-contain transition-transform duration-150"
            />
            <div className="absolute top-2 left-2 bg-slate-950/80 backdrop-blur-sm border border-slate-800 text-[10px] text-slate-300 px-2 py-1 rounded">
              Raw Acoustic Backscatter: Port & Starboard
            </div>
            <div className="absolute bottom-2 right-2 bg-slate-950/80 backdrop-blur-sm border border-slate-800 text-[10px] text-slate-400 px-2 py-1 rounded font-mono">
              640 × 512 px (445 kHz)
            </div>
          </div>
        </div>

        {/* Column 2: OpenCV Object Detection & AI Bounding Boxes */}
        <div id="col-opencv-detections" className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-col">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <h4 className="text-sm font-semibold text-slate-200">
                Column 2: YOLOv8-OBB & U-Net Detections
              </h4>
            </div>
            <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/50 px-2 py-0.5 rounded">
              {activeTargets.length} Detections
            </span>
          </div>

          {/* Canvas Wrapper with Interactive Hover and Click */}
          <div className="relative overflow-hidden rounded-lg bg-slate-950 border border-slate-800/80 aspect-[640/512] flex items-center justify-center cursor-crosshair">
            <canvas
              ref={cvCanvasRef}
              id="opencv-annotated-canvas"
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMouseMove}
              onMouseLeave={() => setHoveredTarget(null)}
              style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'center center' }}
              className="w-full h-full object-contain transition-transform duration-150"
            />

            {/* Hover Tooltip Overlay */}
            {hoveredTarget && (
              <div className="absolute top-2 right-2 bg-slate-900/95 border border-slate-700 shadow-xl backdrop-blur-md p-2.5 rounded-lg text-xs max-w-xs pointer-events-none transition-all">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="font-bold text-slate-100">{hoveredTarget.id}: {hoveredTarget.name}</span>
                  <span
                    className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                    style={{ backgroundColor: `${hoveredTarget.color_hex}33`, color: hoveredTarget.color_hex }}
                  >
                    {hoveredTarget.severity}
                  </span>
                </div>
                {hoveredTarget.bbox_obb && (
                  <p className="text-[11px] text-cyan-300 font-mono">
                    YOLO-OBB Angle θ: {hoveredTarget.bbox_obb.angle_deg}°
                  </p>
                )}
                {hoveredTarget.shadow_metrics && (
                  <p className="text-[11px] text-sky-300 font-mono">
                    Calculated Height: {hoveredTarget.shadow_metrics.estimated_target_height_m}m | Conf: {hoveredTarget.shadow_metrics.shadow_confidence_pct}%
                  </p>
                )}
                <p className="text-[11px] text-slate-400">
                  Confidence: <span className="font-mono text-slate-200">{(hoveredTarget.confidence * 100).toFixed(1)}%</span> | Depth: <span className="font-mono text-slate-200">{hoveredTarget.depth_meters}m</span>
                </p>
                <p className="text-[10px] text-slate-500 mt-1 italic line-clamp-2">
                  {hoveredTarget.description}
                </p>
              </div>
            )}

            <div className="absolute bottom-2 left-2 bg-slate-950/80 backdrop-blur-sm border border-slate-800 text-[10px] text-cyan-400 px-2 py-1 rounded flex items-center gap-1.5">
              <Crosshair className="w-3 h-3" />
              <span>Click any box to inspect & focus map</span>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Target Quick Banner with Shadow Math Breakdown */}
      {activeFocusTarget && (
        <div className="bg-gradient-to-r from-slate-900 to-cyan-950/40 border border-cyan-800/80 rounded-xl p-4 text-xs space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-cyan-400" />
              <span className="text-slate-300">
                Focused Detection: <strong className="text-cyan-300">{activeFocusTarget.id} — {activeFocusTarget.name}</strong>
              </span>
              <span
                className="text-[10px] font-bold px-2 py-0.5 rounded"
                style={{ backgroundColor: `${activeFocusTarget.color_hex}33`, color: activeFocusTarget.color_hex }}
              >
                {activeFocusTarget.severity} ({activeFocusTarget.risk_level})
              </span>
            </div>
            <button
              onClick={() => onSelectTarget(null)}
              className="text-slate-400 hover:text-slate-200 underline text-xs"
            >
              Clear Selection
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {/* OBB Geometrical Orientation */}
            <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-lg space-y-1">
              <div className="flex items-center gap-1.5 text-cyan-400 font-semibold">
                <Compass className="w-3.5 h-3.5" />
                <span>YOLOv8-OBB Geometry</span>
              </div>
              <p className="text-slate-300">
                Orientation Angle θ: <strong className="text-cyan-300 font-mono">{activeFocusTarget.bbox_obb?.angle_deg ?? 0}°</strong>
              </p>
              <p className="text-slate-400 text-[11px]">
                Dimensions: <span className="font-mono text-slate-200">{activeFocusTarget.dimensions}</span>
              </p>
              <p className="text-slate-400 text-[11px]">
                Dataset Origin: <span className="text-cyan-300">{activeFocusTarget.dataset_source || 'SeabedObjects-KLSG'}</span>
              </p>
            </div>

            {/* Acoustic Shadow Math */}
            <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-lg space-y-1">
              <div className="flex items-center gap-1.5 text-sky-400 font-semibold">
                <Layers className="w-3.5 h-3.5" />
                <span>Task 3 Shadow Verification Math</span>
              </div>
              <p className="text-slate-300">
                3D Target Height: <strong className="text-sky-300 font-mono">{activeFocusTarget.shadow_metrics?.estimated_target_height_m ?? 2.5}m</strong>
              </p>
              <p className="text-slate-400 text-[11px]">
                Shadow Length ($L_s$): <span className="font-mono text-slate-200">{activeFocusTarget.shadow_metrics?.shadow_length_m ?? 8.0}m</span> | Slant Range ($R$): <span className="font-mono text-slate-200">{activeFocusTarget.shadow_metrics?.slant_range_m ?? 35.0}m</span>
              </p>
              <p className="text-slate-400 text-[11px]">
                Shadow Confidence: <strong className="text-emerald-400 font-mono">{activeFocusTarget.shadow_metrics?.shadow_confidence_pct ?? 94}%</strong> (Verified 3D)
              </p>
            </div>

            {/* Geotagging & Action */}
            <div className="bg-slate-950/60 border border-slate-800/80 p-2.5 rounded-lg space-y-1">
              <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
                <Crosshair className="w-3.5 h-3.5" />
                <span>WGS84 GPS Coordinates</span>
              </div>
              <p className="text-slate-300 font-mono">
                {activeFocusTarget.latitude.toFixed(6)}° N, {Math.abs(activeFocusTarget.longitude).toFixed(6)}° W
              </p>
              <p className="text-slate-400 text-[11px] line-clamp-2">
                Action: <span className="text-slate-200">{activeFocusTarget.action_recommendation || activeFocusTarget.description}</span>
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
