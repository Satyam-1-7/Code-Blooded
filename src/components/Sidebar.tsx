import React, { useRef, useState } from 'react';
import {
  Upload,
  Sliders,
  Layers,
  RefreshCw,
  FileText,
  CheckSquare,
  Square,
  Compass,
  Zap,
  Database,
  SlidersHorizontal,
} from 'lucide-react';
import { SeverityLevel, TowfishNav } from '../types';
import { SonarColorPalette } from '../utils/sonarSynthetic';

interface SidebarProps {
  onImageUploaded: (file: File | null, sampleId?: string) => void;
  fileName: string | null;
  minConfidence: number;
  setMinConfidence: (val: number) => void;
  selectedSeverities: SeverityLevel[];
  setSelectedSeverities: React.Dispatch<React.SetStateAction<SeverityLevel[]>>;
  palette: SonarColorPalette;
  setPalette: (p: SonarColorPalette) => void;
  onResetToDemo: () => void;
  isCustomImage: boolean;
  towfishNav: TowfishNav;
  setTowfishNav: React.Dispatch<React.SetStateAction<TowfishNav>>;
  leeWindowSize: number;
  setLeeWindowSize: (val: number) => void;
  claheClipLimit: number;
  setClaheClipLimit: (val: number) => void;
  onRunLivePipeline: () => void;
  isProcessing: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  onImageUploaded,
  fileName,
  minConfidence,
  setMinConfidence,
  selectedSeverities,
  setSelectedSeverities,
  palette,
  setPalette,
  onResetToDemo,
  isCustomImage,
  towfishNav,
  setTowfishNav,
  leeWindowSize,
  setLeeWindowSize,
  claheClipLimit,
  setClaheClipLimit,
  onRunLivePipeline,
  isProcessing,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<'ingest' | 'filters' | 'geonav'>('ingest');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      onImageUploaded(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageUploaded(file);
      }
    }
  };

  const toggleSeverity = (sev: SeverityLevel) => {
    if (selectedSeverities.includes(sev)) {
      if (selectedSeverities.length > 1) {
        setSelectedSeverities(selectedSeverities.filter((s) => s !== sev));
      }
    } else {
      setSelectedSeverities([...selectedSeverities, sev]);
    }
  };

  const loadDatasetSample = (sampleId: string) => {
    onImageUploaded(null, sampleId);
  };

  return (
    <aside
      id="sidebar-container"
      className="w-full lg:w-84 bg-slate-900 border-r border-slate-800 p-4 lg:p-5 flex flex-col gap-4 shrink-0 h-auto lg:min-h-screen overflow-y-auto"
    >
      {/* Streamlit & Mission Control Header */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5 text-cyan-400 font-semibold text-xs tracking-wider uppercase">
            <Layers className="w-3.5 h-3.5" />
            <span>FastAPI & AI Pipeline</span>
          </div>
          <span className="text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800/80 px-2 py-0.5 rounded font-mono">
            v2.4.0
          </span>
        </div>
        <h2 className="text-base font-bold text-slate-100">Sonar Vision Control</h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Side-scan sonar pre-processing, YOLOv8-OBB, U-Net & PyXTF Geotagging.
        </p>
      </div>

      {/* Mode Navigation Tabs */}
      <div className="grid grid-cols-3 gap-1 bg-slate-950/70 p-1 rounded-lg border border-slate-800 text-xs">
        <button
          onClick={() => setActiveTab('ingest')}
          className={`py-1.5 px-2 rounded-md font-medium transition ${
            activeTab === 'ingest' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Ingestion
        </button>
        <button
          onClick={() => setActiveTab('filters')}
          className={`py-1.5 px-2 rounded-md font-medium transition ${
            activeTab === 'filters' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Lee / CLAHE
        </button>
        <button
          onClick={() => setActiveTab('geonav')}
          className={`py-1.5 px-2 rounded-md font-medium transition ${
            activeTab === 'geonav' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Towfish GPS
        </button>
      </div>

      {/* TAB 1: Ingestion & Benchmark Datasets */}
      {activeTab === 'ingest' && (
        <div className="space-y-4">
          {/* Feature 1: File Upload (Drag & Drop) */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
              <span>Upload Sonar Scan / XTF</span>
              <span className="text-[10px] text-cyan-400 font-mono">PNG / JPG / XTF</span>
            </label>

            <div
              id="file-upload-dropzone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-700 hover:border-cyan-500 bg-slate-950/60 hover:bg-slate-800/60 rounded-xl p-3.5 text-center cursor-pointer transition-all duration-200 group"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png, image/jpeg, image/jpg, .xtf"
                onChange={handleFileChange}
                className="hidden"
                id="sonar-file-input"
              />
              <div className="w-9 h-9 rounded-full bg-cyan-950/60 border border-cyan-800/80 text-cyan-400 mx-auto flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
                <Upload className="w-4 h-4" />
              </div>
              <p className="text-xs font-medium text-slate-200 group-hover:text-cyan-300">
                {fileName ? fileName : 'Drop sonar scan or XTF here'}
              </p>
              <p className="text-[10px] text-slate-500 mt-1">
                Supports SeabedObjects-KLSG & WATERS
              </p>
            </div>

            {isCustomImage && (
              <button
                id="reset-sample-btn"
                onClick={onResetToDemo}
                className="w-full flex items-center justify-center gap-1.5 py-1 px-3 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 bg-slate-800/50 hover:bg-slate-800 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reset to Standard Track</span>
              </button>
            )}
          </div>

          {/* Quick Dataset Samples */}
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span>Benchmark Dataset Samples</span>
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              <button
                onClick={() => loadDatasetSample('klsg-mine-uxo')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800 border border-slate-800 text-left transition"
              >
                <div className="text-[11px] font-bold text-red-400">KLSG Mine UXO</div>
                <div className="text-[10px] text-slate-400">900 kHz Cylindrical</div>
              </button>
              <button
                onClick={() => loadDatasetSample('klsg-shipwreck')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800 border border-slate-800 text-left transition"
              >
                <div className="text-[11px] font-bold text-amber-400">KLSG Shipwreck</div>
                <div className="text-[10px] text-slate-400">445 kHz Keel Frame</div>
              </button>
              <button
                onClick={() => loadDatasetSample('klsg-pipeline')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800 border border-slate-800 text-left transition"
              >
                <div className="text-[11px] font-bold text-emerald-400">KLSG Pipeline</div>
                <div className="text-[10px] text-slate-400">445 kHz Gas Trunk</div>
              </button>
              <button
                onClick={() => loadDatasetSample('waters-ghostnet')}
                className="p-2 rounded-lg bg-slate-950/80 hover:bg-slate-800 border border-slate-800 text-left transition"
              >
                <div className="text-[11px] font-bold text-amber-400">WATERS Net</div>
                <div className="text-[10px] text-slate-400">U-Net Ghost Net</div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Task 1 OpenCV Lee Filter & CLAHE Settings */}
      {activeTab === 'filters' && (
        <div className="space-y-4">
          {/* Lee Speckle Window */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
                <span>Lee Filter Window ($W$)</span>
              </label>
              <span className="text-xs font-mono font-bold text-cyan-400">
                {leeWindowSize} × {leeWindowSize} px
              </span>
            </div>
            <div className="grid grid-cols-4 gap-1">
              {[3, 5, 7, 9].map((size) => (
                <button
                  key={size}
                  onClick={() => setLeeWindowSize(size)}
                  className={`py-1 rounded text-xs font-mono transition ${
                    leeWindowSize === size
                      ? 'bg-cyan-600 text-white font-bold'
                      : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                  }`}
                >
                  {size}×{size}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-slate-500 leading-tight">
              Adaptive variance filtering: $W = \max(0, 1 - C_u^2 / C_i^2)$
            </p>
          </div>

          {/* CLAHE Clip Limit */}
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300">CLAHE Clip Limit</label>
              <span className="text-xs font-mono font-bold text-cyan-400">
                {claheClipLimit.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="1.0"
              max="5.0"
              step="0.2"
              value={claheClipLimit}
              onChange={(e) => setClaheClipLimit(parseFloat(e.target.value))}
              className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>1.0 (Mild)</span>
              <span>3.0 (Std)</span>
              <span>5.0 (High Contrast)</span>
            </div>
          </div>

          {/* Palette Selector */}
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <label className="text-xs font-semibold text-slate-300">Colormap Palette</label>
            <div className="grid grid-cols-2 gap-1.5">
              {(
                [
                  { id: 'amber', name: 'Amber Copper', color: 'bg-amber-600' },
                  { id: 'cyan', name: 'Deep Cyan', color: 'bg-cyan-600' },
                  { id: 'green', name: 'Phosphor Green', color: 'bg-emerald-600' },
                  { id: 'grayscale', name: 'Grayscale', color: 'bg-slate-400' },
                ] as const
              ).map((item) => (
                <button
                  key={item.id}
                  id={`palette-btn-${item.id}`}
                  onClick={() => setPalette(item.id)}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs font-medium border transition ${
                    palette === item.id
                      ? 'border-cyan-500 bg-cyan-950/40 text-cyan-200'
                      : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  <span className={`w-2 h-2 rounded-full ${item.color}`} />
                  <span className="truncate">{item.name}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Task 3 PyXTF Towfish Georeferencing Nav */}
      {activeTab === 'geonav' && (
        <div className="space-y-3 text-xs">
          <div className="space-y-1">
            <label className="text-[11px] font-semibold text-slate-300 flex items-center gap-1">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              <span>Towfish Heading (θ)</span>
            </label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                value={towfishNav.heading_degrees}
                onChange={(e) =>
                  setTowfishNav({ ...towfishNav, heading_degrees: parseFloat(e.target.value) || 0 })
                }
                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono"
              />
              <span className="text-slate-400 font-mono">deg</span>
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-[11px] font-semibold text-slate-300">Towfish Altitude ($H$)</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                step="0.5"
                value={towfishNav.altitude_meters}
                onChange={(e) =>
                  setTowfishNav({ ...towfishNav, altitude_meters: parseFloat(e.target.value) || 0 })
                }
                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 font-mono"
              />
              <span className="text-slate-400 font-mono">m</span>
            </div>
            <p className="text-[10px] text-slate-500">Altitude above seafloor used for height calc</p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-slate-400">Towfish Lat</label>
              <input
                type="number"
                step="0.0001"
                value={towfishNav.latitude}
                onChange={(e) =>
                  setTowfishNav({ ...towfishNav, latitude: parseFloat(e.target.value) || 0 })
                }
                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono text-[11px]"
              />
            </div>
            <div>
              <label className="text-[10px] text-slate-400">Towfish Lon</label>
              <input
                type="number"
                step="0.0001"
                value={towfishNav.longitude}
                onChange={(e) =>
                  setTowfishNav({ ...towfishNav, longitude: parseFloat(e.target.value) || 0 })
                }
                className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 font-mono text-[11px]"
              />
            </div>
          </div>
        </div>
      )}

      {/* Global Confidence & Severity Thresholds */}
      <div className="space-y-2 pt-2 border-t border-slate-800/80">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            <span>AI Confidence Cutoff</span>
          </label>
          <span className="text-xs font-mono font-bold text-cyan-400">
            {(minConfidence * 100).toFixed(0)}%
          </span>
        </div>
        <input
          id="confidence-slider"
          type="range"
          min="0.50"
          max="0.99"
          step="0.01"
          value={minConfidence}
          onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
          className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Severity Filter */}
      <div className="space-y-1.5 pt-1 border-t border-slate-800/80">
        <label className="text-xs font-semibold text-slate-300">Filter Risk Severity</label>
        <div className="space-y-1">
          {(
            [
              { id: 'Red', label: 'Red (High Risk / UXO)', color: 'text-red-400', badge: 'bg-red-500/20 border-red-500/30' },
              { id: 'Yellow', label: 'Yellow (Medium Risk / Debris)', color: 'text-amber-400', badge: 'bg-amber-500/20 border-amber-500/30' },
              { id: 'Green', label: 'Green (Low Risk / Pipeline)', color: 'text-emerald-400', badge: 'bg-emerald-500/20 border-emerald-500/30' },
            ] as const
          ).map((item) => {
            const isChecked = selectedSeverities.includes(item.id);
            return (
              <button
                key={item.id}
                id={`severity-toggle-${item.id}`}
                onClick={() => toggleSeverity(item.id)}
                className={`w-full flex items-center justify-between p-1.5 rounded-lg border text-xs text-left transition ${
                  isChecked
                    ? `${item.badge} border`
                    : 'border-slate-800/60 bg-slate-950/20 text-slate-500'
                }`}
              >
                <div className="flex items-center gap-2">
                  {isChecked ? (
                    <CheckSquare className={`w-3.5 h-3.5 ${item.color}`} />
                  ) : (
                    <Square className="w-3.5 h-3.5 text-slate-600" />
                  )}
                  <span className={isChecked ? 'text-slate-200 font-medium text-[11px]' : 'text-slate-500 text-[11px]'}>
                    {item.label}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Run Complete FastAPI Pipeline Button */}
      <button
        onClick={onRunLivePipeline}
        disabled={isProcessing}
        className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-900/30 transition active:scale-95 disabled:opacity-50 mt-auto"
      >
        <Zap className={`w-4 h-4 ${isProcessing ? 'animate-spin' : ''}`} />
        <span>{isProcessing ? 'Processing Pipeline...' : 'Run Pipeline (/upload-sonar)'}</span>
      </button>
    </aside>
  );
};
