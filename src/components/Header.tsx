import React, { useState } from 'react';
import { Waves, ShieldAlert, Sparkles, Activity, MapPin, Compass, Cpu, X, Layers } from 'lucide-react';
import { SonarTarget, SonarSurveyData } from '../types';

interface HeaderProps {
  surveyData: SonarSurveyData;
  activeTargets: SonarTarget[];
}

export const Header: React.FC<HeaderProps> = ({
  surveyData,
  activeTargets,
}) => {
  const [showAiModal, setShowAiModal] = useState<boolean>(false);
  const highRiskCount = activeTargets.filter(t => t.severity === 'Red').length;
  const medRiskCount = activeTargets.filter(t => t.severity === 'Yellow').length;
  const lowRiskCount = activeTargets.filter(t => t.severity === 'Green').length;

  return (
    <header className="space-y-4">
      {/* Top Banner */}
      <div className="flex items-center justify-between bg-slate-900/90 border border-slate-800 p-4 sm:p-5 rounded-2xl">
        <div className="flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-950">
            <Waves className="w-6 h-6" />
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
            Sonar Vision-X
          </h1>
        </div>
      </div>

      {/* 4 Summary Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Metric 1: Total Targets */}
        <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">Total Targets</span>
            <div className="text-2xl font-bold text-slate-100 font-mono mt-0.5">{activeTargets.length}</div>
            <span className="text-[10px] text-slate-500">Active acoustic hits</span>
          </div>
          <div className="w-10 h-10 rounded-lg bg-slate-800/80 flex items-center justify-center text-cyan-400">
            <Activity className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 2: High Risk (Red) */}
        <div className="bg-slate-900/80 border border-red-900/40 p-4 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[11px] font-medium text-red-400 uppercase tracking-wider">High Risk (Red)</span>
            <div className="text-2xl font-bold text-red-400 font-mono mt-0.5">{highRiskCount}</div>
            <span className="text-[10px] text-red-300/80">Aircraft / Shipwreck</span>
          </div>
          <div className="w-10 h-10 rounded-lg bg-red-950/60 border border-red-800/50 flex items-center justify-center text-red-400">
            <ShieldAlert className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 3: Medium Risk (Yellow) */}
        <div className="bg-slate-900/80 border border-amber-900/40 p-4 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[11px] font-medium text-amber-400 uppercase tracking-wider">Medium Risk (Yellow)</span>
            <div className="text-2xl font-bold text-amber-400 font-mono mt-0.5">{medRiskCount}</div>
            <span className="text-[10px] text-amber-300/80">Seabed Debris / Contacts</span>
          </div>
          <div className="w-10 h-10 rounded-lg bg-amber-950/60 border border-amber-800/50 flex items-center justify-center text-amber-400">
            <Sparkles className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 4: Low Risk (Green) */}
        <div className="bg-slate-900/80 border border-emerald-900/40 p-4 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[11px] font-medium text-emerald-400 uppercase tracking-wider">Nominal Assets</span>
            <div className="text-2xl font-bold text-emerald-400 font-mono mt-0.5">{lowRiskCount}</div>
            <span className="text-[10px] text-emerald-300/80">Marine Biomass / Fish</span>
          </div>
          <div className="w-10 h-10 rounded-lg bg-emerald-950/60 border border-emerald-800/50 flex items-center justify-center text-emerald-400">
            <Activity className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* AI Engine & Trained YOLO Modal */}
      {showAiModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 space-y-5 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-cyan-950 border border-cyan-700/60 text-cyan-400">
                  <Cpu className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-100">AI Model Architecture & Inference Engine</h3>
                  <p className="text-xs text-slate-400">Trained YOLO Neural Network Active: <code className="text-cyan-300 font-mono">backend/best.pt</code></p>
                </div>
              </div>
              <button
                onClick={() => setShowAiModal(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans">
              {/* Box 1: Trained YOLO Model */}
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-cyan-400 flex items-center gap-1.5">
                    <Layers className="w-4 h-4" />
                    YOLOv8 Detection
                  </span>
                  <span className="font-mono text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800 px-1.5 py-0.5 rounded">
                    best.pt Active
                  </span>
                </div>
                <p className="text-slate-300 text-[11px] leading-relaxed">
                  Trained deep neural network detector integrated directly with the backend FastAPI pipeline.
                </p>
                <div className="pt-2 border-t border-slate-900 space-y-1 font-mono text-[10px] text-slate-400">
                  <div>• Weights: backend/best.pt (22.5 MB)</div>
                  <div>• Classes: aircraft, fish, other, shipwreck</div>
                  <div>• Framework: Ultralytics PyTorch</div>
                </div>
              </div>

              {/* Box 2: Acoustic Shadow 3D Verification */}
              <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-amber-400 flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4" />
                    Acoustic Shadow Math
                  </span>
                  <span className="font-mono text-[10px] bg-amber-950 text-amber-300 border border-amber-800 px-1.5 py-0.5 rounded">
                    h = (H·Ls)/(R+Ls)
                  </span>
                </div>
                <p className="text-slate-300 text-[11px] leading-relaxed">
                  Real-time hydrodynamic raycasting verifies 3D target relief height and rejects flat seabed false positives.
                </p>
                <div className="pt-2 border-t border-slate-900 space-y-1 font-mono text-[10px] text-slate-400">
                  <div>• Georeferencing: PyXTF / WGS84</div>
                  <div>• Filtering: Lee Speckle + CLAHE</div>
                  <div>• Scorecard: Multi-Class Softmax Vector</div>
                </div>
              </div>
            </div>

            {/* Hardware & Latency Benchmarks */}
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-2 text-xs">
              <div className="flex items-center justify-between font-mono">
                <span className="text-slate-400">Backend Engine:</span>
                <span className="text-emerald-400 font-bold">FastAPI + PyTorch (best.pt)</span>
              </div>
              <div className="flex items-center justify-between font-mono">
                <span className="text-slate-400">Average Inference Latency:</span>
                <span className="text-cyan-400 font-bold">32.5 ms (Real-Time 30+ FPS)</span>
              </div>
              <div className="flex items-center justify-between font-mono">
                <span className="text-slate-400">Target Pipeline:</span>
                <span className="text-slate-200 font-bold">Sonar Ingestion → Lee/CLAHE → YOLO → Geotagging</span>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setShowAiModal(false)}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold text-xs rounded-lg transition"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
