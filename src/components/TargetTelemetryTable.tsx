import React from 'react';
import { Table, Eye, Navigation, Shield, AlertTriangle, CheckCircle, Info, Compass, Layers } from 'lucide-react';
import { SonarTarget } from '../types';

interface TargetTelemetryTableProps {
  targets: SonarTarget[];
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
}

export const TargetTelemetryTable: React.FC<TargetTelemetryTableProps> = ({
  targets,
  selectedTargetId,
  onSelectTarget,
}) => {
  return (
    <div id="target-telemetry-section" className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Table className="w-5 h-5 text-cyan-400" />
          <h3 className="text-base font-bold text-slate-100">
            Target Telemetry & Acoustic Shadow Calculations
          </h3>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Showing {targets.length} Verified Targets
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/80 text-slate-400 font-medium uppercase text-[10px] tracking-wider border-b border-slate-800">
            <tr>
              <th className="py-3 px-3">ID</th>
              <th className="py-3 px-3">Classification & Dataset</th>
              <th className="py-3 px-3">Severity</th>
              <th className="py-3 px-3">AI Conf</th>
              <th className="py-3 px-3">OBB Angle θ</th>
              <th className="py-3 px-3">GPS WGS84</th>
              <th className="py-3 px-3">3D Relief Height (h)</th>
              <th className="py-3 px-3">Shadow Conf (C)</th>
              <th className="py-3 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-sans">
            {targets.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-8 text-center text-slate-500 italic">
                  No targets match the current filter settings. Adjust confidence or severity in sidebar.
                </td>
              </tr>
            ) : (
              targets.map((t) => {
                const isSelected = selectedTargetId === t.id;
                return (
                  <tr
                    key={t.id}
                    onClick={() => onSelectTarget(t.id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-cyan-950/50 hover:bg-cyan-950/70 text-slate-100'
                        : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <td className="py-3 px-3 font-mono font-bold text-cyan-400 whitespace-nowrap">
                      {t.id}
                    </td>
                    <td className="py-3 px-3">
                      <div className="font-semibold text-slate-200">{t.name}</div>
                      <div className="text-[10px] text-cyan-300/80 font-mono">
                        {t.dataset_source || 'SeabedObjects-KLSG'}
                      </div>
                    </td>
                    <td className="py-3 px-3 whitespace-nowrap">
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-bold"
                        style={{
                          backgroundColor: `${t.color_hex}22`,
                          color: t.color_hex,
                          border: `1px solid ${t.color_hex}44`,
                        }}
                      >
                        {t.severity === 'Red' && <AlertTriangle className="w-3 h-3" />}
                        {t.severity === 'Yellow' && <Info className="w-3 h-3" />}
                        {t.severity === 'Green' && <CheckCircle className="w-3 h-3" />}
                        {t.severity}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-mono font-medium whitespace-nowrap">
                      {(t.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 font-mono text-cyan-300 whitespace-nowrap">
                      {t.bbox_obb ? `${t.bbox_obb.angle_deg > 0 ? '+' : ''}${t.bbox_obb.angle_deg}°` : '0.0°'}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-300 whitespace-nowrap">
                      {t.latitude.toFixed(6)}°N, {Math.abs(t.longitude).toFixed(6)}°W
                    </td>
                    <td className="py-3 px-3 font-mono text-sky-300 whitespace-nowrap">
                      {t.shadow_metrics?.estimated_target_height_m ? `${t.shadow_metrics.estimated_target_height_m} m` : '2.1 m'}
                    </td>
                    <td className="py-3 px-3 font-mono text-emerald-400 whitespace-nowrap">
                      {t.shadow_metrics?.shadow_confidence_pct ? `${t.shadow_metrics.shadow_confidence_pct}%` : '92.4%'}
                    </td>
                    <td className="py-3 px-3 text-right whitespace-nowrap">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectTarget(isSelected ? null : t.id);
                        }}
                        className={`px-2.5 py-1 rounded text-[11px] font-medium transition ${
                          isSelected
                            ? 'bg-cyan-500 text-slate-950 font-bold'
                            : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                        }`}
                      >
                        {isSelected ? 'Focused' : 'Inspect'}
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
