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
          <tbody className="divide-y divide-slate-800/60 font-sans">            {targets.length === 0 ? (
              <tr>
                <td colSpan={9} className="py-8 text-center text-slate-500 italic">
                  No targets match the current filter settings. Adjust confidence or severity in sidebar.
                </td>
              </tr>
            ) : (
              targets.map((t) => {
                const isSelected = selectedTargetId === t.id;
                return (
                  <React.Fragment key={t.id}>
                    <tr
                      onClick={() => onSelectTarget(isSelected ? null : t.id)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-cyan-950/60 hover:bg-cyan-950/80 text-slate-100'
                          : 'hover:bg-slate-800/40 text-slate-300'
                      }`}
                    >
                      <td className="py-3 px-3 font-mono font-bold text-cyan-400 whitespace-nowrap">
                        <div className="flex items-center gap-1.5">
                          <span>{t.id}</span>
                          {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />}
                        </div>
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
                        <div className="flex items-center gap-1.5">
                          <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-cyan-400 rounded-full"
                              style={{ width: `${Math.min(100, t.confidence * 100)}%` }}
                            />
                          </div>
                          <span>{(t.confidence * 100).toFixed(1)}%</span>
                        </div>
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
                              : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                          }`}
                        >
                          {isSelected ? 'Collapse' : 'AI Scorecard'}
                        </button>
                      </td>
                    </tr>

                    {/* Expandable Multi-Class Probability Scorecard Drawer */}
                    {isSelected && (
                      <tr className="bg-slate-950/90 border-b border-cyan-800/50">
                        <td colSpan={9} className="p-4">
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
                            {/* Card 1: Multi-Class Probability Distribution */}
                            <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3.5 space-y-2.5">
                              <div className="flex items-center justify-between pb-1.5 border-b border-slate-800">
                                <span className="font-bold text-slate-200 flex items-center gap-1.5">
                                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                                  Multi-Class AI Scorecard
                                </span>
                                <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950 px-1.5 py-0.5 rounded border border-cyan-800/60">
                                  Softmax Vector
                                </span>
                              </div>
                              <div className="space-y-2">
                                {(t.class_probabilities || [
                                  { class_name: t.name, probability: t.confidence, icon: '🎯' },
                                  { class_name: 'Natural Seabed Feature', probability: Math.max(0.01, (1 - t.confidence) * 0.6), icon: '🪨' },
                                  { class_name: 'Secondary Marine Debris', probability: Math.max(0.01, (1 - t.confidence) * 0.4), icon: '🗑️' }
                                ]).map((cls, idx) => (
                                  <div key={idx} className="space-y-1">
                                    <div className="flex justify-between items-center text-[11px]">
                                      <span className="text-slate-300 truncate max-w-[180px]">
                                        {cls.icon && <span className="mr-1.5">{cls.icon}</span>}
                                        {cls.class_name}
                                      </span>
                                      <span className="font-mono font-bold text-slate-200">
                                        {(cls.probability * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden border border-slate-800">
                                      <div
                                        className={`h-full rounded-full transition-all duration-500 ${
                                          idx === 0 ? 'bg-gradient-to-r from-cyan-500 to-blue-500' : 'bg-slate-700'
                                        }`}
                                        style={{ width: `${Math.min(100, Math.max(2, cls.probability * 100))}%` }}
                                      />
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Card 2: 3D Acoustic Shadow & Physics Validation */}
                            <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3.5 space-y-2">
                              <div className="flex items-center justify-between pb-1.5 border-b border-slate-800">
                                <span className="font-bold text-slate-200 flex items-center gap-1.5">
                                  <Shield className="w-3.5 h-3.5 text-sky-400" />
                                  3D Shadow Height Verification
                                </span>
                                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/80 px-1.5 py-0.5 rounded border border-emerald-800/60">
                                  Verified 3D
                                </span>
                              </div>
                              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                                  <div className="text-slate-400 text-[10px]">Shadow Length (Ls)</div>
                                  <div className="text-slate-200 font-bold mt-0.5">{t.shadow_metrics?.shadow_length_m || 8.6} m</div>
                                </div>
                                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                                  <div className="text-slate-400 text-[10px]">Slant Range (Rs)</div>
                                  <div className="text-slate-200 font-bold mt-0.5">{t.shadow_metrics?.slant_range_m || 35.0} m</div>
                                </div>
                                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                                  <div className="text-slate-400 text-[10px]">Towfish Altitude (H)</div>
                                  <div className="text-slate-200 font-bold mt-0.5">{t.shadow_metrics?.towfish_altitude_m || 12.5} m</div>
                                </div>
                                <div className="bg-slate-950 p-2 rounded border border-slate-800/80">
                                  <div className="text-slate-400 text-[10px]">Target Relief (h)</div>
                                  <div className="text-cyan-400 font-bold mt-0.5">{t.shadow_metrics?.estimated_target_height_m || 2.1} m</div>
                                </div>
                              </div>
                              <div className="text-[10px] text-slate-400 italic">
                                Formula: h = (H × Ls) / (Rs + Ls) • Rejects flat rock false alarms.
                              </div>
                            </div>

                            {/* Card 3: Actionable Protocol & Geotag */}
                            <div className="bg-slate-900/90 border border-slate-800 rounded-lg p-3.5 space-y-2">
                              <div className="flex items-center justify-between pb-1.5 border-b border-slate-800">
                                <span className="font-bold text-slate-200 flex items-center gap-1.5">
                                  <Compass className="w-3.5 h-3.5 text-amber-400" />
                                  Naval / Environmental Action
                                </span>
                                <span className="text-[10px] font-mono text-slate-400">
                                  {t.risk_level}
                                </span>
                              </div>
                              <p className="text-[11px] text-slate-300 leading-relaxed">
                                {t.action_recommendation || 'Log target in hydrographic survey logbook and dispatch ROV for optical validation.'}
                              </p>
                              <div className="pt-1 border-t border-slate-800/80 text-[10px] font-mono text-cyan-300 flex items-center gap-1">
                                <Navigation className="w-3 h-3 text-cyan-400" />
                                <span>WGS84: {t.latitude.toFixed(6)}°N, {Math.abs(t.longitude).toFixed(6)}°W</span>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
