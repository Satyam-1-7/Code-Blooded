import React from 'react';
import { Waves, ShieldAlert, Sparkles, Activity, Layers, MapPin, Table as TableIcon, Download, Grid } from 'lucide-react';
import { SonarTarget, SonarSurveyData } from '../types';

export type ConsoleTab = 'all' | 'dual' | 'map' | 'telemetry' | 'export';

interface HeaderProps {
  surveyData: SonarSurveyData;
  activeTargets: SonarTarget[];
  activeTab: ConsoleTab;
  setActiveTab: (tab: ConsoleTab) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTargets,
  activeTab,
  setActiveTab,
}) => {
  const highRiskCount = activeTargets.filter(t => t.severity === 'Red').length;
  const medRiskCount = activeTargets.filter(t => t.severity === 'Yellow').length;
  const lowRiskCount = activeTargets.filter(t => t.severity === 'Green').length;

  const tabs: { id: ConsoleTab; label: string; icon: React.ReactNode; badge?: number | string }[] = [
    { id: 'all', label: 'All Sections', icon: <Grid className="w-3.5 h-3.5" /> },
    { id: 'dual', label: 'Acoustic Dual-View', icon: <Layers className="w-3.5 h-3.5" />, badge: activeTargets.length },
    { id: 'map', label: 'Geospatial Map', icon: <MapPin className="w-3.5 h-3.5" />, badge: activeTargets.length },
    { id: 'telemetry', label: 'Target Telemetry', icon: <TableIcon className="w-3.5 h-3.5" /> },
    { id: 'export', label: 'Export Reports', icon: <Download className="w-3.5 h-3.5" /> },
  ];

  return (
    <header className="bg-slate-950/60 p-4 sm:p-5 space-y-4 border-b border-slate-800/80">
      {/* Top Row: Title + Integrated Telemetry Bar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-950/50">
            <Waves className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
              Sonar Vision-X
            </h1>
            <p className="text-[11px] text-slate-400 font-mono">
              Autonomous Acoustic AI & Subsea Telemetry Console
            </p>
          </div>
        </div>

        {/* Integrated Metrics Status Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 bg-slate-900/90 border border-slate-800 rounded-xl divide-x divide-slate-800/80 shadow-inner">
          {/* Total Targets */}
          <div className="px-3.5 py-2 flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-slate-800/80 flex items-center justify-center text-cyan-400 flex-shrink-0">
              <Activity className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-semibold tracking-wider text-slate-400">Total</div>
              <div className="text-base font-bold font-mono text-slate-100 leading-tight">{activeTargets.length}</div>
            </div>
          </div>

          {/* High Risk (Red) */}
          <div className="px-3.5 py-2 flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-red-950/70 border border-red-800/40 flex items-center justify-center text-red-400 flex-shrink-0">
              <ShieldAlert className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-semibold tracking-wider text-red-400">High (Red)</div>
              <div className="text-base font-bold font-mono text-red-400 leading-tight">{highRiskCount}</div>
            </div>
          </div>

          {/* Medium Risk (Yellow) */}
          <div className="px-3.5 py-2 flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-amber-950/70 border border-amber-800/40 flex items-center justify-center text-amber-400 flex-shrink-0">
              <Sparkles className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-semibold tracking-wider text-amber-400">Med (Yellow)</div>
              <div className="text-base font-bold font-mono text-amber-400 leading-tight">{medRiskCount}</div>
            </div>
          </div>

          {/* Low Risk (Green) */}
          <div className="px-3.5 py-2 flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-emerald-950/70 border border-emerald-800/40 flex items-center justify-center text-emerald-400 flex-shrink-0">
              <Activity className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[10px] uppercase font-semibold tracking-wider text-emerald-400">Nominal</div>
              <div className="text-base font-bold font-mono text-emerald-400 leading-tight">{lowRiskCount}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tab Strip */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 pt-1 border-t border-slate-800/70">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer whitespace-nowrap ${
                isActive
                  ? 'bg-cyan-600 text-white shadow-md shadow-cyan-950/50'
                  : 'bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-slate-100 border border-slate-800'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
              {tab.badge !== undefined && (
                <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded-full ${
                  isActive ? 'bg-cyan-900/80 text-cyan-200' : 'bg-slate-800 text-slate-400'
                }`}>
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </header>
  );
};
