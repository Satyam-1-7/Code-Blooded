import React from 'react';
import { Waves, ShieldAlert, Sparkles, Activity, MapPin, Compass } from 'lucide-react';
import { SonarTarget, SonarSurveyData } from '../types';

interface HeaderProps {
  surveyData: SonarSurveyData;
  activeTargets: SonarTarget[];
}

export const Header: React.FC<HeaderProps> = ({
  surveyData,
  activeTargets,
}) => {
  const highRiskCount = activeTargets.filter(t => t.severity === 'Red').length;
  const medRiskCount = activeTargets.filter(t => t.severity === 'Yellow').length;
  const lowRiskCount = activeTargets.filter(t => t.severity === 'Green').length;

  return (
    <header className="space-y-4">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/90 border border-slate-800 p-5 rounded-2xl">
        <div className="flex items-center gap-3.5">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-950">
            <Waves className="w-7 h-7" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
                Sonar Vision-X
              </h1>
              <span className="text-[10px] font-mono font-bold bg-cyan-950 border border-cyan-700/60 text-cyan-300 px-2 py-0.5 rounded-full">
                Mission Control
              </span>
            </div>
            <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
              Subsea Sonar Imagery Target Classification, Geospatial Mapping & Automated Mission Reporting
            </p>
          </div>
        </div>

        {/* Survey Info Pills */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-800 text-xs font-mono text-slate-300">
            <MapPin className="w-3.5 h-3.5 text-cyan-400" />
            <span>{surveyData.survey_id}</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-950/70 border border-slate-800 text-xs font-mono text-cyan-300">
            <Compass className="w-3.5 h-3.5 text-cyan-400" />
            <span>445 kHz Hydroacoustic</span>
          </div>
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
            <span className="text-[10px] text-red-300/80">UXO / Naval Mine</span>
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
            <span className="text-[10px] text-amber-300/80">Submerged Debris</span>
          </div>
          <div className="w-10 h-10 rounded-lg bg-amber-950/60 border border-amber-800/50 flex items-center justify-center text-amber-400">
            <Compass className="w-5 h-5" />
          </div>
        </div>

        {/* Metric 4: Low Risk (Green) */}
        <div className="bg-slate-900/80 border border-emerald-900/40 p-4 rounded-xl flex items-center justify-between">
          <div>
            <span className="text-[11px] font-medium text-emerald-400 uppercase tracking-wider">Low Risk (Green)</span>
            <div className="text-2xl font-bold text-emerald-400 font-mono mt-0.5">{lowRiskCount}</div>
            <span className="text-[10px] text-emerald-300/80">Natural Outcrop</span>
          </div>
          <div className="w-10 h-10 rounded-lg bg-emerald-950/60 border border-emerald-800/50 flex items-center justify-center text-emerald-400">
            <Sparkles className="w-5 h-5" />
          </div>
        </div>
      </div>
    </header>
  );
};
