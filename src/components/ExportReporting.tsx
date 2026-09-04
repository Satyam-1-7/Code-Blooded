import React from 'react';
import { Download, FileSpreadsheet, FileText, CheckCircle2, ShieldAlert } from 'lucide-react';
import { SonarSurveyData, SonarTarget } from '../types';
import { exportTargetsToCSV, exportReportLabStyledPDF } from '../utils/exportUtils';

interface ExportReportingProps {
  surveyData: SonarSurveyData;
  activeTargets: SonarTarget[];
}

export const ExportReporting: React.FC<ExportReportingProps> = ({
  surveyData,
  activeTargets
}) => {
  const handleCSVDownload = () => {
    exportTargetsToCSV(surveyData, activeTargets);
  };

  const handlePDFDownload = () => {
    exportReportLabStyledPDF(surveyData, activeTargets);
  };

  return (
    <div id="export-reporting-section" className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
      <div>
        <div className="flex items-center gap-2">
          <Download className="w-5 h-5 text-cyan-400" />
          <h3 className="text-base font-bold text-slate-100">
            Export & Inspection Reporting
          </h3>
        </div>
        <p className="text-xs text-slate-400 mt-0.5">
          Generate survey deliverables: raw CSV telemetry dataset and formatted PDF inspection report.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* CSV Export Card */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
          <div className="space-y-1.5 mb-4">
            <div className="flex items-center gap-2 text-emerald-400">
              <FileSpreadsheet className="w-5 h-5" />
              <h4 className="text-sm font-semibold text-slate-200">CSV Telemetry Summary</h4>
            </div>
            <p className="text-xs text-slate-400">
              Complete tabular dataset containing target IDs, severity ratings, WGS84 coordinates, depths, and dimensions.
            </p>
            <div className="text-[11px] font-mono text-slate-500 pt-1">
              Format: <span className="text-slate-300">.csv (UTF-8)</span> • {activeTargets.length} records
            </div>
          </div>

          <button
            id="download-csv-btn"
            onClick={handleCSVDownload}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-md shadow-emerald-950/40 transition transform active:scale-95 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            <span>Download CSV Summary</span>
          </button>
        </div>

        {/* PDF Export Card (ReportLab styled) */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 flex flex-col justify-between hover:border-slate-700 transition">
          <div className="space-y-1.5 mb-4">
            <div className="flex items-center gap-2 text-red-400">
              <FileText className="w-5 h-5" />
              <h4 className="text-sm font-semibold text-slate-200">ReportLab PDF Report</h4>
            </div>
            <p className="text-xs text-slate-400">
              Formatted hydrographic inspection document with executive summary, color-coded telemetry matrix, and safety directives.
            </p>
            <div className="text-[11px] font-mono text-slate-500 pt-1">
              Format: <span className="text-slate-300">.pdf (Letter)</span> • Hydrographic Standard
            </div>
          </div>

          <button
            id="download-pdf-btn"
            onClick={handlePDFDownload}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-xs font-semibold bg-red-600 hover:bg-red-500 text-white shadow-md shadow-red-950/40 transition transform active:scale-95 cursor-pointer"
          >
            <Download className="w-4 h-4" />
            <span>Download PDF Report</span>
          </button>
        </div>
      </div>
    </div>
  );
};
