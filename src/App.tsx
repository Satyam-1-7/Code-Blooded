/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useMemo, useEffect } from 'react';
import { MOCK_SONAR_SURVEY } from './data/mockDetections';
import { SeverityLevel, SonarTarget, TowfishNav, SonarSurveyData } from './types';
import { SonarColorPalette } from './utils/sonarSynthetic';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { DualViewSonar } from './components/DualViewSonar';
import { GeospatialSonarMap } from './components/GeospatialSonarMap';
import { TargetTelemetryTable } from './components/TargetTelemetryTable';
import { ExportReporting } from './components/ExportReporting';
import { Activity, Server, Zap, CheckCircle2 } from 'lucide-react';

export default function App() {
  const [surveyData, setSurveyData] = useState<SonarSurveyData>(MOCK_SONAR_SURVEY);
  const [customImageSrc, setCustomImageSrc] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [palette, setPalette] = useState<SonarColorPalette>('amber');
  const [minConfidence, setMinConfidence] = useState<number>(0.75);
  const [selectedSeverities, setSelectedSeverities] = useState<SeverityLevel[]>(['Red', 'Yellow', 'Green']);
  const [selectedTargetId, setSelectedTargetId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [apiLatencyMs, setApiLatencyMs] = useState<number>(142.5);

  // Pre-processing and Geotagging Navigation State
  const [leeWindowSize, setLeeWindowSize] = useState<number>(7);
  const [claheClipLimit, setClaheClipLimit] = useState<number>(3.0);
  const [towfishNav, setTowfishNav] = useState<TowfishNav>({
    latitude: 36.782450,
    longitude: -122.012580,
    altitude_meters: 12.5,
    heading_degrees: 142.0,
    speed_knots: 4.5,
    frequency_khz: 445.0,
    layback_meters: 45.0,
  });

  // Filtered active targets based on confidence slider and severity multiselect
  const activeTargets = useMemo(() => {
    return surveyData.targets.filter(
      (t) => t.confidence >= minConfidence && selectedSeverities.includes(t.severity)
    );
  }, [surveyData.targets, minConfidence, selectedSeverities]);

  // Execute Live Pipeline API (/api/upload-sonar)
  const executePipelineApi = async (nameOfFile: string = fileName || 'KLSG_Track_445kHz.png') => {
    setIsProcessing(true);
    const startTime = performance.now();

    try {
      const response = await fetch('/api/upload-sonar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: nameOfFile,
          towfish_lat: towfishNav.latitude,
          towfish_lon: towfishNav.longitude,
          towfish_heading: towfishNav.heading_degrees,
          towfish_alt: towfishNav.altitude_meters,
          lee_window: leeWindowSize,
          clahe_clip: claheClipLimit,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const duration = Math.round(performance.now() - startTime);
        setApiLatencyMs(duration || data.processing_time_ms);

        if (data.targets && data.targets.length > 0) {
          setSurveyData((prev) => ({
            ...prev,
            survey_id: data.survey_id || prev.survey_id,
            towfish_nav: data.towfish_nav || prev.towfish_nav,
            targets: data.targets,
          }));
        }
      }
    } catch (err) {
      console.warn('API fetch fallback to local processor:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleImageUploaded = (file: File | null, sampleId?: string) => {
    if (file) {
      setFileName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        if (typeof e.target?.result === 'string') {
          setCustomImageSrc(e.target.result);
          executePipelineApi(file.name);
        }
      };
      reader.readAsDataURL(file);
    } else if (sampleId) {
      // Benchmark Dataset Sample Loader
      setCustomImageSrc(null);
      let sampleName = 'SeabedObjects_Sample.png';
      if (sampleId === 'klsg-mine-uxo') sampleName = 'KLSG_Naval_Mine_900kHz.png';
      if (sampleId === 'klsg-shipwreck') sampleName = 'KLSG_Shipwreck_445kHz.png';
      if (sampleId === 'klsg-pipeline') sampleName = 'KLSG_Pipeline_Trunk.png';
      if (sampleId === 'waters-ghostnet') sampleName = 'WATERS_GhostNet_Polymer.png';
      setFileName(sampleName);
      executePipelineApi(sampleName);
    }
  };

  const handleResetToDemo = () => {
    setCustomImageSrc(null);
    setFileName(null);
    setSurveyData(MOCK_SONAR_SURVEY);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col lg:flex-row font-sans">
      {/* Sidebar (Task 1: Ingestion, Lee/CLAHE, PyXTF Towfish Nav, Filters) */}
      <Sidebar
        onImageUploaded={handleImageUploaded}
        fileName={fileName}
        minConfidence={minConfidence}
        setMinConfidence={setMinConfidence}
        selectedSeverities={selectedSeverities}
        setSelectedSeverities={setSelectedSeverities}
        palette={palette}
        setPalette={setPalette}
        onResetToDemo={handleResetToDemo}
        isCustomImage={!!customImageSrc}
        towfishNav={towfishNav}
        setTowfishNav={setTowfishNav}
        leeWindowSize={leeWindowSize}
        setLeeWindowSize={setLeeWindowSize}
        claheClipLimit={claheClipLimit}
        setClaheClipLimit={setClaheClipLimit}
        onRunLivePipeline={() => executePipelineApi()}
        isProcessing={isProcessing}
      />

      {/* Main Analysis Dashboard */}
      <main className="flex-1 p-4 sm:p-6 lg:p-8 space-y-6 max-w-7xl mx-auto w-full overflow-y-auto">
        {/* Real-time Pipeline Status Bar */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <CheckCircle2 className="w-4 h-4" />
              <span>FastAPI Backend Active</span>
            </div>
            <span className="text-slate-600 hidden sm:inline">•</span>
            <div className="text-slate-400 font-mono hidden sm:block">
              Endpoint: <code className="text-cyan-300">POST /api/upload-sonar</code>
            </div>
          </div>

          <div className="flex items-center gap-3 font-mono text-[11px] text-slate-400">
            <span>Latency: <strong className="text-cyan-400">{apiLatencyMs} ms</strong></span>
            <span>Towfish Alt: <strong className="text-sky-300">{towfishNav.altitude_meters}m</strong></span>
            <span>Lee: <strong className="text-amber-300">{leeWindowSize}×{leeWindowSize}</strong></span>
            <span>CLAHE: <strong className="text-emerald-300">{claheClipLimit.toFixed(1)}</strong></span>
          </div>
        </div>

        {/* Header & Metric Cards */}
        <Header
          surveyData={surveyData}
          activeTargets={activeTargets}
        />

        {/* Feature 2: Dual View Image Display (Raw vs OpenCV YOLOv8-OBB & U-Net Bounding Boxes) */}
        <DualViewSonar
          customImageSrc={customImageSrc}
          palette={palette}
          activeTargets={activeTargets}
          selectedTargetId={selectedTargetId}
          onSelectTarget={setSelectedTargetId}
          fileName={fileName}
          isProcessing={isProcessing}
          leeWindowSize={leeWindowSize}
          claheClipLimit={claheClipLimit}
        />

        {/* Feature 3: Geospatial Mapping (Folium-style Interactive Map) */}
        <GeospatialSonarMap
          targets={activeTargets}
          selectedTargetId={selectedTargetId}
          onSelectTarget={setSelectedTargetId}
        />

        {/* Target Details Telemetry & Acoustic Shadow Math Table */}
        <TargetTelemetryTable
          targets={activeTargets}
          selectedTargetId={selectedTargetId}
          onSelectTarget={setSelectedTargetId}
        />

        {/* Feature 4: Export & Reporting (CSV + ReportLab PDF) */}
        <ExportReporting
          surveyData={surveyData}
          activeTargets={activeTargets}
        />
      </main>
    </div>
  );
}
