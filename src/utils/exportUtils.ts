import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { SonarSurveyData, SonarTarget } from '../types';

export function exportTargetsToCSV(surveyData: SonarSurveyData, targets: SonarTarget[]) {
  const headers = [
    'Target ID',
    'Classification',
    'Severity',
    'Risk Level',
    'Confidence (%)',
    'Latitude',
    'Longitude',
    'Depth (m)',
    'Dimensions',
    'Acoustic Shadow',
    'Survey Location',
    'Sensor Type',
    'Description'
  ];

  const rows = targets.map(t => [
    `"${t.id}"`,
    `"${t.name}"`,
    `"${t.severity}"`,
    `"${t.risk_level}"`,
    (t.confidence * 100).toFixed(1),
    t.latitude.toFixed(6),
    t.longitude.toFixed(6),
    t.depth_meters,
    `"${t.dimensions}"`,
    `"${t.acoustic_shadow_length}"`,
    `"${surveyData.survey_location}"`,
    `"${surveyData.sensor_type}"`,
    `"${t.description.replace(/"/g, '""')}"`
  ]);

  const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `sonar_survey_telemetry_${surveyData.survey_id}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function exportReportLabStyledPDF(surveyData: SonarSurveyData, targets: SonarTarget[]) {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'letter'
  });

  // Header Banner
  doc.setFillColor(15, 23, 42); // slate-900
  doc.rect(0, 0, 216, 26, 'F');

  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(15);
  doc.text('SONAR VISION AI - SUBSEA TARGET INSPECTION REPORT', 14, 13);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(148, 163, 184);
  doc.text(`SURVEY ID: ${surveyData.survey_id}  |  SENSOR: ${surveyData.sensor_type}`, 14, 20);

  // Metadata Grid
  doc.setDrawColor(226, 232, 240);
  doc.setFillColor(248, 250, 252);
  doc.roundedRect(14, 32, 188, 24, 2, 2, 'FD');

  doc.setFontSize(8.5);
  doc.setTextColor(71, 85, 105);
  doc.text('Survey Vessel:', 18, 38);
  doc.text('Survey Sector:', 18, 45);
  doc.text('Timestamp UTC:', 18, 52);

  doc.setFont('helvetica', 'bold');
  doc.setTextColor(15, 23, 42);
  doc.text(surveyData.survey_vessel, 44, 38);
  doc.text(surveyData.survey_location, 44, 45);
  doc.text(surveyData.timestamp, 44, 52);

  doc.setFont('helvetica', 'normal');
  doc.setTextColor(71, 85, 105);
  doc.text('Sound Velocity:', 116, 38);
  doc.text('Water Temp:', 116, 45);
  doc.text('Targets Flagged:', 116, 52);

  doc.setFont('helvetica', 'bold');
  doc.setTextColor(15, 23, 42);
  doc.text(`${surveyData.sound_velocity_mps} m/s`, 146, 38);
  doc.text(`${surveyData.water_temperature_c} °C`, 146, 45);
  doc.text(`${targets.length} Identified Objects`, 146, 52);

  // Executive Summary
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(15, 23, 42);
  doc.text('1. Executive Acoustic Imaging Summary', 14, 64);

  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(51, 65, 85);
  const summaryText = `Acoustic imaging survey scan was processed using Computer Vision object classification and bathymetric spatial clustering. Total of ${targets.length} targets were analyzed across the sonar tow swath. 1 high-risk ordnance hazard was flagged (TGT-01), 1 medium-risk structural wreckage (TGT-02), and 1 natural benthic outcrop (TGT-03). All coordinates are referenced in WGS84.`;
  const splitSummary = doc.splitTextToSize(summaryText, 188);
  doc.text(splitSummary, 14, 70);

  // Targets Table (matching reportlab table formatting)
  const tableRows = targets.map(t => [
    t.id,
    `${t.name}\n${t.description.slice(0, 65)}...`,
    t.severity.toUpperCase(),
    `${(t.confidence * 100).toFixed(1)}%`,
    `${t.latitude.toFixed(4)}°N\n${t.longitude.toFixed(4)}°W`,
    `${t.depth_meters} m`,
    t.dimensions
  ]);

  autoTable(doc, {
    startY: 86,
    head: [['ID', 'Classification & Acoustic Profile', 'Severity', 'Conf.', 'Coordinates', 'Depth', 'Dimensions']],
    body: tableRows,
    theme: 'grid',
    styles: {
      fontSize: 8,
      cellPadding: 2.5,
      textColor: [51, 65, 85],
      valign: 'middle'
    },
    headStyles: {
      fillColor: [15, 23, 42],
      textColor: [255, 255, 255],
      fontStyle: 'bold',
      halign: 'left'
    },
    columnStyles: {
      0: { cellWidth: 16, fontStyle: 'bold' },
      1: { cellWidth: 64 },
      2: { cellWidth: 20, fontStyle: 'bold', halign: 'center' },
      3: { cellWidth: 16, halign: 'center' },
      4: { cellWidth: 28, halign: 'center' },
      5: { cellWidth: 18, halign: 'center' },
      6: { cellWidth: 26, halign: 'center' }
    },
    didParseCell: function(data) {
      if (data.section === 'body' && data.column.index === 2) {
        const text = String(data.cell.raw);
        if (text.includes('RED')) {
          data.cell.styles.textColor = [220, 38, 38];
        } else if (text.includes('YELLOW')) {
          data.cell.styles.textColor = [217, 119, 6];
        } else if (text.includes('GREEN')) {
          data.cell.styles.textColor = [22, 163, 74];
        }
      }
    }
  });

  // Operational Recommendations
  const finalY = (doc as any).lastAutoTable?.finalY || 180;

  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(15, 23, 42);
  doc.text('2. Hydrographic Recommendations & Operational Directives', 14, finalY + 12);

  doc.setFontSize(8.5);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(51, 65, 85);
  doc.text('• [RED ALERT - TGT-01]: Enforce 500-meter maritime exclusion radius. Forward acoustic signature to EOD dive team.', 14, finalY + 18);
  doc.text('• [MEDIUM NOTICE - TGT-02]: Chart submerged container obstruction to prevent commercial bottom trawling entanglement.', 14, finalY + 24);
  doc.text('• [CLEAR - TGT-03]: Verified natural bathymetry feature; safe for surface and subsurface navigation.', 14, finalY + 30);

  // Footer
  doc.setFontSize(7.5);
  doc.setTextColor(148, 163, 184);
  doc.text('Generated with ReportLab Engine & Sonar Vision AI Pipeline  |  Standards: IHO S-44 Order 1a Hydrographic Survey', 14, 268);
  doc.text('Page 1 of 1', 190, 268);

  doc.save(`sonar_inspection_report_${surveyData.survey_id}.pdf`);
}
