import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { MapPin, Navigation, Compass, Maximize2 } from 'lucide-react';
import { SonarTarget } from '../types';

interface GeospatialSonarMapProps {
  targets: SonarTarget[];
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
}

// Icons for each target type
const TARGET_ICONS: Record<string, string> = {
  naval_mine_uxo: '💣',
  shipwreck_wreckage: '🚢',
  subsea_pipeline: '⚙️',
  ghost_net_waters: '🪸',
  debris: '🗑️',
};

export const GeospatialSonarMap: React.FC<GeospatialSonarMapProps> = ({
  targets,
  selectedTargetId,
  onSelectTarget
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersGroupRef = useRef<L.LayerGroup | null>(null);
  const markerRefsRef = useRef<Map<string, L.Marker>>(new Map());

  // Initialize map once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [36.7783, -122.0125],
      zoom: 13,
      zoomControl: true,
      attributionControl: false,
    });

    // Base tile layers
    const oceanLayer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 16, attribution: 'Esri, GEBCO, NOAA' }
    );
    const satelliteLayer = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, attribution: 'Esri, Maxar' }
    );
    const darkLayer = L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      { maxZoom: 19, subdomains: 'abcd', attribution: 'CARTO' }
    );
    const osmLayer = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: 'OpenStreetMap' }
    );

    satelliteLayer.addTo(map); // Default to Satellite for better sonar context

    L.control.layers(
      {
        '🌊 Ocean Bathymetry': oceanLayer,
        '🛰️ Satellite Imagery': satelliteLayer,
        '🌌 Dark Nautical': darkLayer,
        '🗺️ Standard Map': osmLayer,
      },
      undefined,
      { position: 'topright' }
    ).addTo(map);

    // Survey towfish trajectory line
    L.polyline(
      [
        [36.7720, -122.0200],
        [36.7750, -122.0165],
        [36.7783, -122.0125],
        [36.7820, -122.0080],
        [36.7860, -122.0030],
      ] as L.LatLngExpression[],
      { color: '#38bdf8', weight: 2.5, dashArray: '8, 6', opacity: 0.7 }
    ).addTo(map).bindTooltip('Towfish Survey Track', { sticky: true });

    // Vessel start marker
    const vesselIcon = L.divIcon({
      className: '',
      html: `<div style="background:#0369a1;width:14px;height:14px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 10px #38bdf8;"></div>`,
      iconSize: [14, 14],
      iconAnchor: [7, 7],
    });
    L.marker([36.7720, -122.0200], { icon: vesselIcon })
      .bindTooltip('Survey Swath Start (WP-01)')
      .addTo(map);

    markersGroupRef.current = L.layerGroup().addTo(map);
    mapInstanceRef.current = map;
  }, []);

  // Update markers whenever targets list or selection changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    const markersGroup = markersGroupRef.current;
    if (!map || !markersGroup) return;

    markersGroup.clearLayers();
    markerRefsRef.current.clear();

    if (targets.length === 0) return;

    const latLngs: L.LatLng[] = [];

    targets.forEach((target, idx) => {
      const isSelected = selectedTargetId === target.id;
      const color = target.color_hex;
      const icon = TARGET_ICONS[target.target_type ?? ''] ?? '📍';
      const shortNum = String(idx + 1);

      const markerSize = isSelected ? 38 : 30;
      const innerSize = isSelected ? 26 : 20;

      const customIcon = L.divIcon({
        className: '',
        html: `
          <div style="position:relative;width:${markerSize}px;height:${markerSize}px;display:flex;align-items:center;justify-content:center;cursor:pointer;">
            <!-- Pulsing ring -->
            <div style="position:absolute;width:${markerSize}px;height:${markerSize}px;border-radius:50%;background:${color};opacity:0.28;animation:ping 1.8s cubic-bezier(0,0,0.2,1) infinite;"></div>
            <!-- Selection ring -->
            ${isSelected ? `<div style="position:absolute;width:${markerSize + 6}px;height:${markerSize + 6}px;border-radius:50%;border:2.5px solid #38bdf8;top:${-3}px;left:${-3}px;box-shadow:0 0 12px #38bdf8;"></div>` : ''}
            <!-- Main pin circle -->
            <div style="
              position:relative;
              width:${innerSize}px;height:${innerSize}px;
              border-radius:50%;
              background:${color};
              border:2.5px solid #ffffff;
              box-shadow:0 2px 12px rgba(0,0,0,0.6),0 0 6px ${color}88;
              display:flex;align-items:center;justify-content:center;
              font-size:${isSelected ? 13 : 10}px;
              font-weight:800;
              color:#fff;
              text-shadow:0 1px 2px rgba(0,0,0,0.8);
            ">${shortNum}</div>
            <!-- Type icon badge -->
            <div style="
              position:absolute;
              top:-6px;right:-6px;
              width:16px;height:16px;
              background:#1e293b;
              border:1.5px solid ${color};
              border-radius:4px;
              font-size:9px;
              display:flex;align-items:center;justify-content:center;
            ">${icon}</div>
          </div>
        `,
        iconSize: [markerSize, markerSize],
        iconAnchor: [markerSize / 2, markerSize / 2],
        popupAnchor: [0, -(markerSize / 2) - 4],
      });

      const latlng = L.latLng(target.latitude, target.longitude);
      latLngs.push(latlng);

      const marker = L.marker(latlng, { icon: customIcon, zIndexOffset: isSelected ? 1000 : 0 });

      // Rich popup
      const popupHtml = `
        <div style="font-family:ui-sans-serif,system-ui,sans-serif;min-width:230px;color:#0f172a;padding:2px;">
          <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e2e8f0;padding-bottom:7px;margin-bottom:8px;">
            <div>
              <div style="font-size:10px;color:#64748b;font-weight:600;letter-spacing:0.05em;">${target.id}</div>
              <strong style="font-size:13px;color:#0f172a;line-height:1.3;">${target.name}</strong>
            </div>
            <span style="background:${color};color:${target.severity === 'Yellow' ? '#1e293b' : '#fff'};font-size:10px;font-weight:700;padding:3px 8px;border-radius:6px;white-space:nowrap;margin-left:8px;">
              ${target.severity === 'Red' ? '🔴' : target.severity === 'Yellow' ? '🟡' : '🟢'} ${target.severity.toUpperCase()}
            </span>
          </div>
          <table style="width:100%;font-size:11px;line-height:1.7;color:#334155;border-collapse:collapse;">
            <tr><td style="font-weight:600;padding-right:8px;white-space:nowrap;">Confidence</td><td>${(target.confidence * 100).toFixed(1)}% ✓</td></tr>
            <tr><td style="font-weight:600;padding-right:8px;">Position</td><td>${target.latitude.toFixed(5)}°N, ${Math.abs(target.longitude).toFixed(5)}°W</td></tr>
            <tr><td style="font-weight:600;padding-right:8px;">Water Depth</td><td>${target.depth_meters} m</td></tr>
            <tr><td style="font-weight:600;padding-right:8px;">Dimensions</td><td>${target.dimensions}</td></tr>
            <tr><td style="font-weight:600;padding-right:8px;">Shadow</td><td>${target.acoustic_shadow_length}</td></tr>
          </table>
          <div style="margin-top:8px;font-size:10.5px;color:#64748b;font-style:italic;border-top:1px dashed #e2e8f0;padding-top:6px;line-height:1.5;">
            ${target.description}
          </div>
          <div style="margin-top:6px;font-size:10px;color:#0369a1;font-weight:600;">
            ⚡ ${target.action_recommendation ?? 'No action required.'}
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml, { maxWidth: 300, className: 'sonar-popup' });
      marker.on('click', () => onSelectTarget(target.id));
      markersGroup.addLayer(marker);
      markerRefsRef.current.set(target.id, marker);

      // Uncertainty radius circle
      markersGroup.addLayer(
        L.circle(latlng, {
          radius: isSelected ? 60 : 35,
          color,
          fillColor: color,
          fillOpacity: isSelected ? 0.22 : 0.12,
          weight: isSelected ? 2 : 1,
          dashArray: isSelected ? undefined : '4, 4',
        })
      );
    });

    // ── Auto-fit all markers into view ──────────────────────────────────
    if (latLngs.length > 0) {
      const bounds = L.latLngBounds(latLngs).pad(0.35);
      map.fitBounds(bounds, { animate: true, duration: 0.9, maxZoom: 15 });
    }

    // Open popup / pan to selected target
    if (selectedTargetId) {
      const selMarker = markerRefsRef.current.get(selectedTargetId);
      const selTarget = targets.find((t) => t.id === selectedTargetId);
      if (selMarker && selTarget) {
        map.panTo([selTarget.latitude, selTarget.longitude], { animate: true, duration: 0.7 });
        selMarker.openPopup();
      }
    }
  }, [targets, selectedTargetId, onSelectTarget]);

  // Fit-all button handler
  const handleFitAll = () => {
    const map = mapInstanceRef.current;
    if (!map || targets.length === 0) return;
    const bounds = L.latLngBounds(targets.map((t) => L.latLng(t.latitude, t.longitude))).pad(0.35);
    map.fitBounds(bounds, { animate: true, duration: 0.9, maxZoom: 15 });
  };

  return (
    <div id="geospatial-map-section" className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3">
      {/* Map Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-slate-100">
              Geospatial Bathymetry &amp; Target Mapping
            </h3>
            {targets.length > 0 && (
              <span className="text-[11px] font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                {targets.length} target{targets.length !== 1 ? 's' : ''} plotted
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Interactive Folium mapping — seabed targets plotted by GPS coordinates and risk severity.
          </p>
        </div>

        {/* Controls row */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Fit-all button */}
          <button
            onClick={handleFitAll}
            className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 hover:text-cyan-300 text-xs px-3 py-1.5 rounded-lg transition-colors"
            title="Fit all targets into view"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            Fit All
          </button>

          {/* Legend */}
          <div className="flex items-center gap-3 bg-slate-950/80 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
              <span className="text-slate-300">High</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-sm shadow-amber-500/50" />
              <span className="text-slate-300">Med</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
              <span className="text-slate-300">Low</span>
            </div>
            <div className="hidden sm:flex items-center gap-1.5 pl-2 border-l border-slate-800 text-cyan-400">
              <Navigation className="w-3 h-3" />
              <span>Survey Track</span>
            </div>
          </div>
        </div>
      </div>

      {/* Map Canvas */}
      <div className="relative rounded-lg overflow-hidden border border-slate-800 h-[420px] w-full bg-slate-950">
        <div ref={mapContainerRef} className="w-full h-full z-0" id="folium-sonar-map" />

        {/* WGS84 badge */}
        <div className="absolute bottom-3 left-3 z-[400] bg-slate-900/90 backdrop-blur-md border border-slate-700 px-2.5 py-1.5 rounded-lg text-[11px] text-slate-300 shadow-lg flex items-center gap-1.5">
          <Compass className="w-4 h-4 text-cyan-400" />
          <span className="font-mono">WGS84 • Sector B</span>
        </div>

        {/* No targets overlay */}
        {targets.length === 0 && (
          <div className="absolute inset-0 z-[500] flex items-center justify-center bg-slate-950/70 backdrop-blur-sm">
            <div className="text-center text-slate-400">
              <MapPin className="w-10 h-10 mx-auto mb-2 opacity-30" />
              <p className="text-sm font-medium">No targets to display</p>
              <p className="text-xs mt-1 opacity-60">Upload a sonar image to detect targets</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
