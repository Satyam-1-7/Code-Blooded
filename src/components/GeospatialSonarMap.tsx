import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { MapPin, Navigation, Compass, Layers } from 'lucide-react';
import { SonarTarget } from '../types';

interface GeospatialSonarMapProps {
  targets: SonarTarget[];
  selectedTargetId: string | null;
  onSelectTarget: (id: string | null) => void;
}

export const GeospatialSonarMap: React.FC<GeospatialSonarMapProps> = ({
  targets,
  selectedTargetId,
  onSelectTarget
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const markersGroupRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      // Default center around Monterey Bay canyon targets
      const centerLat = 36.7783;
      const centerLng = -122.0125;

      const map = L.map(mapContainerRef.current, {
        center: [centerLat, centerLng],
        zoom: 14,
        zoomControl: true,
        attributionControl: false
      });

      // CartoDB Dark/Positron base tiles for authentic marine hydrographic look
      L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
        subdomains: 'abcd'
      }).addTo(map);

      // Survey vessel towfish trajectory line
      const trackPoints: L.LatLngExpression[] = [
        [36.7720, -122.0200],
        [36.7750, -122.0165],
        [36.7783, -122.0125],
        [36.7820, -122.0080],
        [36.7850, -122.0040]
      ];

      L.polyline(trackPoints, {
        color: '#0284c7',
        weight: 3,
        dashArray: '6, 8',
        opacity: 0.85
      }).addTo(map);

      // Start & End markers for survey line
      const startIcon = L.divIcon({
        className: 'custom-vessel-icon',
        html: `<div style="background-color:#0369a1; width:12px; height:12px; border-radius:50%; border:2px solid #ffffff; box-shadow:0 0 8px #0284c7;"></div>`,
        iconSize: [12, 12],
        iconAnchor: [6, 6]
      });
      L.marker([36.7720, -122.0200], { icon: startIcon }).bindTooltip("Survey Swath Start (WP-01)").addTo(map);

      markersGroupRef.current = L.layerGroup().addTo(map);
      mapInstanceRef.current = map;
    }

    return () => {
      // Keep map instance alive across rerenders
    };
  }, []);

  // Update target markers when targets or selectedTargetId change
  useEffect(() => {
    const map = mapInstanceRef.current;
    const markersGroup = markersGroupRef.current;
    if (!map || !markersGroup) return;

    markersGroup.clearLayers();

    targets.forEach((target) => {
      const isSelected = selectedTargetId === target.id;
      const color = target.color_hex;

      // Custom pulsing HTML marker
      const customIcon = L.divIcon({
        className: 'custom-sonar-marker',
        html: `
          <div style="position: relative; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
            <div style="position: absolute; width: 28px; height: 28px; border-radius: 50%; background-color: ${color}; opacity: 0.35; animation: ping 2s cubic-bezier(0, 0, 0.2, 1) infinite;"></div>
            <div style="position: relative; width: 18px; height: 18px; border-radius: 50%; background-color: ${color}; border: 2.5px solid #ffffff; box-shadow: 0 2px 8px rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 9px;">
              ${target.id.split('-')[1] || 'T'}
            </div>
            ${isSelected ? `<div style="position: absolute; top: -6px; right: -6px; width: 8px; height: 8px; border-radius: 50%; background-color: #38bdf8; border: 1.5px solid #ffffff;"></div>` : ''}
          </div>
        `,
        iconSize: [28, 28],
        iconAnchor: [14, 14]
      });

      const marker = L.marker([target.latitude, target.longitude], { icon: customIcon });

      // Custom Folium-style HTML Popup
      const popupHtml = `
        <div style="font-family: ui-sans-serif, system-ui, sans-serif; min-width: 220px; color: #0f172a; padding: 2px;">
          <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 8px;">
            <strong style="font-size: 13px; color: #0f172a;">${target.id}: ${target.name}</strong>
            <span style="background-color: ${color}; color: ${target.severity === 'Yellow' ? '#1e293b' : '#ffffff'}; font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 4px;">
              ${target.severity.toUpperCase()}
            </span>
          </div>
          <div style="font-size: 11px; line-height: 1.5; color: #334155;">
            <div><b>Confidence:</b> ${(target.confidence * 100).toFixed(1)}%</div>
            <div><b>Position:</b> ${target.latitude.toFixed(4)}°N, ${Math.abs(target.longitude).toFixed(4)}°W</div>
            <div><b>Water Depth:</b> ${target.depth_meters} meters</div>
            <div><b>Target Dimensions:</b> ${target.dimensions}</div>
            <div><b>Acoustic Shadow:</b> ${target.acoustic_shadow_length}</div>
            <div style="margin-top: 6px; font-size: 10.5px; color: #64748b; font-style: italic; border-top: 1px dashed #e2e8f0; padding-top: 4px;">
              ${target.description}
            </div>
          </div>
        </div>
      `;

      marker.bindPopup(popupHtml, { maxWidth: 280 });
      marker.on('click', () => {
        onSelectTarget(target.id);
      });

      markersGroup.addLayer(marker);

      // Uncertainty acoustic circle
      const circle = L.circle([target.latitude, target.longitude], {
        radius: 40,
        color: color,
        fillColor: color,
        fillOpacity: 0.18,
        weight: 1.5
      });
      markersGroup.addLayer(circle);

      if (isSelected) {
        map.panTo([target.latitude, target.longitude], { animate: true, duration: 0.8 });
        marker.openPopup();
      }
    });
  }, [targets, selectedTargetId, onSelectTarget]);

  return (
    <div id="geospatial-map-section" className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3">
      {/* Map Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-cyan-400" />
            <h3 className="text-base font-bold text-slate-100">
              Geospatial Bathymetry & Target Mapping
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Interactive Folium mapping displaying seabed targets plotted by coordinates and risk severity.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3 bg-slate-950/80 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
            <span className="text-slate-300">High (Red)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-sm shadow-amber-500/50" />
            <span className="text-slate-300">Med (Yellow)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
            <span className="text-slate-300">Low (Green)</span>
          </div>
          <div className="hidden sm:flex items-center gap-1.5 pl-2 border-l border-slate-800 text-cyan-400">
            <Navigation className="w-3 h-3" />
            <span>Survey Swath</span>
          </div>
        </div>
      </div>

      {/* Interactive Leaflet/Folium Canvas Container */}
      <div className="relative rounded-lg overflow-hidden border border-slate-800 h-[380px] w-full bg-slate-950">
        <div ref={mapContainerRef} className="w-full h-full z-0" id="folium-sonar-map" />

        {/* Compass Overlay Badge */}
        <div className="absolute top-3 right-3 z-[400] bg-slate-900/90 backdrop-blur-md border border-slate-700 p-2 rounded-lg text-[11px] text-slate-300 shadow-lg flex items-center gap-1.5">
          <Compass className="w-4 h-4 text-cyan-400 animate-spin-slow" />
          <span className="font-mono">WGS84 • Sector B</span>
        </div>
      </div>
    </div>
  );
};
