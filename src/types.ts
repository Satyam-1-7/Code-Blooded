export type SeverityLevel = 'Red' | 'Yellow' | 'Green';

export interface BoundingBoxOBB {
  cx: number;
  cy: number;
  w: number;
  h: number;
  angle_deg: number;
}

export interface ShadowMetrics {
  shadow_length_m: number;
  slant_range_m: number;
  towfish_altitude_m: number;
  estimated_target_height_m: number;
  shadow_contrast_index: number;
  shadow_confidence_pct: number;
  verified_3d: boolean;
}

export interface PolygonPoint {
  x: number;
  y: number;
}

export interface SonarTarget {
  id: string;
  name: string;
  target_type?: string;
  dataset_source?: string;
  severity: SeverityLevel;
  risk_level: string;
  confidence: number;
  color_hex: string;
  color_bgr: [number, number, number];
  color_rgb: [number, number, number];
  bbox: [number, number, number, number]; // [x, y, width, height]
  bbox_obb?: BoundingBoxOBB;
  latitude: number;
  longitude: number;
  depth_meters: number;
  dimensions: string;
  acoustic_shadow_length: string;
  description: string;
  action_recommendation?: string;
  shadow_metrics?: ShadowMetrics;
  unet_segmentation_polygon?: PolygonPoint[];
}

export interface TowfishNav {
  latitude: number;
  longitude: number;
  altitude_meters: number;
  heading_degrees: number;
  speed_knots: number;
  frequency_khz?: number;
  layback_meters?: number;
}

export interface PreprocessingConfig {
  leeWindowSize: number; // 3, 5, 7, 9
  speckleCu: number;     // 0.52 standard
  claheClipLimit: number; // 1.0 - 5.0
  colormap: 'amber' | 'copper' | 'grayscale' | 'cool_blue';
}

export interface SonarSurveyData {
  survey_id: string;
  timestamp: string;
  sensor_type: string;
  survey_location: string;
  survey_vessel: string;
  water_temperature_c: number;
  sound_velocity_mps: number;
  towfish_nav?: TowfishNav;
  targets: SonarTarget[];
}
