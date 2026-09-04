"""
Sonar Image Analysis Platform
A Streamlit Web Application for Marine & Subsea Sonar Object Detection, Geospatial Mapping, and Automated Reporting.
"""

import io
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

# Optional/Standard imports with safe fallbacks
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Sonar Vision AI - Subsea Target Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .severity-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .severity-badge-med {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .severity-badge-low {
        background-color: #dcfce7;
        color: #166534;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MOCK DETECTION JSON DATA (3 TARGETS: RED, YELLOW, GREEN)
# ==============================================================================
SAMPLE_DETECTIONS_JSON = {
    "survey_id": "SURV-2026-PAC-042",
    "timestamp": "2026-09-01T14:30:00Z",
    "sensor_type": "High-Resolution Dual-Frequency Side-Scan Sonar",
    "survey_location": "Monterey Bay Submarine Canyon Sector B",
    "targets": [
        {
            "id": "TGT-01",
            "name": "Naval Mine / UXO Anomaly",
            "severity": "Red",
            "risk_level": "High Risk",
            "confidence": 0.942,
            "color_hex": "#EF4444",
            "color_bgr": (0, 0, 239),
            "color_rgb": (239, 68, 68),
            "bbox": [140, 110, 110, 85],  # [x, y, width, height]
            "latitude": 36.7783,
            "longitude": -122.0125,
            "depth_meters": 84.5,
            "dimensions": "1.4m x 0.8m",
            "acoustic_shadow_length": "4.2m",
            "description": "High-intensity specular acoustic reflection with distinct conical acoustic shadow. Potential unexploded ordnance."
        },
        {
            "id": "TGT-02",
            "name": "Sunken Cargo Container / Metal Debris",
            "severity": "Yellow",
            "risk_level": "Medium Risk",
            "confidence": 0.876,
            "color_hex": "#F59E0B",
            "color_bgr": (11, 158, 245),
            "color_rgb": (245, 158, 11),
            "bbox": [390, 190, 160, 110],  # [x, y, width, height]
            "latitude": 36.7820,
            "longitude": -122.0080,
            "depth_meters": 112.0,
            "dimensions": "6.1m x 2.4m",
            "acoustic_shadow_length": "8.7m",
            "description": "Rectilinear geometric signature with hard acoustic edges indicative of structural shipping debris."
        },
        {
            "id": "TGT-03",
            "name": "Natural Basalt Outcrop / Biogenic Reef",
            "severity": "Green",
            "risk_level": "Low Risk",
            "confidence": 0.914,
            "color_hex": "#10B981",
            "color_bgr": (129, 185, 16),
            "color_rgb": (16, 185, 129),
            "bbox": [210, 360, 140, 95],  # [x, y, width, height]
            "latitude": 36.7745,
            "longitude": -122.0160,
            "depth_meters": 65.2,
            "dimensions": "12.0m x 4.5m",
            "acoustic_shadow_length": "3.1m",
            "description": "Diffused textural backscatter characteristic of natural geological shelf formation. Safe for navigation."
        }
    ]
}


# ==============================================================================
# 3. HELPER FUNCTIONS: SYNTHETIC SONAR GENERATOR & OPENCV BOUNDING BOXES
# ==============================================================================
def create_synthetic_sonar_image(width=640, height=512) -> Image.Image:
    """Generate a realistic side-scan sonar image with seafloor texture, nadir track, and target anomalies."""
    np.random.seed(42)
    
    # Base acoustic seabed texture (gradient + speckled speckle noise)
    gradient = np.linspace(50, 110, height)[:, None] * np.ones((1, width))
    noise = np.random.normal(0, 18, (height, width))
    sonar_array = np.clip(gradient + noise, 10, 245).astype(np.uint8)
    
    # Add nadir zone (central dark line representing the water column right below the towfish)
    nadir_x = width // 2
    nadir_width = 32
    sonar_array[:, nadir_x - nadir_width//2 : nadir_x + nadir_width//2] = np.random.normal(15, 5, (height, nadir_width)).clip(5, 30)

    # Convert to PIL and add high-backscatter highlights and acoustic shadows
    img = Image.fromarray(sonar_array).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw synthetic targets matching mock JSON
    for tgt in SAMPLE_DETECTIONS_JSON["targets"]:
        x, y, w, h = tgt["bbox"]
        # Highlight (bright target return)
        draw.ellipse([x + 10, y + 10, x + w - 20, y + h - 20], fill=(220, 230, 245))
        # Acoustic shadow behind target
        shadow_len = int(w * 0.8)
        draw.rectangle([x + w - 15, y + 15, x + w + shadow_len, y + h - 15], fill=(12, 16, 24))

    return img


def draw_bounding_boxes_opencv(image_pil: Image.Image, detections: list) -> Image.Image:
    """Draw color-coded bounding boxes and target tags using OpenCV."""
    img_np = np.array(image_pil)
    
    if cv2 is not None:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]

        for tgt in detections:
            x, y, w, h = tgt["bbox"]
            # Scale if image dimensions differ
            scale_x = img_w / 640.0
            scale_y = img_h / 512.0
            
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
            
            bgr_color = tgt.get("color_bgr", (0, 255, 0))
            
            # Draw rectangular bounding box
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr_color, 2)
            
            # Label banner
            label = f"{tgt['id']}: {tgt['name']} ({tgt['confidence']*100:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Text background badge
            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = y1
            cv2.rectangle(img_bgr, (x1, badge_y1), (x1 + text_w + 10, badge_y2), bgr_color, -1)
            
            # Text label
            text_color = (255, 255, 255) if tgt["severity"] in ["Red", "Blue"] else (15, 23, 42)
            cv2.putText(img_bgr, label, (x1 + 5, y1 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)
            
        # Convert back to RGB PIL Image
        img_annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_annotated_rgb)
    else:
        # Fallback to PIL ImageDraw if OpenCV is not installed
        annotated_img = image_pil.copy()
        draw = ImageDraw.Draw(annotated_img)
        img_w, img_h = annotated_img.size
        
        for tgt in detections:
            x, y, w, h = tgt["bbox"]
            scale_x = img_w / 640.0
            scale_y = img_h / 512.0
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
            rgb_color = tgt.get("color_rgb", (255, 0, 0))
            
            draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=3)
            label = f"{tgt['id']}: {tgt['name']} ({tgt['confidence']*100:.1f}%)"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(label)*8, y1], fill=rgb_color)
            draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(255, 255, 255))
            
        return annotated_img


# ==============================================================================
# 4. HELPER FUNCTIONS: REPORTLAB PDF REPORT GENERATOR
# ==============================================================================
def generate_pdf_report(detections_data: dict) -> bytes:
    """Generate a downloadable PDF inspection report using reportlab."""
    if not REPORTLAB_AVAILABLE:
        # Graceful fallback text buffer if reportlab is unavailable
        buffer = io.BytesIO()
        buffer.write(b"%PDF-1.4 Mock Sonar Report Data")
        buffer.seek(0)
        return buffer.getvalue()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )
    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Sonar Vision AI - Subsea Target Inspection Report", title_style))
    story.append(Paragraph(f"<b>Survey ID:</b> {detections_data['survey_id']} &nbsp;|&nbsp; <b>Sensor:</b> {detections_data['sensor_type']} &nbsp;|&nbsp; <b>Sector:</b> {detections_data['survey_location']}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    # Executive Summary Box
    summary_text = (
        f"Acoustic imaging inspection completed. Total detected seabed targets: <b>{len(detections_data['targets'])}</b>. "
        "Analysis identified 1 critical high-risk navigational hazard, 1 medium-risk structural debris anomaly, "
        "and 1 natural geological formation. Geospatial positioning and target profiles are detailed below."
    )
    story.append(Paragraph("<b>Executive Summary:</b>", heading2_style))
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 12))

    # Targets Table
    story.append(Paragraph("<b>Identified Target Telemetry:</b>", heading2_style))
    
    table_data = [
        ["ID", "Classification", "Severity", "Confidence", "Coordinates", "Depth", "Dimensions"]
    ]

    for t in detections_data["targets"]:
        table_data.append([
            t["id"],
            Paragraph(f"<b>{t['name']}</b><br/><font size=7 color='#64748b'>{t['description'][:60]}...</font>", body_style),
            t["severity"].upper(),
            f"{t['confidence']*100:.1f}%",
            f"{t['latitude']:.4f}°N<br/>{t['longitude']:.4f}°W",
            f"{t['depth_meters']} m",
            t["dimensions"]
        ])

    col_widths = [45, 170, 60, 60, 85, 55, 65]
    t_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ])

    # Add custom background colors for severity badges
    for row_idx, tgt in enumerate(detections_data["targets"], start=1):
        if tgt["severity"] == "Red":
            t_style.add('TEXTCOLOR', (2, row_idx), (2, row_idx), colors.HexColor("#dc2626"))
            t_style.add('FONTNAME', (2, row_idx), (2, row_idx), 'Helvetica-Bold')
        elif tgt["severity"] == "Yellow":
            t_style.add('TEXTCOLOR', (2, row_idx), (2, row_idx), colors.HexColor("#d97706"))
            t_style.add('FONTNAME', (2, row_idx), (2, row_idx), 'Helvetica-Bold')
        elif tgt["severity"] == "Green":
            t_style.add('TEXTCOLOR', (2, row_idx), (2, row_idx), colors.HexColor("#16a34a"))
            t_style.add('FONTNAME', (2, row_idx), (2, row_idx), 'Helvetica-Bold')

    t_table.setStyle(t_style)
    story.append(t_table)
    story.append(Spacer(1, 16))

    # Operational Recommendations
    story.append(Paragraph("<b>Operational Recommendations & Safety Actions:</b>", heading2_style))
    recommendations = (
        "1. <b>TGT-01 (Naval Mine / UXO):</b> Establish 500m safety exclusion zone. Notify Regional Maritime Operations Center.<br/>"
        "2. <b>TGT-02 (Debris Cluster):</b> Log hazard to nautical chart database. Safe for ROV inspection.<br/>"
        "3. <b>TGT-03 (Reef Shelf):</b> Nominal bathymetric feature. No hazard mitigation required."
    )
    story.append(Paragraph(recommendations, body_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<i>Report generated automatically by Sonar Vision AI Analysis Pipeline. Verified hydrographic standards.</i>", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
# 5. MAIN STREAMLIT APP LAYOUT & LOGIC
# ==============================================================================
def main():
    # --- Header ---
    st.markdown('<div class="main-title">🌊 Sonar Vision AI Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Acoustic Sonar Imagery Target Classification, Geospatial Bathymetry Mapping & Mission Reporting</div>', unsafe_allow_html=True)

    # --- Sidebar: Feature 1 - File Upload ---
    st.sidebar.header("📁 Sonar Data Ingestion")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Sonar Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop raw side-scan, synthetic aperture, or forward-looking sonar scan images."
    )

    use_sample = st.sidebar.checkbox("Use Demo Synthetic Sonar Scan", value=(uploaded_file is None))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Detection Filters")
    min_confidence = st.sidebar.slider("Confidence Threshold", min_value=0.50, max_value=0.99, value=0.75, step=0.01)
    selected_severities = st.sidebar.multiselect("Filter Severity", ["Red", "Yellow", "Green"], default=["Red", "Yellow", "Green"])

    # Load Image
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        image_source_label = f"Uploaded File: `{uploaded_file.name}`"
    elif use_sample:
        raw_image = create_synthetic_sonar_image()
        image_source_label = "Demo Survey Scan: `Sector_B_SideScan_Sonar.png`"
    else:
        st.info("👈 Please upload a sonar image (JPG/PNG) in the sidebar or check 'Use Demo Synthetic Sonar Scan'.")
        return

    # Filter detections based on sidebar inputs
    active_detections = [
        t for t in SAMPLE_DETECTIONS_JSON["targets"]
        if t["confidence"] >= min_confidence and t["severity"] in selected_severities
    ]

    # Metrics Summary Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Targets Detected", len(active_detections))
    with m2:
        high_risk_count = sum(1 for t in active_detections if t["severity"] == "Red")
        st.metric("High Risk / Critical (Red)", high_risk_count, delta="Immediate Alert" if high_risk_count > 0 else None, delta_color="inverse")
    with m3:
        med_risk_count = sum(1 for t in active_detections if t["severity"] == "Yellow")
        st.metric("Medium Risk (Yellow)", med_risk_count)
    with m4:
        low_risk_count = sum(1 for t in active_detections if t["severity"] == "Green")
        st.metric("Low Risk / Natural (Green)", low_risk_count)

    st.markdown("---")

    # --- Feature 2: Dual View Image Display ---
    st.subheader("🔍 Dual View Acoustic Inspection")
    st.caption(f"Source: {image_source_label}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Column 1: Raw Sonar Image**")
        st.image(raw_image, use_container_width=True, caption="Unprocessed Acoustic Backscatter Input")

    with col2:
        st.markdown("**Column 2: OpenCV Object Detection & Color Bounding Boxes**")
        annotated_image = draw_bounding_boxes_opencv(raw_image, active_detections)
        st.image(
            annotated_image,
            use_container_width=True,
            caption=f"Computer Vision Detections ({len(active_detections)} Targets Highlighted)"
        )

    # --- Target Details Table ---
    with st.expander("📋 View Target Telemetry & Detection JSON Data", expanded=True):
        if active_detections:
            df_display = pd.DataFrame([
                {
                    "Target ID": t["id"],
                    "Classification": t["name"],
                    "Severity": t["severity"],
                    "Confidence": f"{t['confidence']*100:.1f}%",
                    "Latitude": t["latitude"],
                    "Longitude": t["longitude"],
                    "Depth (m)": t["depth_meters"],
                    "Dimensions": t["dimensions"],
                    "Acoustic Shadow": t["acoustic_shadow_length"],
                    "Description": t["description"]
                }
                for t in active_detections
            ])
            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("No targets match current confidence and severity filter settings.")

    st.markdown("---")

    # --- Feature 3: Geospatial Mapping with Folium ---
    st.subheader("🗺️ Geospatial Bathymetry & Target Mapping")
    st.caption("Interactive Folium map plotting detected seabed anomalies color-coded by threat severity.")

    if folium is not None:
        # Calculate center coordinates
        if active_detections:
            avg_lat = sum(t["latitude"] for t in active_detections) / len(active_detections)
            avg_lon = sum(t["longitude"] for t in active_detections) / len(active_detections)
        else:
            avg_lat, avg_lon = 36.7783, -122.0125

        # Initialize folium map with dark ocean/marine tiles
        sonar_map = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=14,
            tiles="CartoDB positron",
            control_scale=True
        )

        # Optional: Add OpenStreetMap as alternative layer
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap Standard').add_to(sonar_map)

        # Plot survey vessel path / track line
        survey_track = [
            [36.7720, -122.0200],
            [36.7760, -122.0150],
            [36.7800, -122.0100],
            [36.7840, -122.0050],
        ]
        folium.PolyLine(
            survey_track,
            color="#3b82f6",
            weight=3,
            dash_array='5, 10',
            tooltip="Survey Vessel Towfish Trajectory"
        ).add_to(sonar_map)

        # Color mapping for folium marker icons
        marker_color_map = {
            "Red": "red",
            "Yellow": "orange",
            "Green": "green"
        }

        # Plot target markers
        for tgt in active_detections:
            popup_html = f"""
            <div style="font-family: sans-serif; width: 220px;">
                <h4 style="margin: 0 0 6px 0; color: #0f172a;">{tgt['id']}: {tgt['name']}</h4>
                <p style="margin: 2px 0;"><b>Severity:</b> <span style="color:{tgt['color_hex']}; font-weight:bold;">{tgt['severity']} ({tgt['risk_level']})</span></p>
                <p style="margin: 2px 0;"><b>Confidence:</b> {tgt['confidence']*100:.1f}%</p>
                <p style="margin: 2px 0;"><b>Coordinates:</b> {tgt['latitude']:.4f}°N, {tgt['longitude']:.4f}°W</p>
                <p style="margin: 2px 0;"><b>Depth:</b> {tgt['depth_meters']} meters</p>
                <p style="margin: 2px 0;"><b>Dimensions:</b> {tgt['dimensions']}</p>
                <p style="margin: 6px 0 0 0; font-size: 11px; color: #64748b;">{tgt['description']}</p>
            </div>
            """
            
            folium.Marker(
                location=[tgt["latitude"], tgt["longitude"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{tgt['id']} - {tgt['name']} ({tgt['severity']} Risk)",
                icon=folium.Icon(color=marker_color_map.get(tgt["severity"], "blue"), icon="info-sign")
            ).add_to(sonar_map)

            # Draw acoustic uncertainty circle around target
            folium.Circle(
                location=[tgt["latitude"], tgt["longitude"]],
                radius=45,
                color=tgt["color_hex"],
                fill=True,
                fill_color=tgt["color_hex"],
                fill_opacity=0.2,
                weight=1
            ).add_to(sonar_map)

        folium.LayerControl().add_to(sonar_map)

        if st_folium is not None:
            st_folium(sonar_map, width="100%", height=420)
        else:
            st.write("Displaying interactive folium map:")
            st.components.v1.html(sonar_map._repr_html_(), height=420)
    else:
        st.warning("Folium package not installed. Displaying raw coordinate table.")
        st.write(pd.DataFrame(active_detections)[["id", "name", "severity", "latitude", "longitude", "depth_meters"]])

    st.markdown("---")

    # --- Feature 4: Export & Reporting (CSV + ReportLab PDF) ---
    st.subheader("📥 Export & Inspection Reporting")
    st.write("Generate and download comprehensive survey deliverables for bathymetric records and defense intelligence.")

    export_col1, export_col2 = st.columns(2)

    # 1. CSV Export
    with export_col1:
        st.markdown("##### 📊 Telemetry CSV Summary")
        st.caption("Structured tabular dataset containing coordinates, confidence ratings, and acoustic dimensions.")
        
        csv_df = pd.DataFrame(active_detections)
        if not csv_df.empty:
            # Flatten or select columns for clean CSV
            csv_export_df = csv_df[[
                "id", "name", "severity", "risk_level", "confidence",
                "latitude", "longitude", "depth_meters", "dimensions",
                "acoustic_shadow_length", "description"
            ]]
            csv_bytes = csv_export_df.to_csv(index=False).encode("utf-8")
        else:
            csv_bytes = b"id,name,severity,confidence\n"

        st.download_button(
            label="⬇️ Download CSV Target Summary",
            data=csv_bytes,
            file_name=f"sonar_detections_{SAMPLE_DETECTIONS_JSON['survey_id']}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. PDF ReportLab Export
    with export_col2:
        st.markdown("##### 📄 Official Hydrographic PDF Report")
        st.caption("Formatted inspection document with telemetry table, risk breakdown, and nautical recommendations.")
        
        export_payload = {
            "survey_id": SAMPLE_DETECTIONS_JSON["survey_id"],
            "sensor_type": SAMPLE_DETECTIONS_JSON["sensor_type"],
            "survey_location": SAMPLE_DETECTIONS_JSON["survey_location"],
            "targets": active_detections
        }
        
        pdf_bytes = generate_pdf_report(export_payload)

        st.download_button(
            label="⬇️ Download PDF Inspection Report",
            data=pdf_bytes,
            file_name=f"sonar_survey_report_{SAMPLE_DETECTIONS_JSON['survey_id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
