import streamlit as st
import pandas as pd
import pickle
import os
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from datetime import datetime, timedelta
import tempfile
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
import time
import plotly.express as px
import sklearn
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
import asyncio
import sys

# Memory optimization
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define classes for PPE detection
PPE_CLASSES = [
    "boots", "gloves", "goggles", "helmet", "no-helm", 
    "no_glove", "no_goggles", "no_helmet", "no_shoes", "person", "vest"
]

# Set page configuration with improved theme
st.set_page_config(
    page_title="🏗️ Construction Management System", 
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Construction Management System v1.0"
    }
)

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #f8f9fa;
        color: #2B2B2B; /* Dark text for light background */
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FDF6E3 !important;
        color: #2B2B2B !important; /* Darker text for visibility */
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #2B2B2B !important;
    }
    
    /* Button styling */
    .stButton>button {
        border: 1px solid #3498db;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        border-color: #2980b9;
    }
    
    /* Primary button */
    .stButton>button:focus:not(:active) {
        background-color: #2980b9;
        border-color: #2980b9;
    }
    
    /* Secondary button */
    .css-1x8cf1d {
        background-color: #6c757d;
        border-color: #6c757d;
        color: white;
    }
    
    /* Cards and containers */
    .css-1aumxhk {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        color: #2B2B2B;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        color: #2B2B2B;
    }
    
    /* Tab styling */
    [role="tab"] {
        padding: 0.5rem 1rem !important;
        border-radius: 5px 5px 0 0 !important;
        color: #2B2B2B !important;
    }
    
    [role="tab"][aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #3498db;
    }
    
    /* Success messages */
    .stAlert .st-b7 {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
    
    /* Error messages */
    .stAlert .st-b8 {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
    
    /* Custom cards */
    .custom-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #2B2B2B;
    }
    
    /* Custom badges */
    .custom-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        color: white;
        background-color: #3498db;
    }
    
    /* Custom headers */
    .custom-header {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Define the DetectionTracker class for PPE detection
class DetectionTracker:
    def __init__(self):
        self.total_detections = 0
        self.class_counts = {cls: 0 for cls in PPE_CLASSES}
        self.processing_time = 0
        self.frame_count = 0
        self.person_safety_violations = []
        self.seen_persons = []  # Track unique persons by bounding box

    def is_new_person(self, person_box, threshold=50):
        """Check if this person is new by comparing bounding boxes"""
        for seen_box in self.seen_persons:
            if np.allclose(person_box, seen_box, atol=threshold):
                return False
        self.seen_persons.append(person_box)
        return True
    
    def reset(self):
        """Reset all tracking data"""
        self.total_detections = 0
        self.class_counts = {cls: 0 for cls in PPE_CLASSES}
        self.processing_time = 0
        self.frame_count = 0
        self.person_safety_violations = []
        self.seen_persons = []

# Generator Model Definition
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()

        def down_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.bottleneck = down_block(512, 512)

        # Upsampling
        self.up1 = up_block(512, 512, dropout=True)
        self.up2 = up_block(1024, 512, dropout=True)
        self.up3 = up_block(1024, 512, dropout=True)
        self.up4 = up_block(1024, 256)
        self.up5 = up_block(512, 128)
        self.up6 = up_block(256, 64)

        # Final output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        b = self.bottleneck(d6)

        # Decoder with skip connections
        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))

        return self.final(torch.cat([u6, d1], 1))

@st.cache_resource
def load_generator():
    """Load the pre-trained generator model"""
    model = Generator().to(device)
    generator_path = r"C:\minor\floor\generator_epoch_1668.pth"
    
    if not os.path.exists(generator_path):
        st.error(f"Error: {generator_path} not found!")
        return None

    try:
        model.load_state_dict(torch.load(generator_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Convert uploaded image to model input format"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

def tensor_to_image(tensor):
    """Convert model output tensor to PIL Image"""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image + 1) / 2  # Scale from [-1, 1] to [0, 1]
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

# Define all the functions for PPE detection
def generate_pdf_report(tracker, output_file):
    """Generate a KPI dashboard-style PDF report"""
    pdf_path = os.path.join(tempfile.gettempdir(), f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Worker Safety KPI Dashboard", styles['Heading1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Paragraph(f"Output Video: {os.path.basename(output_file)}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Calculate KPIs
    total_incidents = len(tracker.person_safety_violations)
    
    # Summary section
    elements.append(Paragraph("Summary", styles['Heading2']))
    summary_data = [["Total Persons", "Total Violations", "Violation Rate"]]
    
    # Calculate violation rate
    if len(tracker.seen_persons) > 0:
        violation_rate = f"{(len(tracker.person_safety_violations) / len(tracker.seen_persons)) * 100:.1f}%"
    else:
        violation_rate = "0%"
        
    summary_data.append([
        str(len(tracker.seen_persons)), 
        str(total_incidents),
        violation_rate
    ])
    
    summary_table = Table(summary_data)
    summary_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ])
    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # Type of Incident
    type_counts = {
        "No Helmet": 0, "No Goggles": 0, "No Gloves": 0, 
        "No Boots": 0, "No Vest": 0
    }
    for violation in tracker.person_safety_violations:
        details = violation["details"]
        for violation_type in type_counts.keys():
            if violation_type in details:
                type_counts[violation_type] += 1

    # Type of Incident
    elements.append(Paragraph("Type of Incident", styles['Heading2']))
    type_data = [["Type", "Count"]]
    for incident_type, count in type_counts.items():
        type_data.append([incident_type, str(count)])
    
    type_table = Table(type_data)
    type_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ])
    elements.append(type_table)
    elements.append(Spacer(1, 12))

    # Safety Violations Timeline
    elements.append(Paragraph("Safety Violations Timeline", styles['Heading2']))
    if tracker.person_safety_violations:
        violation_data = [["Timestamp", "Event", "Details"]]
        for event in tracker.person_safety_violations:
            violation_data.append([
                event["timestamp"],
                event["event"],
                event["details"]
            ])
        
        violation_table = Table(violation_data)
        violation_table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ])
        elements.append(violation_table)
    else:
        elements.append(Paragraph("No safety violations detected.", styles['Normal']))

    # Add footer
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(f"Report generated by PPE Detection System on {datetime.now().strftime('%Y-%m-%d')}", styles['Italic']))

    # Build the PDF
    doc.build(elements)
    print(f"PDF report generated: {pdf_path}")
    
    return pdf_path

def process_frame(frame, model, tracker, start_time, fps, current_frame):
    """Process a single frame and return the annotated frame with detections"""
    elapsed_seconds = current_frame / fps if fps > 0 else 0
    current_time = (start_time + timedelta(seconds=elapsed_seconds)).strftime("%H:%M:%S.%f")[:-3]
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    tracker.frame_count += 1
    tracker.total_detections += len(detections)

    # Process detections
    current_frame_detections = {cls: [] for cls in PPE_CLASSES}

    if hasattr(detections, 'class_id') and hasattr(detections, 'xyxy'):
        for i, class_id in enumerate(detections.class_id):
            class_name = model.names[class_id] if class_id < len(model.names) else f"unknown_{class_id}"
            if class_name in PPE_CLASSES:
                tracker.class_counts[class_name] += 1
                current_frame_detections[class_name].append(detections.xyxy[i])
    
    # Draw bounding boxes for all detected objects
    for class_name, boxes in current_frame_detections.items():
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if class_name == "person" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Check safety compliance for persons (only for new persons)
    if "person" in current_frame_detections and len(current_frame_detections["person"]) > 0:
        for person_box in current_frame_detections["person"]:
            if tracker.is_new_person(person_box):
                # Track violations for this person
                person_violations = set()

                # Check for helmet
                has_helmet = any(
                    (person_box[0] <= h_box[2] and person_box[2] >= h_box[0] and
                     person_box[1] <= h_box[3] and person_box[3] >= h_box[1])
                    for h_box in current_frame_detections.get("helmet", [])
                )
                if not has_helmet and "No Helmet" not in person_violations:
                    person_violations.add("No Helmet")
                    tracker.person_safety_violations.append({
                        "timestamp": current_time,
                        "event": "Safety Violation",
                        "details": "No Helmet"
                    })
            
                # Check for goggles
                has_goggles = any(
                    (person_box[0] <= g_box[2] and person_box[2] >= g_box[0] and
                     person_box[1] <= g_box[3] and person_box[3] >= g_box[1])
                    for g_box in current_frame_detections.get("goggles", [])
                )
                if not has_goggles and "No Goggles" not in person_violations:
                    person_violations.add("No Goggles")
                    tracker.person_safety_violations.append({
                        "timestamp": current_time,
                        "event": "Safety Violation",
                        "details": "No Goggles"
                    })
            
                # Check for boots
                has_boots = any(
                    (person_box[0] <= b_box[2] and person_box[2] >= b_box[0] and
                     person_box[1] <= b_box[3] and person_box[3] >= b_box[1])
                    for b_box in current_frame_detections.get("boots", [])
                )
                if not has_boots and "No Boots" not in person_violations:
                    person_violations.add("No Boots")
                    tracker.person_safety_violations.append({
                        "timestamp": current_time,
                        "event": "Safety Violation",
                        "details": "No Boots"
                    })

                # Check for gloves
                has_gloves = any(
                    (person_box[0] <= g_box[2] and person_box[2] >= g_box[0] and
                     person_box[1] <= g_box[3] and person_box[3] >= g_box[1])
                    for g_box in current_frame_detections.get("gloves", [])
                )
                if not has_gloves and "No Gloves" not in person_violations:
                    person_violations.add("No Gloves")
                    tracker.person_safety_violations.append({
                        "timestamp": current_time,
                        "event": "Safety Violation",
                        "details": "No Gloves"
                    })

                # Check for vest
                has_vest = any(
                    (person_box[0] <= v_box[2] and person_box[2] >= v_box[0] and
                     person_box[1] <= v_box[3] and person_box[3] >= v_box[1])
                    for v_box in current_frame_detections.get("vest", [])
                )
                if not has_vest and "No Vest" not in person_violations:
                    person_violations.add("No Vest")
                    tracker.person_safety_violations.append({
                        "timestamp": current_time,
                        "event": "Safety Violation",
                        "details": "No Vest"
                    })

    # Check for explicit "no_xxx" detections in each frame
    for class_name, boxes in current_frame_detections.items():
        if class_name.startswith("no_") and len(boxes) > 0:
            violation_type = f"No {class_name.replace('no_', '').capitalize()}"
            tracker.person_safety_violations.append({
                "timestamp": current_time,
                "event": "Safety Violation",
                "details": violation_type
            })

    # Apply annotation
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    return annotated_frame

def create_stats_charts(tracker):
    # Create a DataFrame for violation types
    if not tracker.person_safety_violations:
        return None, None
    
    # Count violation types
    violation_types = {"No Helmet": 0, "No Goggles": 0, "No Gloves": 0, "No Boots": 0, "No Vest": 0}
    for violation in tracker.person_safety_violations:
        details = violation["details"]
        if details in violation_types:
            violation_types[details] += 1
    
    df_violations = pd.DataFrame({
        "Violation Type": list(violation_types.keys()),
        "Count": list(violation_types.values())
    })
    
    # Create pie chart for violation types
    fig_pie = px.pie(
        df_violations, 
        values="Count", 
        names="Violation Type",
        title="Safety Violations by Type",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Create bar chart for violations
    fig_bar = px.bar(
        df_violations,
        x="Violation Type",
        y="Count",
        title="Safety Violations Count",
        color="Violation Type",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    return fig_pie, fig_bar

# Function to load the supplier recommendation model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_components = pickle.load(f)
    return model_components

# Function to recommend suppliers
def recommend_suppliers(model_components, material_type, location, min_rating, 
                       max_delivery_time, min_cost, max_cost, top_n=5):
    """
    Recommend suppliers based on specified criteria
    """
    # Extract components
    nn_model = model_components['nn_model']
    features = model_components['features']
    suppliers_df = model_components['suppliers_df']
    materials = model_components['materials']
    locations = model_components['locations']
    scaler = model_components['scaler']
    
    # Check if material type and location are valid
    if material_type not in materials:
        raise ValueError(f"Material type '{material_type}' not found. Available types: {materials}")
    
    if location not in locations:
        raise ValueError(f"Location '{location}' not found. Available locations: {locations}")
    
    # Create a query feature vector
    import numpy as np
    query_features = np.zeros(features.shape[1])
    
    # Set material type and location
    mat_col = features.columns.get_indexer([f'mat_{material_type}'])[0]
    loc_col = features.columns.get_indexer([f'loc_{location}'])[0]
    query_features[mat_col] = 1
    query_features[loc_col] = 1
    
    # Set normalized rating (only set a value if min_rating is specified)
    if min_rating is not None:
        rating_col = features.columns.get_indexer(['Rating'])[0]
        # Scale the rating to match the normalized data
        norm_min_rating = scaler.transform([[min_rating, 0, 0, 0]])[0][0]
        query_features[rating_col] = norm_min_rating
    
    # Set normalized delivery time (only if max_delivery_time is specified)
    if max_delivery_time is not None:
        delivery_col = features.columns.get_indexer(['Average Delivery Time (days)'])[0]
        # For delivery time, lower is better, so we use a low value in the query
        # We invert the scaling as we want suppliers with delivery time <= max_delivery_time
        norm_max_delivery = scaler.transform([[0, 0, max_delivery_time, 0]])[0][2]
        query_features[delivery_col] = norm_max_delivery
    
    # Find nearest neighbors
    distances, indices = nn_model.kneighbors([query_features], n_neighbors=len(features))
    
    # Get all recommended suppliers
    recommended_indices = indices[0]
    
    # Convert indices to original dataframe indices
    recommended_suppliers = suppliers_df.iloc[recommended_indices].copy()
    
    # Filter by rating if specified
    if min_rating is not None:
        recommended_suppliers = recommended_suppliers[recommended_suppliers['Rating'] >= min_rating]
    
    # Filter by delivery time if specified
    if max_delivery_time is not None:
        recommended_suppliers = recommended_suppliers[
            recommended_suppliers['Average Delivery Time (days)'] <= max_delivery_time
        ]
    
    # Filter by cost range if specified
    if min_cost is not None and max_cost is not None:
        recommended_suppliers = recommended_suppliers[
            (recommended_suppliers['Cost per unit (INR)'] >= min_cost) & 
            (recommended_suppliers['Cost per unit (INR)'] <= max_cost)
        ]
    
    # Filter by exact material type and location (as NN might return close but not exact matches)
    recommended_suppliers = recommended_suppliers[
        (recommended_suppliers['Material Type'] == material_type) & 
        (recommended_suppliers['Location'] == location)
    ]
    
    # Sort by rating (descending), delivery time (ascending), and cost (ascending)
    recommended_suppliers = recommended_suppliers.sort_values(
        by=['Rating', 'Average Delivery Time (days)', 'Cost per unit (INR)'], 
        ascending=[False, True, True]
    )
    
    # Return top N suppliers
    return recommended_suppliers.head(top_n)

def floor_plan_generator_page():
    st.markdown("""
    <div class='custom-card'>
        <h2 class='custom-header'>🏠 Floor Plan Generator</h2>
        <p>Upload a building footprint image to generate a corresponding architectural floor plan.
        This model was trained on the Hugging Face Floor Plan dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    generator = load_generator()
    if generator is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.subheader("Input Footprint")
            uploaded_file = st.file_uploader(
                "Choose a footprint image (JPG/PNG)",
                type=["jpg", "jpeg", "png"],
                key="uploader"
            )

            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file).convert("RGB")
                    st.image(input_image, caption="Uploaded Footprint", use_column_width=True)

                    if st.button("Generate Floor Plan", type="primary"):
                        with st.spinner("Generating floor plan..."):
                            input_tensor = preprocess_image(input_image)
                            with torch.no_grad():
                                output_tensor = generator(input_tensor)

                            output_image = tensor_to_image(output_tensor)

                            with col2:
                                st.subheader("Generated Floor Plan")
                                st.image(output_image, use_column_width=True)

                                img_bytes = io.BytesIO()
                                output_image.save(img_bytes, format="PNG")
                                st.download_button(
                                    label="Download Floor Plan",
                                    data=img_bytes.getvalue(),
                                    file_name="generated_floor_plan.png",
                                    mime="image/png",
                                    key="download"
                                )
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

    with st.sidebar:
        with st.container():
            st.header("About")
            st.markdown("""
            <div class='custom-card'>
                <p>This AI model generates architectural floor plans from building footprints.</p>
                <p><b>How it works:</b></p>
                <ol>
                    <li>Upload a footprint image</li>
                    <li>Click "Generate Floor Plan"</li>
                    <li>View and download the result</li>
                </ol>
                <p><b>Tips for best results:</b></p>
                <ul>
                    <li>Use clear, well-defined footprint images</li>
                    <li>Images should have good contrast</li>
                    <li>Square-ish aspect ratios work best</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**Model Info**")
            st.text(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            st.text("Model: generator.pth")

def delay_predictor_page():
    st.markdown("""
    <div class='custom-card'>
        <h2 class='custom-header'>⏱️ Construction Project Delay Predictor</h2>
        <p>This app predicts the expected delay (in days) for construction projects in India based on various project parameters.</p>
    </div>
    """, unsafe_allow_html=True)

    # Main page for user input
    st.header('Project Parameters')
    st.markdown("Enter your project details below to get a delay prediction.")

    def user_input_features():
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox('City', ['Bangalore', 'Mumbai', 'Ahmedabad', 'Hyderabad', 
                                        'Delhi', 'Chennai', 'Kolkata', 'Pune'])
            project_type = st.selectbox('Project Type', ['Residential', 'Bridge', 'Railway', 
                                                       'Commercial', 'Metro', 'Hospital', 
                                                       'Road', 'Industrial'])
            project_size = st.number_input('Project Size (sq. m)', min_value=1000, max_value=100000, value=25000)
            labor_count = st.number_input('Labor Count', min_value=50, max_value=2000, value=500)
        
        with col2:
            equipment_count = st.number_input('Equipment Count', min_value=5, max_value=50, value=25)
            avg_temp = st.number_input('Average Temperature (°C)', min_value=15.0, max_value=45.0, value=25.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=150.0)
            milestone = st.selectbox('Milestone', ['Inspection', 'Structural Completion', 
                                                  'Roofing', 'Handover', 'Foundation', 'Finishing'])
            external_factor = st.selectbox('External Factor', ['Unknown', 'Land dispute', 
                                                              'Funding issues', 'Environmental clearance delay'])
        
        data = {
            'City': city,
            'Project Type': project_type,
            'Project Size (sq. m)': project_size,
            'Labor Count': labor_count,
            'Equipment Count': equipment_count,
            'Avg Temperature (°C)': avg_temp,
            'Rainfall (mm)': rainfall,
            'Milestone': milestone,
            'External Factor': external_factor
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Display user input parameters
    st.subheader('Your Project Details')
    st.dataframe(input_df, use_container_width=True)

    # Load the model
    model_path = r"C:\Users\sanje\Downloads\Delay\Delay\construction_delay_model (1).pkl"
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.session_state['delay_model'] = model
        else:
            st.error(f"Model file not found at: {model_path}")
            return
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Make prediction (but don't show it yet)
    if 'delay_model' in st.session_state:
        prediction = st.session_state['delay_model'].predict(input_df)
    else:
        st.warning("Model not loaded. Please check the model path.")
        return

    # Button to show all results
    if st.button('Predict Delay and Show Analysis', type="primary"):
        # Show prediction
        st.subheader('Prediction Result')
        
        delay_days = round(prediction[0], 1)
        if delay_days <= 30:
            st.success(f"Predicted Construction Delay: **{delay_days} days** (Low Risk)")
        elif 30 < delay_days <= 90:
            st.warning(f"Predicted Construction Delay: **{delay_days} days** (Moderate Risk)")
        else:
            st.error(f"Predicted Construction Delay: **{delay_days} days** (High Risk)")
        
        # Detailed analysis
        st.subheader('Detailed Delay Analysis')
        
        # Risk assessment
        if delay_days <= 30:
            st.markdown("""
            <div class='custom-card'>
                <h4>✅ Low Delay Risk</h4>
                <p>The project is likely to experience minimal delays (≤ 30 days).</p>
                <p><b>Recommended Actions:</b></p>
                <ul>
                    <li>Maintain current operations</li>
                    <li>Monitor for any changes in conditions</li>
                    <li>Standard risk management procedures are sufficient</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif 30 < delay_days <= 90:
            st.markdown("""
            <div class='custom-card'>
                <h4>⚠️ Moderate Delay Risk</h4>
                <p>The project may experience significant delays (31-90 days).</p>
                <p><b>Recommended Actions:</b></p>
                <ul>
                    <li>Review resource allocation</li>
                    <li>Assess external factors</li>
                    <li>Consider buffer time in schedule</li>
                    <li>Weekly progress monitoring recommended</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='custom-card'>
                <h4>❌ High Delay Risk</h4>
                <p>Significant delays are predicted (> 90 days).</p>
                <p><b>Recommended Actions:</b></p>
                <ul>
                    <li>Immediate risk mitigation required</li>
                    <li>Review project planning</li>
                    <li>Daily progress monitoring</li>
                    <li>Consider additional resources</li>
                    <li>Stakeholder communication plan needed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Factor analysis
        st.subheader('Key Contributing Factors')
        factors = []
        
        if input_df['External Factor'].iloc[0] != 'Unknown':
            factors.append(f"External factor: {input_df['External Factor'].iloc[0]}")
        if input_df['Rainfall (mm)'].iloc[0] > 300:
            factors.append("High rainfall (above 300mm)")
        if input_df['Labor Count'].iloc[0] < 100:
            factors.append("Low labor count")
        if input_df['Equipment Count'].iloc[0] < 10:
            factors.append("Insufficient equipment")
        
        if factors:
            st.markdown("""
            <div class='custom-card'>
                <p>The following factors are contributing to the predicted delay:</p>
                <ul>
            """ + "\n".join([f"<li>{factor}</li>" for factor in factors]) + """
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No extreme risk factors identified. Delay is primarily due to standard project variables.")
        
        # Industry benchmarks
        st.subheader('Industry Benchmark Statistics')
        
        stats = pd.DataFrame({
            'Metric': ['Average Delay', 'Minimum Delay', 'Maximum Delay', 'Your Project'],
            'Days': [60, 0, 365, delay_days]
        })
        
        fig = px.bar(
            stats, 
            x='Metric', 
            y='Days', 
            color='Metric',
            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
            title="Comparison with Industry Benchmarks"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='custom-card'>
            <ul>
                <li><b>Average delay</b> in similar projects: 60 days</li>
                <li><b>Best case scenario</b>: 0 days (on-time completion)</li>
                <li><b>Worst case scenario</b>: 365 days</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='custom-card'>
        <p><i>Note: This prediction is based on historical data and machine learning models. 
        Actual results may vary based on real-world conditions not captured in the model.</i></p>
    </div>
    """, unsafe_allow_html=True)

# Main application with navigation
def main():
    st.sidebar.title("🏗️ Construction Management System")
    
    # Sidebar with icons and better organization
    with st.sidebar.expander("🔍 Navigation", expanded=True):
        app_mode = st.radio(
            "Choose Module",
            ["Supplier Recommendation", "PPE Detection", "Floor Plan Generator", "Delay Predictor"],
            index=0
        )
    
    # Add system info in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ System Info", expanded=False):
        st.markdown("""
        **Version:** 1.0.0  
        **Last Updated:** June 2024  
        **Developed by:** Your Team  
        """)
    
    # Add feedback section
    st.sidebar.markdown("---")
    with st.sidebar.expander("📝 Feedback", expanded=False):
        st.text_area("Share your feedback", height=100, key="feedback")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")
    
    # Page routing with improved headers
    if app_mode == "Supplier Recommendation":
        st.header("")
        supplier_recommendation_page()
    elif app_mode == "PPE Detection":
        st.header("")
        ppe_detection_page()
    elif app_mode == "Floor Plan Generator":
        st.header("")
        floor_plan_generator_page()
    elif app_mode == "Delay Predictor":
        st.header("")
        delay_predictor_page()

# Enhanced Supplier Recommendation Page
def supplier_recommendation_page():
    st.markdown("""
    <div class='custom-card'>
        <h2 class='custom-header'>🏭 Supplier Recommendation System</h2>
        <p>Find the best construction material suppliers based on your specific requirements. 
        Select your criteria below and get personalized recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model loading
    st.sidebar.header("Model Configuration")
    
    # Model path - FIXED with raw string prefix to handle Windows paths correctly
    model_path = st.sidebar.text_input("Model Path", r"C:\Users\sanje\Downloads\supplier_recommendation_model.pkl")
    
    # Information about file paths
    st.sidebar.info("""
    When entering Windows file paths, use one of these formats:
    - Forward slashes: C:/Users/sanje/Downloads/model.pkl
    - Double backslashes: C:\\Users\\sanje\\Downloads\\model.pkl
    """)
    
    # Load model button
    if st.sidebar.button("Load Model", type="primary"):
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                st.session_state['model'] = model
                st.sidebar.success(f"Model loaded successfully!")
                
                # Display available materials and locations
                st.sidebar.subheader("Available Options")
                st.sidebar.write(f"Materials: {', '.join(model['materials'])}")
                st.sidebar.write(f"Locations: {', '.join(model['locations'])}")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")
        else:
            st.sidebar.error(f"Model file not found at: {model_path}")
    
    # Check if model is loaded
    if 'model' not in st.session_state:
        st.warning("Please load the model first using the sidebar.")
    else:
        # Main form for recommendations
        st.header("Specify Your Requirements")
        
        # Create two columns layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Material and location selection
            material_type = st.selectbox("Select Material Type", options=st.session_state['model']['materials'])
            location = st.selectbox("Select Location", options=st.session_state['model']['locations'])
            
            # Number of recommendations
            top_n = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)
        
        with col2:
            # Rating, delivery time and cost range
            min_rating = st.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
            max_delivery_time = st.slider("Maximum Delivery Time (days)", min_value=1, max_value=30, value=15)
            
            # Cost range with min and max values
            cost_range = st.slider("Cost Range (INR)", 
                                  min_value=1000, 
                                  max_value=10000, 
                                  value=(2000, 5000),
                                  step=100)
            min_cost, max_cost = cost_range
        
        # Get recommendations button
        if st.button("Get Recommendations", type="primary"):
            try:
                # Get recommendations
                recommendations = recommend_suppliers(
                    st.session_state['model'],
                    material_type,
                    location,
                    min_rating,
                    max_delivery_time,
                    min_cost,
                    max_cost,
                    top_n
                )
                
                # Display recommendations
                if len(recommendations) > 0:
                    st.success(f"Found {len(recommendations)} suppliers matching your criteria!")
                    
                    # Format the dataframe for display
                    display_df = recommendations.copy()
                    display_df = display_df.reset_index(drop=True)
                    
                    # Round numerical values for cleaner display
                    display_df['Rating'] = display_df['Rating'].round(1)
                    display_df['Cost per unit (INR)'] = display_df['Cost per unit (INR)'].round(2)
                    
                    # Display as table
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader("Summary Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Average Rating", f"{display_df['Rating'].mean():.1f}")
                    
                    with stats_col2:
                        st.metric("Average Delivery Time", f"{display_df['Average Delivery Time (days)'].mean():.1f} days")
                    
                    with stats_col3:
                        st.metric("Average Cost", f"₹{display_df['Cost per unit (INR)'].mean():.2f}")
                    
                    # Visualizations
                    st.subheader("Supplier Comparison")
                    
                    fig1 = px.bar(
                        display_df,
                        x='Supplier Name',
                        y='Rating',
                        title='Supplier Ratings',
                        color='Rating',
                        color_continuous_scale='Bluered'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    fig2 = px.scatter(
                        display_df,
                        x='Cost per unit (INR)',
                        y='Average Delivery Time (days)',
                        size='Rating',
                        color='Supplier Name',
                        title='Cost vs Delivery Time',
                        hover_name='Supplier Name'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    
                else:
                    st.warning("No suppliers found matching your criteria. Try adjusting your parameters.")
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='custom-card'>
        <p><i>Supplier Recommendation System - Powered by Machine Learning</i></p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced PPE Detection Page
def ppe_detection_page():
    st.markdown("""
    <div class='custom-card'>
        <h2 class='custom-header'>🦺 PPE Detection System</h2>
        <p>Monitor and detect PPE (Personal Protective Equipment) compliance in real-time to ensure worker safety on construction sites.</p>
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;'>
            <span class='custom-badge' style='background-color: #d4edda;'>🪖 Helmets</span>
            <span class='custom-badge' style='background-color: #d4edda;'>👓 Goggles</span>
            <span class='custom-badge' style='background-color: #d4edda;'>🧤 Gloves</span>
            <span class='custom-badge' style='background-color: #d4edda;'>👢 Boots</span>
            <span class='custom-badge' style='background-color: #d4edda;'>🦺 Vests</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar options
    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.markdown("### Input Source")
    input_source = st.sidebar.radio("Select detection method:", ["Webcam", "Upload Video"])
    
    # Model paths configuration in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Configuration")
    webcam_model_path = st.sidebar.text_input("Webcam Model Path", r"C:\minor\integra\best (1).pt")
    upload_model_path = st.sidebar.text_input("Upload Video Model Path", r"C:\Users\sanje\Downloads\last.pt")
    
    # Add model information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ System Information")
    st.sidebar.markdown("""
    - *Webcam Model*: Optimized for real-time processing
    - *Upload Model*: Higher accuracy for detailed analysis
    - *Version*: 1.0.0
    """)
    
    # Load models
    @st.cache_resource
    def load_webcam_model():
        try:
            if os.path.exists(webcam_model_path):
                return YOLO(webcam_model_path)
            else:
                st.error(f"Webcam model not found at: {webcam_model_path}")
                return None
        except Exception as e:
            st.error(f"Error loading webcam model: {e}")
            return None
    
    @st.cache_resource
    def load_upload_model():
        try:
            if os.path.exists(upload_model_path):
                return YOLO(upload_model_path)
            else:
                st.error(f"Upload model not found at: {upload_model_path}")
                return None
        except Exception as e:
            st.error(f"Error loading upload model: {e}")
            return None
    
    # Load appropriate model based on input source
    if input_source == "Webcam":
        with st.spinner("Loading webcam model (optimized for real-time)..."):
            model = load_webcam_model()
    else:
        with st.spinner("Loading upload model (higher accuracy)..."):
            model = load_upload_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model paths.")
        st.stop()
    
    # Initialize tracker in session state if not exists
    if 'tracker' not in st.session_state:
        st.session_state.tracker = DetectionTracker()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📷 Detection", "📊 Analytics", "📄 Report"])
    
    with tab1:
        if input_source == "Webcam":
            st.subheader("Webcam Feed")
            st.info("Using real-time optimized model for webcam processing")
            
            col1, col2 = st.columns(2)
            with col1:
                run_detection = st.button("Start Detection", type="primary")
            with col2:
                stop_detection = st.button("Stop Detection")
            
            stframe = st.empty()
            
            if run_detection:
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("Error: Could not open webcam.")
                    st.stop()
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30  # Default to 30 fps if webcam doesn't report FPS
                
                start_time = datetime.now()
                frame_count = 0
                
                # Create temp file for video saving
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                output_file = temp_video.name
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                
                # Reset tracker for new session
                st.session_state.tracker.reset()
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        annotated_frame = process_frame(frame, model, st.session_state.tracker, start_time, fps, frame_count)
                        out.write(annotated_frame)
                        
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        stframe.image(rgb_frame, channels="RGB", use_column_width=True)
                        
                        if stop_detection:
                            break
                        
                        time.sleep(0.01)  # Small delay to prevent UI from freezing
                except Exception as e:
                    st.error(f"Error during webcam processing: {e}")
                finally:
                    cap.release()
                    out.release()
                    st.session_state.output_file = output_file
                    st.success(f"Detection complete! Processed {frame_count} frames.")
        
        elif input_source == "Upload Video":
            st.subheader("Upload Video")
            st.info("Using high-accuracy model for video processing")
            
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            
            if uploaded_file is not None:
                # Save uploaded file to temp location
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                
                st.video(uploaded_file)
                process_button = st.button("Process Video", type="primary")
                
                if process_button:
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Error: Could not open video file.")
                        st.stop()
                    
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create temp file for output
                    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    output_file = temp_output.name
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                    
                    # Reset tracker for new session
                    st.session_state.tracker.reset()
                    
                    start_time = datetime.now()
                    frame_count = 0
                    
                    try:
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            annotated_frame = process_frame(frame, model, st.session_state.tracker, start_time, fps, frame_count)
                            out.write(annotated_frame)
                            
                            # Update progress
                            progress = int(frame_count / total_frames * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")
                            
                    except Exception as e:
                        st.error(f"Error during video processing: {e}")
                    finally:
                        cap.release()
                        out.release()
                        st.session_state.output_file = output_file
                        st.success(f"Processing complete! Processed {frame_count} frames.")
                        
                        # Show processed video
                        st.subheader("Processed Video")
                        
                        # Create a download button for the processed video
                        with open(output_file, 'rb') as file:
                            st.download_button(
                                label="Download Processed Video",
                                data=file,
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
    
    with tab2:
        st.subheader("Analytics Dashboard")
        
        if 'tracker' in st.session_state:
            tracker = st.session_state.tracker
            
            # Display basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Persons Detected", len(tracker.seen_persons))
            with col2:
                st.metric("Total Violations", len(tracker.person_safety_violations))
            with col3:
                if len(tracker.seen_persons) > 0:
                    violation_rate = len(tracker.person_safety_violations) / len(tracker.seen_persons) * 100
                    st.metric("Violation Rate", f"{violation_rate:.1f}%")
                else:
                    st.metric("Violation Rate", "0%")
            
            # Create and display charts
            if len(tracker.person_safety_violations) > 0:
                # Count violation types
                violation_types = {
                    "No Helmet": 0, 
                    "No Goggles": 0, 
                    "No Gloves": 0, 
                    "No Boots": 0, 
                    "No Vest": 0
                }
                
                for violation in tracker.person_safety_violations:
                    details = violation["details"]
                    if details in violation_types:
                        violation_types[details] += 1
                
                # Create pie chart
                fig_pie = px.pie(
                    names=list(violation_types.keys()),
                    values=list(violation_types.values()),
                    title="Safety Violations by Type",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                # Create bar chart
                fig_bar = px.bar(
                    x=list(violation_types.keys()),
                    y=list(violation_types.values()),
                    title="Safety Violations Count",
                    color=list(violation_types.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={'x': 'Violation Type', 'y': 'Count'}
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Show violations table
                st.subheader("Safety Violations Log")
                violations_df = pd.DataFrame(tracker.person_safety_violations)
                st.dataframe(violations_df, use_container_width=True)
            else:
                if len(tracker.seen_persons) > 0:
                    st.success("All workers are properly equipped with PPE - no violations detected!")
                else:
                    st.info("No detection data available. Run detection first.")
        else:
            st.info("No detection data available. Run detection first.")
    
    with tab3:
        st.subheader("Safety Report")
        
        if 'tracker' in st.session_state and 'output_file' in st.session_state:
            tracker = st.session_state.tracker
            output_file = st.session_state.output_file
            
            # Store the generated PDF path in session state to keep it available
            if 'pdf_report_path' not in st.session_state:
                st.session_state.pdf_report_path = None
            
            col1, col2 = st.columns([1, 2])
            with col1:
                generate_report = st.button("Generate PDF Report", type="primary")
            
            if generate_report:
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_file = generate_pdf_report(tracker, output_file)
                        st.session_state.pdf_report_path = pdf_file
                        st.success("PDF report generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating PDF report: {e}")
            
            # Always show the download button if we have a report
            if st.session_state.pdf_report_path and os.path.exists(st.session_state.pdf_report_path):
                with col2:
                    with open(st.session_state.pdf_report_path, 'rb') as file:
                        st.download_button(
                            label="Download PDF Report",
                            data=file,
                            file_name="safety_report.pdf",
                            mime="application/pdf",
                            key="download_report"
                        )
                
                # Show PDF preview
                st.subheader("Report Preview")
                st.info("A downloadable PDF report has been generated with detailed safety compliance information.")
                st.markdown("""
                <div class='custom-card'>
                    <p>The PDF report includes:</p>
                    <ul>
                        <li>Summary of safety statistics</li>
                        <li>Breakdown of violation types</li>
                        <li>Complete timeline of all safety violations</li>
                        <li>Timestamps and details for each incident</li>
                    </ul>
                    <p>Click the 'Download PDF Report' button above to save this report.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No detection data available. Run detection first.")

if __name__ == "__main__":
    main()
