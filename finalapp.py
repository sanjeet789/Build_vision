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

# Define classes for PPE detection
PPE_CLASSES = [
    "boots", "gloves", "goggles", "helmet", "no-helm", 
    "no_glove", "no_goggles", "no_helmet", "no_shoes", "person", "vest"
]

# Set page configuration
st.set_page_config(
    page_title="Construction Management System", 
    layout="wide",
    page_icon="🏗️",
    initial_sidebar_state="expanded"
)

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
        violation_rate = f"{(len(tracker.person_safety_violations) / len(tracker.seen_persons) * 100):.1f}%"
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

# Main application with navigation
def main():
    st.sidebar.title("🏗️ Construction Management System")
    app_mode = st.sidebar.selectbox("Choose the module", 
                                   ["Supplier Recommendation", "PPE Detection"])
    
    if app_mode == "Supplier Recommendation":
        supplier_recommendation_page()
    elif app_mode == "PPE Detection":
        ppe_detection_page()

# Supplier Recommendation Page
def supplier_recommendation_page():
    st.title("Construction Supplier Recommendation System")
    st.write("""
    This application helps you find the best suppliers for construction materials based on your specific requirements.
    Select your criteria below and get personalized recommendations.
    """)
    
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
    if st.sidebar.button("Load Model"):
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
        if st.button("Get Recommendations"):
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
                    
                else:
                    st.warning("No suppliers found matching your criteria. Try adjusting your parameters.")
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Supplier Recommendation System - Powered by Machine Learning")

# PPE Detection Page
def ppe_detection_page():
    st.title("🦺 PPE Detection System")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
    <h3>Worker Safety Monitoring System</h3>
    <p>Monitor and detect PPE (Personal Protective Equipment) compliance in real-time. This system identifies:</p>
    <ul>
      <li>🪖 <b>Helmets</b> - Head protection</li>
      <li>👓 <b>Goggles</b> - Eye protection</li>
      <li>🧤 <b>Gloves</b> - Hand protection</li>
      <li>👢 <b>Boots</b> - Foot protection</li>
      <li>🦺 <b>Vests</b> - High-visibility clothing</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar options
    st.sidebar.header("📋 Settings")
    
    st.sidebar.markdown("### Input Source")
    input_source = st.sidebar.radio("Select detection method:", ["Webcam", "Upload Video"])
    
    # Add model information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ System Information")
    st.sidebar.markdown("""
    - *Model*: YOLOv8 Custom
    - *Classes*: Helmet, Goggles, Gloves, Boots, Vest, Person
    - *Version*: 1.0.0
    """)
    
    # Add help section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🆘 Help")
    with st.sidebar.expander("How to use this app"):
        st.markdown("""
        1. Select input source (Webcam or Video Upload)
        2. For webcam, click "Start Detection"
        3. For video upload, upload a file and click "Process Video"
        4. View results in the Analytics tab
        5. Generate PDF reports in the Report tab
        """)
        
    with st.sidebar.expander("About PPE Detection"):
        st.markdown("""
        Personal Protective Equipment (PPE) is crucial for workplace safety.
        
        This application uses computer vision to detect whether workers are
        wearing the required safety equipment, helping to maintain compliance
        with safety regulations.
        """)
    
    # Load model at startup
    @st.cache_resource
    def load_ppe_model():
        return YOLO(r"C:\minor\integra\best (1).pt")  # You can make this path configurable
    
    try:
        with st.spinner("Loading PPE detection model..."):
            model = load_ppe_model()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        st.stop()
    
    # Initialize tracker
    tracker = DetectionTracker()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Detection", "Analytics", "Report"])
    
    with tab1:
        if input_source == "Webcam":
            st.subheader("Webcam Feed")
            run_detection = st.button("Start Detection")
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
                tracker.reset()
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        annotated_frame = process_frame(frame, model, tracker, start_time, fps, frame_count)
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
                    st.session_state.tracker = tracker
                    st.success(f"Detection complete! Processed {frame_count} frames.")
        
        elif input_source == "Upload Video":
            st.subheader("Upload Video")
            uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
            
            if uploaded_file is not None:
                # Save uploaded file to temp location
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
                
                st.video(uploaded_file)
                process_button = st.button("Process Video")
                
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
                    tracker.reset()
                    
                    start_time = datetime.now()
                    frame_count = 0
                    
                    try:
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            annotated_frame = process_frame(frame, model, tracker, start_time, fps, frame_count)
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
                        st.session_state.tracker = tracker
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
        
        if hasattr(st.session_state, 'tracker'):
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
            fig_pie, fig_bar = create_stats_charts(tracker)
            if fig_pie and fig_bar:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Show violations table
                if tracker.person_safety_violations:
                    st.subheader("Safety Violations Log")
                    violations_df = pd.DataFrame(tracker.person_safety_violations)
                    st.dataframe(violations_df, use_container_width=True)
                else:
                    st.info("No safety violations detected.")
            else:
                st.info("No data available for visualization. Run detection first.")
        else:
            st.info("No detection data available. Run detection first.")
    
    with tab3:
        st.subheader("Safety Report")
        
        if hasattr(st.session_state, 'tracker') and hasattr(st.session_state, 'output_file'):
            tracker = st.session_state.tracker
            output_file = st.session_state.output_file
            
            # Store the generated PDF path in session state to keep it available
            if 'pdf_report_path' not in st.session_state:
                st.session_state.pdf_report_path = None
            
            col1, col2 = st.columns([1, 2])
            with col1:
                generate_report = st.button("Generate PDF Report")
            
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
                The PDF report includes:
                - Summary of safety statistics
                - Breakdown of violation types 
                - Complete timeline of all safety violations
                - Timestamps and details for each incident
                
                Click the 'Download PDF Report' button above to save this report.
                """)
        else:
            st.info("No detection data available. Run detection first.")

if __name__ == "__main__":
    main()
