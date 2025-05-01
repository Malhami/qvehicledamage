import streamlit as st
from openai import AzureOpenAI
import base64
import cv2
import tempfile
import io
from PIL import Image
import json
import pandas as pd
import plotly.express as px
import config
import os
import hashlib
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Smart Damage Analyzer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more attractive interface
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4F8BF9;
        --secondary-color: #FF5A5F;
        --background-color: #F8F9FB;
        --text-color: #333333;
        --accent-color: #19CDD7;
    }
    
    /* Overall styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 30px;
        color: var(--primary-color);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .main-header p {
        font-size: 1rem;
        color: #888;
    }
    
    /* Card styling */
    .card {
        border-radius: 10px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    .card-header {
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #3a7ad5;
        transform: scale(1.03);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: var(--primary-color);
    }
    
    /* Analysis results styling */
    .analysis-header {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        color: white;
        padding: 15px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    .analysis-content {
        background-color: white;
        padding: 20px;
        border-radius: 0 0 8px 8px;
        border: 1px solid #eee;
        border-top: none;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: var(--accent-color);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border: 1px solid #eee;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Custom separator */
    .separator {
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        margin: 30px 0;
        border-radius: 2px;
    }
    
    /* Success/info/warning boxes */
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 5px solid #28a745;
    }
    
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 5px solid #17a2b8;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 5px solid #ffc107;
    }
    
    /* Dashboard stats */
    .stat-card {
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
        margin-bottom: 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stat-value {
        font-size: 1.5rem !important;
        font-weight: 600;
        color: var(--primary-color);
        word-break: break-word;
        max-width: 100%;
        overflow-wrap: break-word;
        padding: 0 5px;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #2C3E50;
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: white;
    }
    
    /* Dark mode toggle switch */
    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 60px;
        height: 34px;
    }
    
    .toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #ccc;
        transition: .4s;
        border-radius: 34px;
    }
    
    .slider:before {
        position: absolute;
        content: "";
        height: 26px;
        width: 26px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    
    input:checked + .slider {
        background-color: var(--primary-color);
    }
    
    input:checked + .slider:before {
        transform: translateX(26px);
    }
</style>
""", unsafe_allow_html=True)

# Configure Azure OpenAI client
AZURE_OPENAI_ENDPOINT = config.endpoint
AZURE_OPENAI_DEPLOYMENT_NAME = config.deployment
AZURE_OPENAI_KEY = config.subscription_key

if not AZURE_OPENAI_KEY:
    st.error("Azure OpenAI API key not found in environment variables. Please set AZURE_OPENAI_KEY.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-12-01-preview",
)

# Create a directory for saved analyses if it doesn't exist
ANALYSES_DIR = "saved_analyses"
if not os.path.exists(ANALYSES_DIR):
    os.makedirs(ANALYSES_DIR)

# Helper functions
def load_damage_types():
    """Load predefined damage types and their descriptions."""
    return {
        "Dent": "A depression in the surface of the vehicle.",
        "Scratch": "A mark or line made by scraping or cutting the surface.",
        "Crack": "A line on the surface of the vehicle indicating it is broken but not separated.",
        "Shatter": "Broken glass or plastic components.",
        "Puncture": "A hole made by a sharp object.",
        "Burn": "Damage caused by fire or extreme heat.",
        "Water Damage": "Damage caused by exposure to water or flooding.",
        "Collision": "Damage from impact with another vehicle or object.",
        "Vandalism": "Intentional damage caused by a third party.",
        "Hail": "Damage caused by hail impacts."
    }

def get_car_manufacturers():
    """Get a list of common car manufacturers."""
    return [
        "Audi", "BMW", "Chevrolet", "Citroën", "Dacia", "Fiat", "Ford", "Honda", 
        "Hyundai", "Jaguar", "Jeep", "Kia", "Land Rover", "Lexus", "Mazda", 
        "Mercedes-Benz", "Mitsubishi", "Nissan", "Opel", "Peugeot", "Porsche", 
        "Renault", "Seat", "Škoda", "Subaru", "Suzuki", "Tesla", "Toyota", 
        "Volkswagen", "Volvo"
    ]

def generate_report_id():
    """Generate a unique report ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d%H%M%S") + hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]

def save_analysis(report_id, analysis_data):
    """Save analysis results to a JSON file."""
    filename = os.path.join(ANALYSES_DIR, f"{report_id}.json")
    with open(filename, 'w') as f:
        json.dump(analysis_data, f)
    return filename

def load_analysis(report_id):
    """Load a saved analysis by report ID."""
    filename = os.path.join(ANALYSES_DIR, f"{report_id}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def get_saved_analysis_reports():
    """Get a list of all saved analysis reports."""
    reports = []
    for filename in os.listdir(ANALYSES_DIR):
        if filename.endswith('.json'):
            report_id = filename.replace('.json', '')
            try:
                data = load_analysis(report_id)
                if data:
                    reports.append({
                        'id': report_id,
                        'date': data.get('timestamp', 'Unknown'),
                        'vehicle': data.get('vehicle_type', 'Unknown'),
                        'estimate': data.get('cost_estimate', 'N/A')
                    })
            except Exception as e:
                print(f"Error loading report {report_id}: {e}")
    return reports

def parse_analysis_result(raw_analysis):
    """Parse the raw analysis text to extract structured data."""
    data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'raw_analysis': raw_analysis,
        'vehicle_type': 'Unknown',
        'damage_types': [],
        'fraud_likelihood': 'Low',
        'cost_estimate': 'Unknown',
        'severity': 'Medium'
    }
    
    # Remove markdown syntax for cleaner text processing
    clean_analysis = raw_analysis.replace('*', '').replace('#', '').replace('**', '')
    
    # Extract vehicle type
    if "vehicle type:" in clean_analysis.lower():
        lines = clean_analysis.split('\n')
        for line in lines:
            if "vehicle type:" in line.lower() or line.strip().startswith(("This appears to be", "The vehicle is")):
                data['vehicle_type'] = line.strip()
                break
    else:
        # Try to find vehicle references in the first few lines
        first_lines = clean_analysis.split('\n')[:5]
        for line in first_lines:
            # Look for common vehicle make references
            for make in get_car_manufacturers():
                if make.lower() in line.lower():
                    data['vehicle_type'] = line.strip()
                    break

    # Extract damage types
    damage_types = load_damage_types()
    for damage_type in damage_types:
        if damage_type.lower() in clean_analysis.lower():
            data['damage_types'].append(damage_type)
    
    # If no damage types found, try to extract them differently
    if not data['damage_types']:
        damage_indicators = ["damage", "damaged", "damages"]
        lines = clean_analysis.split('\n')
        for line in lines:
            if any(indicator in line.lower() for indicator in damage_indicators):
                # Extract words around damage mentions that might be damage types
                words = line.split()
                for i, word in enumerate(words):
                    if any(indicator in word.lower() for indicator in damage_indicators) and i > 0:
                        potential_damage = words[i-1].strip('.,():;')
                        if len(potential_damage) > 3 and potential_damage.lower() not in ["the", "and", "has", "with", "this"]:
                            data['damage_types'].append(potential_damage.capitalize())
    
    # Ensure we have at least one damage type
    if not data['damage_types']:
        data['damage_types'] = ["Damage"]
    
    # Extract fraud likelihood
    if "fraudulent" in clean_analysis.lower():
        if "not fraudulent" in clean_analysis.lower() or "low likelihood of fraud" in clean_analysis.lower() or "no indication of fraud" in clean_analysis.lower():
            data['fraud_likelihood'] = 'Low'
        elif "possibly fraudulent" in clean_analysis.lower() or "medium likelihood" in clean_analysis.lower() or "potential fraud" in clean_analysis.lower():
            data['fraud_likelihood'] = 'Medium'
        elif "highly likely fraudulent" in clean_analysis.lower() or "high likelihood" in clean_analysis.lower() or "signs of fraud" in clean_analysis.lower():
            data['fraud_likelihood'] = 'High'
    
    # Extract cost estimate - look for euro symbols, currency references
    import re
    
    # Look for currency patterns with regex
    currency_pattern = r'€\s*\d+[.,]?\d*|(?:eur|euro)s?\s*\d+[.,]?\d*|\d+[.,]?\d*\s*(?:eur|euro)s?'
    currency_matches = re.findall(currency_pattern, clean_analysis.lower())
    if currency_matches:
        data['cost_estimate'] = currency_matches[0]
    else:
        # Look for cost estimate lines
        cost_lines = [line for line in clean_analysis.split('\n') 
                    if "€" in line or 
                    any(term in line.lower() for term in ["eur", "euro", "cost estimate", "repair cost", "estimate"])]
        if cost_lines:
            data['cost_estimate'] = cost_lines[0].strip()
    
    # Determine severity
    severity_words = {
        'minor': 0,
        'light': 0,
        'small': 0,
        'moderate': 1,
        'medium': 1,
        'intermediate': 1,
        'significant': 2,
        'substantial': 2, 
        'considerable': 2,
        'severe': 3,
        'extensive': 3,
        'major': 3,
        'heavy': 3,
        'critical': 3
    }
    
    severity_score = 0
    for word, score in severity_words.items():
        if word in clean_analysis.lower():
            severity_score = max(severity_score, score)
    
    severity_mapping = {0: 'Minor', 1: 'Moderate', 2: 'Significant', 3: 'Severe'}
    data['severity'] = severity_mapping.get(severity_score, 'Moderate')
    
    return data

def analyze_image(image_bytes, additional_instructions=""):
    """Analyzes an image using Azure OpenAI Vision with custom instructions."""
    
    # Convert bytes to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt = """Analyze the damage and identify the different damage types. Mention the car type at the beginning of the analysis if you're sure about identifying it.
    
    Estimate if the image is fraudulent or not, and present the results in a well-formatted way.
    Do a fix cost estimation if possible, assuming this is for a Netherlands-based insurance company.
    
    Be very specific about:
    1. Type of damage (dent, scratch, broken parts, etc.)
    2. Severity of damage (minor, moderate, severe)
    3. Affected areas of the vehicle
    4. Cost estimate range in Euros
    5. Likelihood of fraud (low, medium, high) with brief reasoning
    """
    
    if additional_instructions:
        prompt += f"\n\nAdditional instructions: {additional_instructions}"

    content = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
        )
        
        analysis_text = response.choices[0].message.content
        
        # Parse the analysis for structured data
        structured_data = parse_analysis_result(analysis_text)
        
        return analysis_text, structured_data
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return f"Error: {str(e)}", None

def analyze_video(video_bytes, additional_instructions=""):
    """Analyzes a video by extracting frames and analyzing them."""

    # Save video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    # Use OpenCV to read the video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return "Unable to read the video file.", None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    # Calculate number of frames to analyze (more for longer videos)
    num_frames = min(max(int(duration / 5), 3), 10)  # Between 3 and 10 frames
    
    frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    # Progress bar for video analysis
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Select frames to analyze
    frames_to_analyze = []
    for i, frame_idx in enumerate(frame_indices):
        status_text.text(f"Extracting frame {i+1}/{num_frames}...")
        progress_bar.progress((i+1) / (2*num_frames))  # First half of progress is extraction
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert frame to JPEG bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            frames_to_analyze.append(image_bytes)
    
    cap.release()
    os.unlink(temp_video_path)  # Clean up temp file

    # Display frames in a grid
    cols = st.columns(min(3, len(frames_to_analyze)))
    for idx, image_bytes in enumerate(frames_to_analyze):
        cols[idx % 3].image(image_bytes, caption=f'Frame {idx + 1}', use_container_width =True)

    # Analyze each selected frame
    analyses = []
    all_structured_data = []
    
    for idx, image_bytes in enumerate(frames_to_analyze):
        status_text.text(f"Analyzing frame {idx+1}/{num_frames}...")
        progress_bar.progress(0.5 + (idx+1) / (2*num_frames))  # Second half of progress is analysis
        
        frame_text = f"Frame {idx + 1} ({frame_indices[idx]/frame_count*100:.1f}% through video)"
        analysis_text, structured_data = analyze_image(
            image_bytes, 
            f"This is {frame_text}. {additional_instructions}"
        )
        
        analyses.append(f"**{frame_text} Analysis:**\n{analysis_text}\n")
        if structured_data:
            structured_data['frame'] = idx + 1
            structured_data['frame_position'] = f"{frame_indices[idx]/frame_count*100:.1f}%"
            all_structured_data.append(structured_data)
    
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    
    # Aggregate structured data
    aggregated_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'raw_analysis': "\n".join(analyses),
        'vehicle_type': all_structured_data[0]['vehicle_type'] if all_structured_data else 'Unknown',
        'damage_types': [],
        'fraud_likelihood': 'Low',
        'cost_estimate': 'Unknown',
        'severity': 'Medium',
        'frames_analyzed': len(frames_to_analyze),
        'frame_analyses': all_structured_data
    }
    
    # Combine damage types from all frames
    all_damage_types = set()
    for data in all_structured_data:
        all_damage_types.update(data['damage_types'])
    aggregated_data['damage_types'] = list(all_damage_types)
    
    # Use the highest fraud likelihood
    fraud_levels = {'Low': 0, 'Medium': 1, 'High': 2}
    highest_fraud = 0
    for data in all_structured_data:
        fraud_level = fraud_levels.get(data['fraud_likelihood'], 0)
        highest_fraud = max(highest_fraud, fraud_level)
    reverse_fraud_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
    aggregated_data['fraud_likelihood'] = reverse_fraud_levels[highest_fraud]
    
    # Use the highest severity
    severity_levels = {'Minor': 0, 'Moderate': 1, 'Significant': 2, 'Severe': 3}
    highest_severity = 0
    for data in all_structured_data:
        severity_level = severity_levels.get(data['severity'], 1)
        highest_severity = max(highest_severity, severity_level)
    reverse_severity_levels = {0: 'Minor', 1: 'Moderate', 2: 'Significant', 3: 'Severe'}
    aggregated_data['severity'] = reverse_severity_levels[highest_severity]
    
    return "\n".join(analyses), aggregated_data

def display_analysis_dashboard(analysis_data):
    """Display a comprehensive dashboard for the analysis results."""
    
    if not analysis_data:
        st.warning("No analysis data available to display.")
        return
    
    # Clean any markdown characters from values
    for key in analysis_data:
        if isinstance(analysis_data[key], str):
            analysis_data[key] = analysis_data[key].replace('*', '').replace('#', '')
    
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # Summary header
    st.markdown(f"""
    <div class="analysis-header" style="font-size: 1.1rem;">
        Analysis Summary Report #{analysis_data.get('report_id', 'Unknown')}
    </div>
    """, unsafe_allow_html=True)
    
    # Main summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    # Helper function to clean text (remove markdown and special characters)
    def clean_text(text):
        if not text:
            return "Unknown"
        # Remove markdown syntax and special characters
        cleaned = text.replace('*', '').replace('#', '').replace(':', ' ').strip()
        # If it looks like a long sentence, truncate it
        if len(cleaned) > 15:
            parts = cleaned.split()
            if len(parts) > 2:
                cleaned = ' '.join(parts[-2:])  # Just use last 2 words
        return cleaned
    
    # Get vehicle type (simplified)
    vehicle_type = analysis_data.get('vehicle_type', 'Unknown')
    if isinstance(vehicle_type, str):
        if ':' in vehicle_type:
            vehicle_type = vehicle_type.split(':')[-1].strip()
        vehicle_type = clean_text(vehicle_type)
        if len(vehicle_type) > 10:  # If still too long, truncate
            vehicle_type = vehicle_type[:10]
    
    # Clean up cost estimate
    cost_estimate = analysis_data.get('cost_estimate', 'Unknown')
    if isinstance(cost_estimate, str):
        # Extract just the numeric part and currency symbol if possible
        if 'Cost Estimate:' in cost_estimate:
            cost_estimate = cost_estimate.replace('Cost Estimate:', '').strip()
        cost_estimate = clean_text(cost_estimate)
        # If it contains euro symbol or EUR, keep it simple
        if '€' in cost_estimate or 'EUR' in cost_estimate.upper():
            # Try to extract just the amount
            import re
            amount_match = re.search(r'€?\s*(\d+[.,]?\d*)', cost_estimate)
            if amount_match:
                cost_estimate = f"€{amount_match.group(1)}"
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size: 1.5rem;">{vehicle_type}</div>
            <div class="stat-label">Vehicle Type</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size: 1.5rem;">{clean_text(analysis_data.get('severity', 'Medium'))}</div>
            <div class="stat-label">Damage Severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size: 1.5rem;">{cost_estimate}</div>
            <div class="stat-label">Cost Estimate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        fraud_colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        fraud_level = clean_text(analysis_data.get('fraud_likelihood', 'Low'))
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value" style="font-size: 1.5rem; color: {fraud_colors.get(fraud_level, '#28a745')}">{fraud_level}</div>
            <div class="stat-label">Fraud Likelihood</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Damage types visualization
    st.markdown("### Damage Types Identified")
    damage_types = analysis_data.get('damage_types', [])
    
    if damage_types:
        # Create a DataFrame for the damage types
        df_damage = pd.DataFrame({
            'Damage Type': damage_types,
            'Count': [1] * len(damage_types)  # Each type has a count of 1
        })
        
        # Horizontal bar chart for damage types
        fig = px.bar(
            df_damage, 
            y='Damage Type', 
            x='Count', 
            color='Damage Type',
            orientation='h',
            height=max(300, 50 * len(damage_types)),
            title="Damage Types",
            labels={'Count': 'Identified in # Frames' if 'frame_analyses' in analysis_data else 'Present'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No specific damage types were identified.")
    
    # If video analysis, show per-frame data
    if 'frame_analyses' in analysis_data and analysis_data['frame_analyses']:
        st.markdown("### Frame-by-Frame Analysis")
        
        # Create a DataFrame for the frame data
        frame_data = []
        for frame in analysis_data['frame_analyses']:
            frame_data.append({
                'Frame': frame['frame'],
                'Position': frame['frame_position'],
                'Damage Types': ', '.join(frame['damage_types']),
                'Severity': frame['severity'],
                'Fraud Likelihood': frame['fraud_likelihood']
            })
        
        df_frames = pd.DataFrame(frame_data)
        st.dataframe(df_frames, use_container_width=True)
        
        # Timeline visualization
        st.markdown("### Damage Severity Timeline")
        severity_values = {'Minor': 1, 'Moderate': 2, 'Significant': 3, 'Severe': 4}
        
        df_timeline = pd.DataFrame([{
            'Frame': f"{frame['frame']} ({frame['frame_position']})",
            'Severity Score': severity_values.get(frame['severity'], 2),
            'Severity': frame['severity']
        } for frame in analysis_data['frame_analyses']])
        
        fig = px.line(
            df_timeline, 
            x='Frame', 
            y='Severity Score',
            markers=True,
            labels={'Severity Score': 'Severity Level', 'Frame': 'Video Position'},
            title="Damage Severity Throughout Video",
            color_discrete_sequence=['#4F8BF9']
        )
        fig.update_layout(yaxis=dict(tickvals=list(severity_values.values()), ticktext=list(severity_values.keys())))
        st.plotly_chart(fig, use_container_width=True)
    
    # Full detailed analysis
    with st.expander("View Full Analysis Details", expanded=False):
        st.markdown("### Complete Analysis Report")
        st.markdown(analysis_data.get('raw_analysis', 'No detailed analysis available.'))

def display_confirmation(report_id):
    """Display a confirmation message with the report ID."""
    st.markdown(f"""
    <div class="success-box">
        <h3>✅ Analysis Completed Successfully!</h3>
        <p>Your damage analysis report has been created with Report ID: <strong>{report_id}</strong></p>
        <p>You can reference this ID to retrieve this analysis in the future.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.image("logo.jpg", caption=None)  # Replace with your logo
        st.markdown("<h2 style='color: white;'>Smart Damage Analyzer Pro</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #CCC;'>AI-powered insurance damage assessment tool</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        app_mode = st.radio("Navigation", 
            options=["New Analysis", "Past Reports", "Settings"],
            index=0)
        
        st.markdown("---")
        st.markdown("<p style='color: #CCC; font-size: 0.8rem;'>© 2025 Smart Solutions</p>", unsafe_allow_html=True)
    
    # Main area content
    if app_mode == "New Analysis":
        st.markdown("""
        <div class="main-header">
            <h1>Smart Damage Analyzer Pro</h1>
            <p>Upload images or videos of vehicle damage for AI-powered analysis and cost estimation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for image and video analysis
        tab1, tab2 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis"])
        
        with tab1:
            st.markdown("""
            <div class="card">
                <div class="card-header">Image Upload</div>
                <p>Upload a clear photo of the vehicle damage for instant analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # User inputs for image analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_file = st.file_uploader("Upload an image of the damaged vehicle", 
                                                type=["jpg", "jpeg", "png"], key="image_uploader")
            
            with col2:
                vehicle_make = st.selectbox("Vehicle Make (Optional)", 
                                         ["Unknown"] + get_car_manufacturers(),
                                         index=0)
                
                analysis_focus = st.multiselect("Analysis Focus (Optional)",
                                             ["General Assessment", "Fraud Detection", "Cost Estimation", 
                                              "Repair Recommendations", "Parts Identification"],
                                             default=["General Assessment"])
            
            additional_instructions = ""
            if vehicle_make != "Unknown":
                additional_instructions += f"The vehicle is a {vehicle_make}. "
            
            if analysis_focus and "General Assessment" not in analysis_focus:
                focus_str = ", ".join(analysis_focus)
                additional_instructions += f"Please focus your analysis on: {focus_str}. "
            
            # Submit button with styling
            analyze_btn = st.button("Analyze Damage", key="analyze_image_btn", use_container_width=True)
            
            if uploaded_file is not None:
                st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(uploaded_file, caption='Uploaded Image', use_container_width =True)
                
                with col2:
                    st.markdown("""
                    <div class="info-box">
                        <h4>Image Analysis Tips</h4>
                        <ul>
                            <li>Ensure good lighting</li>
                            <li>Capture multiple angles if possible</li>
                            <li>Include a reference object for scale</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                if analyze_btn:
                    with st.spinner('Analyzing your image... This may take a moment'):
                        image_bytes = uploaded_file.read()
                        analysis_text, analysis_data = analyze_image(image_bytes, additional_instructions)
                        
                        if analysis_data:
                            # Generate a report ID
                            report_id = generate_report_id()
                            analysis_data['report_id'] = report_id
                            
                            # Save the analysis
                            save_analysis(report_id, analysis_data)
                            
                            # Display confirmation
                            display_confirmation(report_id)
                            
                            # Display the analysis dashboard
                            display_analysis_dashboard(analysis_data)
                        else:
                            st.error("Analysis failed. Please try again with a different image.")
        
        with tab2:
            st.markdown("""
            <div class="card">
                <div class="card-header">Video Upload</div>
                <p>Upload a short video of the vehicle damage for comprehensive frame-by-frame analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # User inputs for video analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                video_file = st.file_uploader("Upload a video of the damaged vehicle", 
                                            type=["mp4", "avi", "mov"], key="video_uploader")
            
            with col2:
                video_vehicle_make = st.selectbox("Vehicle Make (Optional)", 
                                               ["Unknown"] + get_car_manufacturers(),
                                               index=0, key="video_vehicle_make")
                
                video_analysis_focus = st.multiselect("Analysis Focus (Optional)",
                                                   ["General Assessment", "Fraud Detection", "Cost Estimation", 
                                                    "Repair Recommendations", "Damage Progression"],
                                                   default=["General Assessment"], key="video_analysis_focus")
            
            video_instructions = ""
            if video_vehicle_make != "Unknown":
                video_instructions += f"The vehicle is a {video_vehicle_make}. "
            
            if video_analysis_focus and "General Assessment" not in video_analysis_focus:
                focus_str = ", ".join(video_analysis_focus)
                video_instructions += f"Please focus your analysis on: {focus_str}. "
            
            # Submit button with styling
            analyze_video_btn = st.button("Analyze Video", key="analyze_video_btn", use_container_width=True)
            
            if video_file is not None:
                st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                
                st.video(video_file)
                
                st.markdown("""
                <div class="info-box">
                    <h4>Video Analysis Information</h4>
                    <p>Video analysis will extract key frames and analyze each one separately, then provide an aggregated report.</p>
                    <p>This process may take several minutes depending on video length.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if analyze_video_btn:
                    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                    st.subheader("Video Analysis Processing")
                    
                    video_bytes = video_file.read()
                    analysis_text, analysis_data = analyze_video(video_bytes, video_instructions)
                    
                    if analysis_data:
                        # Generate a report ID
                        report_id = generate_report_id()
                        analysis_data['report_id'] = report_id
                        
                        # Save the analysis
                        save_analysis(report_id, analysis_data)
                        
                        # Display confirmation
                        display_confirmation(report_id)
                        
                        # Display the analysis dashboard
                        display_analysis_dashboard(analysis_data)
                    else:
                        st.error("Video analysis failed. Please try again with a different video.")
    
    elif app_mode == "Past Reports":
        st.markdown("""
        <div class="main-header">
            <h1>Analysis History</h1>
            <p>View and manage your previous damage analysis reports</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get saved reports
        reports = get_saved_analysis_reports()
        
        if not reports:
            st.markdown("""
            <div class="warning-box">
                <h3>No Analysis Reports Found</h3>
                <p>You haven't performed any damage analyses yet. Go to the New Analysis page to create your first report.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display reports in a data table
            df_reports = pd.DataFrame(reports)
            st.dataframe(df_reports, use_container_width=True)
            
            # Report selection
            selected_report = st.selectbox("Select a report to view details:", 
                                        [f"Report #{r['id']} - {r['date']} - {r['vehicle']}" for r in reports])
            
            if selected_report:
                report_id = selected_report.split(" - ")[0].replace("Report #", "")
                report_data = load_analysis(report_id)
                
                if report_data:
                    display_analysis_dashboard(report_data)
                else:
                    st.error("Could not load the selected report.")
    
    elif app_mode == "Settings":
        st.markdown("""
        <div class="main-header">
            <h1>Application Settings</h1>
            <p>Configure your preferences for Smart Damage Analyzer Pro</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Settings sections
        st.markdown("""
        <div class="card">
            <div class="card-header">User Interface Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Theme Preferences")
            st.radio("Color Theme", ["Light", "Dark", "Auto (Follow System)"], index=0)
            st.radio("Layout Density", ["Compact", "Comfortable", "Spacious"], index=1)
        
        with col2:
            st.markdown("##### Notifications")
            st.toggle("Email Notifications", value=True)
            st.toggle("In-App Notifications", value=True)
            st.number_input("Analysis Report Retention (days)", min_value=1, max_value=365, value=30)
        
        st.markdown("""
        <div class="card">
            <div class="card-header">Analysis Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Default Analysis Parameters")
            st.multiselect("Default Analysis Focus", 
                        ["General Assessment", "Fraud Detection", "Cost Estimation", 
                         "Repair Recommendations", "Parts Identification"],
                        default=["General Assessment", "Cost Estimation"])
            
            st.selectbox("Default Currency", ["EUR (€)", "USD ($)", "GBP (£)"], index=0)
        
        with col2:
            st.markdown("##### Video Analysis")
            st.slider("Frames to Analyze", min_value=3, max_value=20, value=8)
            st.toggle("Auto-Enhance Image Quality", value=True)
        
        st.markdown("""
        <div class="card">
            <div class="card-header">Advanced Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Azure OpenAI Integration")
            st.text_input("Endpoint URL", value="https://your-endpoint.openai.azure.com", disabled=True)
            st.text_input("API Key", value="••••••••••••••••••••••••", type="password", disabled=True)
        
        with col2:
            st.markdown("##### Model Configuration")
            st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            st.slider("Max Tokens", min_value=500, max_value=8000, value=4000, step=500)
        
        # Save settings button
        st.button("Save Settings", use_container_width=True)
        
        # Reset settings option
        with st.expander("Reset to Default Settings"):
            st.warning("This will reset all settings to their default values. This action cannot be undone.")
            st.button("Reset All Settings", type="primary")

if __name__ == "__main__":
    main()