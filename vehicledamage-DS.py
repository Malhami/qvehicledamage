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
import os
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="Smart Damage Analyzer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #4F8BF9;
        --secondary-color: #FF5A5F;
        --background-color: #F8F9FB;
        --text-color: #333333;
        --accent-color: #19CDD7;
    }
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 20px;
    }
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
    .card {
        border-radius: 10px;
        box-shadow: 0 6px 16Co: 0 4px 12px rgba(0, 0, 0, 0.1);
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
    .stProgress > div > div {
        background-color: var(--accent-color);
    }
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
    .separator {
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        margin: 30px 0;
        border-radius: 2px;
    }
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
    [data-testid="stSidebar"] {
        background-color: #2C3E50;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")

if not AZURE_OPENAI_KEY:
    st.error("Azure OpenAI API key not found in environment variables. Please set AZURE_OPENAI_KEY.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-12-01-preview",
)

# Create directory for saved analyses
ANALYSES_DIR = "saved_analyses"
if not os.path.exists(ANALYSES_DIR):
    os.makedirs(ANALYSES_DIR)

# Helper functions (unchanged from original code)
def load_damage_types():
    return {
        "Dent": "A depression in the surface of the vehicle.",
        "Scratch": "A mark or line made by scraping or cutting the surface.",
        "Crack": "A line on the surface indicating it is broken but not separated.",
        "Shatter": "Broken glass or plastic components.",
        "Puncture": "A hole made by a sharp object.",
        "Burn": "Damage caused by fire or extreme heat.",
        "Water Damage": "Damage caused by exposure to water or flooding.",
        "Collision": "Damage from impact with another vehicle or object.",
        "Vandalism": "Intentional damage caused by a third party.",
        "Hail": "Damage caused by hail impacts."
    }

def get_car_manufacturers():
    return [
        "Audi", "BMW", "Chevrolet", "Citroën", "Dacia", "Fiat", "Ford", "Honda", 
        "Hyundai", "Jaguar", "Jeep", "Kia", "Land Rover", "Lexus", "Mazda", 
        "Mercedes-Benz", "Mitsubishi", "Nissan", "Opel", "Peugeot", "Porsche", 
        "Renault", "Seat", "Škoda", "Subaru", "Suzuki", "Tesla", "Toyota", 
        "Volkswagen", "Volvo"
    ]

def generate_report_id():
    return datetime.now().strftime("%Y%m%d%H%M%S") + hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]

def save_analysis(report_id, analysis_data):
    filename = os.path.join(ANALYSES_DIR, f"{report_id}.json")
    with open(filename, 'w') as f:
        json.dump(analysis_data, f)
    return filename

def load_analysis(report_id):
    filename = os.path.join(ANALYSES_DIR, f"{report_id}.json")
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def get_saved_analysis_reports():
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
    data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'raw_analysis': raw_analysis,
        'vehicle_type': 'Unknown',
        'damage_types': [],
        'fraud_likelihood': 'Low',
        'cost_estimate': 'Unknown',
        'severity': 'Medium'
    }
    clean_analysis = raw_analysis.replace('*', '').replace('#', '').replace('**', '')
    if "vehicle type:" in clean_analysis.lower():
        lines = clean_analysis.split('\n')
        for line in lines:
            if "vehicle type:" in line.lower() or line.strip().startswith(("This appears to be", "The vehicle is")):
                data['vehicle_type'] = line.strip()
                break
    else:
        first_lines = clean_analysis.split('\n')[:5]
        for line in first_lines:
            for make in get_car_manufacturers():
                if make.lower() in line.lower():
                    data['vehicle_type'] = line.strip()
                    break
    damage_types = load_damage_types()
    for damage_type in damage_types:
        if damage_type.lower() in clean_analysis.lower():
            data['damage_types'].append(damage_type)
    if not data['damage_types']:
        damage_indicators = ["damage", "damaged", "damages"]
        lines = clean_analysis.split('\n')
        for line in lines:
            if any(indicator in line.lower() for indicator in damage_indicators):
                words = line.split()
                for i, word in enumerate(words):
                    if any(indicator in word.lower() for indicator in damage_indicators) and i > 0:
                        potential_damage = words[i-1].strip('.,():;')
                        if len(potential_damage) > 3 and potential_damage.lower() not in ["the", "and", "has", "with", "this"]:
                            data['damage_types'].append(potential_damage.capitalize())
    if not data['damage_types']:
        data['damage_types'] = ["Damage"]
    if "fraudulent" in clean_analysis.lower():
        if "not fraudulent" in clean_analysis.lower() or "low likelihood of fraud" in clean_analysis.lower() or "no indication of fraud" in clean_analysis.lower():
            data['fraud_likelihood'] = 'Low'
        elif "possibly fraudulent" in clean_analysis.lower() or "medium likelihood" in clean_analysis.lower() or "potential fraud" in clean_analysis.lower():
            data['fraud_likelihood'] = 'Medium'
        elif "highly likely fraudulent" in clean_analysis.lower() or "high likelihood" in clean_analysis.lower() or "signs of fraud" in clean_analysis.lower():
            data['fraud_likelihood'] = 'High'
    import re
    currency_pattern = r'€\s*\d+[.,]?\d*|(?:eur|euro)s?\s*\d+[.,]?\d*|\d+[.,]?\d*\s*(?:eur|euro)s?'
    currency_matches = re.findall(currency_pattern, clean_analysis.lower())
    if currency_matches:
        data['cost_estimate'] = currency_matches[0]
    else:
        cost_lines = [line for line in clean_analysis.split('\n') 
                      if "€" in line or 
                      any(term in line.lower() for term in ["eur", "euro", "cost estimate", "repair cost", "estimate"])]
        if cost_lines:
            data['cost_estimate'] = cost_lines[0].strip()
    severity_words = {
        'minor': 0, 'light': 0, 'small': 0, 'moderate': 1, 'medium': 1, 'intermediate': 1,
        'significant': 2, 'substantial': 2, 'considerable': 2, 'severe': 3, 'extensive': 3,
        'major': 3, 'heavy': 3, 'critical': 3
    }
    severity_score = 0
    for word, score in severity_words.items():
        if word in clean_analysis.lower():
            severity_score = max(severity_score, score)
    severity_mapping = {0: 'Minor', 1: 'Moderate', 2: 'Significant', 3: 'Severe'}
    data['severity'] = severity_mapping.get(severity_score, 'Moderate')
    return data

def analyze_image(image_bytes, additional_instructions=""):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    prompt = """Analyze the damage and identify the different damage types. Mention the car type at the beginning of the analysis if you're sure about identifying it.
    
    Estimate if the image is fraudulent or not, and present the results in a well-formatted way.
    Do a fix cost estimation in Qatari Riyal  if possible, assuming this is for a Qatar-based insurance company.
    
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
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
        )
        analysis_text = response.choices[0].message.content
        structured_data = parse_analysis_result(analysis_text)
        return analysis_text, structured_data
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return f"Error: {str(e)}", None

def analyze_video(video_bytes, additional_instructions=""):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return "Unable to read the video file.", None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    num_frames = min(max(int(duration / 5), 3), 10)
    frame_indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    progress_bar = st.progress(0)
    status_text = st.empty()
    frames_to_analyze = []
    for i, frame_idx in enumerate(frame_indices):
        status_text.text(f"Extracting frame {i+1}/{num_frames}...")
        progress_bar.progress((i+1) / (2*num_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            frames_to_analyze.append(image_bytes)
    cap.release()
    os.unlink(temp_video_path)
    cols = st.columns(min(3, len(frames_to_analyze)))
    for idx, image_bytes in enumerate(frames_to_analyze):
        cols[idx % 3].image(image_bytes, caption=f'Frame {idx + 1}', use_container_width=True)
    analyses = []
    all_structured_data = []
    for idx, image_bytes in enumerate(frames_to_analyze):
        status_text.text(f"Analyzing frame {idx+1}/{num_frames}...")
        progress_bar.progress(0.5 + (idx+1) / (2*num_frames))
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
    all_damage_types = set()
    for data in all_structured_data:
        all_damage_types.update(data['damage_types'])
    aggregated_data['damage_types'] = list(all_damage_types)
    fraud_levels = {'Low': 0, 'Medium': 1, 'High': 2}
    highest_fraud = 0
    for data in all_structured_data:
        fraud_level = fraud_levels.get(data['fraud_likelihood'], 0)
        highest_fraud = max(highest_fraud, fraud_level)
    reverse_fraud_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
    aggregated_data['fraud_likelihood'] = reverse_fraud_levels[highest_fraud]
    severity_levels = {'Minor': 0, 'Moderate': 1, 'Significant': 2, 'Severe': 3}
    highest_severity = 0
    for data in all_structured_data:
        severity_level = severity_levels.get(data['severity'], 1)
        highest_severity = max(highest_severity, severity_level)
    reverse_severity_levels = {0: 'Minor', 1: 'Moderate', 2: 'Significant', 3: 'Severe'}
    aggregated_data['severity'] = reverse_severity_levels[highest_severity]
    return "\n".join(analyses), aggregated_data

# New function for multi-vehicle analysis
def analyze_images(image_bytes_list, additional_instructions=""):
    base64_images = [base64.b64encode(img).decode('utf-8') for img in image_bytes_list]
    prompt = """As a police officer investigating a multi-vehicle incident, analyze the damage shown in these images of the same vehicle. Identify the different damage types and provide a comprehensive assessment.

    Mention the car type if identifiable.

    Be specific about:
    1. Types of damage (dent, scratch, broken parts, etc.)
    2. Severity of damage (minor, moderate, severe)
    3. Affected areas of the vehicle
    4. Direction or angle of impact if discernible
    5. Any evidence suggesting the sequence of events or fault

    Since there are multiple images, consider all of them to provide an overall assessment of the vehicle's damage relevant to a police investigation."""
    if additional_instructions:
        prompt += f"\n\nAdditional instructions: {additional_instructions}"
    content = [
        {"type": "text", "text": prompt}
    ] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} 
        for base64_image in base64_images
    ]
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0.7,
            top_p=0.95,
        )
        analysis_text = response.choices[0].message.content
        structured_data = parse_analysis_result(analysis_text)
        return analysis_text, structured_data
    except Exception as e:
        st.error(f"Error analyzing images: {str(e)}")
        return f"Error: {str(e)}", None

def display_analysis_dashboard(analysis_data):
    if not analysis_data:
        st.warning("No analysis data available to display.")
        return
    for key in analysis_data:
        if isinstance( analysis_data[key], str):
            analysis_data[key] = analysis_data[key].replace('*', '').replace('#', '')
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="analysis-header" style="font-size: 1.1rem;">
        Analysis Summary Report #{analysis_data.get('report_id', 'Unknown')}
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    def clean_text(text):
        if not text:
            return "Unknown"
        cleaned = text.replace('*', '').replace('#', '').replace(':', ' ').strip()
        if len(cleaned) > 15:
            parts = cleaned.split()
            if len(parts) > 2:
                cleaned = ' '.join(parts[-2:])
        return cleaned
    vehicle_type = analysis_data.get('vehicle_type', 'Unknown')
    if isinstance(vehicle_type, str):
        if ':' in vehicle_type:
            vehicle_type = vehicle_type.split(':')[-1].strip()
        vehicle_type = clean_text(vehicle_type)
        if len(vehicle_type) > 10:
            vehicle_type = vehicle_type[:10]
    cost_estimate = analysis_data.get('cost_estimate', 'Unknown')
    if isinstance(cost_estimate, str):
        if 'Cost Estimate:' in cost_estimate:
            cost_estimate = cost_estimate.replace('Cost Estimate:', '').strip()
        cost_estimate = clean_text(cost_estimate)
        if '€' in cost_estimate or 'EUR' in cost_estimate.upper():
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
    st.markdown("### Damage Types Identified")
    damage_types = analysis_data.get('damage_types', [])
    if damage_types:
        df_damage = pd.DataFrame({
            'Damage Type': damage_types,
            'Count': [1] * len(damage_types)
        })
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
    if 'frame_analyses' in analysis_data and analysis_data['frame_analyses']:
        st.markdown("### Frame-by-Frame Analysis")
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
    with st.expander("View Full Analysis Details", expanded=False):
        st.markdown("### Complete Analysis Report")
        st.markdown(analysis_data.get('raw_analysis', 'No detailed analysis available.'))

def display_confirmation(report_id):
    st.markdown(f"""
    <div class="success-box">
        <h3>✅ Analysis Completed Successfully!</h3>
        <p>Your damage analysis report has been created with Report ID: <strong>{report_id}</strong></p>
        <p>You can reference this ID to retrieve this analysis in the future.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.image("logo.jpg", caption=None)
        st.markdown("<h2 style='color: white;'>Smart Damage Analyzer Pro</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #CCC;'>AI-powered insurance damage assessment tool</p>", unsafe_allow_html=True)
        st.markdown("---")
        app_mode = st.radio("Navigation", 
                            options=["New Analysis", "Past Reports", "Settings"],
                            index=0)
        st.markdown("---")
        st.markdown("<p style='color: #CCC; font-size: 0.8rem;'>© 2025 Smart Solutions</p>", unsafe_allow_html=True)
    
    if app_mode == "New Analysis":
        st.markdown("""
        <div class="main-header">
            <h1>Smart Damage Analyzer Pro</h1>
            <p>Upload images or videos of vehicle damage for AI-powered analysis and cost estimation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Updated tabs to include Multi-Vehicle Analysis
        tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis", "🚗 Multi-Vehicle Analysis"])
        
        with tab1:
            st.markdown("""
            <div class="card">
                <div class="card-header">Image Upload</div>
                <p>Upload a clear photo of the vehicle damage for instant analysis.</p>
            </div>
            """, unsafe_allow_html=True)
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
            analyze_btn = st.button("Analyze Damage", key="analyze_image_btn", use_container_width=True)
            if uploaded_file is not None:
                st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
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
                            report_id = generate_report_id()
                            analysis_data['report_id'] = report_id
                            save_analysis(report_id, analysis_data)
                            display_confirmation(report_id)
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
                        report_id = generate_report_id()
                        analysis_data['report_id'] = report_id
                        save_analysis(report_id, analysis_data)
                        display_confirmation(report_id)
                        display_analysis_dashboard(analysis_data)
                    else:
                        st.error("Video analysis failed. Please try again with a different video.")
        
        with tab3:
            st.markdown("""
            <div class="card">
                <div class="card-header">Multi-Vehicle Analysis</div>
                <p>Upload images or videos from multiple vehicles involved in an incident. The AI will analyze the damage and cross-reference patterns to assist in fault determination as a police officer.</p>
            </div>
            """, unsafe_allow_html=True)
            num_vehicles = st.number_input("Number of vehicles involved", min_value=2, max_value=5, value=2, step=1)
            for i in range(num_vehicles):
                st.markdown(f"### Vehicle {i+1}")
                vehicle_id = st.text_input(f"Vehicle Identifier (e.g., License Plate, Make/Model)", key=f"vehicle_id_{i}")
                media_type = st.selectbox(f"Media Type", ["Images", "Video"], key=f"media_type_{i}")
                if media_type == "Images":
                    uploaded_images = st.file_uploader(f"Upload images for Vehicle {i+1}", 
                                                       type=["jpg", "jpeg", "png"], 
                                                       accept_multiple_files=True, 
                                                       key=f"images_{i}")
                elif media_type == "Video":
                    uploaded_video = st.file_uploader(f"Upload video for Vehicle {i+1}", 
                                                      type=["mp4", "avi", "mov"], 
                                                      key=f"video_{i}")
            analyze_multi_btn = st.button("Analyze Multi-Vehicle Incident", key="analyze_multi_btn")
            if analyze_multi_btn:
                vehicle_analyses = []
                for i in range(num_vehicles):
                    vehicle_id = st.session_state.get(f"vehicle_id_{i}", f"Vehicle {i+1}")
                    media_type = st.session_state.get(f"media_type_{i}", "Images")
                    if media_type == "Images":
                        uploaded_images = st.session_state.get(f"images_{i}", [])
                        if uploaded_images:
                            image_bytes_list = [img.read() for img in uploaded_images]
                            analysis_text, analysis_data = analyze_images(image_bytes_list)
                            if analysis_data:
                                analysis_data['vehicle_id'] = vehicle_id
                                vehicle_analyses.append(analysis_data)
                        else:
                            st.error(f"No images uploaded for Vehicle {i+1}")
                            continue
                    elif media_type == "Video":
                        uploaded_video = st.session_state.get(f"video_{i}", None)
                        if uploaded_video:
                            video_bytes = uploaded_video.read()
                            analysis_text, analysis_data = analyze_video(video_bytes)
                            if analysis_data:
                                analysis_data['vehicle_id'] = vehicle_id
                                vehicle_analyses.append(analysis_data)
                        else:
                            st.error(f"No video uploaded for Vehicle {i+1}")
                            continue
                if len(vehicle_analyses) >= 2:
                    with st.spinner("Analyzing multi-vehicle incident... This may take several minutes."):
                        comparison_prompt = """As a Qatari police officer investigating a multi-vehicle collision, analyze the following damage reports from vehicles involved in the same incident to determine fault:

                        Based on the damage patterns, please:
                        1. Identify which vehicle likely initiated the impact
                        2. Assess the sequence of events
                        3. Highlight any evidence supporting your fault determination
                        4. Note any inconsistencies or suspicious patterns that may require further investigation

                        Provide a detailed explanation of your reasoning. Make sure to follow the Qatari traffic laws and regulations."""
                        for i, data in enumerate(vehicle_analyses):
                            comparison_prompt += f"\n\n**Vehicle {i+1} ({data['vehicle_id']}):**\n{data['raw_analysis']}"
                        try:
                            comparison_response = client.chat.completions.create(
                                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                                messages=[{"role": "user", "content": comparison_prompt}],
                                max_tokens=4096,
                                temperature=0.7,
                                top_p=0.95,
                            )
                            comparison_text = comparison_response.choices[0].message.content
                            report_id = generate_report_id()
                            multi_vehicle_data = {
                                'report_id': report_id,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'vehicle_analyses': vehicle_analyses,
                                'comparison_analysis': comparison_text
                            }
                            save_analysis(report_id, multi_vehicle_data)
                            display_confirmation(report_id)
                            st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                            st.subheader("Individual Vehicle Analyses")
                            for i, data in enumerate(vehicle_analyses):
                                st.markdown(f"### Vehicle {i+1}: {data['vehicle_id']}")
                                display_analysis_dashboard(data)
                            st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                            st.subheader("Fault Determination Analysis")
                            st.markdown(comparison_text)
                        except Exception as e:
                            st.error(f"Error performing comparison analysis: {str(e)}")
                else:
                    st.warning("At least two vehicles with valid media are required for multi-vehicle analysis.")
    
    elif app_mode == "Past Reports":
        st.markdown("""
        <div class="main-header">
            <h1>Analysis History</h1>
            <p>View and manage your previous damage analysis reports</p>
        </div>
        """, unsafe_allow_html=True)
        reports = get_saved_analysis_reports()
        if not reports:
            st.markdown("""
            <div class="warning-box">
                <h3>No Analysis Reports Found</h3>
                <p>You haven't performed any damage analyses yet. Go to the New Analysis page to create your first report.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            df_reports = pd.DataFrame(reports)
            st.dataframe(df_reports, use_container_width=True)
            selected_report = st.selectbox("Select a report to view details:", 
                                           [f"Report #{r['id']} - {r['date']} - {r['vehicle']}" for r in reports])
            if selected_report:
                report_id = selected_report.split(" - ")[0].replace("Report #", "")
                report_data = load_analysis(report_id)
                if report_data:
                    if 'vehicle_analyses' in report_data:
                        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                        st.subheader("Individual Vehicle Analyses")
                        for i, data in enumerate(report_data['vehicle_analyses']):
                            st.markdown(f"### Vehicle {i+1}: {data['vehicle_id']}")
                            display_analysis_dashboard(data)
                        st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
                        st.subheader("Fault Determination Analysis")
                        st.markdown(report_data.get('comparison_analysis', 'No comparison analysis available.'))
                    else:
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
        st.button("Save Settings", use_container_width=True)
        with st.expander("Reset to Default Settings"):
            st.warning("This will reset all settings to their default values. This action cannot be undone.")
            st.button("Reset All Settings", type="primary")

if __name__ == "__main__":
    main()