import streamlit as st
from openai import AzureOpenAI
import base64
import cv2
import tempfile
import io
from PIL import Image
import config

# Configure Azure OpenAI client
AZURE_OPENAI_ENDPOINT = config.endpoint
AZURE_OPENAI_DEPLOYMENT_NAME = config.deployment
AZURE_OPENAI_KEY = config.subscription_key

if not AZURE_OPENAI_KEY:
    raise ValueError("Azure OpenAI API key not found in environment variables. Please set AZURE_OPENAI_KEY.")


client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2023-12-01-preview",
)

def analyze_image(image_bytes):
    """Analyzes an image using Azure OpenAI Vision."""

    # Convert bytes to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    content = [
        {
            "type": "text",
            "text": """Analyze the damage and identify the different damage types. Mention the car type at the begining of the analysis if youre sure about knowing it Estimate if the image is fraudulent or not, and present the results in a well-formatted way. Do a fix cost estimation if possible, assuing this is for a Netherlands based country"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    ]

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=4096,
        temperature=1,
        top_p=0.95,
    )

    return response.choices[0].message.content

def analyze_video(video_bytes):
    """Analyzes a video by extracting frames and analyzing them."""

    # Save video bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    # Use OpenCV to read the video
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return "Unable to read the video file."

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select frames to analyze (e.g., every nth frame)
    frames_to_analyze = []
    nth_frame = max(frame_count // 10, 1)  # Adjust as needed
    for i in range(0, frame_count, nth_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert frame to JPEG bytes
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            frames_to_analyze.append(image_bytes)
    cap.release()

    # Analyze each selected frame
    analyses = []
    for idx, image_bytes in enumerate(frames_to_analyze):
        st.image(image_bytes, caption=f'Analyzed Frame {idx + 1}')
        analysis = analyze_image(image_bytes)
        analyses.append(f"**Frame {idx + 1} Analysis:**\n{analysis}\n")

    return "\n".join(analyses)

def main():
    st.title("Insurance Damage Analyzer")

    input_type = st.selectbox("Select input type:", ("Image", "Video"))

    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image of the damaged item", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption='Uploaded Image')

            with st.spinner('Analyzing the image...'):
                analysis = analyze_image(image_bytes)

            st.subheader("Analysis Results:")
            st.write(analysis)

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Upload a video of the damaged item", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            video_bytes = uploaded_file.read()
            st.video(video_bytes)

            with st.spinner('Analyzing the video...'):
                analysis = analyze_video(video_bytes)

            st.subheader("Analysis Results:")
            st.write(analysis)

if __name__ == "__main__":
    main()