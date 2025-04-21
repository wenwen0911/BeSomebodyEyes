import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pyttsx3
import translators as ts
import tempfile
import os

# Set up the page
st.set_page_config(page_title="Scene Describer", page_icon="👁️", layout="wide")

# Initialize device (GPU/CPU)
@st.cache_resource
def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

device = setup_device()

# Load models with caching
@st.cache_resource
def load_models():
    # Load YOLO models
    model_yolo1 = YOLO("best.pt")  # Your custom model
    model_yolo2 = YOLO('yolov8m.pt')  # Standard YOLO model
    model_yolo2.to(device)
    
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    
    # Load transforms
    transforms = torch.hub.load('intel-isl/MiDaS','transforms')
    transform = transforms.small_transform
    
    return model_yolo1, model_yolo2, midas, transform

model_yolo1, model_yolo2, midas, transform = load_models()

# Initialize TTS engine
@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    return engine

engine = init_tts()

# Helper functions
def get_position(center, image_width):
    if center < image_width * 0.4:          #left
        return "left"
    if center > image_width * 0.6:           #right
        return "right"
    else: 
        return "center"

def get_english_voice(engine):
    """Get an English voice"""
    voices = engine.getProperty("voices")
    for voice in voices:
        if "en-US" in voice.languages:
            return voice
        if "English" in voice.name or "English" in voice.name.title():
            return voice
    return voices[0]  # Return default voice if no English found

def get_chinese_voice(engine):
    """Get a Chinese voice"""
    voices = engine.getProperty("voices")
    for voice in voices:
        if "zh-CN" in voice.languages:
            return voice
        if "Chinese" in voice.name or "Mandarin" in voice.name.title():
            return voice
    return voices[0]  # Return default voice if no Chinese found

def tts(text, gender="male", speed="normal"):
    """Text-to-speech with voice and speed options"""
    engine = init_tts()
    
    # Set voice
    if gender == "male":
        for voice in engine.getProperty("voices"):
            if "male" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
    else:  # female
        for voice in engine.getProperty("voices"):
            if "female" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break
    
    # Set speed
    rate = engine.getProperty("rate")
    if speed == "slow":
        engine.setProperty("rate", rate * 0.7)
    else:
        engine.setProperty("rate", rate)
    
    engine.say(text)
    engine.runAndWait()

def process_image(image_file, scale=1.0, camera_to_person=0.1):
    """Process the uploaded image and return description, image, and depth map"""
    # Read image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        st.error("Error: Could not read the image file.")
        return None, None, None
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    # Depth estimation
    with torch.inference_mode():
        depth_pred = midas(input_batch)
        depth_pred = torch.nn.functional.interpolate(
            depth_pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalize Depth Map
    depth_map = depth_pred.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Run object detection
    results1 = model_yolo1(img_rgb, save=False, conf=0.2)[0]
    results2 = model_yolo2(img_rgb, save=False, conf=0.5)[0]

    # Process detected objects
    description = []
    
    # Process results from both models
    for result, model in [(results1, model_yolo1), (results2, model_yolo2)]:
        for box in result.boxes:
            # Skip objects with low confidence
            if box.conf < 0.1:
                continue
                
            # Extract bounding box coordinates
            left, top, right, bottom = box.xyxy[0].tolist()
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            # Extract depth value of object
            object_depth_value = depth_map[top:bottom, left:right]
            average_depth = np.median(object_depth_value)
            
            # Convert inverse depth value to distance
            distance = scale / (average_depth + 1e-6)  # Avoid division by zero
            total_distance = camera_to_person + distance

            # Find horizontal position of object
            center = (left + right) / 2
            image_width = img.shape[1]
            position = get_position(center, image_width)

            # Get object name
            object_name = model.names[int(box.cls)]
            
            # Add to description
            description.append(f"There is a {object_name} at your {position}, {total_distance:.2f} meters away.")

    return description, img_rgb, depth_map

def display_results(description, img, depth_map):
    """Display results in Streamlit"""
    # Display the original image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Display the description
    st.subheader("Scene Description")
    for desc in description:
        st.write(f"- {desc}")
    
    # Display depth map
    depth_map_display = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colormap = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)
    st.image(depth_colormap, caption="Depth Map", use_column_width=True)

# Streamlit UI
def main():
    st.title("👁️ Scene Describer for Visually Impaired")
    st.markdown("""
    This application helps visually impaired users understand their surroundings by:
    - Detecting objects in an image
    - Estimating their distance and position
    - Providing both text and audio descriptions
    """)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        scale = st.slider("Depth Scale Factor", 0.1, 10.0, 1.0, 0.1)
        camera_to_person = st.slider("Camera to Person Distance (meters)", 0.0, 2.0, 0.1, 0.1)
        
        st.header("Audio Settings")
        language = st.radio("Language", ["English", "Chinese"])
        if language == "English":
            voice_type = st.radio("Voice Type", ["Male (Normal)", "Male (Slow)", "Female (Normal)", "Female (Slow)"])
        else:
            voice_type = "Chinese"
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            # Process the image
            description, img, depth_map = process_image(uploaded_file, scale, camera_to_person)
            
            if description:
                # Display results
                display_results(description, img, depth_map)
                
                # Prepare text description
                full_description = ' '.join(description)
                
                # Audio output section
                st.subheader("Audio Description")
                st.write("Click the button below to hear the description")
                
                # Create a temporary audio file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    temp_path = tmpfile.name
                
                # Generate and play audio
                if st.button("Play Audio Description"):
                    with st.spinner("Generating audio..."):
                        # Set voice properties based on selection
                        if language == "Chinese":
                            chinese_text = ts.translate_text(full_description, to_language='zh')
                            chinese_voice = get_chinese_voice(engine)
                            engine.setProperty("voice", chinese_voice.id)
                            engine.save_to_file(chinese_text, temp_path)
                        else:
                            english_voice = get_english_voice(engine)
                            engine.setProperty("voice", english_voice.id)
                            
                            # Set speed
                            rate = engine.getProperty("rate")
                            if "Slow" in voice_type:
                                engine.setProperty("rate", rate * 0.7)
                            
                            engine.save_to_file(full_description, temp_path)
                        
                        engine.runAndWait()
                        
                        # Play the audio file
                        st.audio(temp_path, format="audio/wav")
                    
                    # Clean up
                    os.unlink(temp_path)

if __name__ == "__main__":
    main()
