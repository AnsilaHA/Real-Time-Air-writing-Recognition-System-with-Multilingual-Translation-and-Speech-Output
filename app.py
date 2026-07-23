import sys


# Silence Windows-specific asyncio proactor event loop ConnectionResetError (WinError 10054) console spam
if sys.platform == 'win32':
    import asyncio
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost
        
        def _patched_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, OSError) as e:
                if isinstance(e, ConnectionResetError) or getattr(e, 'winerror', None) == 10054:
                    if self._sock is not None:
                        try:
                            self._sock.close()
                        except Exception:
                            pass
                else:
                    raise

        _ProactorBasePipeTransport._call_connection_lost = _patched_call_connection_lost
    except Exception:
        pass

import streamlit as st
import cv2
import numpy as np
import os
import time
import json
import tensorflow as tf
import threading
import queue
import atexit


def cleanup_webcam():
    if "webcam_manager" in st.session_state:
        try:
            st.session_state.webcam_manager.stop()
        except Exception:
            pass

atexit.register(cleanup_webcam)

# Configure Streamlit page layout and theme for high-fidelity SaaS presentation
st.set_page_config(
    page_title="Air-Writing AI Studio",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Append src directory for modular python imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import existing backend modules and helper functions
from hand_tracker import HandTracker
from trajectory_manager import TrajectoryManager
from preprocess import preprocess_single_trajectory
from sentence_builder import SentenceBuilder
from translation import TranslationEngine
from speech import SpeechGenerator
from predict import (
    apply_temperature_scaling,
    detect_hand_gesture,
    validate_trajectory,
    extract_geometric_features,
    validate_character,
    calculate_trajectory_orientation,
    CLASS_THRESHOLDS
)

# ----------------------------------------------------
# 1. Caching Resource Functions (Prevent Model Reloading)
# ----------------------------------------------------
@st.cache_resource
def load_recognition_model(model_path: str, mapping_path: str):
    """
    Loads Keras model and label mapping configs exactly once.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Label mapping not found at: {mapping_path}")
        
    model = tf.keras.models.load_model(model_path)
    with open(mapping_path, "r") as f:
        mapping_config = json.load(f)
        
    classes = mapping_config["classes"]
    idx_to_label = {int(k): v for k, v in mapping_config["idx_to_label"].items()}
    return model, classes, idx_to_label

@st.cache_resource
def load_completeness_stats(stats_path: str):
    """
    Loads character completeness stats exactly once.
    """
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return json.load(f)
    return None

# Load configurations
MODEL_PATH = "models/best_model.keras"
MAPPING_PATH = os.path.join("data", "processed", "label_mapping.json")
STATS_PATH = os.path.join("data", "processed", "completeness_stats.json")

try:
    model, classes, idx_to_label = load_recognition_model(MODEL_PATH, MAPPING_PATH)
    completeness_stats = load_completeness_stats(STATS_PATH)
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load AI system models: {e}. Ensure models are trained first.")
    model_loaded = False

# ----------------------------------------------------
# 2. Initialize Session State variables
# ----------------------------------------------------
if "tracker" not in st.session_state:
    st.session_state.tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
if "trajectory" not in st.session_state:
    st.session_state.trajectory = TrajectoryManager()
if "translation_engine" not in st.session_state:
    st.session_state.translation_engine = TranslationEngine()
if "speech_generator" not in st.session_state:
    st.session_state.speech_generator = SpeechGenerator(autoplay=True)
if "sentence_builder" not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()

# Persistent dynamic UI states
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "English"
if "latest_nlp_result" not in st.session_state:
    st.session_state.latest_nlp_result = None
if "latest_translation_result" not in st.session_state:
    st.session_state.latest_translation_result = None
if "latest_speech_result" not in st.session_state:
    st.session_state.latest_speech_result = None
if "latest_pred_char" not in st.session_state:
    st.session_state.latest_pred_char = ""
if "latest_pred_conf" not in st.session_state:
    st.session_state.latest_pred_conf = 0.0
if "latest_pred_status" not in st.session_state:
    st.session_state.latest_pred_status = "Unknown"
if "validation_status" not in st.session_state:
    st.session_state.validation_status = ""
if "fps" not in st.session_state:
    st.session_state.fps = 0
if "recognition_time" not in st.session_state:
    st.session_state.recognition_time = 0.0
if "system_status" not in st.session_state:
    st.session_state.system_status = "Ready"
if "recorded_points" not in st.session_state:
    st.session_state.recorded_points = []
if "gesture_state" not in st.session_state:
    st.session_state.gesture_state = "idle"
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# ----------------------------------------------------
# 3. Audio Control Callbacks
# ----------------------------------------------------
def speech_play():
    if st.session_state.latest_speech_result and st.session_state.latest_speech_result.get("success"):
        st.session_state.speech_generator.play_audio()
        st.session_state.system_status = "Speaking"

def speech_pause():
    st.session_state.speech_generator.pause_audio()

def speech_stop():
    st.session_state.speech_generator.stop_audio()
    st.session_state.system_status = "Ready"

def speech_replay():
    res = st.session_state.speech_generator.replay_audio()
    if res:
        st.session_state.system_status = "Speaking"

def clear_all_inputs():
    if "webcam_manager" in st.session_state:
        st.session_state.webcam_manager.clear_all()
    else:
        st.session_state.trajectory.clear()
        st.session_state.recorded_points = []
    st.session_state.sentence_builder.clear()
    st.session_state.latest_nlp_result = None
    st.session_state.latest_translation_result = None
    st.session_state.latest_speech_result = None
    st.session_state.latest_pred_char = ""
    st.session_state.latest_pred_conf = 0.0
    st.session_state.latest_pred_status = "Unknown"
    st.session_state.validation_status = ""
    st.session_state.system_status = "Ready"
    st.session_state.gesture_state = "idle"

def delete_last_callback():
    """Remove only the last character from the current word and its canvas stroke."""
    # Only act if there is a character in the current word
    if st.session_state.sentence_builder.get_current_word():
        st.session_state.sentence_builder.delete_last_character()
        if "webcam_manager" in st.session_state:
            st.session_state.webcam_manager.delete_last_character_trajectory()
        else:
            st.session_state.trajectory.delete_last_character_trajectory()
        # Reset prediction display — the last accepted char is gone
        st.session_state.latest_pred_char = ""
        st.session_state.latest_pred_conf = 0.0
        st.session_state.latest_pred_status = "Unknown"
        st.session_state.validation_status = ""

def clear_word_callback():
    """Remove the entire current word and all its canvas strokes."""
    st.session_state.sentence_builder.clear_current_word()
    if "webcam_manager" in st.session_state:
        st.session_state.webcam_manager.clear_all()
    else:
        st.session_state.trajectory.clear()
        st.session_state.recorded_points = []
    # Move gesture state back to idle so the next index raise starts fresh
    st.session_state.gesture_state = "waiting_for_index"
    # Reset prediction display
    st.session_state.latest_pred_char = ""
    st.session_state.latest_pred_conf = 0.0
    st.session_state.latest_pred_status = "Unknown"
    st.session_state.validation_status = ""

def delete_last_word_callback():
    """Remove only the last completed word from the current sentence."""
    if st.session_state.sentence_builder.get_current_sentence():
        st.session_state.sentence_builder.delete_last_word()

# ----------------------------------------------------
# 4. Inject Premium Dark Theme Styles (CSS)
# ----------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* Global Font Settings & Body styling */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* Hide default Streamlit components */
header { visibility: hidden !important; }
footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

.stApp {
    background-color: #0F172A !important;
    background-image: radial-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 0);
    background-size: 32px 32px;
    color: #F8FAFC !important;
}

/* Full Bleed Viewport Container Constraints for Single Screen */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 95% !important;
}

/* Glassmorphism Card Panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(30, 41, 59, 0.6) !important;
    backdrop-filter: blur(16px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 14px !important;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.3s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 8px 32px 0 rgba(59, 130, 246, 0.15) !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
    transform: translateY(-2px);
}

/* Header style */
.app-header {
    text-align: center;
    margin-bottom: 25px;
    margin-top: 5px;
}
.app-title {
    font-size: 32px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: 1.5px;
    margin: 0;
    padding: 0;
    text-transform: uppercase;
    background: linear-gradient(135deg, #F8FAFC 40%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-subtitle {
    font-size: 13px;
    font-weight: 400;
    color: #94A3B8;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Card Header elements */
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 8px;
}
.header-icon {
    font-size: 20px;
    color: #3B82F6; /* Primary Accent */
}
.card-title {
    font-weight: 600;
    font-size: 14px;
    color: #F8FAFC;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Sub-labels & Values */
.label-text {
    font-size: 11px;
    color: #94A3B8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}

/* Specific elements inside Recognition Card */
.recognition-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.char-val {
    font-size: 38px;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1;
}
.confidence-box {
    margin-top: 10px;
}
.progress-bar-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 9px;
    height: 8px;
    overflow: hidden;
    margin-top: 6px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.progress-bar-fill {
    background: linear-gradient(90deg, #1E40AF, #3B82F6);
    height: 100%;
    border-radius: 9px;
    transition: width 0.4s ease;
}
.status-badge {
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid transparent;
}
.status-success {
    background: rgba(34, 197, 94, 0.15);
    color: #22C55E;
    border-color: rgba(34, 197, 94, 0.3);
}
.status-waiting {
    background: rgba(148, 163, 184, 0.15);
    color: #94A3B8;
    border-color: rgba(148, 163, 184, 0.3);
}

/* Live Text elements */
.text-group {
    margin-bottom: 8px;
}
.word-val {
    font-size: 24px;
    font-weight: 700;
    color: #3B82F6;
}
.sentence-val {
    font-size: 16px;
    font-weight: 500;
    color: #F8FAFC;
}
.card-divider {
    height: 1px;
    background-color: rgba(255, 255, 255, 0.08);
    margin: 12px 0;
}

/* Live Text Action Toolbar styles */
.live-text-toolbar-marker {
    display: none;
}

/* Base style for all action buttons in the Live Text toolbar */
.st-key-delete_last_btn button,
.st-key-clear_word_btn button,
.st-key-delete_last_word_btn button,
.st-key-reset_canvas_btn button {
    border-radius: 12px !important; /* Rounded corners 10-12px */
    font-size: 13px !important;
    font-weight: 600 !important;
    height: 40px !important;
    font-family: 'Poppins', sans-serif !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
}

/* Delete Last button & Delete Last Word button: Secondary/Blue */
.st-key-delete_last_btn button,
.st-key-delete_last_word_btn button {
    background-color: rgba(59, 130, 246, 0.1) !important;
    color: #3B82F6 !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
}
.st-key-delete_last_btn button:hover:not(:disabled),
.st-key-delete_last_word_btn button:hover:not(:disabled) {
    background-color: rgba(59, 130, 246, 0.2) !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.3) !important;
    transform: translateY(-1px) !important;
}
.st-key-delete_last_btn button:active:not(:disabled),
.st-key-delete_last_word_btn button:active:not(:disabled) {
    transform: translateY(0px) !important;
    background-color: rgba(59, 130, 246, 0.25) !important;
}

/* Clear Word button: Orange/Amber */
.st-key-clear_word_btn button {
    background-color: rgba(245, 158, 11, 0.1) !important;
    color: #F59E0B !important;
    border: 1px solid rgba(245, 158, 11, 0.4) !important;
}
.st-key-clear_word_btn button:hover:not(:disabled) {
    background-color: rgba(245, 158, 11, 0.2) !important;
    border-color: #F59E0B !important;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.3) !important;
    transform: translateY(-1px) !important;
}
.st-key-clear_word_btn button:active:not(:disabled) {
    transform: translateY(0px) !important;
    background-color: rgba(245, 158, 11, 0.25) !important;
}

/* Reset button: Red/Destructive */
.st-key-reset_canvas_btn button {
    background-color: rgba(239, 68, 68, 0.1) !important;
    color: #EF4444 !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
}
.st-key-reset_canvas_btn button:hover:not(:disabled) {
    background-color: rgba(239, 68, 68, 0.2) !important;
    border-color: #EF4444 !important;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.3) !important;
    transform: translateY(-1px) !important;
}
.st-key-reset_canvas_btn button:active:not(:disabled) {
    transform: translateY(0px) !important;
    background-color: rgba(239, 68, 68, 0.25) !important;
}

/* Disabled styling for the action buttons */
.st-key-delete_last_btn button:disabled,
.st-key-clear_word_btn button:disabled,
.st-key-delete_last_word_btn button:disabled,
.st-key-reset_canvas_btn button:disabled {
    background-color: rgba(255, 255, 255, 0.02) !important;
    color: rgba(148, 163, 184, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
    transform: none !important;
}

/* AI Enhanced text */
.nlp-val {
    font-size: 18px;
    font-weight: 600;
    color: #22C55E; /* Success accent */
}

/* Translation card output */
.translation-output {
    font-size: 20px;
    font-weight: 700;
    color: #F59E0B; /* Warning/Amber Accent */
    margin-top: 10px;
    min-height: 30px;
}

/* Speech status text */
.speech-status {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 600;
    margin-top: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.speech-status-active {
    color: #22C55E;
}

/* Webcam glowing border & aspect ratio */
.webcam-placeholder-card {
    border: 1px solid rgba(59, 130, 246, 0.3);
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
    border-radius: 18px;
    background: #1E293B;
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    aspect-ratio: 16 / 9; /* Enforced responsive 16:9 aspect ratio */
    margin-top: 10px;
    overflow: hidden;
}

/* Force Streamlit's element container and full screen frame wrappers to fill the entire column width */
div[data-testid="stFullScreenFrame"] {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
}

div[data-testid="stFullScreenFrame"] > div,
div.e1plw2qp2:has(div.stImage),
div.e1plw2qp2:has(div[data-testid="stImage"]),
div.element-container:has(div.stImage),
div.element-container:has(div[data-testid="stImage"]) {
    width: 100% !important;
    max-width: 100% !important;
    height: auto !important;
}

[data-testid="stImage"], div.stImage {
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.2) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    background: #1E293B !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    aspect-ratio: 16 / 9 !important; /* Fixed aspect ratio matching placeholder */
    margin-top: 10px !important;
    animation: blue-glow-pulse 3s infinite ease-in-out !important;
}

/* Force intermediate divs/wrappers inside the image widget to occupy full width and height */
div[data-testid="stImageContainer"],
[data-testid="stImage"] div, 
[data-testid="stImage"] > div,
div.stImage div {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

[data-testid="stImage"] img, div.stImage img {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    object-fit: cover !important; /* Scale and crop excess areas dynamically */
    border-radius: 0px !important;
}

@keyframes blue-glow-pulse {
    0% {
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.6);
    }
    50% {
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.6);
        border-color: rgba(59, 130, 246, 1);
    }
    100% {
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        border-color: rgba(59, 130, 246, 0.6);
    }
}

.webcam-placeholder-content {
    text-align: center;
    color: #94A3B8;
}
.webcam-placeholder-icon {
    font-size: 48px;
    margin-bottom: 12px;
    color: #3B82F6;
    opacity: 0.8;
}

/* History Card */
.history-list {
    margin-top: 10px;
}
.history-item {
    font-size: 14px;
    font-weight: 500;
    color: #F8FAFC;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.03);
    margin-bottom: 8px;
}
.history-empty {
    font-size: 13px;
    color: #94A3B8;
    text-align: center;
    padding: 15px 0;
}

/* General Placeholder style */
.empty-placeholder {
    font-size: 14px;
    color: #94A3B8;
    font-style: italic;
}

/* Premium Rounded Outline Buttons styling */
div.stButton > button {
    background-color: transparent !important;
    color: #3B82F6 !important;
    border: 1px solid rgba(59, 130, 246, 0.6) !important;
    border-radius: 18px !important;
    padding: 6px 12px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    height: 38px !important;
    box-shadow: none !important;
}
div.stButton > button:hover {
    background-color: rgba(59, 130, 246, 0.08) !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 0 12px rgba(59, 130, 246, 0.3) !important;
}
div.stButton > button:active {
    background-color: rgba(59, 130, 246, 0.16) !important;
}

/* Dropdown styling */
div[data-testid="stSelectbox"] > div {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    color: #F8FAFC !important;
}
div[data-testid="stSelectbox"] div[role="combobox"] {
    background-color: transparent !important;
    border: none !important;
    color: #F8FAFC !important;
}
div[data-testid="stSelectbox"] span, div[data-testid="stSelectbox"] p {
    color: #F8FAFC !important;
}

/* Blinking terminal cursor */
.terminal-cursor {
    display: inline-block;
    width: 2px;
    height: 15px;
    background-color: #3B82F6;
    margin-left: 4px;
    vertical-align: middle;
    animation: blink 1s step-end infinite;
}
@keyframes blink {
    from, to { background-color: transparent }
    50% { background-color: #3B82F6 }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 5. Helper Rendering Functions (UI HTML components)
# ----------------------------------------------------
def get_prediction_card_html(char, conf, status):
    if status == "Accepted":
        char_val = char if char else "Waiting for input..."
        conf_percent = int(conf * 100)
        progress_html = f"""
        <div class="confidence-box">
            <div class="label-text">Confidence</div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {conf_percent}%;"></div>
            </div>
            <div style="font-size: 11px; text-align: right; color: #94A3B8; margin-top: 4px; font-weight: 500;">{conf_percent}%</div>
        </div>
        """
        status_badge = f'<span class="status-badge status-success">Accepted</span>'
        
        return f"""
        <div class="recognition-row">
            <div>
                <div class="label-text">Current Character</div>
                <div class="char-val">{char_val}</div>
            </div>
            <div>
                {status_badge}
            </div>
        </div>
        {progress_html}
        """
    elif status == "Rejected":
        return """
        <div style="text-align: center; padding: 5px 0;">
            <div style="font-size: 15px; font-weight: 700; color: #EF4444; margin-bottom: 2px;">Character Rejected</div>
            <div style="font-size: 11px; color: #94A3B8;">Please write again.</div>
        </div>
        """
    else:
        return """
        <div style="text-align: center; padding: 10px 0;">
            <div class="empty-placeholder">Waiting for input...</div>
        </div>
        """

def get_live_text_card_html(word, sentence):
    if not word and not sentence:
        return """
        <div style="text-align: center; padding: 10px 0;">
            <div class="empty-placeholder">No text recognized yet.</div>
        </div>
        """
    else:
        word_display = word if word else "-"
        sentence_display = sentence if sentence else "-"
        cursor_html = '<span class="terminal-cursor"></span>' if word else ''
        return f"""
        <div class="text-group">
            <div class="label-text">Current Word</div>
            <div class="word-val">{word_display}{cursor_html}</div>
        </div>
        <div class="card-divider"></div>
        <div class="text-group">
            <div class="label-text">Current Sentence</div>
            <div class="sentence-val">{sentence_display}</div>
        </div>
        """

def get_nlp_card_html(nlp_res):
    if nlp_res and nlp_res.get("corrected"):
        corrected = nlp_res["corrected"]
        return f'<div class="nlp-val">{corrected}</div>'
    else:
        return '<div class="empty-placeholder" style="text-align: center;">Waiting for completed sentence...</div>'

def get_translation_card_html(translated_text):
    if not translated_text:
        return '<div class="empty-placeholder">Translation will appear here.</div>'
    else:
        return f'<div class="translation-output">{translated_text}</div>'

def get_speech_status_html(p_status):
    if p_status == "Playing":
        return '<div class="speech-status speech-status-active">Playing Audio...</div>'
    elif p_status.startswith("Offline speech") or "unavailable" in p_status.lower():
        return f'<div class="speech-status" style="color: #EF4444; border-color: #EF4444; background: rgba(239,68,68,0.1);">{p_status}</div>'
    else:
        return '<div class="speech-status">Ready for Speech</div>'

def get_history_card_html(history_list):
    if not history_list:
        return """
        <div class="history-empty">
            <div style="font-weight: 600; margin-bottom: 2px; color: #F8FAFC;">No sentences yet.</div>
            <div style="color: #94A3B8;">Start writing to create your first sentence.</div>
        </div>
        """
    else:
        recent = history_list[-5:]
        items_html = ""
        for idx, sentence in enumerate(recent):
            number = idx + 1
            items_html += f'<div class="history-item">{number}. {sentence}</div>'
        return f'<div class="history-list">{items_html}</div>'

# OpenCV Frame Floating Overlay Badge Drawer
def draw_gesture_badge(img: cv2.Mat, gesture_name: str) -> cv2.Mat:
    """
    Renders a premium transparent floating badge on the top-right corner of the video frame.
    States: Writing (Blue), Character Complete (Green), Word Complete (Amber),
    Sentence Complete (Green), Idle (Gray).
    """
    h, w, c = img.shape
    
    # Define colors in BGR matching design guidelines
    colors = {
        "idle": (184, 163, 148),        # #94A3B8 in BGR
        "writing": (246, 130, 59),      # #3B82F6 in BGR
        "completion": (94, 197, 34),     # #22C55E in BGR
        "open_palm": (11, 158, 245),     # #F59E0B in BGR
        "thumbs_up": (94, 197, 34)       # #22C55E in BGR
    }
    
    labels = {
        "idle": "IDLE",
        "writing": "WRITING",
        "completion": "CHARACTER COMPLETE",
        "open_palm": "WORD COMPLETE",
        "thumbs_up": "SENTENCE COMPLETE"
    }
    
    g_key = gesture_name.lower().strip()
    if g_key not in colors:
        g_key = "idle"
        
    color = colors[g_key]
    label = labels[g_key]
    
    # Calculate badge dimension based on text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    padding_x = 12
    padding_y = 8
    badge_w = text_w + padding_x * 2
    badge_h = text_h + padding_y * 2
    
    x1 = w - badge_w - 20
    y1 = 20
    x2 = w - 20
    y2 = y1 + badge_h
    
    # Blend background badge box transparency
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw label text
    text_x = x1 + padding_x
    text_y = y2 - padding_y + 1
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

class WebcamManager:
    """
    Manages the webcam video capture and MediaPipe hand landmark tracking
    in a dedicated background thread to guarantee smooth frame rates.
    """
    def __init__(self, tracker, detect_hand_gesture_fn, trajectory=None):
        self.tracker = tracker
        self.detect_hand_gesture_fn = detect_hand_gesture_fn
        self.cap = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Thread-safe storage for the latest processed state
        self.latest_frame = None
        self.latest_landmarks = []
        self.latest_gesture = "idle"
        self.latest_fps = 0
        self.latest_hand_detected = False
        
        # Stroke and trajectory tracking inside the background thread to prevent dropped points
        self.recorded_points = []
        self.trajectory = trajectory if trajectory is not None else TrajectoryManager()
        self.gesture_state = "idle"
        
        # Gesture history for debouncing
        self.gesture_history = []
        self.debounce_frames = 5
        
        # Event flags for the main thread
        self.inference_pending = False
        self.open_palm_pending = False
        self.thumbs_up_pending = False
        
    def start(self):
        with self.lock:
            if self.running:
                return
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam hardware.")
            self.running = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
            
        # Release the camera FIRST. This breaks the blocking cap.read() call in the thread loop instantly.
        if self.cap:
            self.cap.release()
            
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        with self.lock:
            self.cap = None
            
    def _loop(self):
        prev_time = time.time()
        hand_present_prev = False
        prev_gesture = "other"
        
        while True:
            with self.lock:
                if not self.running:
                    break
            
            # Read from camera. If cap was released, this returns ret=False immediately.
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Execute hand landmarker on the background thread
            frame, hand_detected = self.tracker.find_hands(frame, draw=True)
            
            landmarks = []
            raw_gesture = "idle"
            if hand_detected:
                landmarks = self.tracker.get_landmarks(frame, hand_idx=0)
                raw_gesture = self.detect_hand_gesture_fn(landmarks)
                
            # Debounce gesture
            self.gesture_history.append(raw_gesture)
            if len(self.gesture_history) > self.debounce_frames:
                self.gesture_history.pop(0)
                
            # Majority vote gesture
            if len(self.gesture_history) > 0:
                current_gesture = max(set(self.gesture_history), key=self.gesture_history.count)
            else:
                current_gesture = "idle"
                
            # Update state and trajectory under lock
            with self.lock:
                self.latest_landmarks = landmarks
                self.latest_gesture = current_gesture
                self.latest_fps = int(fps)
                self.latest_hand_detected = hand_detected
                
                if hand_detected:
                    if current_gesture == "writing":
                        if self.gesture_state in ["idle", "waiting_for_index"]:
                            self.gesture_state = "writing"
                            self.recorded_points = []
                            self.trajectory.clear_current()
                            
                        # Extract index fingertip landmark (8)
                        index_tip = self.tracker.get_landmark_by_id(landmarks, 8)
                        if index_tip:
                            px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                            norm_x, norm_y, norm_z = index_tip["x"], index_tip["y"], index_tip["z"]
                            
                            # Exponential Moving Average (EMA) Coordinate Smoothing
                            alpha = 0.65
                            if len(self.recorded_points) > 0:
                                prev_pt = self.recorded_points[-1]
                                px_x = int(alpha * px_x + (1 - alpha) * prev_pt[0])
                                px_y = int(alpha * px_y + (1 - alpha) * prev_pt[1])
                                norm_x = alpha * norm_x + (1 - alpha) * prev_pt[2]
                                norm_y = alpha * norm_y + (1 - alpha) * prev_pt[3]
                                norm_z = alpha * norm_z + (1 - alpha) * prev_pt[4]
                                
                            self.recorded_points.append([px_x, px_y, norm_x, norm_y, norm_z])
                            self.trajectory.add_point(px_x, px_y, norm_x, norm_y, norm_z)
                            
                    elif current_gesture == "completion":
                        if self.gesture_state == "writing":
                            self.gesture_state = "waiting_for_index"
                            self.inference_pending = True
                            
                    elif current_gesture == "open_palm":
                        if prev_gesture != "open_palm":
                            self.gesture_state = "waiting_for_index"
                            self.open_palm_pending = True
                            self.recorded_points = []
                            self.trajectory.clear()
                            
                    elif current_gesture == "thumbs_up":
                        if prev_gesture != "thumbs_up":
                            self.gesture_state = "idle"
                            self.thumbs_up_pending = True
                            self.recorded_points = []
                            self.trajectory.clear()
                    else:
                        if self.gesture_state == "writing":
                            self.trajectory.trigger_new_stroke()
                            
                    prev_gesture = current_gesture
                    hand_present_prev = True
                else:
                    prev_gesture = "other"
                    if hand_present_prev:
                        if self.gesture_state == "writing":
                            self.trajectory.trigger_new_stroke()
                        hand_present_prev = False
                        
                # Draw pointer and trajectory on the frame
                if current_gesture == "writing":
                    index_tip = self.tracker.get_landmark_by_id(landmarks, 8)
                    if index_tip:
                        cv2.circle(frame, (index_tip["px_x"], index_tip["px_y"]), 6, (0, 0, 255), -1)
                elif current_gesture in ["completion", "open_palm", "thumbs_up"]:
                    index_tip = self.tracker.get_landmark_by_id(landmarks, 8)
                    if index_tip:
                        cv2.circle(frame, (index_tip["px_x"], index_tip["px_y"]), 6, (0, 255, 0), -1)
                else:
                    if landmarks:
                        index_tip = self.tracker.get_landmark_by_id(landmarks, 8)
                        if index_tip:
                            cv2.circle(frame, (index_tip["px_x"], index_tip["px_y"]), 6, (0, 255, 0), -1)
                            
                frame = self.trajectory.draw_trajectory(frame, color=(246, 130, 59), thickness=5)
                frame = draw_gesture_badge(frame, current_gesture)
                
                # Store the fully drawn frame
                self.latest_frame = frame
                
            time.sleep(0.005) # Yield thread slice
            
    def get_latest_state(self):
        """
        Thread-safe getter for the latest frame, landmarks, gesture, FPS, hand detection status,
        recorded points, and the current gesture state.
        """
        with self.lock:
            return (
                self.latest_frame.copy() if self.latest_frame is not None else None,
                self.latest_landmarks,
                self.latest_gesture,
                self.latest_fps,
                self.latest_hand_detected,
                list(self.recorded_points),
                self.gesture_state
            )
            
    def clear_all(self):
        with self.lock:
            self.trajectory.clear()
            self.recorded_points = []
            self.gesture_state = "idle"
            self.gesture_history = []
            self.inference_pending = False
            self.open_palm_pending = False
            self.thumbs_up_pending = False
            
    def clear_current(self):
        with self.lock:
            self.trajectory.clear_current()
            self.recorded_points = []
            self.gesture_state = "waiting_for_index"
            
    def delete_last_character_trajectory(self):
        with self.lock:
            self.trajectory.delete_last_character_trajectory()
            
    def save_current_character_trajectory(self):
        with self.lock:
            self.trajectory.save_current_character()
            
    def __del__(self):
        self.stop()

# ----------------------------------------------------
# 6. UI Layout Setup
# ----------------------------------------------------
# Centered application header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">AIR WRITING RECOGNITION SYSTEM</h1>
    <div class="app-subtitle">Real-Time AI Based Air Writing Recognition using Hand Gestures</div>
</div>
""", unsafe_allow_html=True)

# Split into left (35%) and right (65%) columns
left_col, right_col = st.columns([0.35, 0.65], gap="large")

# --- LEFT COLUMN (Cards 1 to 4) ---
with left_col:
    # Card 1: Recognition
    with st.container(border=True):
        st.markdown("""
            <div class="card-header">
                <i class="material-icons header-icon">psychology</i>
                <span class="card-title">Recognition</span>
            </div>
        """, unsafe_allow_html=True)
        pred_placeholder = st.empty()
        
    # Card 2: Live Text
    with st.container(border=True):
        st.markdown("""
            <div class="card-header">
                <i class="material-icons header-icon">keyboard</i>
                <span class="card-title">Live Text</span>
            </div>
        """, unsafe_allow_html=True)
        text_placeholder = st.empty()
        
        # Action buttons toolbar
        st.markdown('<div class="live-text-toolbar-marker"></div>', unsafe_allow_html=True)
        
        current_word = st.session_state.sentence_builder.get_current_word()
        current_sentence = st.session_state.sentence_builder.get_current_sentence()
        is_word_empty = (len(current_word) == 0)
        is_sentence_empty = (len(current_sentence.strip()) == 0)
        
        # Word Actions Row
        st.markdown('<div class="label-text" style="font-weight:600; margin-bottom:6px;">Word Actions</div>', unsafe_allow_html=True)
        btn_cols_word = st.columns(2)
        btn_cols_word[0].button(
            "← Delete Last", 
            key="delete_last_btn", 
            on_click=delete_last_callback, 
            disabled=is_word_empty, 
            width="stretch"
        )
        btn_cols_word[1].button(
            "🗑 Clear Word", 
            key="clear_word_btn", 
            on_click=clear_word_callback, 
            disabled=is_word_empty, 
            width="stretch"
        )
        
        # Sentence Actions Row
        st.markdown('<div class="label-text" style="font-weight:600; margin-top:10px; margin-bottom:6px;">Sentence Actions</div>', unsafe_allow_html=True)
        btn_cols_sent = st.columns(2)
        btn_cols_sent[0].button(
            "← Delete Last Word", 
            key="delete_last_word_btn", 
            on_click=delete_last_word_callback, 
            disabled=is_sentence_empty, 
            width="stretch"
        )
        btn_cols_sent[1].button(
            "↺ Reset", 
            key="reset_canvas_btn", 
            on_click=clear_all_inputs, 
            width="stretch"
        )
        
    # Card 3: AI Enhanced Text
    with st.container(border=True):
        st.markdown("""
            <div class="card-header">
                <i class="material-icons header-icon">auto_awesome</i>
                <span class="card-title">AI Enhanced Text</span>
            </div>
        """, unsafe_allow_html=True)
        nlp_placeholder = st.empty()
        
    # Card 4: Translation
    with st.container(border=True):
        st.markdown("""
            <div class="card-header">
                <i class="material-icons header-icon">translate</i>
                <span class="card-title">Translation</span>
            </div>
        """, unsafe_allow_html=True)
        
        target_languages = [
            "English", "Hindi", "Kannada", "Tamil", "Telugu", "Malayalam", "Marathi",
            "French", "German", "Korean", "Spanish", "Japanese"
        ]
        selected_lang = st.selectbox(
            "Select Translation Language",
            options=target_languages,
            index=target_languages.index(st.session_state.target_lang) if st.session_state.target_lang in target_languages else 0,
            key="translation_lang_select",
            label_visibility="collapsed"
        )
        translation_placeholder = st.empty()
        
    # Trigger translation dynamically on dropdown language switch
    if selected_lang != st.session_state.target_lang:
        st.session_state.target_lang = selected_lang
        if st.session_state.latest_nlp_result and st.session_state.latest_nlp_result.get("corrected"):
            st.session_state.system_status = "Translating"
            trans_res = st.session_state.translation_engine.translate(
                st.session_state.latest_nlp_result["corrected"], 
                selected_lang
            )
            st.session_state.latest_translation_result = trans_res
            
            st.session_state.system_status = "Speaking"
            speech_res = st.session_state.speech_generator.generate_speech(
                trans_res["translated"], 
                selected_lang
            )
            st.session_state.latest_speech_result = speech_res
            st.session_state.system_status = "Ready"

# --- RIGHT COLUMN (Webcam Feed only) ---
with right_col:
    # Camera active state and button toggle
    btn_label = "Stop Camera" if st.session_state.camera_active else "Start Camera"
    if st.button(btn_label, key="webcam_toggle_btn", width="stretch"):
        st.session_state.camera_active = not st.session_state.camera_active
        st.rerun()
        
    # Always render using the same placeholder to prevent recreation layout jumps
    video_placeholder = st.empty()
    
    if not st.session_state.camera_active:
        video_placeholder.markdown("""
            <div class="webcam-placeholder-card">
                <div class="webcam-placeholder-content">
                    <i class="material-icons webcam-placeholder-icon">videocam_off</i>
                    <div style="font-size: 18px; font-weight: 600; color: #F8FAFC; margin-bottom: 6px;">Camera Preview</div>
                    <div style="font-size: 13px; color: #94A3B8; margin-bottom: 4px;">Camera is currently off</div>
                    <div style="font-size: 12px; color: #64748B;">Click "Start Camera" to begin</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- BOTTOM FULL WIDTH SECTION 1: Speech Playback Controls ---
st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("""
        <div class="card-header">
            <i class="material-icons header-icon">volume_up</i>
            <span class="card-title">Speech Playback Controls</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Horizontally aligned and centered buttons setup
    sp_cols = st.columns([2, 1, 1, 1, 1, 1, 2])
    sp_cols[1].button("Play", key="speech_play", on_click=speech_play, width="stretch")
    sp_cols[2].button("Pause", key="speech_pause", on_click=speech_pause, width="stretch")
    sp_cols[3].button("Replay", key="speech_replay", on_click=speech_replay, width="stretch")
    sp_cols[4].button("Stop", key="speech_stop", on_click=speech_stop, width="stretch")
    
    audio_path = st.session_state.latest_speech_result.get("audio_path", "") if st.session_state.latest_speech_result else ""
    if audio_path and os.path.exists(audio_path):
        with open(audio_path, "rb") as f:
            sp_cols[5].download_button(
                label="Download",
                data=f,
                file_name=os.path.basename(audio_path),
                mime="audio/wav" if audio_path.endswith(".wav") else "audio/mpeg",
                width="stretch"
            )
    else:
        sp_cols[5].button("Download", disabled=True, width="stretch")
        
    speech_status_placeholder = st.empty()

# --- BOTTOM FULL WIDTH SECTION 2: Recent Sentence History ---
st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("""
        <div class="card-header">
            <i class="material-icons header-icon">history</i>
            <span class="card-title">History</span>
        </div>
    """, unsafe_allow_html=True)
    history_placeholder = st.empty()

# Initialize placeholder outputs on first render
pred_placeholder.markdown(get_prediction_card_html(
    st.session_state.latest_pred_char,
    st.session_state.latest_pred_conf,
    st.session_state.latest_pred_status
), unsafe_allow_html=True)

text_placeholder.markdown(get_live_text_card_html(
    st.session_state.sentence_builder.get_current_word(),
    st.session_state.sentence_builder.get_current_sentence()
), unsafe_allow_html=True)

nlp_placeholder.markdown(get_nlp_card_html(st.session_state.latest_nlp_result), unsafe_allow_html=True)

translated_text = st.session_state.latest_translation_result.get("translated", "") if st.session_state.latest_translation_result else ""
translation_placeholder.markdown(get_translation_card_html(translated_text), unsafe_allow_html=True)

speech_status_placeholder.markdown(get_speech_status_html(
    st.session_state.speech_generator.get_playback_status()
), unsafe_allow_html=True)

history_placeholder.markdown(get_history_card_html(
    st.session_state.sentence_builder.get_history()
), unsafe_allow_html=True)

# ----------------------------------------------------
# 7. Webcam Real-Time Video Thread Loop
# ----------------------------------------------------
# If camera is inactive, ensure the background manager is stopped
if not st.session_state.camera_active:
    if "webcam_manager" in st.session_state:
        st.session_state.webcam_manager.stop()
        del st.session_state.webcam_manager

if st.session_state.camera_active:
    if "webcam_manager" not in st.session_state:
        try:
            manager = WebcamManager(st.session_state.tracker, detect_hand_gesture, st.session_state.trajectory)
            manager.start()
            st.session_state.webcam_manager = manager
        except Exception as e:
            right_col.error(f"Webcam hardware is currently locked or unavailable: {e}")
            st.session_state.camera_active = False
            st.rerun()
            
    manager = st.session_state.webcam_manager
    
    # Initialize HTML caches to avoid redundant Streamlit updates
    last_pred_html = None
    last_text_html = None
    last_nlp_html = None
    last_translation_html = None
    last_speech_html = None
    last_history_html = None
    
    prev_gesture = "other"
    hand_present_prev = False
    
    while st.session_state.camera_active:
        # Get the latest state thread-safely from WebcamManager
        latest_state = manager.get_latest_state()
        if latest_state[0] is None:
            time.sleep(0.01)
            continue
            
        (
            latest_frame,
            latest_landmarks,
            latest_gesture,
            fps,
            hand_detected,
            recorded_points,
            gesture_state
        ) = latest_state
        
        st.session_state.fps = fps
        st.session_state.recorded_points = recorded_points
        st.session_state.gesture_state = gesture_state
        
        # 1. Inference Event Trigger
        if manager.inference_pending:
            with manager.lock:
                manager.inference_pending = False
                
            st.session_state.system_status = "Processing"
            inf_start = time.time()
            
            # Validate path length, bounding box diagonals, velocities
            is_valid, err_reason, dpr = validate_trajectory(
                st.session_state.recorded_points,
                min_len=10,
                min_bbox_diagonal=40.0,
                min_path_distance=35.0,
                min_dpr=0.10
            )
            
            should_rerun = False
            if not is_valid:
                st.session_state.latest_pred_char = "UNKNOWN"
                st.session_state.latest_pred_conf = 0.0
                st.session_state.latest_pred_status = "Rejected"
                st.session_state.validation_status = f"Rejected: {err_reason}"
                manager.clear_current()
            else:
                try:
                    raw_traj = np.array(st.session_state.recorded_points, dtype=np.float32)
                    preprocessed = preprocess_single_trajectory(raw_traj, target_len=64, smooth_window=3, mode='resample')
                    input_tensor = np.expand_dims(preprocessed, axis=0)
                    
                    # Run prediction
                    raw_preds = model(input_tensor, training=False).numpy()[0]
                    preds = apply_temperature_scaling(raw_preds, 1.4)
                    
                    sorted_idx = np.argsort(preds)
                    pred_idx = sorted_idx[-1]
                    second_idx = sorted_idx[-2]
                    
                    confidence = float(preds[pred_idx])
                    second_confidence = float(preds[second_idx])
                    margin = confidence - second_confidence
                    predicted_label = idx_to_label[pred_idx]
                    
                    # Shoelace area check to overrule O vs 0
                    if predicted_label in ['O', '0']:
                        orientation = calculate_trajectory_orientation(preprocessed)
                        if orientation > 0:
                            predicted_label = '0'
                        else:
                            predicted_label = 'O'
                            
                    original_label = predicted_label
                    
                    # Geometric feature checks
                    feats = extract_geometric_features(preprocessed)
                    
                    # Loop closure check
                    is_closed_loop = True
                    if predicted_label in ['O', '0']:
                        loop_closure = feats["loop_closure"]
                        if completeness_stats is not None and predicted_label in completeness_stats:
                            stats = completeness_stats[predicted_label]["loop_closure"]
                            loop_threshold = stats["p95"] + 0.30
                        else:
                            loop_threshold = 0.40
                        is_closed_loop = (loop_closure <= loop_threshold)
                        if not is_closed_loop:
                            predicted_label = "UNKNOWN"
                            
                    # Straight line validation
                    is_straight_line = (feats["dpr"] > 0.90)
                    if is_straight_line and predicted_label != "UNKNOWN":
                        dx = st.session_state.recorded_points[-1][0] - st.session_state.recorded_points[0][0]
                        dy = st.session_state.recorded_points[-1][1] - st.session_state.recorded_points[0][1]
                        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
                        if angle <= 35.0:
                            if predicted_label not in ['I', '1']:
                                predicted_label = "UNKNOWN"
                        else:
                            predicted_label = "UNKNOWN"
                            
                    # Geometric completeness verification
                    is_complete = True
                    is_lowercase = predicted_label.islower() and predicted_label.isalpha()
                    
                    if predicted_label != "UNKNOWN" and completeness_stats is not None:
                        tol = 0.40 if is_lowercase else 0.25
                        min_pr = 0.70 if is_lowercase else 0.80
                        is_complete, _, _ = validate_character(feats, completeness_stats.get(predicted_label, {}), tolerance=tol, min_pass_ratio=min_pr)
                        
                    # Commit classification outputs
                    confidence_thresh = CLASS_THRESHOLDS.get(predicted_label, 0.70)
                    margin_thresh = 0.10 if is_lowercase else 0.15
                    
                    if predicted_label == "UNKNOWN" or not is_closed_loop:
                        st.session_state.latest_pred_char = original_label
                        st.session_state.latest_pred_conf = confidence
                        st.session_state.latest_pred_status = "Rejected"
                        st.session_state.validation_status = "Rejected: Incomplete shape loop"
                        manager.clear_current()
                    elif not is_complete:
                        st.session_state.latest_pred_char = original_label
                        st.session_state.latest_pred_conf = confidence
                        st.session_state.latest_pred_status = "Rejected"
                        st.session_state.validation_status = "Rejected: Structural features invalid"
                        manager.clear_current()
                    elif confidence < confidence_thresh:
                        st.session_state.latest_pred_char = predicted_label
                        st.session_state.latest_pred_conf = confidence
                        st.session_state.latest_pred_status = "Rejected"
                        st.session_state.validation_status = f"Rejected: Low confidence prediction (<{confidence_thresh*100:.0f}%)"
                        manager.clear_current()
                    elif margin < margin_thresh:
                        st.session_state.latest_pred_char = predicted_label
                        st.session_state.latest_pred_conf = confidence
                        st.session_state.latest_pred_status = "Rejected"
                        st.session_state.validation_status = f"Rejected: Close class ambiguities (<{margin_thresh*100:.0f}%)"
                        manager.clear_current()
                    else:
                        st.session_state.latest_pred_char = predicted_label
                        st.session_state.latest_pred_conf = confidence
                        st.session_state.latest_pred_status = "Accepted"
                        st.session_state.validation_status = ""
                        st.session_state.sentence_builder.append_character(predicted_label)
                        manager.save_current_character_trajectory()
                        should_rerun = True
                        
                except Exception as e:
                    print(f"Error during recognition: {e}")
                    st.session_state.latest_pred_status = "Rejected"
                    manager.clear_current()
                    
            st.session_state.recognition_time = (time.time() - inf_start) * 1000 # ms
            st.session_state.system_status = "Ready"
            if should_rerun:
                st.rerun()
                
        # 2. Open Palm Event Trigger
        elif manager.open_palm_pending:
            with manager.lock:
                manager.open_palm_pending = False
            should_rerun = False
            if st.session_state.sentence_builder.get_current_word():
                st.session_state.sentence_builder.finish_word()
                should_rerun = True
            st.session_state.system_status = "Ready"
            if should_rerun:
                st.rerun()
                
        # 3. Thumbs Up Event Trigger
        elif manager.thumbs_up_pending:
            with manager.lock:
                manager.thumbs_up_pending = False
            should_rerun = False
            if st.session_state.sentence_builder.get_current_sentence() or st.session_state.sentence_builder.get_current_word():
                if st.session_state.sentence_builder.get_current_word():
                    st.session_state.sentence_builder.finish_word()
                    
                st.session_state.system_status = "Processing"
                nlp_res = st.session_state.sentence_builder.finish_sentence()
                st.session_state.latest_nlp_result = nlp_res
                
                if nlp_res and nlp_res.get("corrected"):
                    st.session_state.system_status = "Translating"
                    trans_res = st.session_state.translation_engine.translate(
                        nlp_res["corrected"], 
                        st.session_state.target_lang
                    )
                    st.session_state.latest_translation_result = trans_res
                    
                    st.session_state.system_status = "Speaking"
                    speech_res = st.session_state.speech_generator.generate_speech(
                        trans_res["translated"], 
                        st.session_state.target_lang
                    )
                    st.session_state.latest_speech_result = speech_res
                should_rerun = True
                
            st.session_state.system_status = "Ready"
            if should_rerun:
                st.rerun()
                
        # Display frame image inside placeholder and update panels
        try:
            frame_rgb = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, width="stretch")
            
            # Dynamic updates of Left Panel Placeholders (only render if contents changed)
            pred_html = get_prediction_card_html(
                st.session_state.latest_pred_char,
                st.session_state.latest_pred_conf,
                st.session_state.latest_pred_status
            )
            if pred_html != last_pred_html:
                pred_placeholder.markdown(pred_html, unsafe_allow_html=True)
                last_pred_html = pred_html
                
            curr_word = st.session_state.sentence_builder.get_current_word()
            curr_sent = st.session_state.sentence_builder.get_current_sentence()
            text_html = get_live_text_card_html(curr_word, curr_sent)
            if text_html != last_text_html:
                text_placeholder.markdown(text_html, unsafe_allow_html=True)
                last_text_html = text_html
                
            nlp_html = get_nlp_card_html(st.session_state.latest_nlp_result)
            if nlp_html != last_nlp_html:
                nlp_placeholder.markdown(nlp_html, unsafe_allow_html=True)
                last_nlp_html = nlp_html
                
            translated_text = st.session_state.latest_translation_result.get("translated", "") if st.session_state.latest_translation_result else ""
            translation_html = get_translation_card_html(translated_text)
            if translation_html != last_translation_html:
                translation_placeholder.markdown(translation_html, unsafe_allow_html=True)
                last_translation_html = translation_html
                
            p_status = st.session_state.speech_generator.get_playback_status()
            speech_html = get_speech_status_html(p_status)
            if speech_html != last_speech_html:
                speech_status_placeholder.markdown(speech_html, unsafe_allow_html=True)
                last_speech_html = speech_html
                
            recent_history = st.session_state.sentence_builder.get_history()
            history_html = get_history_card_html(recent_history)
            if history_html != last_history_html:
                history_placeholder.markdown(history_html, unsafe_allow_html=True)
                last_history_html = history_html
        except Exception:
            # Catch closed socket/websocket errors on server exit or tab closing
            break
            
        time.sleep(0.01)
        
    video_placeholder.empty()

