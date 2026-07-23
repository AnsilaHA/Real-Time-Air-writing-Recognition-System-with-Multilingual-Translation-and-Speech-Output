import os
import sys
import json
import argparse
import time
import numpy as np
import cv2
import tensorflow as tf

# Ensure the src directory is on the path for importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_tracker import HandTracker
from trajectory_manager import TrajectoryManager
from preprocess import preprocess_single_trajectory
from sentence_builder import SentenceBuilder
from translation import TranslationEngine
from speech import SpeechGenerator

def load_recognition_system(model_path: str, mapping_path: str):
    """
    Loads the trained model and label mapping configuration.
    
    Returns:
        model: Loaded Keras model object.
        classes: List of string labels.
        idx_to_label: Dict mapping index to label.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at: {model_path}. "
                                f"Please train the model first.")
                                
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Label mapping JSON file not found at: {mapping_path}.")
        
    print(f"[Info] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"[Info] Loading label mapping from: {mapping_path}")
    with open(mapping_path, "r") as f:
        mapping_config = json.load(f)
        
    classes = mapping_config["classes"]
    idx_to_label = {int(k): v for k, v in mapping_config["idx_to_label"].items()}
    
    return model, classes, idx_to_label

def draw_hud(img: cv2.Mat, writing_mode: bool, latest_pred: str, 
             confidence: float, history: list, confidence_threshold: float,
             validation_status: str = "", margin: float = 0.0, 
             margin_threshold: float = 0.15, entropy: float = 0.0,
             max_entropy: float = 1.8, dpr: float = 1.0,
             current_word: str = "", current_sentence: str = "",
             nlp_result: dict = None, translation_result: dict = None,
             target_lang: str = "Hindi", speech_result: dict = None,
             speech_generator = None) -> cv2.Mat:
    """
    Renders a premium Head-Up Display (HUD) overlay on the OpenCV frame.
    Includes Prediction Card (Left), Word Card (Left Middle), Sentence Card (Left Bottom),
    NLP Post-Processing Card (Right), and Sentence History bar (Bottom).
    """
    h, w, c = img.shape
    
    # 1. Header Bar
    cv2.rectangle(img, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(img, "AIR-WRITING SENTENCE RECOGNITION SYSTEM", (20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
    # 2. Writing Status Card
    status_w = 200
    status_x = w - status_w - 20
    if writing_mode:
        # Pulsing effect for red recording indicator
        alpha = int(127 * (np.sin(cv2.getTickCount() / cv2.getTickFrequency() * 10) + 1.0))
        cv2.rectangle(img, (status_x, 10), (w - 20, 40), (0, 0, 200 + alpha // 4), -1)
        cv2.putText(img, "● WRITING", (status_x + 35, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (status_x, 10), (w - 20, 40), (60, 60, 60), -1)
        cv2.putText(img, "STANDBY", (status_x + 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # 3. Main Prediction Overlay Card (Top Left - "Current Character")
    # If there is a validation error (rejection reason), draw a red card
    if validation_status:
        card_w = 340
        card_h = 110
        card_x = 20
        card_y = 70
        
        # Semi-transparent overlay background card
        overlay = img.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 40), -1)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 50, 150), 2)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        # Large "?" symbol
        cv2.putText(img, "?", (card_x + 25, card_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (100, 100, 255), 5, cv2.LINE_AA)
                    
        # Error Details
        cv2.putText(img, "UNKNOWN / REJECTED", (card_x + 100, card_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2, cv2.LINE_AA)
        
        # Split reason if it's too long
        reason_text = validation_status
        if len(reason_text) > 28:
            cv2.putText(img, reason_text[:28], (card_x + 100, card_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, reason_text[28:], (card_x + 100, card_y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, reason_text, (card_x + 100, card_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            
    elif latest_pred:
        card_w = 340
        card_h = 110
        card_x = 20
        card_y = 70
        
        # Semi-transparent overlay background card
        overlay = img.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (100, 100, 100), 2)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        # Large Predicted Character
        if latest_pred == "UNKNOWN":
            cv2.putText(img, "UNKNOWN", (card_x + 15, card_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(img, latest_pred, (card_x + 20, card_y + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5, cv2.LINE_AA)
                    
        # Confidence Score & Labels
        conf_pct = confidence * 100
        margin_pct = margin * 100
        text_y = card_y + 25
        cv2.putText(img, f"Class (Char): {latest_pred}", (card_x + 100, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        if latest_pred == "UNKNOWN":
            cv2.putText(img, "Conf (Calib): N/A", (card_x + 100, text_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Margin: N/A", (card_x + 100, text_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f"Conf (Calib): {conf_pct:.1f}%", (card_x + 100, text_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"Margin: {margin_pct:.1f}%", (card_x + 100, text_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Entropy: {entropy:.2f} (<{max_entropy})", (card_x + 100, text_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"DPR: {dpr:.2f}", (card_x + 100, text_y + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
                        
        # Mini Confidence Bar
        bar_x = card_x + 100
        bar_y = card_y + 98
        bar_max_w = 210
        bar_h = 8
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (50, 50, 50), -1)
        if latest_pred != "UNKNOWN":
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_max_w * confidence), bar_y + bar_h), (0, 255, 0), -1)

    # 4. Word Formation Card (Below Prediction Card on Left)
    card_w = 340
    card_h = 100
    card_x = 20
    card_y = 195
    
    # Semi-transparent overlay background card
    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (35, 20, 20), -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (180, 50, 50), 2)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    # Draw title
    cv2.putText(img, "CURRENT WORD", (card_x + 15, card_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
                
    word_disp = current_word if current_word else "[Empty]"
    font_scale = 1.0
    if len(word_disp) > 8:
        font_scale = max(0.5, 1.0 - (len(word_disp) - 8) * 0.08)
    cv2.putText(img, word_disp, (card_x + 15, card_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2, cv2.LINE_AA)

    # 5. Sentence Formation Card (Below Word Card on Left)
    card_w = 480
    card_h = 100
    card_x = 20
    card_y = 310
    
    # Semi-transparent overlay background card
    overlay = img.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 35, 20), -1)
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (50, 180, 50), 2)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    # Draw title
    cv2.putText(img, "CURRENT SENTENCE", (card_x + 15, card_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
                
    sent_disp = current_sentence if current_sentence else "[Empty]"
    font_scale = 0.95
    if len(sent_disp) > 20:
        font_scale = max(0.45, 0.95 - (len(sent_disp) - 20) * 0.018)
    cv2.putText(img, sent_disp, (card_x + 15, card_y + 75),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

    # 6. NLP Post-Processing Card (Right Side)
    nlp_x = w - 420
    nlp_y = 65
    nlp_w = 400
    nlp_h = 240
    
    overlay = img.copy()
    cv2.rectangle(overlay, (nlp_x, nlp_y), (nlp_x + nlp_w, nlp_y + nlp_h), (40, 30, 20), -1)
    cv2.rectangle(overlay, (nlp_x, nlp_y), (nlp_x + nlp_w, nlp_y + nlp_h), (255, 140, 0), 2)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    cv2.putText(img, "NLP POST-PROCESSING", (nlp_x + 15, nlp_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2, cv2.LINE_AA)
                
    if nlp_result:
        # Correction Status
        status = nlp_result.get("status", "No Corrections Needed")
        status_color = (0, 255, 0) if "No" in status else (255, 180, 0)
        cv2.putText(img, f"Status: {status}", (nlp_x + 15, nlp_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
                    
        # Original text
        orig_text = nlp_result.get("original", "")
        cv2.putText(img, f"Orig: {orig_text}", (nlp_x + 15, nlp_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
                    
        # Corrected text
        corr_text = nlp_result.get("corrected", "")
        cv2.putText(img, "Corrected Sentence:", (nlp_x + 15, nlp_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        corr_disp = corr_text if corr_text else "[Empty]"
        if len(corr_disp) > 35:
            cv2.putText(img, corr_disp[:35], (nlp_x + 15, nlp_y + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, corr_disp[35:70], (nlp_x + 15, nlp_y + 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, corr_disp, (nlp_x + 15, nlp_y + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)
                        
        # Confidences
        spell_conf = nlp_result.get("spelling_confidence", 1.0) * 100
        gram_conf = nlp_result.get("grammar_confidence", 1.0) * 100
        cv2.putText(img, f"Spell: {spell_conf:.0f}% | Gram: {gram_conf:.0f}%", (nlp_x + 15, nlp_y + 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                    
        # Alternative suggestions (abbreviated single line)
        suggestions_map = nlp_result.get("suggestions", {})
        if suggestions_map:
            s_keys = list(suggestions_map.keys())[:2]
            s_text = " | ".join([f"{k}->{suggestions_map[k][0]}" for k in s_keys])
            cv2.putText(img, f"Suggs: {s_text}", (nlp_x + 15, nlp_y + 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Suggs: None (All words valid)", (nlp_x + 15, nlp_y + 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
    else:
        # Placeholder standby instructions
        cv2.putText(img, "Status: Standby", (nlp_x + 15, nlp_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "Draw and complete characters.", (nlp_x + 15, nlp_y + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "Complete word: Open Palm gesture.", (nlp_x + 15, nlp_y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "Finalize sentence: Thumbs Up gesture.", (nlp_x + 15, nlp_y + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    # 7. Multilingual Translation Card (Right Side, below NLP)
    trans_x = w - 420
    trans_y = 315
    trans_w = 400
    trans_h = 180
    
    overlay = img.copy()
    cv2.rectangle(overlay, (trans_x, trans_y), (trans_x + trans_w, trans_y + trans_h), (40, 20, 40), -1)
    cv2.rectangle(overlay, (trans_x, trans_y), (trans_x + trans_w, trans_y + trans_h), (255, 0, 255), 2)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    cv2.putText(img, f"TRANSLATION: {target_lang.upper()}", (trans_x + 15, trans_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 255), 2, cv2.LINE_AA)
                
    if translation_result:
        # Translation Status
        status = translation_result.get("status", "Failed")
        status_color = (0, 255, 0) if "Translated" in status or "No Translation" in status else (255, 180, 0)
        cv2.putText(img, f"Status: {status}", (trans_x + 15, trans_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
                    
        # Translation Confidence
        conf = translation_result.get("confidence", 0.0) * 100
        cv2.putText(img, f"Confidence: {conf:.0f}%", (trans_x + 15, trans_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
                    
        # Translated Text
        translated_text = translation_result.get("translated", "")
        is_latin = target_lang in ["English", "French", "German", "Spanish"]
        disp_text = translated_text if is_latin else "[Non-Latin script: Logged to Console]"
        
        cv2.putText(img, "Translated Sentence:", (trans_x + 15, trans_y + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        if len(disp_text) > 35:
            cv2.putText(img, disp_text[:35], (trans_x + 15, trans_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, disp_text[35:70], (trans_x + 15, trans_y + 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, disp_text, (trans_x + 15, trans_y + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        # Placeholder standby instructions
        cv2.putText(img, "Status: Standby", (trans_x + 15, trans_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "Press [1-0/L] keys to switch language.", (trans_x + 15, trans_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "Translations will show here upon", (trans_x + 15, trans_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "finalizing the sentence.", (trans_x + 15, trans_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    # 7.5 Speech Generation Card (Right Side, below Translation)
    speech_x = w - 420
    speech_y = 505
    speech_w = 400
    speech_h = 165
    
    overlay = img.copy()
    cv2.rectangle(overlay, (speech_x, speech_y), (speech_x + speech_w, speech_y + speech_h), (20, 40, 40), -1)
    cv2.rectangle(overlay, (speech_x, speech_y), (speech_x + speech_w, speech_y + speech_h), (0, 255, 255), 2)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    autoplay_status = "ON" if (speech_generator and speech_generator.get_autoplay()) else "OFF"
    cv2.putText(img, f"SPEECH GENERATION (AUTO: {autoplay_status})", (speech_x + 15, speech_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
                
    if speech_result:
        # Success status
        success = speech_result.get("success", False)
        status_text = "Idle"
        if speech_generator:
            status_text = speech_generator.get_playback_status()
            
        status_color = (0, 255, 0) if status_text == "Playing" else ((0, 255, 255) if status_text == "Paused" else (255, 255, 0))
        cv2.putText(img, f"Playback: {status_text}", (speech_x + 15, speech_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
                    
        # Engine and speed
        engine_name = speech_result.get("engine", "None")
        gen_time = speech_result.get("generation_time", 0.0)
        cv2.putText(img, f"Engine: {engine_name} | Latency: {gen_time:.1f} ms", (speech_x + 15, speech_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
                    
        # Filename
        audio_path = speech_result.get("audio_path", "")
        file_name = os.path.basename(audio_path) if audio_path else "None"
        cv2.putText(img, f"File: {file_name}", (speech_x + 15, speech_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
                    
        cv2.putText(img, "Press [P] Play/Rep, [Space] Pause, [S] Stop, [V] Auto.", (speech_x + 15, speech_y + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        # Placeholder standby instructions
        cv2.putText(img, "Status: Standby", (speech_x + 15, speech_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "Playback controls standby.", (speech_x + 15, speech_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, "Speech auto-plays on thumbs up", (speech_x + 15, speech_y + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "when AutoPlay is enabled.", (speech_x + 15, speech_y + 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    # 8. Recognition History Bar (Bottom)
    cv2.rectangle(img, (0, h - 140), (w, h), (15, 15, 15), -1)
    cv2.line(img, (0, h - 140), (w, h - 140), (50, 50, 50), 1)
    
    # Render last 3 sentences
    history_lines = history[-3:]
    for idx, line in enumerate(history_lines):
        line_num = len(history) - len(history_lines) + idx + 1
        cv2.putText(img, f"SENTENCE {line_num}: {line}", (20, h - 105 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    
    # Render controls instructions
    controls_txt = "Gesture: [Index] Draw | [Index+Middle] Comp Char | [Open Palm] Word | [Thumbs Up] Sentence | [1-0/L] Lang | [P/Space/S/V] TTS | [C] Clear | [Q] Exit"
    cv2.putText(img, controls_txt, (w - 1100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                
    return img

def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies temperature scaling to soft probabilities to calibrate confidence.
    """
    if temperature <= 1.0:
        return probs
    # Avoid log(0) with a tiny epsilon
    logits = np.log(probs + 1e-8)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # shift for numerical stability
    return exp_logits / np.sum(exp_logits)

def detect_hand_gesture(landmarks: list) -> str:
    """
    Analyzes landmarks of a single hand to determine the current gesture.
    
    Returns:
        "writing" (Index raised, others folded)
        "completion" (Index & Middle raised, others folded)
        "open_palm" (All 4 major fingers raised)
        "thumbs_up" (Thumb raised, all other 4 folded)
        "other" (any other configuration)
    """
    if not landmarks or len(landmarks) < 21:
        return "other"
        
    try:
        # Standard landmarks:
        # Index: TIP (8), PIP (6)
        # Middle: TIP (12), PIP (10)
        # Ring: TIP (16), PIP (14)
        # Pinky: TIP (20), PIP (18)
        # Thumb: TIP (4), MCP (2)
        
        lm_dict = {lm["id"]: lm for lm in landmarks}
        
        # Check if index is raised: TIP y is less than PIP y
        index_raised = lm_dict[8]["y"] < lm_dict[6]["y"]
        # Check if middle is raised: TIP y is less than PIP y
        middle_raised = lm_dict[12]["y"] < lm_dict[10]["y"]
        # Check if ring is raised: TIP y is less than PIP y
        ring_raised = lm_dict[16]["y"] < lm_dict[14]["y"]
        # Check if pinky is raised: TIP y is less than PIP y
        pinky_raised = lm_dict[20]["y"] < lm_dict[18]["y"]
        # Check if thumb is pointing up: TIP y is less than MCP y
        thumb_up = lm_dict[4]["y"] < lm_dict[2]["y"]
        
        # 1. Thumbs Up (Finish Sentence)
        if thumb_up and not index_raised and not middle_raised and not ring_raised and not pinky_raised:
            return "thumbs_up"
            
        # 2. Open Palm (Finish Word)
        if index_raised and middle_raised and ring_raised and pinky_raised:
            return "open_palm"
            
        # 3. Index + Middle (Finish Character)
        if index_raised and middle_raised and not ring_raised and not pinky_raised:
            return "completion"
            
        # 4. Index Only (Writing Mode)
        if index_raised and not middle_raised and not ring_raised and not pinky_raised:
            return "writing"
            
        return "other"
    except Exception:
        return "other"

def validate_trajectory(recorded_points: list, 
                        min_len: int = 10, 
                        min_bbox_diagonal: float = 40.0, 
                        min_path_distance: float = 35.0, 
                        horizontal_line_ratio: float = 5.0,
                        min_height_for_2d: float = 25.0,
                        min_dpr: float = 0.10,
                        max_velocity: float = 150.0) -> tuple:
    """
    Validates a list of recorded points before running prediction.
    Each point is a list/tuple: [px_x, px_y, norm_x, norm_y, norm_z]
    
    Returns:
        (is_valid, error_reason, dpr)
    """
    if len(recorded_points) < min_len:
        return False, f"Too short (under {min_len} frames)", 0.0
        
    x_coords = [p[0] for p in recorded_points]
    y_coords = [p[1] for p in recorded_points]
    
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    w = xmax - xmin
    h = ymax - ymin
    
    # 1. Bounding box diagonal check
    diagonal = np.sqrt(w**2 + h**2)
    if diagonal < min_bbox_diagonal:
        return False, f"BBox too small ({diagonal:.1f}px < {min_bbox_diagonal}px)", 0.0
        
    # 2. Cumulative path distance & Velocity check
    total_dist = 0.0
    for i in range(1, len(recorded_points)):
        dx = recorded_points[i][0] - recorded_points[i-1][0]
        dy = recorded_points[i][1] - recorded_points[i-1][1]
        step_dist = np.sqrt(dx**2 + dy**2)
        
        # Max velocity check (tracking glitch / jump)
        if step_dist > max_velocity:
            return False, f"Tracking jump ({step_dist:.1f}px > {max_velocity}px)", 0.0
            
        total_dist += step_dist
        
    if total_dist < min_path_distance:
        return False, f"Too static (path {total_dist:.1f}px < {min_path_distance}px)", 0.0
        
    # 3. Displacement-to-Path Ratio (DPR) check (for scribbles)
    disp_x = recorded_points[-1][0] - recorded_points[0][0]
    disp_y = recorded_points[-1][1] - recorded_points[0][1]
    displacement = np.sqrt(disp_x**2 + disp_y**2)
    
    dpr = displacement / max(total_dist, 1.0)
    # We only apply DPR check for longer trajectories where scribbles are common
    if len(recorded_points) > 30 and dpr < min_dpr:
        # Check if it is a clean closed loop (like 'O' or '0')
        is_closed_loop = (displacement < 0.30 * diagonal)
        is_complex_scribble = (total_dist / diagonal) > 5.0
        if is_closed_loop and not is_complex_scribble:
            # Accept: it is a clean closed loop, not a scribble
            pass
        else:
            return False, f"Scribble detected (DPR {dpr:.2f} < {min_dpr})", dpr
        
    # 4. Horizontal line check (high aspect ratio and very small height)
    ratio = w / max(h, 1.0)
    if ratio > horizontal_line_ratio and h < min_height_for_2d:
        return False, f"Horizontal line (ratio {ratio:.1f}x, height {h:.1f}px)", dpr
        
    return True, "", dpr

def extract_geometric_features(preprocessed_traj: np.ndarray) -> dict:
    """
    Extracts geometric features from a preprocessed trajectory of shape (L, 3).
    Only uses x and y coordinates.
    """
    pts = preprocessed_traj[:, :2] # L x 2
    L = len(pts)
    
    # 1. Bounding Box
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    xmin, xmax = np.min(x_coords), np.max(x_coords)
    ymin, ymax = np.min(y_coords), np.max(y_coords)
    w = float(xmax - xmin)
    h = float(ymax - ymin)
    bbox_diagonal = float(np.sqrt(w**2 + h**2))
    bbox_area = float(w * h)
    aspect_ratio = float(w / max(h, 1e-6))
    
    # 2. Path Length & Steps
    diffs = np.diff(pts, axis=0) # (L-1) x 2
    step_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    path_length = float(np.sum(step_lengths))
    
    # 3. Start-End Relationship
    disp_vec = pts[-1] - pts[0]
    start_end_dist = float(np.sqrt(np.sum(disp_vec**2)))
    dpr = float(start_end_dist / max(path_length, 1e-6))
    loop_closure = float(start_end_dist / max(bbox_diagonal, 1e-6))
    
    # 4. Direction Changes
    # Count sign changes in dx and dy, filtering out tiny changes
    dx = diffs[:, 0]
    dy = diffs[:, 1]
    sign_threshold = 0.01
    
    def count_sign_changes(arr):
        changes = 0
        prev_sign = 0
        for val in arr:
            if abs(val) > sign_threshold:
                current_sign = np.sign(val)
                if prev_sign != 0 and current_sign != prev_sign:
                    changes += 1
                prev_sign = current_sign
        return changes
        
    dir_changes_x = count_sign_changes(dx)
    dir_changes_y = count_sign_changes(dy)
    total_dir_changes = dir_changes_x + dir_changes_y
    
    # 5. Curvature / Cumulative Angle Change
    cumulative_angle = 0.0
    for i in range(L - 2):
        v1 = pts[i+1] - pts[i]
        v2 = pts[i+2] - pts[i+1]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_theta = np.dot(v1, v2) / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle = np.arccos(cos_theta)
            cumulative_angle += angle
            
    # 6. Quadrant Coverage
    q1 = np.any((x_coords >= 0) & (y_coords >= 0))
    q2 = np.any((x_coords < 0) & (y_coords >= 0))
    q3 = np.any((x_coords < 0) & (y_coords < 0))
    q4 = np.any((x_coords >= 0) & (y_coords < 0))
    quadrant_coverage = int(q1) + int(q2) + int(q3) + int(q4)
    
    # 7. Self-Intersections (Line Segment Intersections)
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
    intersections = 0
    for i in range(L - 1):
        for j in range(i + 2, L - 1):
            if intersect(pts[i], pts[i+1], pts[j], pts[j+1]):
                intersections += 1
                
    return {
        "path_length": path_length,
        "aspect_ratio": aspect_ratio,
        "dpr": dpr,
        "start_end_dist": start_end_dist,
        "loop_closure": loop_closure,
        "dir_changes": total_dir_changes,
        "cumulative_angle": float(cumulative_angle),
        "quadrant_coverage": quadrant_coverage,
        "intersections": intersections,
        "bbox_diagonal": bbox_diagonal,
        "bbox_area": bbox_area
    }

ABS_TOLERANCES = {
    "path_length": 0.25,
    "aspect_ratio": 0.15,
    "dpr": 0.10,
    "start_end_dist": 0.20,
    "loop_closure": 0.20,
    "dir_changes": 2.0,
    "cumulative_angle": 5.0,
    "quadrant_coverage": 1,
    "intersections": 2.0,
    "bbox_diagonal": 0.10,
    "bbox_area": 0.10
}

CLASS_THRESHOLDS = {
    "0": 0.60,
    # Lower confidence thresholds for lowercase letters to account for flatter probability spreads
    "a": 0.55, "c": 0.55, "e": 0.55, "k": 0.55, "o": 0.55, "q": 0.55, "u": 0.55, "v": 0.55, "r": 0.55
}

def validate_character(features: dict, class_stats: dict, tolerance: float = 0.25, min_pass_ratio: float = 0.80) -> tuple:
    """
    Compares extracted features against reference statistics for the predicted class.
    
    Returns:
        (is_valid, pass_ratio, list_of_failed_features)
    """
    pass_count = 0
    total_features = len(features)
    failed_details = []
    
    for key, val in features.items():
        if key not in class_stats:
            pass_count += 1
            continue
            
        stats = class_stats[key]
        p5 = stats["p5"]
        p95 = stats["p95"]
        
        range_width = max(p95 - p5, 1e-5)
        abs_tol = ABS_TOLERANCES.get(key, 0.1)
        
        lower_bound = p5 - max(tolerance * range_width, abs_tol)
        upper_bound = p95 + max(tolerance * range_width, abs_tol)
        
        # Clip values to physically realistic limits
        if key == "quadrant_coverage":
            lower_bound = max(lower_bound, 1.0)
        elif key == "intersections":
            lower_bound = max(lower_bound, 0.0)
            
        if lower_bound <= val <= upper_bound:
            pass_count += 1
        else:
            failed_details.append(f"{key}: {val:.2f} not in [{lower_bound:.2f}, {upper_bound:.2f}]")
            
    pass_ratio = pass_count / total_features
    is_valid = pass_ratio >= min_pass_ratio
    return is_valid, pass_ratio, failed_details

def calculate_trajectory_orientation(pts: np.ndarray) -> float:
    """
    Computes the signed area of a 2D trajectory path using the Shoelace formula.
    In a screen coordinate system (y going down):
    - A positive value (> 0) indicates a Clockwise (CW) rotation direction.
    - A negative value (< 0) indicates a Counter-Clockwise (CCW) rotation direction.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    val = 0.0
    for i in range(len(pts) - 1):
        val += (x[i] * y[i+1] - x[i+1] * y[i])
    # Close the loop
    val += (x[-1] * y[0] - x[0] * y[-1])
    return val

def print_rejected_zero_debug(predicted_class: str, confidence: float, margin: float, 
                              entropy: float, feats: dict, raw_disp: float, 
                              raw_diag: float, raw_w: float, raw_h: float, 
                              rejection_reason: str, pass_ratio: float = 0.0):
    """
    Prints detailed debugging information for a rejected '0' or 'O' gesture.
    """
    print("\n" + "="*60)
    print("           REJECTED CLOSED-LOOP CHARACTER DEBUGGING          ")
    print("="*60)
    print(f"Predicted Class           : {predicted_class}")
    print(f"Confidence (Calibrated)   : {confidence*100:.1f}%")
    print(f"Margin                    : {margin*100:.1f}%")
    print(f"Entropy                   : {entropy:.4f}")
    if feats:
        print(f"Quality Score (Pass Rate) : {pass_ratio*100:.1f}%")
        print(f"Displacement (Smoothed)   : {feats['start_end_dist']:.4f}")
        print(f"Diagonal (Smoothed)       : {feats['bbox_diagonal']:.4f}")
        print(f"Disp/Diag Ratio (Smoothed): {feats['loop_closure']:.4f}")
        print(f"DPR                       : {feats['dpr']:.4f}")
        print(f"Total Curvature (Rad)     : {feats['cumulative_angle']:.4f}")
        print(f"Direction Changes         : {feats['dir_changes']}")
        print(f"Self-Intersections        : {feats['intersections']}")
        print(f"Quadrant Coverage         : {feats['quadrant_coverage']}/4")
    else:
        print(f"Quality Score             : N/A")
        print(f"Displacement (Raw)        : {raw_disp:.1f}px")
        print(f"Diagonal (Raw)            : {raw_diag:.1f}px")
        print(f"Disp/Diag Ratio (Raw)     : {raw_disp/max(raw_diag, 1e-6):.4f}")
        print(f"DPR                       : N/A")
        print(f"Total Curvature           : N/A")
        print(f"Direction Changes         : N/A")
        print(f"Self-Intersections        : N/A")
        print(f"Quadrant Coverage         : N/A")
    print(f"Raw BBox Width & Height   : {raw_w:.1f}px x {raw_h:.1f}px")
    print(f"Rejection Reason          : {rejection_reason}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Air-Writing Real-Time Recognition Interface")
    parser.add_argument("--model", type=str, default="models/best_model.keras",
                        help="Path to trained Keras model file (defaults to the recommended model)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold for predictions (0.0 to 1.0)")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="Margin threshold between Top-1 and Top-2 probabilities (0.0 to 1.0)")
    parser.add_argument("--min-len", type=int, default=10,
                        help="Minimum trajectory length in frames")
    parser.add_argument("--min-bbox", type=float, default=40.0,
                        help="Minimum bounding box diagonal in pixels")
    parser.add_argument("--min-dist", type=float, default=35.0,
                        help="Minimum path distance in pixels")
    parser.add_argument("--min-dpr", type=float, default=0.10,
                        help="Minimum displacement-to-path ratio (0.0 to 1.0)")
    parser.add_argument("--max-vel", type=float, default=150.0,
                        help="Maximum single-frame pixel jump velocity")
    parser.add_argument("--temp", type=float, default=1.4,
                        help="Temperature scaling factor for confidence calibration (>= 1.0)")
    parser.add_argument("--max-entropy", type=float, default=1.8,
                        help="Maximum entropy threshold for unknown detection")
    args = parser.parse_args()
    
    mapping_path = os.path.join("data", "processed", "label_mapping.json")
    
    # 1. Load Recognition Engine
    try:
        model, classes, idx_to_label = load_recognition_system(args.model, mapping_path)
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        print("Please verify your files or run: python src/train.py")
        return
        
    print(f"[Success] Recognition System Initialized. Confidence Threshold: {args.threshold * 100}%")
    
    # Load completeness statistics for post-prediction validation
    stats_path = os.path.join("data", "processed", "completeness_stats.json")
    if os.path.exists(stats_path):
        print(f"[Info] Loading character completeness reference stats from: {stats_path}")
        with open(stats_path, "r") as f:
            completeness_stats = json.load(f)
    else:
        print("[Warning] Completeness stats file not found. Post-prediction validation will be skipped.")
        completeness_stats = None
    
    # Initialize trackers
    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7    )
    trajectory = TrajectoryManager()
    
    # Open camera stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam camera capture.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State tracking variables
    writing_mode = False
    recorded_points = []
    gesture_state = "idle" # "idle", "writing", "waiting_for_index"
    
    # Predictions and history state
    latest_predicted_class = ""
    latest_confidence = 0.0
    latest_margin = 0.0
    latest_entropy = 0.0
    latest_dpr = 0.0
    validation_status = ""
    latest_nlp_result = None
    
    # Initialize Translation Engine & State Variables
    translation_engine = TranslationEngine()
    target_languages = [
        "English", "Hindi", "Kannada", "Tamil", "Telugu", "Malayalam", 
        "Marathi", "French", "German", "Korean", "Spanish", "Japanese"
    ]
    active_lang_idx = 0  # Default English
    latest_translation_result = None
    
    # Initialize Speech Generator & State Variables
    speech_generator = SpeechGenerator()
    latest_speech_result = None
    
    # Initialize SentenceBuilder for sentence and word formation
    sentence_builder = SentenceBuilder()
    
    print("\n" + "="*60)
    print("      AIR WRITING RECOGNITION INTERFACE IS ONLINE      ")
    print("="*60)
    print(" Controls:")
    print("   [Gesture: Index Only]       : Writing Mode (Draw stroke)")
    print("   [Gesture: Index + Middle]   : Finish Current Character")
    print("   [Gesture: Open Palm]        : Finish Current Word")
    print("   [Gesture: Thumbs Up]        : Finish Current Sentence")
    print("   [C / c]                     : Clear history and sentence")
    print("   [Backspace]                 : Delete last character")
    print("   [P / p]                     : Play/Replay latest speech")
    print("   [Spacebar]                  : Pause/Resume audio playback")
    print("   [S / s]                     : Stop audio playback")
    print("   [V / v]                     : Toggle speech autoplay")
    print("   [Q / q]                     : Quit application")
    print("="*60 + "\n")
    
    hand_present_prev = False
    prev_gesture = "other"
    
    while True:
        
        success, frame = cap.read()
        if not success:
            print("[Error] Failed to read webcam video frame.")
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Track hand landmarks
        frame, hand_detected = tracker.find_hands(frame, draw=True)
        
        if hand_detected:
            landmarks = tracker.get_landmarks(frame, hand_idx=0)
            gesture = detect_hand_gesture(landmarks)
            
            if gesture == "writing":
                # Start new character trajectory if transitioning from idle or waiting
                if gesture_state in ["idle", "waiting_for_index"]:
                    gesture_state = "writing"
                    writing_mode = True
                    recorded_points = []
                    trajectory.clear()
                    latest_predicted_class = ""
                    latest_confidence = 0.0
                    latest_margin = 0.0
                    latest_entropy = 0.0
                    latest_dpr = 0.0
                    validation_status = ""
                    print("[Info] Writing mode activated via Index gesture.")
                
                # Extract Index Finger Tip (Landmark 8)
                index_tip = tracker.get_landmark_by_id(landmarks, 8)
                if index_tip:
                    px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                    norm_x, norm_y, norm_z = index_tip["x"], index_tip["y"], index_tip["z"]
                    
                    # Render active pointer (Red fingertip)
                    cv2.circle(frame, (px_x, px_y), 8, (0, 0, 255), -1)
                    
                    # Record coordinate points
                    recorded_points.append([px_x, px_y, norm_x, norm_y, norm_z])
                    trajectory.add_point(px_x, px_y, norm_x, norm_y, norm_z)
                    
            elif gesture == "completion":
                # Stop and trigger character recognition if transitioning from writing state
                if gesture_state == "writing":
                    gesture_state = "waiting_for_index"
                    writing_mode = False
                    print(f"[Info] Character completion gesture detected (Index+Middle). Captured {len(recorded_points)} frames. Running inference...")
                    
                    # Check trajectory validation
                    is_valid, err_reason, dpr = validate_trajectory(
                        recorded_points,
                        min_len=args.min_len,
                        min_bbox_diagonal=args.min_bbox,
                        min_path_distance=args.min_dist,
                        horizontal_line_ratio=5.0,
                        min_height_for_2d=25.0,
                        min_dpr=args.min_dpr,
                        max_velocity=args.max_vel
                    )
                    
                    latest_dpr = dpr
                    
                    if not is_valid:
                        print(f"[Warning] Trajectory rejected: {err_reason}")
                        validation_status = f"Rejected: {err_reason}"
                        latest_predicted_class = "UNKNOWN"
                        latest_confidence = 0.0
                        latest_margin = 0.0
                        latest_entropy = 0.0
                        trajectory.clear()
                        recorded_points = []
                    else:
                        # Run inference pipeline
                        try:
                            raw_traj = np.array(recorded_points, dtype=np.float32)
                            
                            # Preprocess trajectory: Smooth -> Normalize -> Resample to 64
                            preprocessed = preprocess_single_trajectory(
                                raw_traj, 
                                target_len=64, 
                                smooth_window=3, 
                                mode='resample'
                            )
                            
                            # Add Batch Dimension: shape (1, 64, 3)
                            input_tensor = np.expand_dims(preprocessed, axis=0)
                            
                            # Model Predict using direct calling for speed
                            raw_preds = model(input_tensor, training=False).numpy()[0]
                            
                            # Apply Temperature Scaling Calibration
                            preds = apply_temperature_scaling(raw_preds, args.temp)
                            
                            # Calculate Shannon Entropy
                            entropy = -np.sum(preds * np.log(preds + 1e-8))
                            latest_entropy = float(entropy)
                            
                            sorted_idx = np.argsort(preds)
                            pred_idx = sorted_idx[-1]
                            second_idx = sorted_idx[-2]
                            
                            confidence = float(preds[pred_idx])
                            second_confidence = float(preds[second_idx])
                            margin = confidence - second_confidence
                            predicted_label = idx_to_label[pred_idx]
                            
                            # Overrule O, o, and 0 classification based on drawing orientation
                            # Clockwise drawing → digit zero ('0')
                            # Counter-clockwise drawing → letter O or o (determined by model's prediction)
                            if predicted_label in ['O', 'o', '0']:
                                orientation = calculate_trajectory_orientation(preprocessed)
                                if orientation > 0:
                                    # Clockwise → Zero digit '0'
                                    if predicted_label != '0':
                                        print(f"[Info] Overriding predicted '{predicted_label}' to '0' based on clockwise drawing direction (orientation={orientation:.4f})")
                                        predicted_label = '0'
                                else:
                                    # Counter-clockwise → Letter 'O' or 'o'
                                    # If model said '0', promote to uppercase 'O' (safer default for ambiguous CCW)
                                    if predicted_label == '0':
                                        print(f"[Info] Overriding predicted '0' to 'O' based on anti-clockwise drawing direction (orientation={orientation:.4f})")
                                        predicted_label = 'O'
                                    # If model said 'O' or 'o', keep that case preference as-is
                                        
                            original_label = predicted_label
                            
                            # --- GEOMETRIC AND SEMANTIC FILTERS ---
                            x_coords = [p[0] for p in recorded_points]
                            y_coords = [p[1] for p in recorded_points]
                            xmin, xmax = min(x_coords), max(x_coords)
                            ymin, ymax = min(y_coords), max(y_coords)
                            w = xmax - xmin
                            h = ymax - ymin
                            diagonal = np.sqrt(w**2 + h**2)
                            
                            disp_x = recorded_points[-1][0] - recorded_points[0][0]
                            disp_y = recorded_points[-1][1] - recorded_points[0][1]
                            displacement = np.sqrt(disp_x**2 + disp_y**2)
                            
                            # Calculate total path distance
                            total_dist = 0.0
                            for idx_p in range(1, len(recorded_points)):
                                dx_p = recorded_points[idx_p][0] - recorded_points[idx_p-1][0]
                                dy_p = recorded_points[idx_p][1] - recorded_points[idx_p-1][1]
                                total_dist += np.sqrt(dx_p**2 + dy_p**2)
                                
                            dpr_val = displacement / max(total_dist, 1.0)
                            
                            # Extract preprocessed geometric features
                            feats = extract_geometric_features(preprocessed)
                            
                            # Pre-calculate completeness for debugging output
                            if completeness_stats is not None and original_label in completeness_stats:
                                _, debug_pass_ratio, _ = validate_character(
                                    feats,
                                    completeness_stats[original_label],
                                    tolerance=0.25,
                                    min_pass_ratio=0.80
                                )
                            else:
                                debug_pass_ratio = 0.0
                                
                            # 1. Closed-loop validation for 'O', 'o', and '0' using stats-based threshold
                            is_closed_loop = True
                            loop_closure_reason = ""
                            if predicted_label in ['O', 'o', '0']:
                                loop_closure = feats["loop_closure"]
                                # Use class-specific stats if available, else fallback
                                if completeness_stats is not None and predicted_label in completeness_stats:
                                    stats = completeness_stats[predicted_label]["loop_closure"]
                                    loop_threshold = stats["p95"] + 0.30
                                elif completeness_stats is not None and 'O' in completeness_stats:
                                    # For 'o', fall back to 'O' stats if 'o' stats not yet computed
                                    stats = completeness_stats['O']["loop_closure"]
                                    loop_threshold = stats["p95"] + 0.30
                                else:
                                    loop_threshold = 0.40  # Fallback
                                    
                                is_closed_loop = (loop_closure <= loop_threshold)
                                if not is_closed_loop:
                                    loop_closure_reason = f"Loop closure check failed (ratio {loop_closure:.3f} > threshold {loop_threshold:.3f})"
                                    print(f"[Warning] Override '{predicted_label}' to 'UNKNOWN' - {loop_closure_reason}")
                                    predicted_label = "UNKNOWN"
                                    
                            # 2. Straight line check (e.g. diagonal/horizontal slashes or lines)
                            is_straight_line = (dpr_val > 0.90)
                            if is_straight_line and predicted_label != "UNKNOWN":
                                dx = recorded_points[-1][0] - recorded_points[0][0]
                                dy = recorded_points[-1][1] - recorded_points[0][1]
                                angle_with_vertical = np.degrees(np.arctan2(abs(dx), abs(dy)))
                                
                                # If it's a vertical-ish line, it must only be predicted as 'I', 'l', or '1'
                                # ('l' lowercase L has the same vertical stroke shape as 'I' and '1')
                                if angle_with_vertical <= 35.0:
                                    if predicted_label not in ['I', 'l', '1']:
                                        print(f"[Warning] Override '{predicted_label}' to 'UNKNOWN' - straight vertical line mapped to non-line class")
                                        predicted_label = "UNKNOWN"
                                else:
                                    # Horizontal or diagonal lines are not letters
                                    print(f"[Warning] Override '{predicted_label}' to 'UNKNOWN' - horizontal/diagonal straight line (angle={angle_with_vertical:.1f} deg)")
                                    predicted_label = "UNKNOWN"
                            
                            # 3. Character Completeness Validation (only for non-UNKNOWN predictions)
                            is_complete = True
                            completeness_reason = ""
                            if predicted_label != "UNKNOWN" and completeness_stats is not None:
                                # Apply wider tolerances and lower pass ratio for lowercase letters to prevent tremor rejections
                                is_lowercase = predicted_label.islower() and predicted_label.isalpha()
                                tol = 0.40 if is_lowercase else 0.25
                                min_pr = 0.70 if is_lowercase else 0.80
                                
                                is_complete, pass_ratio, fails = validate_character(
                                    feats, 
                                    completeness_stats.get(predicted_label, {}), 
                                    tolerance=tol, 
                                    min_pass_ratio=min_pr
                                )
                                if not is_complete:
                                    completeness_reason = f"Incomplete Gesture (features pass: {pass_ratio*100:.0f}%, failed: {', '.join(fails)})"
                            
                            # Check confidence, margin, and entropy thresholding
                            if original_label in ['O', 'o', '0'] and not is_closed_loop:
                                validation_status = f"Rejected: {loop_closure_reason}"
                                latest_predicted_class = "UNKNOWN"
                                latest_confidence = 0.0
                                latest_margin = 0.0
                                print_rejected_zero_debug(
                                    original_label, confidence, margin, latest_entropy, feats,
                                    displacement, diagonal, w, h, loop_closure_reason, pass_ratio=debug_pass_ratio
                                )
                            elif predicted_label == "UNKNOWN":
                                validation_status = ""
                                latest_predicted_class = "UNKNOWN"
                                latest_confidence = confidence
                                latest_margin = margin
                            elif not is_complete:
                                reject_reason = completeness_reason
                                print(f"[Warning] Prediction '{predicted_label}' ignored: {reject_reason}")
                                validation_status = f"Rejected: {reject_reason}"
                                latest_predicted_class = "UNKNOWN"
                                latest_confidence = 0.0
                                latest_margin = 0.0
                                if original_label in ['O', 'o', '0']:
                                    print_rejected_zero_debug(
                                        original_label, confidence, margin, latest_entropy, feats,
                                        displacement, diagonal, w, h, reject_reason, pass_ratio=debug_pass_ratio
                                    )
                            elif confidence < CLASS_THRESHOLDS.get(predicted_label, args.threshold):
                                req_thresh = CLASS_THRESHOLDS.get(predicted_label, args.threshold)
                                reject_reason = f"Low Confidence ({confidence*100:.1f}% < {req_thresh*100:.0f}%)"
                                print(f"[Warning] Prediction '{predicted_label}' ignored: {reject_reason}")
                                validation_status = f"Rejected: {reject_reason}"
                                latest_predicted_class = "UNKNOWN"
                                latest_confidence = 0.0
                                latest_margin = 0.0
                                if original_label in ['O', 'o', '0']:
                                    print_rejected_zero_debug(
                                        original_label, confidence, margin, latest_entropy, feats,
                                        displacement, diagonal, w, h, reject_reason, pass_ratio=debug_pass_ratio
                                    )
                            elif margin < args.margin:
                                reject_reason = f"Low Margin ({margin*100:.1f}% < {args.margin*100:.0f}%)"
                                print(f"[Warning] Prediction '{predicted_label}' ignored: {reject_reason}")
                                validation_status = f"Rejected: {reject_reason}"
                                latest_predicted_class = "UNKNOWN"
                                latest_confidence = 0.0
                                latest_margin = 0.0
                                if original_label in ['O', 'o', '0']:
                                    print_rejected_zero_debug(
                                        original_label, confidence, margin, latest_entropy, feats,
                                        displacement, diagonal, w, h, reject_reason, pass_ratio=debug_pass_ratio
                                    )
                            elif entropy > args.max_entropy:
                                reject_reason = f"High Entropy ({entropy:.2f} > {args.max_entropy})"
                                print(f"[Warning] Prediction '{predicted_label}' ignored: {reject_reason}")
                                validation_status = f"Rejected: {reject_reason}"
                                latest_predicted_class = "UNKNOWN"
                                latest_confidence = 0.0
                                latest_margin = 0.0
                                if original_label in ['O', 'o', '0']:
                                    print_rejected_zero_debug(
                                        original_label, confidence, margin, latest_entropy, feats,
                                        displacement, diagonal, w, h, reject_reason, pass_ratio=debug_pass_ratio
                                    )
                            else:
                                validation_status = ""
                                latest_predicted_class = predicted_label
                                latest_confidence = confidence
                                latest_margin = margin
                                sentence_builder.append_character(predicted_label)
                                print(f"[Success] Predicted: '{predicted_label}' | Calibrated Confidence: {confidence*100:.1f}% | Margin: {margin*100:.1f}% | Entropy: {entropy:.2f}")
                                
                        except Exception as e:
                            print(f"[Error] Inference failed: {e}")
                            validation_status = "Rejected: Inference Error"
                            
                        # Reset points and canvas
                        recorded_points = []
                        trajectory.clear()
                else:
                    # In standby / waiting state, render pointer (Green pointer)
                    index_tip = tracker.get_landmark_by_id(landmarks, 8)
                    if index_tip:
                        px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                        cv2.circle(frame, (px_x, px_y), 8, (0, 255, 0), -1)
                        
            elif gesture == "open_palm":
                # Trigger finish word only on state transition to open_palm
                if prev_gesture != "open_palm":
                    if sentence_builder.get_current_word():
                        print(f"[Info] Open Palm gesture detected. Finalizing word '{sentence_builder.get_current_word()}'...")
                        sentence_builder.finish_word()
                    gesture_state = "waiting_for_index"
                    writing_mode = False
                    recorded_points = []
                    trajectory.clear()
                
                # Show green pointer for standby
                index_tip = tracker.get_landmark_by_id(landmarks, 8)
                if index_tip:
                    px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                    cv2.circle(frame, (px_x, px_y), 8, (0, 255, 0), -1)
                    
            elif gesture == "thumbs_up":
                # Trigger finish sentence only on state transition to thumbs_up
                if prev_gesture != "thumbs_up":
                    if sentence_builder.get_current_sentence() or sentence_builder.get_current_word():
                        if sentence_builder.get_current_word():
                            sentence_builder.finish_word()
                        latest_nlp_result = sentence_builder.finish_sentence()
                        if latest_nlp_result and latest_nlp_result.get("corrected"):
                            target_lang = target_languages[active_lang_idx]
                            latest_translation_result = translation_engine.translate(
                                latest_nlp_result["corrected"], 
                                target_lang
                            )
                            print(f"[Translation] {target_lang} -> {latest_translation_result['translated']}")
                            
                            # Generate Speech for the translated output
                            latest_speech_result = speech_generator.generate_speech(
                                latest_translation_result["translated"],
                                target_lang
                            )
                        
                    gesture_state = "idle"
                    writing_mode = False
                    recorded_points = []
                    trajectory.clear()
                    latest_predicted_class = ""
                    latest_confidence = 0.0
                    latest_margin = 0.0
                    latest_entropy = 0.0
                    latest_dpr = 0.0
                    validation_status = ""
                
                # Show green pointer for standby
                index_tip = tracker.get_landmark_by_id(landmarks, 8)
                if index_tip:
                    px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                    cv2.circle(frame, (px_x, px_y), 8, (0, 255, 0), -1)
            else:
                # Other gesture: standby green pointer, trigger new stroke if writing was active
                index_tip = tracker.get_landmark_by_id(landmarks, 8)
                if index_tip:
                    px_x, px_y = index_tip["px_x"], index_tip["px_y"]
                    cv2.circle(frame, (px_x, px_y), 8, (0, 255, 0), -1)
                if gesture_state == "writing":
                    trajectory.trigger_new_stroke()
                    
            prev_gesture = gesture
            hand_present_prev = True
        else:
            prev_gesture = "other"
            # Hand lost: trigger new stroke if writing was active
            if hand_present_prev:
                if gesture_state == "writing":
                    trajectory.trigger_new_stroke()
                hand_present_prev = False

        # Render the drawing canvas overlay
        frame = trajectory.draw_trajectory(frame, color=(0, 0, 255), thickness=5)
        
        # Render HUD elements
        frame = draw_hud(
            frame, 
            writing_mode, 
            latest_predicted_class, 
            latest_confidence, 
            sentence_builder.get_history(),
            args.threshold,
            validation_status=validation_status,
            margin=latest_margin,
            margin_threshold=args.margin,
            entropy=latest_entropy,
            max_entropy=args.max_entropy,
            dpr=latest_dpr,
            current_word=sentence_builder.get_current_word(),
            current_sentence=sentence_builder.get_current_sentence(),
            nlp_result=latest_nlp_result,
            translation_result=latest_translation_result,
            target_lang=target_languages[active_lang_idx],
            speech_result=latest_speech_result,
            speech_generator=speech_generator
        )
        
        # Display composite frame
        cv2.imshow("Air-Writing Real-Time Recognition", frame)
        
        # Keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Manual keys:
        if key == ord('c') or key == ord('C'):
            gesture_state = "idle"
            writing_mode = False
            trajectory.clear()
            recorded_points = []
            latest_predicted_class = ""
            latest_confidence = 0.0
            latest_margin = 0.0
            latest_entropy = 0.0
            latest_dpr = 0.0
            validation_status = ""
            latest_nlp_result = None
            latest_translation_result = None
            if speech_generator:
                speech_generator.stop_audio()
            latest_speech_result = None
            sentence_builder.clear()
            print("[Info] Canvas, active prediction, and sentence history cleared.")
            
        elif (key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'), ord('0')]) or (key in [ord('l'), ord('L')]):
            if key == ord('0'):
                active_lang_idx = 9  # Korean
            elif key in [ord('l'), ord('L')]:
                active_lang_idx = (active_lang_idx + 1) % len(target_languages)
            else:
                active_lang_idx = key - ord('1')
                
            target_lang = target_languages[active_lang_idx]
            print(f"[Info] Target translation language switched to: {target_lang}")
            if latest_nlp_result and latest_nlp_result.get("corrected"):
                latest_translation_result = translation_engine.translate(
                    latest_nlp_result["corrected"], 
                    target_lang
                )
                print(f"[Translation] {target_lang} -> {latest_translation_result['translated']}")
                
                # Re-generate speech dynamically on target language change
                latest_speech_result = speech_generator.generate_speech(
                    latest_translation_result["translated"],
                    target_lang
                )
                
        elif key == ord('v') or key == ord('V'):
            new_autoplay = not speech_generator.get_autoplay()
            speech_generator.set_autoplay(new_autoplay)
            print(f"[Info] Speech Autoplay set to: {new_autoplay}")
            
        elif key == ord('p') or key == ord('P'):
            if latest_speech_result and latest_speech_result.get("success"):
                speech_generator.play_audio()
                print("[Info] Replaying latest sentence speech...")
            else:
                print("[Warning] No valid sentence audio available to play.")
                
        elif key == ord('s') or key == ord('S'):
            speech_generator.stop_audio()
            print("[Info] Audio playback stopped.")
            
        elif key == 32:  # Spacebar
            if speech_generator:
                speech_generator.pause_audio()
            
        elif key == 8:
            # Backspace - Delete last character in current word, or sentence
            sentence_builder.delete_last_character()
            print(f"[Info] Deleted last character. Current word: '{sentence_builder.get_current_word()}', Current sentence: '{sentence_builder.get_current_sentence()}'")
                
        elif key == ord('q') or key == ord('Q'):
            print("[Info] Terminating recognition session...")
            break
            
    # Cleanup resources
    if speech_generator:
        speech_generator.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main()