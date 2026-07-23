# Air-Writing Recognition System with Multilingual Speech Synthesis

An advanced real-time gesture-recognition application that enables users to write characters in the air using their webcam. The system captures finger trajectories, classifies characters using a deep Gated Recurrent Unit (GRU) neural network, applies NLP spell/grammar corrections, translates sentences into 12 target languages, and synthesizes speech output.

This project features a professional local **Streamlit Dashboard** alongside the classic OpenCV terminal-based UI.

---

## 🚀 Key Features

* **Phase 1-4: Hand Tracking & Trajectory Capture**: Powered by MediaPipe Hand Landmarking (Landmark 8 - Index Finger Tip).
* **Phase 5-7: Character Recognition & Spatial Overrules**: Resamples coordinate pathways to `(1, 64, 3)` sequences and runs inference via a trained GRU model in Keras. Automatically calibrates confidence scores using Temperature Scaling. Overrules circular ambiguities (CW/CCW Shoelace area checks for `0` vs `O`).
* **Phase 8: NLP Post-Processing**: Spell-checks compiled sentences using SymSpell, handles contractions, corrects subject-verb agreements, and restores capitalization and terminal punctuation.
* **Phase 9: Multilingual Translation**: Dual offline/online router supporting 12 languages: English, Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, French, German, Korean, Spanish, and Japanese.
* **Phase 10: Speech Synthesis**: Modular Text-to-Speech (TTS) engine using offline `pyttsx3` and online `gTTS` with background pygame music playback (pause, resume, stop, and play controls).
* **Phase 11: Streamlit Integration**: A premium, single-viewport 30/70 double-column dark-themed SaaS dashboard (using Poppins typography and a minimal color scheme) providing title, character recognition results, live text streams, AI enhancements, and translation settings on the left; and a dominant webcam view, compact playback audio controllers, and scrollable recent history cards on the right.

---

## 🛠️ Technology Stack

* **Front-end**: Streamlit (Dashboard GUI) / OpenCV (HUD Render Window)
* **Hand Tracking**: Google MediaPipe (Hand Landmarker API)
* **Sequence Classification**: TensorFlow / Keras (GRU Network)
* **NLP Processing**: SymSpell (minimum edit distance spelling)
* **Translation**: Deep Translator / Googletrans (Online) & Argos / MarianMT (Offline)
* **Audio playback**: Pygame (Non-blocking music streaming)
* **TTS Engines**: pyttsx3 (SAPI5 COM Windows driver) & gTTS (Google Web Wrapper)

---

## 📁 Workspace Directory Structure

```
Air_Writing_Recognition/
│
├── app.py                      # Main Streamlit Dashboard Entrypoint
├── requirements.txt            # System dependencies manifest
├── README.md                   # Main system handbook
│
├── src/                        # Modular Source Code Backend
│   ├── app.py                  # OpenCV CLI Entrypoint (Phase 1 Sandbox)
│   ├── predict.py              # OpenCV CLI Sentence Inference Entrypoint
│   ├── hand_tracker.py         # MediaPipe tracking logic
│   ├── trajectory_manager.py   # Coordinate accumulation and canvas drawing
│   ├── preprocess.py           # Scaling, resample-interpolation, smoothing
│   ├── sentence_builder.py     # Compile characters -> words -> sentences
│   ├── nlp_pipeline.py         # Orchestrates spelling and grammar correction
│   ├── translation.py          # Dual offline/online translation routing
│   ├── speech.py               # Text-to-speech SAPI5/gTTS engine (Pause/Resume/Stop)
│   └── ...                     
│
├── docs/                       # Project Design and Reference Guidelines
│   ├── streamlit_architecture.md # Streamlit pipeline diagrams and sequence lifecycles
│   ├── viva_qa.md              # Project defense questions and answers
│   └── ...
│
├── scratch/                    # Test Verification Scripts
│   ├── test_speech.py          # Unit tests verifying speech pause/play states
│   └── ...
│
├── data/                       # Dataset configurations and mapping keys
└── models/                     # GRU saved weights (.keras format)
```

---

## ⚙️ Setup and Deployment Guide

### Prerequisites
* Windows 10/11 (required for SAPI5 pyttsx3 offline audio drivers).
* Python 3.9 - 3.11.
* A connected USB or integrated webcam.

### Installation
1. Open PowerShell or Command Prompt in the project folder:
   ```bash
   cd d:\My____Projects\Air_Writing_Recognition
   ```
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install system dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
* **Streamlit Professional GUI (Recommended)**:
  ```bash
  streamlit run app.py
  ```
* **Classic OpenCV GUI Interface**:
  ```bash
  python src/predict.py
  ```

---

## 🧪 Verification & Testing

Verify that your hardware and modules work properly by executing the unit tests:
```bash
python scratch/test_speech.py
```
This runs 26 test cases verifying speech generation, language synthesis metrics, and correct toggle pause/resume states.
