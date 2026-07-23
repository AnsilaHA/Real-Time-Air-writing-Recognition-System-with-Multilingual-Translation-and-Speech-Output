

# Real-Time Air-Writing Recognition System with Multilingual Translation and Speech Output

A real-time AI-powered air-writing recognition system that enables users to write English alphanumeric characters in the air using a standard webcam. The system recognizes air-written text, forms words and sentences, performs spell correction, translates the recognized text into multiple languages, and converts it into speech.

## Features

- Real-time air-writing recognition using a webcam
- Hand tracking using MediaPipe Hands
- Character recognition using LSTM and GRU deep learning models
- Word and sentence formation
- Automatic spell correction using SymSpell
- Multilingual text translation
- Text-to-Speech (TTS) support
- Interactive Streamlit web interface
- Translation history management

## Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Streamlit
- NumPy
- SymSpell
- Deep Translator
- Meta NLLB-200 (Offline Translation)
- gTTS
- pyttsx3


## System Workflow

1. Capture live video using webcam.
2. Detect hand landmarks with MediaPipe.
3. Record fingertip trajectory during air writing.
4. Preprocess trajectory data.
5. Recognize characters using trained LSTM/GRU models.
6. Form words and sentences.
7. Correct spelling using SymSpell.
8. Translate text into the selected language.
9. Convert translated text into speech.
10. Display the output in the Streamlit interface.

## Dataset

- Custom air-writing trajectory dataset
- 62 classes:
  - Uppercase letters (A–Z)
  - Lowercase letters (a–z)
  - Digits (0–9)

### Dataset Statistics

- Total Samples: **3,238**
- Training: **2,265**
- Validation: **455**
- Testing: **518**

## Model Performance

### Test Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| LSTM | 98.84% | 98.95% | 98.84% | 98.84% |
| GRU | 97.49% | 97.75% | 97.49% | 97.46% |

### Cross-Validation Performance

| Model | Mean Accuracy | Standard Deviation |
|--------|---------------|--------------------|
| LSTM | 96.29% | 2.07% |
| GRU | 97.26% | 1.13% |

## Installation

### Navigate to Project Folder

```bash
cd Real-Time-Air-Writing-Recognition-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

## Applications

- Touchless Human-Computer Interaction
- Assistive Communication
- Smart Education
- Multilingual Communication
- Healthcare
- Public Interactive Systems

## Future Enhancements

- Continuous handwriting recognition
- Support for multiple users
- Mobile application deployment
- Personalized user adaptation
- Native multilingual script recognition
- Mathematical symbol recognition
- Edge device optimization
- Improved low-light hand tracking

## Author

**Ansila H A**

MCA Student  
Parivarthana Business School, Mysore

## License

This project is developed for academic and educational purposes.
