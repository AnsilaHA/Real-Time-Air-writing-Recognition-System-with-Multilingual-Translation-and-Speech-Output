import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Setup Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "speech_generation.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SpeechGenerator")

# Check imports
HAS_PYTTSX3 = False
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    logger.warning("pyttsx3 not installed. Offline TTS fallback disabled.")

HAS_GTTS = False
try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    logger.warning("gTTS not installed. Online TTS disabled.")

HAS_PYGAME = False
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    logger.warning("pygame not installed. Audio playback disabled.")


def is_internet_available() -> bool:
    """
    Checks if there is an active internet connection.
    Tests TCP connection to Google Public DNS (8.8.8.8) on port 53.
    """
    import socket
    try:
        socket.setdefaulttimeout(1.0)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 53))
        s.close()
        return True
    except Exception:
        return False


def find_local_voice(engine, lang_code: str):
    """
    Finds an installed SAPI5 voice matching the target language code.
    Matches against languages metadata, voice name, and voice ID keywords.
    """
    try:
        voices = engine.getProperty('voices')
        lang_code = lang_code.lower().strip()
        
        # Mappings for common language name matching in voice metadata
        lang_keywords = {
            "en": ["english", "en-us", "en-gb", "zira", "david", "hazel"],
            "fr": ["french", "fr-fr", "fr_", "hortense"],
            "de": ["german", "de-de", "de_", "hedda"],
            "es": ["spanish", "es-es", "es-co", "es_", "helena", "sabella"],
            "hi": ["hindi", "hi-in", "hi_", "kalpana", "hemant"],
            "kn": ["kannada", "kn-in", "kn_"],
            "ta": ["tamil", "ta-in", "ta_"],
            "te": ["telugu", "te-in", "te_"],
            "ml": ["malayalam", "ml-in", "ml_"],
            "mr": ["marathi", "mr-in", "mr_"],
            "ja": ["japanese", "ja-jp", "ja_", "haruka"],
            "ko": ["korean", "ko-kr", "ko_", "heami"]
        }
        
        keywords = lang_keywords.get(lang_code, [lang_code])
        
        # First pass: Check strict language code match in voice languages list
        for voice in voices:
            if hasattr(voice, "languages") and voice.languages:
                for lang in voice.languages:
                    clean_lang = str(lang).replace("_", "-").split("-")[0].lower()
                    if clean_lang == lang_code:
                        return voice
                        
        # Second pass: Check keyword matches in voice name or ID
        for voice in voices:
            name_lower = voice.name.lower()
            id_lower = voice.id.lower()
            for kw in keywords:
                if kw in name_lower or kw in id_lower:
                    return voice
                    
        return None
    except Exception as e:
        logger.error(f"Error scanning local voices: {e}")
        return None


class SpeechGenerator:
    """
    Modular Text-to-Speech (TTS) engine.
    Supports offline (pyttsx3) and online (gTTS) for multilingual targets.
    Integrates background non-blocking playback via pygame.
    """
    def __init__(self, output_dir: str = "output_audio", autoplay: bool = True):
        self.output_dir = output_dir
        self.autoplay_enabled = autoplay
        self.last_audio_path: Optional[str] = None
        self.last_metadata: Dict[str, Any] = {}
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Verify pyttsx3 is available offline
        self.pyttsx3_available = HAS_PYTTSX3

        # Initialize pygame mixer for non-blocking audio play
        self.pygame_mixer_initialized = False
        self.is_paused = False
        if HAS_PYGAME:
            try:
                pygame.mixer.init()
                self.pygame_mixer_initialized = True
                logger.info("pygame mixer initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize pygame mixer: {e}")

        # Language ISO Mappings (gTTS requires standard ISO codes)
        self.lang_codes = {
            "english": "en",
            "hindi": "hi",
            "kannada": "kn",
            "tamil": "ta",
            "telugu": "te",
            "malayalam": "ml",
            "marathi": "mr",
            "french": "fr",
            "german": "de",
            "korean": "ko",
            "spanish": "es",
            "japanese": "ja"
        }

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Returns supported languages with their mapped ISO codes.
        """
        return self.lang_codes.copy()

    def generate_speech(self, text: str, language_name: str) -> Dict[str, Any]:
        """
        Converts translated text into natural speech using pyttsx3 or gTTS.
        Saves files inside output_dir/ with unique timestamps.
        """
        start_time = time.time()
        result = {
            "success": False,
            "audio_path": "",
            "engine": "None",
            "generation_time": 0.0,
            "language": language_name,
            "error": ""
        }

        # 1. Validation checks
        if not text or not text.strip():
            result["error"] = "Empty input text."
            logger.warning("Attempted to generate speech with empty text.")
            return result

        lang_lower = language_name.lower().strip()
        lang_code = self.lang_codes.get(lang_lower)
        if not lang_code:
            result["error"] = f"Unsupported language: {language_name}"
            logger.warning(f"Unsupported language requested: {language_name}")
            return result

        # 2. Determine Speech Engine and verify local voice availability
        engine_to_use = "gTTS"
        online = is_internet_available()

        has_local_voice = False
        matched_voice_id = None
        if HAS_PYTTSX3:
            try:
                temp_engine = pyttsx3.init()
                matched_voice = find_local_voice(temp_engine, lang_code)
                if matched_voice:
                    has_local_voice = True
                    matched_voice_id = matched_voice.id
                temp_engine.stop()
                del temp_engine
            except Exception as e:
                logger.warning(f"Failed to scan local SAPI5 voices: {e}")

        # Choose the engine
        if online:
            # gTTS is primary for all languages in online mode
            engine_to_use = "gTTS"
            logger.info("INFO: Using gTTS")
        else:
            # Offline mode: must use pyttsx3 if local voice is installed, otherwise warn
            if has_local_voice:
                engine_to_use = "pyttsx3"
                logger.info("INFO: Using pyttsx3")
            else:
                logger.warning(f"WARNING: No offline voice installed for {language_name}")
                result["error"] = "Offline speech is unavailable for this language."
                self.last_metadata = result
                return result

        # Generate unique file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "wav" if engine_to_use == "pyttsx3" else "mp3"
        file_name = f"speech_{timestamp}.{ext}"
        audio_path = os.path.abspath(os.path.join(self.output_dir, file_name))

        # 3. Speech synthesis
        try:
            if engine_to_use == "pyttsx3":
                logger.info(f"Synthesizing offline {language_name} speech using pyttsx3 to: {file_name}")
                engine = pyttsx3.init()
                if matched_voice_id:
                    try:
                        engine.setProperty('voice', matched_voice_id)
                    except Exception as voice_err:
                        logger.warning(f"Could not configure custom voice: {voice_err}")
                
                engine.save_to_file(text, audio_path)
                engine.runAndWait()
                
                # Release pyttsx3 engine resources to avoid COM state lock
                try:
                    engine.stop()
                except Exception:
                    pass
                del engine
            else:
                logger.info(f"Synthesizing online {language_name} speech using gTTS to: {file_name}")
                # Check internet or catch connection exceptions gracefully
                try:
                    tts = gTTS(text=text, lang=lang_code, slow=False)
                    tts.save(audio_path)
                except Exception as gtts_err:
                    if HAS_PYTTSX3 and has_local_voice:
                        logger.warning(f"gTTS failed: {gtts_err}. Falling back to pyttsx3 offline engine.")
                        engine_to_use = "pyttsx3"
                        # Re-route file extension to .wav for pyttsx3 output
                        file_name = file_name.rsplit(".", 1)[0] + ".wav"
                        audio_path = os.path.abspath(os.path.join(self.output_dir, file_name))
                        
                        engine = pyttsx3.init()
                        if matched_voice_id:
                            try:
                                engine.setProperty('voice', matched_voice_id)
                            except Exception:
                                pass
                        engine.save_to_file(text, audio_path)
                        engine.runAndWait()
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        del engine
                        result["engine"] = "pyttsx3"
                        logger.info("INFO: Using pyttsx3")
                    else:
                        logger.warning(f"WARNING: No offline voice installed for {language_name}")
                        raise RuntimeError("Offline speech is unavailable for this language.")

            generation_time = (time.time() - start_time) * 1000 # ms
            
            result.update({
                "success": True,
                "audio_path": audio_path,
                "engine": engine_to_use,
                "generation_time": generation_time
            })
            
            self.last_audio_path = audio_path
            self.last_metadata = result
            logger.info(f"Successfully generated speech in {generation_time:.1f}ms via {engine_to_use}.")

            # 4. Optional Autoplay
            if self.autoplay_enabled:
                self.play_audio(audio_path)

        except Exception as e:
            result["error"] = str(e)
            self.last_metadata = result
            logger.error(f"Speech generation failed: {e}")

        return result

    def play_audio(self, file_path: Optional[str] = None) -> bool:
        """
        Plays target speech audio file in background (non-blocking) using pygame.
        """
        target_path = file_path or self.last_audio_path
        if not target_path or not os.path.exists(target_path):
            logger.warning("No valid audio file to play.")
            return False

        if not HAS_PYGAME or not self.pygame_mixer_initialized:
            logger.warning("Cannot play audio: pygame mixer is not available.")
            return False

        try:
            # Stop any currently playing audio and unload to release OS locks
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.unload()

            # Load and play the new file
            pygame.mixer.music.load(target_path)
            pygame.mixer.music.play()
            self.is_paused = False
            logger.info(f"Started playback: {os.path.basename(target_path)}")
            return True
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
            return False

    def stop_audio(self) -> None:
        """
        Stops active audio playback.
        """
        if HAS_PYGAME and self.pygame_mixer_initialized:
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    logger.info("Playback stopped by user.")
                pygame.mixer.music.unload()
                self.is_paused = False
            except Exception as e:
                logger.error(f"Failed to stop playback: {e}")

    def pause_audio(self) -> bool:
        """
        Pauses or resumes active audio playback (toggles pause state).
        Returns True if successful, False otherwise.
        """
        if not HAS_PYGAME or not self.pygame_mixer_initialized:
            logger.warning("Cannot pause audio: pygame mixer is not available.")
            return False

        try:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                logger.info("Playback resumed.")
                return True
            else:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                    self.is_paused = True
                    logger.info("Playback paused.")
                    return True
                else:
                    logger.warning("No active audio playing to pause.")
                    return False
        except Exception as e:
            logger.error(f"Failed to toggle pause: {e}")
            return False

    def replay_audio(self) -> bool:
        """
        Replays the last generated audio file.
        """
        if self.last_audio_path:
            logger.info("Replaying last generated audio...")
            return self.play_audio(self.last_audio_path)
        logger.warning("No audio history available to replay.")
        return False

    def save_audio(self, text: str, language_name: str, file_name: Optional[str] = None) -> str:
        """
        Explicitly generates and saves audio with option to set a custom file name.
        """
        res = self.generate_speech(text, language_name)
        if not res["success"]:
            return ""
        
        orig_path = res["audio_path"]
        if file_name:
            custom_path = os.path.abspath(os.path.join(self.output_dir, file_name))
            try:
                # Rename the file if custom name is requested
                if os.path.exists(custom_path):
                    os.remove(custom_path)
                os.rename(orig_path, custom_path)
                res["audio_path"] = custom_path
                self.last_audio_path = custom_path
                logger.info(f"Saved audio renamed to custom file: {file_name}")
                return custom_path
            except Exception as e:
                logger.error(f"Failed to rename audio file: {e}")
                return orig_path
        return orig_path

    def get_autoplay(self) -> bool:
        """
        Returns autoplay enabled/disabled state.
        """
        return self.autoplay_enabled

    def set_autoplay(self, enabled: bool) -> None:
        """
        Enables or disables autoplay of generated speech.
        """
        self.autoplay_enabled = enabled
        logger.info(f"Autoplay toggled to: {enabled}")

    def get_playback_status(self) -> str:
        """
        Checks if audio is currently playing in background.
        Returns: 'Playing', 'Paused', 'Idle', 'Unavailable', or last error message.
        """
        if self.last_metadata and not self.last_metadata.get("success", False):
            err_msg = self.last_metadata.get("error", "")
            if err_msg:
                return err_msg

        if not HAS_PYGAME or not self.pygame_mixer_initialized:
            return "Unavailable"
        if self.is_paused:
            return "Paused"
        return "Playing" if pygame.mixer.music.get_busy() else "Idle"

    def cleanup(self) -> None:
        """
        Stops mixer and unloads resources.
        """
        self.stop_audio()
        if HAS_PYGAME and self.pygame_mixer_initialized:
            try:
                pygame.mixer.quit()
                self.pygame_mixer_initialized = False
                logger.info("pygame mixer quit completed.")
            except Exception as e:
                logger.error(f"Failed to quit pygame mixer: {e}")

