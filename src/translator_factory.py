import logging
from typing import Optional, Dict, Any

# Configure logger
logger = logging.getLogger("TranslatorFactory")

# Check backend availability
import importlib.util

# Check backend availability without executing imports (prevents Streamlit watcher crashes)
HAS_ARGOS = importlib.util.find_spec("argostranslate") is not None

HAS_NLLB = (
    importlib.util.find_spec("transformers") is not None and
    importlib.util.find_spec("torch") is not None and
    importlib.util.find_spec("sentencepiece") is not None
)

HAS_DEEP_TRANS = importlib.util.find_spec("deep_translator") is not None

HAS_GOOGLETRANS = False
if importlib.util.find_spec("googletrans") is not None:
    try:
        from googletrans import Translator
        HAS_GOOGLETRANS = True
    except Exception:
        pass


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


# Map supported languages to standard NLLB-200 script codes
NLLB_LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ko": "kor_Hang",
    "es": "spa_Latn",
    "ja": "jpn_Jpan"
}


class TranslatorFactory:
    """
    Loads and configures translation backends based on availability and priority:
    NLLB -> Deep Translator (Google) -> GoogleTrans -> Static Fail-safe.
    Supports runtime backend switching.
    """
    def __init__(self):
        self.active_backend = self._determine_default_backend()
        self.nllb_model = None
        self.nllb_tokenizer = None
        logger.info(f"TranslatorFactory initialized. Default backend: {self.active_backend}")

    def _determine_default_backend(self) -> str:
        """
        Determines the highest priority available backend based on internet state.
        """
        if is_internet_available():
            if HAS_DEEP_TRANS:
                return "DeepTranslator"
            elif HAS_GOOGLETRANS:
                return "GoogleTrans"
        
        if HAS_ARGOS:
            return "Argos"
        elif HAS_NLLB:
            return "NLLB"
        return "CacheOnly"

    def get_supported_backends(self) -> list:
        """
        Returns list of all available translation backends in the environment.
        """
        backends = []
        if HAS_ARGOS:
            backends.append("Argos")
        if HAS_NLLB:
            backends.append("NLLB")
        if HAS_DEEP_TRANS:
            backends.append("DeepTranslator")
        if HAS_GOOGLETRANS:
            backends.append("GoogleTrans")
        backends.append("CacheOnly")
        return backends

    def get_backends_in_priority_order(self) -> list:
        """
        Returns a list of available backends sorted by priority depending on current internet status.
        Google Translate (online) takes priority if connected, falling back to local models.
        """
        online_backends = []
        if HAS_DEEP_TRANS:
            online_backends.append("DeepTranslator")
        if HAS_GOOGLETRANS:
            online_backends.append("GoogleTrans")

        offline_backends = []
        if HAS_ARGOS:
            offline_backends.append("Argos")
        if HAS_NLLB:
            offline_backends.append("NLLB")

        if is_internet_available():
            backends = online_backends + offline_backends
        else:
            backends = offline_backends + online_backends

        backends.append("CacheOnly")
        return backends

    def set_backend(self, backend_name: str) -> bool:
        """
        Switches active translation backend. Returns True if switch succeeded.
        """
        supported = self.get_supported_backends()
        if backend_name in supported:
            self.active_backend = backend_name
            logger.info(f"Translation backend switched to: {backend_name}")
            return True
        logger.warning(f"Backend '{backend_name}' not available. Active remains: {self.active_backend}")
        return False

    def get_backend(self) -> str:
        """
        Returns active translation backend name.
        """
        return self.active_backend

    def translate_via_active_backend(self, text: str, target_lang: str) -> Optional[str]:
        """
        Routes the translation query to the active backend.
        """
        if not text:
            return ""
            
        target_lang = target_lang.lower().strip()
        backend = self.active_backend

        # 1. Argos Translate (Offline Local Model)
        if backend == "Argos":
            try:
                import argostranslate.translate
                # Target language standard codes mapping check
                translated = argostranslate.translate.translate(text, "en", target_lang)
                if translated:
                    return translated
            except Exception as e:
                logger.error(f"Argos Translate failed: {e}. Falling back...")
                
        # 2. NLLB (Offline Local Hugging Face Model)
        elif backend == "NLLB" and HAS_NLLB:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                import os
                
                # Check for locally saved models inside 'models/nllb'
                model_dir = os.path.abspath(os.path.join("models", "nllb"))
                if not os.path.exists(model_dir):
                    logger.error(f"Local NLLB model directory not found: {model_dir}")
                    return f"ERROR_MODEL_MISSING:{model_dir}"

                if self.nllb_model is None or self.nllb_tokenizer is None:
                    logger.info("INFO: Using Offline Translation")
                    logger.info(f"Loading local NLLB model from: {model_dir}...")
                    self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
                    logger.info("INFO: Offline model loaded successfully")
                
                # Target language mapping check
                target_lang_token = NLLB_LANG_MAP.get(target_lang)
                if not target_lang_token:
                    logger.warning(f"Offline translation unavailable for language code '{target_lang}'")
                    return f"ERROR_LANG_UNSUPPORTED:{target_lang}"
                
                # Run local inference
                self.nllb_tokenizer.src_lang = "eng_Latn"
                inputs = self.nllb_tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    translated_tokens = self.nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=self.nllb_tokenizer.convert_tokens_to_ids(target_lang_token),
                        max_length=128
                    )
                translated = self.nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                if translated:
                    return translated
            except Exception as e:
                logger.error(f"NLLB failed: {e}. Falling back...")

        # 3. Deep Translator (Online Google Web interface)
        elif backend == "DeepTranslator" and HAS_DEEP_TRANS:
            try:
                from deep_translator import GoogleTranslator
                logger.info("INFO: Using Online Translation")
                translated = GoogleTranslator(source="en", target=target_lang).translate(text)
                if translated:
                    return translated
            except Exception as e:
                logger.error(f"DeepTranslator failed: {e}. Falling back...")

        # 4. GoogleTrans (Online Google API wrapper)
        elif backend == "GoogleTrans" and HAS_GOOGLETRANS:
            try:
                from googletrans import Translator
                logger.info("INFO: Using Online Translation")
                translator = Translator()
                res = translator.translate(text, src="en", dest=target_lang)
                if res and res.text:
                    return res.text
            except Exception as e:
                logger.error(f"GoogleTrans failed: {e}. Falling back...")

        # CacheOnly or Fallback (returns None so cache matrix lookup takes over)
        return None


