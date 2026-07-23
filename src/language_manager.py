from typing import Dict

class LanguageManager:
    """
    Manages supported translation languages, mapping names to ISO codes.
    Allows runtime dynamic language registration.
    """
    def __init__(self):
        # Base supported languages map
        self.languages: Dict[str, str] = {
            "English": "en",
            "Hindi": "hi",
            "Kannada": "kn",
            "Tamil": "ta",
            "Telugu": "te",
            "Malayalam": "ml",
            "Marathi": "mr",
            "French": "fr",
            "German": "de",
            "Korean": "ko",
            "Spanish": "es",
            "Japanese": "ja"
        }

    def get_code(self, name: str) -> str:
        """
        Retrieves the language code for a given language name.
        """
        if not name:
            return ""
        # Match case-insensitive
        for k, v in self.languages.items():
            if k.lower() == name.lower():
                return v
        return ""

    def get_name(self, code: str) -> str:
        """
        Retrieves the language name for a given language code.
        """
        if not code:
            return ""
        code_lower = code.lower().strip()
        for k, v in self.languages.items():
            if v.lower() == code_lower:
                return k
        return ""

    def register_language(self, name: str, code: str) -> None:
        """
        Registers a new language dynamically at runtime.
        """
        if not name or not code:
            return
        self.languages[name.strip()] = code.strip().lower()

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Returns a dictionary of all registered languages.
        """
        return self.languages.copy()
