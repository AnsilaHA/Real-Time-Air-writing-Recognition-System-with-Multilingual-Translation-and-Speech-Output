import time

class WordBuilder:
    """
    A class to handle word formation from real-time character streams.
    Combines recognized characters into words using timing-based logic,
    independent of any dictionary, language model, or spell correction.
    """
    def __init__(self, timeout_seconds: float = 2.0):
        """
        Initializes the WordBuilder instance.
        
        Args:
            timeout_seconds: Maximum idle duration (in seconds) allowed between characters 
                             before the current word is automatically finalized.
        """
        self.current_word = ""
        self.history = []  # Python list storing finalized words
        self.last_activity_time = None  # Timestamp of the last character insertion or active event
        self.timeout_seconds = timeout_seconds

    def add_character(self, char: str):
        """
        Appends a valid character to the current word and updates the last activity timestamp.
        
        Args:
            char: The recognized character string to append.
        """
        if not char or not isinstance(char, str):
            return
            
        # Strip whitespaces and check if it is a single valid character
        char = char.strip()
        if len(char) == 0:
            return

        # Perform a quick check if the word has timed out before appending (fail-safe)
        self.check_timeout()
        
        self.current_word += char
        self.last_activity_time = time.time()

    def finalize_word(self):
        """
        Moves the current word (if non-empty) to the word history and resets it.
        """
        if self.current_word:
            self.history.append(self.current_word)
            self.current_word = ""
        self.last_activity_time = None

    def reset_current_word(self):
        """
        Clears the current word without adding it to the history.
        """
        self.current_word = ""
        self.last_activity_time = None

    def clear_history(self):
        """
        Clears both the current word and the entire word history.
        """
        self.current_word = ""
        self.history = []
        self.last_activity_time = None

    def get_current_word(self) -> str:
        """
        Returns the current active word.
        
        Returns:
            str: The current word string.
        """
        return self.current_word

    def get_history(self) -> list:
        """
        Returns the list of finalized words.
        
        Returns:
            list: List of string words.
        """
        return self.history

    def check_timeout(self):
        """
        Internal check: if a word is in progress and the elapsed time since
        last_activity_time exceeds timeout_seconds, automatically finalize the word.
        """
        if self.current_word and self.last_activity_time is not None:
            if time.time() - self.last_activity_time >= self.timeout_seconds:
                self.finalize_word()

    def update(self):
        """
        Performs the periodic update checks (meant to be run in the main application loop).
        """
        self.check_timeout()

    def get_time_remaining(self) -> float:
        """
        Returns the seconds remaining before the current word is automatically finalized.
        
        Returns:
            float: Time in seconds, or 0.0 if no character is in progress.
        """
        if not self.current_word or self.last_activity_time is None:
            return 0.0
        elapsed = time.time() - self.last_activity_time
        remaining = self.timeout_seconds - elapsed
        return max(0.0, remaining)
