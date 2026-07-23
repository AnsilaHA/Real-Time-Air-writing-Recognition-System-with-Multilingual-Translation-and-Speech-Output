from typing import List
from nlp_pipeline import NLPPipeline

class SentenceBuilder:
    """
    Manages the compilation of recognized characters into words, 
    words into sentences, and sentences into a historical log.
    Includes NLP post-processing on finalized sentences.
    """
    def __init__(self):
        """
        Initializes the SentenceBuilder state.
        """
        self.current_character: str = ""
        self.current_word: str = ""
        self.current_sentence: str = ""
        self.history: List[str] = []
        self.nlp = NLPPipeline()

    def append_character(self, char: str):
        """
        Appends a newly recognized character to the current active word.
        
        Args:
            char: The single character to append.
        """
        if char and isinstance(char, str):
            clean_char = char.strip()
            if len(clean_char) > 0:
                self.current_word += clean_char
                self.current_character = clean_char

    def finish_word(self):
        """
        Finalizes the current word, appends it to the current sentence 
        followed by a space, and resets the current word buffer.
        """
        if self.current_word:
            self.current_sentence += self.current_word + " "
            self.current_word = ""

    def finish_sentence(self) -> dict:
        """
        Finalizes the current sentence by removing any trailing whitespace,
        running the NLP post-processing pipeline, saving the result to history,
        and clearing the sentence buffer.
        """
        # If there's a word in the buffer, finalize it first
        if self.current_word:
            self.finish_word()
            
        nlp_result = None
        if self.current_sentence:
            # Run NLP pipeline process
            nlp_result = self.nlp.process(self.current_sentence)
            
            # Save the corrected sentence to history
            self.history.append(nlp_result["corrected"])
            
            # Reset buffers
            self.current_sentence = ""
            self.current_word = ""
            self.current_character = ""
            
        return nlp_result

    def delete_last_character(self):
        """
        Deletes the last character of the current word only.
        Does nothing if the current word is already empty.
        Never modifies the current sentence or history.
        """
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.current_character = ""

    def delete_last_word(self):
        """
        Removes the last completed word from the current sentence.
        Does not affect current_word or history.
        """
        if self.current_sentence:
            words = [w for w in self.current_sentence.strip().split(" ") if w]
            if words:
                words.pop()
            if words:
                self.current_sentence = " ".join(words) + " "
            else:
                self.current_sentence = ""

    def clear_current_word(self):
        """
        Resets only the current active word buffer.
        """
        self.current_word = ""
        self.current_character = ""

    def clear(self):
        """
        Fully resets all text states and history.
        """
        self.current_character = ""
        self.current_word = ""
        self.current_sentence = ""
        self.history = []

    def get_current_character(self) -> str:
        """
        Returns the latest recognized character.
        """
        return self.current_character

    def get_current_word(self) -> str:
        """
        Returns the active word in progress.
        """
        return self.current_word

    def get_current_sentence(self) -> str:
        """
        Returns the active sentence in progress.
        """
        return self.current_sentence

    def get_history(self) -> List[str]:
        """
        Returns the list of finalized sentences.
        """
        return self.history
