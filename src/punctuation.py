import re

QUESTION_START_WORDS = {
    "who", "what", "where", "when", "why", "how", "is", "are", "do", "does",
    "did", "can", "could", "will", "would", "should", "am", "has", "have",
    "had", "was", "were", "shall", "may"
}

EXCLAMATION_WORDS = {
    "wow", "amazing", "great", "awesome", "cool", "wonderful", "excellent",
    "beautiful", "fantastic", "oh", "incredible"
}

class PunctuationRestorer:
    """
    Handles automatic punctuation restoration including terminal punctuation
    (periods, question marks, exclamation marks) and mid-sentence commas.
    """
    def __init__(self):
        pass

    def restore_punctuation(self, sentence: str) -> str:
        """
        Appends correct terminal punctuation and inserts conversational commas.
        """
        if not sentence:
            return ""
            
        s = sentence.strip()
        if not s:
            return ""
            
        # 1. Restore mid-sentence conversational commas
        s = re.sub(r"\b(hello|hi|hey|wow)\s+(how|what|i|is|are|we|you|thats|that|this)\b", r"\1, \2", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(fine|good|ok)\s+(thank you|thanks)\b", r"\1, \2", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(thanks|thank you)\s+(please|plz)\b", r"\1, \2", s, flags=re.IGNORECASE)
        
        # 2. Check and restore terminal punctuation if missing
        if s[-1] not in [".", "?", "!"]:
            # Tokenize words for semantic inspection
            words_clean = [re.sub(r'[^\w]', '', w.lower()) for w in s.split()]
            
            is_question = False
            is_exclamation = False
            
            if words_clean:
                # Direct question start words
                if words_clean[0] in QUESTION_START_WORDS:
                    # Exception: "Have a...", "Has a...", "Had a..." are wishes/imperatives, not questions
                    is_wish = False
                    if words_clean[0] in ["have", "has", "had"] and len(words_clean) > 1 and words_clean[1] in ["a", "an"]:
                        is_wish = True
                        
                    if not is_wish:
                        is_question = True
                        
                # Greetings followed by a question word
                elif words_clean[0] in ["hello", "hi", "hey"] and len(words_clean) > 1 and words_clean[1] in QUESTION_START_WORDS:
                    is_question = True
                    
                # Exclamation check: contains any exclamation keywords
                for w in words_clean:
                    if w in EXCLAMATION_WORDS:
                        is_exclamation = True
                        break
                        
            if is_question:
                s += "?"
            elif is_exclamation:
                s += "!"
            else:
                s += "."
                
        return s
