import re
from typing import Set, List

try:
    from symspellpy import SymSpell, Verbosity
    import pkg_resources
    HAS_SYMSPELL = True
except ImportError:
    HAS_SYMSPELL = False

try:
    from spellchecker import SpellChecker
    HAS_PYSPELL = True
except ImportError:
    HAS_PYSPELL = False

from grammar_correction import PROPER_NOUNS

# High-frequency dictionary for fallback and boosting accuracy
COMMON_WORDS: Set[str] = {
    "hello", "how", "are", "you", "am", "going", "to", "school", "today", "weather", 
    "is", "very", "good", "fine", "thank", "welcome", "my", "project", "work", 
    "home", "the", "be", "and", "of", "a", "in", "that", "have", "i", "it", 
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", 
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", 
    "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", 
    "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", 
    "like", "time", "no", "just", "him", "know", "take", "people", "into", 
    "year", "your", "good", "some", "could", "them", "see", "other", "than", 
    "then", "now", "look", "only", "come", "its", "over", "think", "also", 
    "back", "after", "use", "two", "how", "our", "work", "first", "well", 
    "way", "even", "new", "want", "because", "any", "these", "give", "day", 
    "most", "us", "write", "practice", "system", "gesture", "air", "writing", 
    "recognition", "model", "letter", "character", "sentence", "history"
}

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the edit distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

class SpellCorrector:
    """
    Handles spell checking and correction for individual words and sentences.
    Integrates SymSpell as the primary correction engine, and falls back to
    pyspellchecker or rule-based matching.
    """
    def __init__(self):
        self.sym_spell = None
        self.pyspell = None
        
        # 1. Initialize SymSpell (Primary)
        if HAS_SYMSPELL:
            try:
                self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
                # Boost common words
                for w in COMMON_WORDS:
                    self.sym_spell.create_dictionary_entry(w, 1000)
            except Exception as e:
                print(f"[Warning] Failed to load SymSpell dictionary: {e}.")
                self.sym_spell = None
                
        # 2. Initialize pyspellchecker (Fallback)
        if not self.sym_spell and HAS_PYSPELL:
            try:
                self.pyspell = SpellChecker()
                self.pyspell.word_frequency.load_words(list(COMMON_WORDS))
            except Exception as e:
                print(f"[Warning] Failed to initialize fallback pyspellchecker: {e}.")
                self.pyspell = None

    def is_valid_word(self, word: str) -> bool:
        """
        Returns True if the word is spelled correctly.
        """
        clean_word = word.strip().lower()
        if not clean_word:
            return True
            
        # Ignore purely numeric or mixed alphanumeric containing digits
        if re.search(r'\d', clean_word):
            return True
            
        # Check custom dictionary list and proper nouns
        if clean_word in COMMON_WORDS or clean_word in PROPER_NOUNS:
            return True
            
        if self.sym_spell:
            try:
                suggestions = self.sym_spell.lookup(clean_word, Verbosity.CLOSEST, max_edit_distance=0)
                if suggestions:
                    return True
            except Exception:
                pass
        elif self.pyspell:
            try:
                return clean_word in self.pyspell
            except Exception:
                pass
                
        return clean_word in COMMON_WORDS

    def correct_word(self, word: str) -> str:
        """
        Corrects a single word if it is misspelled.
        """
        if not word:
            return ""
            
        # If the word is already valid or contains digits, preserve it
        if self.is_valid_word(word) or re.search(r'\d', word):
            return word
            
        clean_word = word.strip().lower()
        
        # Check casing patterns
        is_upper = word.isupper()
        is_title = word.istitle()
        
        corrected = clean_word
        
        # 1. Try SymSpell (Primary - Closest edit distance)
        if self.sym_spell:
            try:
                suggestions = self.sym_spell.lookup(clean_word, Verbosity.CLOSEST, max_edit_distance=2)
                if suggestions:
                    corrected = suggestions[0].term
                    # Safety check: if the corrected suggestion is very different, keep original
                    if levenshtein_distance(clean_word, corrected) > 2:
                        corrected = clean_word
            except Exception:
                pass
                
        # 2. Try pyspellchecker (Fallback)
        elif self.pyspell:
            try:
                res = self.pyspell.correction(clean_word)
                if res:
                    corrected = res
            except Exception:
                pass
                
        # 3. Try Levenshtein Fallback
        else:
            best_word = clean_word
            min_dist = 999
            for w in COMMON_WORDS:
                dist = levenshtein_distance(clean_word, w)
                if dist < min_dist:
                    min_dist = dist
                    best_word = w
            if min_dist <= 2:
                corrected = best_word
                
        # Re-apply casing
        if is_upper:
            return corrected.upper()
        elif is_title:
            return corrected.capitalize()
        return corrected

    def correct_sentence(self, sentence: str) -> str:
        """
        Corrects spelling errors in a full sentence while preserving punctuation and spacing.
        """
        if not sentence or len(sentence.strip()) == 0:
            return sentence
            
        words = sentence.split(" ")
        corrected_words = []
        
        for w in words:
            if not w:
                corrected_words.append("")
                continue
                
            # Parse leading/trailing punctuation boundaries
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            if match:
                lead, core, trail = match.group(1), match.group(2), match.group(3)
                if core:
                    corrected_core = self.correct_word(core)
                    corrected_words.append(f"{lead}{corrected_core}{trail}")
                else:
                    corrected_words.append(w)
            else:
                corrected_words.append(self.correct_word(w))
                
        return " ".join(corrected_words)
