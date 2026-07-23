import re
from typing import Dict, Any, List
from spell_checker import SpellCorrector
from grammar_correction import GrammarCorrector
from suggestion_engine import SuggestionEngine
from punctuation import PunctuationRestorer

# Predefined handwriting recognition error corrections
CUSTOM_CORRECTIONS: Dict[str, str] = {
    "helo": "hello",
    "hllo": "hello",
    "hlo": "hello",
    "yu": "you",
    "u": "you",
    "wnt": "want",
    "wan": "want",
    "gud": "good",
    "thats": "that's",
    "dont": "don't",
    "cant": "can't",
    "im": "i'm",
    "iam": "i am",
    "plz": "please",
    "thx": "thanks",
    "thnk": "thank",
    "hve": "have",
    "youre": "you're",
    "ive": "i've"
}

# Words allowed to repeat consecutively without correction
ALLOWED_REPETITIONS = {"had", "that"}

# Contraction dictionary for post-spelling restoration pass
CONTRACTIONS = {
    "dont": "don't",
    "cant": "can't",
    "im": "i'm",
    "youre": "you're",
    "ive": "i've",
    "thats": "that's",
    "didnt": "didn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wouldnt": "wouldn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "isnt": "isn't",
    "arent": "aren't",
    "havent": "haven't",
    "hasnt": "hasn't"
}

class NLPPipeline:
    """
    Upgraded post-processing pipeline optimized for handwriting recognition errors (A-Z, 0-9).
    Executes stages: Normalization -> Custom Corrections -> Duplicate Removal ->
    Spell Correction (SymSpell) -> Contraction Restoration -> Grammar -> Punctuation -> Capitalization
    """
    def __init__(self):
        self.spell_corrector = SpellCorrector()
        self.grammar_corrector = GrammarCorrector()
        self.suggestion_engine = SuggestionEngine(self.spell_corrector)
        self.punctuation_restorer = PunctuationRestorer()

    def process(self, sentence: str) -> Dict[str, Any]:
        """
        Processes a raw input sentence through the handwriting-optimized NLP pipeline.
        """
        if not sentence or not sentence.strip():
            return {
                "original": "",
                "corrected": "",
                "status": "No Corrections Needed",
                "spelling_confidence": 1.0,
                "grammar_confidence": 1.0,
                "suggestions": {}
            }

        original_sentence = sentence.strip()
        
        # --- Stage 1: Normalize Text (Convert to lowercase) ---
        normalized = original_sentence.lower()
        # Context-specific correction for common handwriting error (e.g. "I AN" -> "I AM")
        normalized = re.sub(r"\bi\s+an\b", "i am", normalized)
        
        # --- Stage 2: Custom Recognition Dictionary ---
        words_normalized = normalized.split()
        words_custom = []
        for w in words_normalized:
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            lead, core, trail = match.group(1) if match else "", match.group(2) if match else w, match.group(3) if match else ""
            if core in CUSTOM_CORRECTIONS:
                words_custom.append(f"{lead}{CUSTOM_CORRECTIONS[core]}{trail}")
            else:
                words_custom.append(w)
        sentence_custom = " ".join(words_custom)
        
        # --- Stage 3: Duplicate Word Removal ---
        words_dup_check = sentence_custom.split()
        words_no_dup = []
        for w in words_dup_check:
            # Strip punctuation for duplicate check comparison
            w_clean = re.sub(r'[^\w]', '', w.lower())
            if words_no_dup:
                prev_clean = re.sub(r'[^\w]', '', words_no_dup[-1].lower())
                if w_clean == prev_clean and w_clean not in ALLOWED_REPETITIONS:
                    # Skip consecutive duplicates unless allowed
                    continue
            words_no_dup.append(w)
        sentence_no_dup = " ".join(words_no_dup)
        
        # --- Stage 4: Spell Correction (Gated by Confidence Threshold) ---
        words_spell_check = sentence_no_dup.split()
        words_corrected = []
        suggestions_map = {}
        spell_conf_sum = 0.0
        
        for w in words_spell_check:
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            lead, core, trail = match.group(1) if match else "", match.group(2) if match else w, match.group(3) if match else ""
            
            if not core:
                words_corrected.append(w)
                spell_conf_sum += 1.0
                continue
                
            is_valid = self.spell_corrector.is_valid_word(core)
            if not is_valid:
                corrected_core = self.spell_corrector.correct_word(core)
                conf = self.suggestion_engine.confidence_score(core, corrected_core)
                
                # If confidence is below 0.70, do NOT automatically replace the word
                if conf >= 0.70:
                    words_corrected.append(f"{lead}{corrected_core}{trail}")
                    spell_conf_sum += conf
                else:
                    # Keep original, return suggestions instead
                    words_corrected.append(w)
                    spell_conf_sum += conf
                    suggs = self.suggestion_engine.get_top_suggestions(w)
                    if suggs:
                        suggestions_map[w] = suggs
            else:
                words_corrected.append(w)
                spell_conf_sum += 1.0
                
        spelling_confidence = spell_conf_sum / max(len(words_spell_check), 1)
        sentence_spelled = " ".join(words_corrected)
        
        # --- Stage 5: Contraction Restoration ---
        words_contraction = sentence_spelled.split()
        words_restored = []
        for w in words_contraction:
            match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", w)
            lead, core, trail = match.group(1) if match else "", match.group(2) if match else w, match.group(3) if match else ""
            if core in CONTRACTIONS:
                words_restored.append(f"{lead}{CONTRACTIONS[core]}{trail}")
            else:
                words_restored.append(w)
        sentence_contraction = " ".join(words_restored)
        
        # --- Stage 6: Grammar Correction ---
        sentence_grammar = self.grammar_corrector.correct_grammar(sentence_contraction)
        
        # Calculate grammar confidence
        num_grammar_changes = 0
        orig_words = sentence_contraction.lower().split()
        gram_words = sentence_grammar.lower().split()
        if len(orig_words) == len(gram_words):
            for w1, w2 in zip(orig_words, gram_words):
                if w1 != w2:
                    num_grammar_changes += 1
        else:
            num_grammar_changes = abs(len(orig_words) - len(gram_words))
        grammar_confidence = max(0.5, 1.0 - (num_grammar_changes * 0.15))
        
        # --- Stage 7: Punctuation Restoration ---
        sentence_punctuated = self.punctuation_restorer.restore_punctuation(sentence_grammar)
        
        # --- Stage 8: Capitalization ---
        sentence_capitalized = self.grammar_corrector.capitalize(sentence_punctuated)
        
        # Apply readability heuristics (e.g. today weather -> today's weather)
        sentence_readability = self.grammar_corrector.improve_readability(sentence_capitalized)
        
        # Final polish capitalization pass
        final_sentence = self.grammar_corrector.capitalize(sentence_readability)
        
        # --- Stage 9: Determine Status ---
        has_low_conf_suggestions = len(suggestions_map) > 0
        text_modified = final_sentence.lower() != original_sentence.lower()
        
        if has_low_conf_suggestions:
            status = "Suggestions Available"
        elif text_modified or spelling_confidence < 1.0:
            status = "Auto-Corrected"
        else:
            status = "No Corrections Needed"
            
        return {
            "original": original_sentence,
            "corrected": final_sentence,
            "status": status,
            "spelling_confidence": spelling_confidence,
            "grammar_confidence": grammar_confidence,
            "suggestions": suggestions_map
        }
