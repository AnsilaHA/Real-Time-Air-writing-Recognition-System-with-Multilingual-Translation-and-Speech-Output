import re
from typing import List
from spell_checker import SpellCorrector, levenshtein_distance, COMMON_WORDS

try:
    from symspellpy import Verbosity
except ImportError:
    pass

class SuggestionEngine:
    """
    Generates alternative spelling suggestions (top-3 candidates) using
    SymSpell or fallback dictionary checks, sorted by edit distance.
    """
    def __init__(self, spell_corrector: SpellCorrector):
        self.spell_corrector = spell_corrector

    def get_top_suggestions(self, word: str) -> List[str]:
        """
        Retrieves the top-3 suggestions for a misspelled word.
        Prioritizes minimum edit distance.
        """
        if not word:
            return []
            
        # Parse punctuation boundaries
        match = re.match(r"^([^\w]*)(.*?)([^\w]*)$", word)
        if not match:
            return []
            
        lead, core, trail = match.group(1), match.group(2), match.group(3)
        if not core:
            return []
            
        # If the word is already valid, suggestions are empty
        if self.spell_corrector.is_valid_word(core):
            return []
            
        clean_core = core.lower()
        candidates = []
        seen = set()
        
        # 1. Fetch candidates from SymSpell (Primary - edit distance sorted)
        if self.spell_corrector.sym_spell:
            try:
                # Verbosity.CLOSEST returns results sorted by edit distance, then frequency
                suggs = self.spell_corrector.sym_spell.lookup(clean_core, Verbosity.CLOSEST, max_edit_distance=2)
                for s in suggs:
                    term = s.term
                    if term not in seen:
                        seen.add(term)
                        candidates.append(term)
            except Exception:
                pass
                
        # 2. Fetch candidates from fallback pyspellchecker
        elif self.spell_corrector.pyspell:
            try:
                cands = self.spell_corrector.pyspell.candidates(clean_core)
                if cands:
                    # Sort candidates by edit distance to input word
                    sorted_cands = sorted(list(cands), key=lambda x: levenshtein_distance(clean_core, x))
                    for c in sorted_cands:
                        if c not in seen:
                            seen.add(c)
                            candidates.append(c)
            except Exception:
                pass
                
        # 3. Fallback matching from COMMON_WORDS
        if len(candidates) < 3:
            s_cands = []
            for w in COMMON_WORDS:
                dist = levenshtein_distance(clean_core, w)
                if dist <= 2:
                    s_cands.append((w, dist))
            # Sort by distance
            s_cands.sort(key=lambda x: x[1])
            for w, _ in s_cands:
                if w not in seen:
                    seen.add(w)
                    candidates.append(w)
                    
        # Extract top-3
        top_3 = candidates[:3]
        
        # Format casing and re-attach punctuation
        formatted_suggestions = []
        is_upper = core.isupper()
        is_title = core.istitle()
        
        for cand in top_3:
            s_core = cand
            if is_upper:
                s_core = cand.upper()
            elif is_title:
                s_core = cand.capitalize()
            formatted_suggestions.append(f"{lead}{s_core}{trail}")
            
        return formatted_suggestions

    def confidence_score(self, original: str, corrected: str) -> float:
        """
        Calculates the similarity confidence score [0.0 - 1.0] between 
        original and corrected words based on Levenshtein distance.
        """
        if not original or not corrected:
            return 0.0
            
        # Clean punctuation to compute distance on actual word content
        orig_core = re.sub(r'[^\w]', '', original).lower()
        corr_core = re.sub(r'[^\w]', '', corrected).lower()
        
        if orig_core == corr_core:
            return 1.0
            
        max_len = max(len(orig_core), len(corr_core))
        if max_len == 0:
            return 1.0
            
        dist = levenshtein_distance(orig_core, corr_core)
        score = 1.0 - (dist / max_len)
        return max(0.0, min(1.0, score))
