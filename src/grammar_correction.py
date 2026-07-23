import re

PROPER_NOUNS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "london", "paris", "tokyo",
    "india", "america", "english", "french", "spanish", "german", "china",
    "sayad", "ansar", "antigravity"
}

class GrammarCorrector:
    """
    Performs grammar correction, capitalization, and readability improvements
    using rule-based parsing algorithms.
    """
    def __init__(self):
        pass

    def correct_grammar(self, sentence: str) -> str:
        """
        Corrects common grammatical mistakes like subject-verb agreement and missing prepositions.
        """
        if not sentence:
            return ""
            
        s = sentence
        
        # 1. Subject-Verb Agreement corrections
        s = re.sub(r"\b(i)\s+is\b", r"\1 am", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(i)\s+are\b", r"\1 am", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(you|we|they)\s+is\b", r"\1 are", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(he|she|it)\s+are\b", r"\1 is", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(we|they|you)\s+was\b", r"\1 were", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(i|he|she|it)\s+were\b", r"\1 was", s, flags=re.IGNORECASE)
        
        # 2. Missing prepositions before destinations
        s = re.sub(r"\b(go|going|goes|went)\s+(school|work|university|college|bed|market|church|hospital)\b", r"\1 to \2", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(want|wants|wanted)\s+(go|run|play|do|write|eat)\b", r"\1 to \2", s, flags=re.IGNORECASE)
        
        # 3. Indefinite article corrections
        s = re.sub(r"\b(a)\s+([aeiouAEIOU][a-zA-Z]*)\b", r"\1n \2", s)
        def replace_an(match):
            article = match.group(1)
            word = match.group(2)
            if word.lower() not in ["hour", "honest", "honor", "heir"]:
                new_art = "A" if article[0].isupper() else "a"
                return f"{new_art} {word}"
            return match.group(0)
            
        s = re.sub(r"\b(an)\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ][a-zA-Z]*)\b", replace_an, s)
        
        return s

    def capitalize(self, sentence: str) -> str:
        """
        Applies sentence capitalization rules.
        """
        if not sentence:
            return ""
            
        s = re.sub(r"\s+", " ", sentence).strip()
        if not s:
            return ""
            
        # 1. Capitalize first letter of the sentence
        s = s[0].upper() + s[1:]
        
        # 2. Capitalize pronoun "I"
        s = re.sub(r"\bi\b", "I", s)
        
        # 3. Capitalize proper nouns
        for pn in PROPER_NOUNS:
            s = re.sub(rf"\b{pn}\b", pn.capitalize(), s, flags=re.IGNORECASE)
            
        # 4. Capitalize first letter after terminal punctuation (. ! ?)
        def cap_match(match):
            punctuation_space = match.group(1)
            char = match.group(2)
            return punctuation_space + char.upper()
            
        s = re.sub(r"([.!?]\s+)([a-z])", cap_match, s)
        
        return s

    def improve_readability(self, sentence: str) -> str:
        """
        Applies readability heuristics to smooth out awkward phrasing.
        """
        if not sentence:
            return ""
            
        s = sentence
        
        # 1. Temporal possessives + verb inserting (e.g. "today weather very good" -> "today's weather is very good")
        s = re.sub(r"\b(today|yesterday|tomorrow)\s+weather\b", r"\1's weather", s, flags=re.IGNORECASE)
        s = re.sub(r"\bweather\s+very\s+good\b", r"weather is very good", s, flags=re.IGNORECASE)
        s = re.sub(r"\bweather\s+very\s+nice\b", r"weather is very nice", s, flags=re.IGNORECASE)
        
        # 2. Subject verb suffixes
        s = re.sub(r"\b(he|she|it)\s+(like|want|need)\s", r"\1 \2s ", s, flags=re.IGNORECASE)
        
        return s
