import torch
import string
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import WORD_TOXICITY_THRESHOLD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. INITIALIZE ENGLISH TRANSFORMER MODEL
# ==========================================
tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier")
model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta_toxicity_classifier").to(DEVICE)
model.eval()

# ==========================================
# 2. ADVANCED HINGLISH/KAGGLE TAXONOMY
# ==========================================
TOXIC_TAXONOMY = {
    "threatening speech": {
        "severity": "HIGH",
        "words": {
            # English Threats
            "kill", "die", "murder", "destroy", "assassinate", "slaughter", 
            "stab", "choke", "strangle", "bomb", "shoot", "execute", "crush", "hang",
            # Hinglish Threats
            "maar", "khatam", "kaat", "goli", "thok", "mita", "gaad",
            "khatm", "uda", "zinda", "nikaal", "phod"
        }
    },
    "verbal abuse": {
        "severity": "MEDIUM",
        "words": {
            # English Slurs & Abuse
            "idiot", "stupid", "moron", "dumb", "trash", "bastard", "asshole", 
            "bitch", "fucker", "slut", "whore", "retard", "cunt", "dick", 
            "dickhead", "motherfucker", "cocksucker", "prick", "scumbag", 
            "douchebag", "wanker", "pussy", "twat", "shithead", "hoe", "skank",
            # Hinglish Slurs & Abuse
            "pagal", "chutiya", "saala", "kutta", "kamina", "harami", 
            "bhainchod", "madarchod", "bhenchod", "randi", "bhadwa", "chinal", 
            "raand", "chodu", "lodu", "gaandu", "bhosdike", "bosadike", "bhosdi",
            "tatti", "haramzaada", "suar", "ullu", "rakhail", "dalal", "kutti"
        }
    },
    "aggressive speech": {
        "severity": "LOW",
        "words": {
            # English Aggression
            "hate", "worst", "shut", "nonsense", "ruin", "loser", 
            "pathetic", "disgusting", "crap", "bullshit", "ugly", 
            "garbage", "worthless", "suck", "freak", "scum",
            # Hinglish Aggression
            "bakwas", "nikal", "drame", "aukaat", "bhag", "dafa",
            "ghatiya", "kachra", "sadakchap", "faaltu", "bekaar", "chal", "bhasad"
        }
    }
}

# ==========================================
# 3. TEXT NORMALIZATION UTILITIES
# ==========================================
def normalize_word(w: str) -> str:
    """Removes punctuation and makes lowercase."""
    return w.strip().lower().translate(str.maketrans("", "", string.punctuation))

def collapse_duplicates(word: str) -> str:
    """
    Collapses 3+ consecutive repeating characters to 1.
    e.g., 'maaaaaaar' -> 'maar', 'paaagal' -> 'pagal'
    """
    word = word.lower().strip()
    return re.sub(r'(.)\1{2,}', r'\1', word)

def get_toxic_label_and_severity(word: str) -> tuple:
    """Matches words against the taxonomy (including substrings)."""
    clean_word = collapse_duplicates(word)
    
    for label, config in TOXIC_TAXONOMY.items():
        if word in config["words"] or clean_word in config["words"]:
            return label, config["severity"]
            
        # Check if a known toxic root exists inside the spoken word (e.g., 'maardunga' -> 'maar')
        if any(bad_root in clean_word for bad_root in config["words"] if len(bad_root) > 2):
            return label, config["severity"]

    return "aggressive speech", "LOW"

# ==========================================
# 4. CORE PIPELINE EXECUTION
# ==========================================
def score_words(words):
    """Passes English words through the RoBERTa model to get dynamic probabilities."""
    if not words:
        return {}

    inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    return {w: float(p) for w, p in zip(words, probs)}


def detect_toxic_words(segments):
    """Main function called by audio_pipeline to find timestamps to BEEP."""
    candidate_words = []
    candidate_map = {}

    # Extract all words from the Whisper timestamps
    for seg in segments:
        for w in seg.get("words", []):
            norm = normalize_word(w["word"])
            if not norm:
                continue
            candidate_words.append(norm)
            candidate_map.setdefault(norm, []).append(w)

    candidate_words = list(set(candidate_words))

    # Score using English Transformer AI
    word_probs = score_words(candidate_words)
    toxic_segments = []

    for word in candidate_words:
        prob = word_probs.get(word, 0.0)
        collapsed = collapse_duplicates(word)
        
        label, severity = get_toxic_label_and_severity(word)
        
        # SYSTEM BYPASS: If word matches our Hinglish Kaggle list, force threat score to 1.0
        is_explicit_violation = any(
            word in config["words"] or collapsed in config["words"] or 
            any(root in collapsed for root in config["words"] if len(root) > 2)
            for config in TOXIC_TAXONOMY.values()
        )

        if is_explicit_violation:
            prob = max(prob, 1.0) 

        # If it passes the threshold, flag it for the audio censor
        if prob >= WORD_TOXICITY_THRESHOLD:
            for wobj in candidate_map[word]:
                toxic_segments.append({
                    "start": float(wobj["start"]),
                    "end": float(wobj["end"]),
                    "word": wobj["word"],
                    "confidence": round(prob, 3),
                    "label": label,
                    "severity": severity 
                })

    # Return chronological list of toxic events
    return sorted(toxic_segments, key=lambda x: x["start"])