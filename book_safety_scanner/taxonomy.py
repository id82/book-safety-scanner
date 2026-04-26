"""Content category taxonomy with keyword patterns and severity thresholds."""

import re
from dataclasses import dataclass

CATEGORIES = ["violence", "sexual_content", "drugs_alcohol", "language", "psychological", "themes"]

CATEGORY_LABELS = {
    "violence": "Violence",
    "sexual_content": "Sexual Content",
    "drugs_alcohol": "Drugs & Alcohol",
    "language": "Strong Language",
    "psychological": "Psychological Horror",
    "themes": "Distressing Themes",
}

# Age band -> max paragraph score that is acceptable (inclusive)
AGE_BANDS: dict[str, float] = {
    "under_7": 1.0,
    "7_10": 2.0,
    "10_12": 2.5,
    "12_plus": 5.0,
}

# Keyword lists per category (word-boundary matched, case-insensitive)
_RAW_KEYWORDS: dict[str, list[str]] = {
    "violence": [
        r"kill(?:ed|ing|s|er)?",
        r"murder(?:ed|ing|s|er)?",
        r"blood(?:y|ied|shed)?",
        r"wound(?:ed|ing|s)?",
        r"weapon(?:s|ry)?",
        r"shoot(?:ing|s)?",
        r"shot",
        r"stab(?:bed|bing|s)?",
        r"slash(?:ed|ing)?",
        r"gore(?:y|ier)?",
        r"brutal(?:ly|ity)?",
        r"torture(?:d|s|r)?",
        r"slaughter(?:ed|ing)?",
        r"assault(?:ed|ing|s)?",
        r"bludgeon(?:ed|ing)?",
        r"decapitat(?:e|ed|ion)",
        r"dismember(?:ed|ing)?",
        r"corpse(?:s)?",
        r"dead\s+body",
        r"blow(?:s)?\s+to\s+(?:the\s+)?head",
        r"death(?:s)?",
        r"dead",
        r"dying",
        r"fight(?:ing|s)?",
        r"battle(?:s|d)?",
        r"war(?:s|fare)?",
        r"attack(?:ed|ing|s|er)?",
        r"punch(?:ed|ing)?",
        r"hit(?:ting)?",
        r"beat(?:en|ing)?",
        r"hurt(?:ing)?",
        r"pain(?:ful)?",
        r"suffer(?:ing|ed|s)?",
        r"agony",
        r"scream(?:ed|ing|s)?",
        r"bleed(?:ing)?",
        r"injur(?:y|ies|ed|ing)",
    ],
    "sexual_content": [
        r"sex(?:ual|ually)?",
        r"naked(?:ness)?",
        r"nude(?:ity)?",
        r"breast(?:s)?",
        r"nipple(?:s)?",
        r"genitals?",
        r"penis",
        r"vagina",
        r"erect(?:ion|ed)?",
        r"orgasm(?:s|ed)?",
        r"intercourse",
        r"fornication",
        r"lust(?:ful|fully)?",
        r"arousal",
        r"aroused",
        r"seduc(?:e|ed|ing|tion|tive)",
        r"provocative(?:ly)?",
        r"sensual(?:ly|ity)?",
        r"erotic(?:ally|ism)?",
        r"pornograph(?:y|ic)",
        r"undress(?:ed|ing)?",
        r"strip(?:ped|ping)?\s+(?:off|down|naked)",
        r"make\s+love",
        r"sleep(?:ing)?\s+with",
        r"bed(?:ded)?\s+(?:her|him|with)",
        r"lover(?:s)?",
        r"mistress",
        r"affair(?:s)?",
        r"intimate(?:ly|cy)?",
        r"caress(?:ed|ing)?",
        r"fondle(?:d|ing)?",
        r"grope(?:d|ing)?",
    ],
    "drugs_alcohol": [
        r"drunk(?:en|ness)?",
        r"alcohol(?:ic|ism)?",
        r"intoxicat(?:e|ed|ing|ion)",
        r"wine",
        r"whiskey",
        r"whisky",
        r"vodka",
        r"cocaine",
        r"heroin",
        r"marijuana",
        r"cannabis",
        r"weed\b",
        r"narcotic(?:s)?",
        r"drug(?:s|ged|ging)?",
        r"overdose(?:d)?",
        r"hallucinat(?:e|ed|ing|ion)",
        r"high\s+on",
        r"stoned",
        r"needle\s+in\s+(?:his|her|the)\s+arm",
        r"shooting\s+up",
        r"snort(?:ing|ed)?",
        r"smuggl(?:e|ing|ed|er)",
        r"dealer\b",
        r"traffick(?:ing|er)",
    ],
    "language": [
        r"fuck(?:ing|ed|er|s)?",
        r"shit(?:ting|ted|s)?",
        r"cunt(?:s)?",
        r"bastard(?:s)?",
        r"bitch(?:es)?",
        r"damn(?:ed|it)?",
        r"ass(?:hole)?(?:s)?",
        r"dick\b",
        r"cock\b",
        r"piss(?:ed)?",
        r"whore(?:s)?",
        r"slut(?:s)?",
        r"nigger(?:s)?",
        r"faggot(?:s)?",
        r"retard(?:ed)?",
        r"crap(?:py)?",
        r"hell\b",
        r"goddamn(?:ed)?",
        r"christ\b",  # used as expletive
        r"bollocks",
        r"bloody\s+hell",
    ],
    "psychological": [
        r"suicide",
        r"suicidal",
        r"self[- ]harm",
        r"self[- ]mutilat(?:e|ion|ing)",
        r"cutting\s+(?:myself|herself|himself)",
        r"slit(?:ting)?\s+(?:wrists|throat)",
        r"hang(?:ing)?\s+(?:himself|herself|myself)",
        r"terror(?:ified|izing|ism)?",
        r"horror(?:ific|ified)?",
        r"nightmare(?:s|ish)?",
        r"haunt(?:ed|ing|s)?",
        r"traumati[sz](?:ed|ing)?",
        r"trauma(?:s|tic)?",
        r"abuse(?:d|r|rs|rs)?",
        r"abusive",
        r"molest(?:ed|ing|er)?",
        r"predator(?:s)?",
        r"threaten(?:ing|ed)?",
        r"bully(?:ing|ied)?",
        r"torment(?:ed|ing)?",
        r"despair(?:ing|ed)?",
        r"helpless(?:ness)?",
        r"abandon(?:ed|ment)?",
        r"neglect(?:ed|ing)?",
        r"kidnap(?:ped|ping|per)?",
        r"hostage(?:s)?",
        r"captiv(?:e|ity)",
    ],
    "themes": [
        r"death\s+of\s+(?:a\s+)?(?:parent|mother|father|child|children)",
        r"orphan(?:s|ed)?",
        r"war\s+crime(?:s)?",
        r"genocide",
        r"starvation",
        r"concentration\s+camp",
        r"slave(?:ry|s)?",
        r"human\s+trafficking",
        r"child\s+(?:labour|labor|abuse|soldier)",
        r"poverty\b",
        r"homeless(?:ness)?",
        r"begging\b",
        r"suicide\s+(?:note|attempt|bomber)",
        r"murder(?:ed)?\s+(?:his|her|their)\s+(?:parent|mother|father|child|family)",
    ],
}

# Pre-compile all patterns
KEYWORD_PATTERNS: dict[str, list[re.Pattern]] = {
    cat: [re.compile(r"\b" + kw + r"\b", re.IGNORECASE) for kw in keywords]
    for cat, keywords in _RAW_KEYWORDS.items()
}


@dataclass
class KeywordHit:
    category: str
    matched_text: str
    pattern: str


def scan_keywords(text: str) -> list[KeywordHit]:
    """Run Pass 1 keyword scan on a text string. Returns all hits."""
    hits: list[KeywordHit] = []
    for category, patterns in KEYWORD_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                hits.append(
                    KeywordHit(
                        category=category,
                        matched_text=match.group(0),
                        pattern=pattern.pattern,
                    )
                )
    return hits


def hits_by_category(hits: list[KeywordHit]) -> dict[str, list[KeywordHit]]:
    result: dict[str, list[KeywordHit]] = {c: [] for c in CATEGORIES}
    for hit in hits:
        result[hit.category].append(hit)
    return result
