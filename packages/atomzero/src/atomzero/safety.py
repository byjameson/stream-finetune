"""Opt-in safety filter for Atom Zero.

The filter is intentionally narrow and conservative: it targets explicit
requests for harmful instructions, not discussion, prevention or news. It is a
guardrail, not a complete moderation system. Enable it for hosted or shared
deployments; local research defaults stay unchanged.

    filter = SafetyFilter()
    verdict = filter.check("how to make a bomb")       # {"category": "violence", ...}
    text, verdict = filter.filter_reply(model_output)
"""
from __future__ import annotations

import re

VERSION = "0.2"

REFUSALS = {
    "sexual_minors": "I can't help with sexual content involving minors.",
    "self_harm": ("I can't help with instructions for self-harm. If you are in "
                  "immediate danger, contact local emergency services or a "
                  "crisis line. I can help with coping or finding support."),
    "violence": "I can't help with instructions for violence or weapons.",
    "hate": "I can't help with hateful or violent content targeting a group.",
    "harassment": "I can't help with harassment, doxxing or blackmail.",
    "illegal": ("I can't help with illegal instructions. I can help with harm "
                "reduction or lawful alternatives."),
    "privacy": ("I can't help with finding or exposing someone's private "
                "information."),
    "malware": "I can't help with malware or unauthorized access.",
    "generic": ("I can't help with that request. I can help with a safe, "
                "lawful alternative."),
}

# Narrow patterns. Each one needs an explicit harmful target or instruction;
# topic words alone (suicide prevention, child abuse reporting, genocide
# history) must not match.
_PATTERNS = {
    "sexual_minors": [
        r"\b(child|minor|underage)\s+(porn|pornography|nudes?|sex|erotica)\b",
        r"\b(porn|nudes?|sex|erotica)\s+(with|involving)\s+(a\s+)?"
        r"(child|minor|underage)\b",
        r"\b(sexual|explicit)\s+(story|content|roleplay|fantasy)\b"
        r".{0,30}\b(child|minor|underage)\b",
        r"\bcsam\b",
    ],
    "self_harm": [
        r"\bhow (do|can) i (kill|hurt|cut|harm) myself\b",
        r"\bhow to (kill|hurt|cut|harm) (myself|yourself)\b",
        r"\b(suicide|self[- ]harm)\s+(method|methods|instruction|instructions)\b",
        r"\bhow to (commit|die by) suicide\b",
        r"\bwie\s+bringe\s+ich\s+mich\s+um\b",
    ],
    "violence": [
        r"\bhow to (make|build|construct)\b.{0,40}\b(bomb|explosive|gun|weapon)\b",
        r"\b(bomb|explosive)\b.{0,30}\b(recipe|instructions|how to make)\b",
        r"\b(shoot|attack)\s+(up|a)\s+(school|church|mosque|synagogue|mall)\b",
        r"\bhow to (kill|murder|stab|shoot)\b",
        r"\bpoison\s+(someone|a person|people)\b",
        r"\bmass shooting plan\b",
        r"\b(plan|planning)\s+(a\s+)?mass shooting\b",
    ],
    "hate": [
        r"\b(kill|exterminate|gas|lynch)\s+(all\s+)?"
        r"(jews|muslims|christians|blacks|whites|asians|gays|trans|"
        r"immigrants|women)\b",
        r"\b(ethnic cleansing|genocide)\b.{0,30}\b(how|plan|support|justify)\b",
        r"\b(how|why|justify|support)\b.{0,30}\b(ethnic cleansing|genocide)\b",
    ],
    "harassment": [
        r"\b(dox|doxx|swat)\s+(someone|him|her|them|a person)\b",
        r"\bblackmail\s+(someone|him|her|them)\b",
        r"\bsextortion\b",
        r"\bhow to (stalk|harass|threaten)\b",
    ],
    "illegal": [
        r"\bhow to (make|synthesize|cook)\s+"
        r"(meth|methamphetamine|fentanyl|heroin|cocaine|explosives)\b",
        r"\bhow to (launder|launder money)\b",
        r"\bcredit card (fraud|skimming)\b",
        r"\bhow to (steal|hotwire)\s+(a car|a credit card|money)\b",
        r"\bbuy (drugs|weapons) online\b",
    ],
    "privacy": [
        r"\bhow to (find|get)\s+someone'?s\s+"
        r"(address|ssn|social security|phone number)\b",
        r"\b(dox|doxx)\s+(someone|him|her|them|a person)\b",
    ],
    "malware": [
        r"\bhow to (write|create|build)\s+"
        r"(ransomware|malware|a virus|a trojan)\b",
        r"\binstall\s+(a\s+)?keylogger\b",
    ],
}

_COMPILED = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in _PATTERNS.items()
}

# Language-agnostic, order-independent intent rules. These are still lexical
# rules (not an ML classifier), but they catch paraphrases and translated
# requests that the English regexes miss.
_INSTRUCTION = ("how", "steps", "explain", "instructions", "nasıl", "nasil",
                "wie", "comment", "como", "cómo", "anleitung", "instrucciones",
                "talimat", "erklär", "come", "hoe", "как", "walk", "guide",
                "show", "recipe")
_BUILD = ("make", "build", "construct", "yap", "bau", "fabriqu", "hac",
          "constru", "herstell", "costru", "faz", "maak", "сдела", "assembl",
          "recipe", "makin")
_WEAPON = ("bomb", "explos", "patlay", "spreng", "gun", "weapon", "silah",
           "arma", "bom", "бомб")
_SELF = ("myself", "yourself", "kendim", "kendin", "mich", "mir", "me",
         "moi", "mi", "себя")
_KILL = ("kill", "hurt", "cut", "harm", "öldür", "oldur", "umbring", "matar",
         "suicid", "tuer", "uccid", "уби")
_SUICIDE = ("suicid", "intihar", "selbstmord")
_CHILD = ("child", "minor", "underage", "çocuk", "cocuk", "kind", "menor",
          "mineur", "enfant", "infant", "дет")
_SEXUAL = ("sex", "porn", "nude", "erotic", "cinsel", "seks", "porno",
           "sexuel", "sexual")
_BENIGN_SEXUAL = ("abuse", "report", "prevent", "protect", "istismar",
                  "ihbar", "koru", "missbrauch", "melden", "abuso",
                  "denunci", "preven", "schutz", "segnal", "minori")
_BENIGN_SELF = ("help", "friend", "support", "prevent", "warning", "signs",
                "destek", "yardım", "arkadaş", "önle", "belirti", "helfen",
                "freund", "verhindern", "anzeichen", "ayudar", "amigo",
                "preven", "señales", "aider", "ami", "prévenir", "signes",
                "aiut", "amico", "segnal", "помощ", "помоч", "друг",
                "предупред")
_HATE_GROUP = ("jews", "muslims", "christians", "immigrants", "women", "gays",
               "trans", "mülteci", "multeci", "muslime", "juden", "inmigrant",
               "mujer", "femme", "kadın", "kadin", "frau", "immigrat",
               "migrant")
_HATE_ACTION = ("kill", "extermin", "gas", "lynch", "öldür", "oldur",
                "matar", "töt", "yok", "uccid")
_HARASS_ACTION = ("dox", "swat", "blackmail", "sextort", "şantaj", "santaj",
                  "chantaj", "erpress", "ricatt")
_HARASS_TARGET = ("someone", "him", "her", "them", "person", "birini",
                  "birine", "jemand", "alguien", "qualcuno", "quelqu")
_DRUG = ("meth", "fentanyl", "heroin", "cocaine", "metamfetamin",
         "uyuşturucu", "uyusturucu", "drogen", "droga", "drogue",
         "metanfetamina", "метамф")
_ILLEGAL_ACTION = ("make", "synthes", "cook", "yap", "herstell", "fabriqu",
                   "hac", "buy", "satın", "kauf", "comprar", "produr",
                   "сдела")
_PRIVATE_ACTION = ("find", "get", "bul", "finden", "encontr", "trouv",
                   "trovar", "найд")
_PRIVATE_TARGET = ("address", "ssn", "phone", "adres", "adresse",
                   "dirección", "direccion", "telefon", "password",
                   "contraseña", "contrase", "indirizz", "адрес")
_MALWARE = ("ransomware", "malware", "virus", "trojan", "fidye",
            "schadsoftware", "malveillant")
_MALWARE_ACTION = ("write", "create", "build", "install", "yaz", "kur",
                   "schreib", "écri", "crear", "instalar", "scriv", "напиш")

_SEMANTIC_RULES = [
    {"category": "sexual_minors", "all": (_CHILD, _SEXUAL),
     "none": _BENIGN_SEXUAL},
    {"category": "self_harm", "all": (_INSTRUCTION, _SUICIDE),
     "none": _BENIGN_SELF},
    {"category": "self_harm", "all": (_INSTRUCTION, _KILL, _SELF),
     "none": _BENIGN_SELF},
    {"category": "violence", "all": (_INSTRUCTION, _BUILD, _WEAPON)},
    {"category": "hate", "all": (_HATE_GROUP, _HATE_ACTION)},
    {"category": "harassment", "all": (_HARASS_ACTION, _HARASS_TARGET)},
    {"category": "illegal", "all": (_INSTRUCTION, _DRUG, _ILLEGAL_ACTION)},
    {"category": "privacy", "all": (_PRIVATE_ACTION, _PRIVATE_TARGET)},
    {"category": "malware", "all": (_MALWARE, _MALWARE_ACTION,
                                    _INSTRUCTION)},
]


def _tokens(value: str) -> list[str]:
    normalized = str(value or "").replace("İ", "i").replace("I", "ı")
    return re.findall(r"[^\W_]+", normalized.lower(), re.UNICODE)


def _has(tokens: list[str], stems) -> bool:
    return any(token.startswith(stem) for token in tokens
               for stem in stems)


def _semantic_check(value: str) -> dict | None:
    tokens = _tokens(value)
    if not tokens:
        return None
    for rule in _SEMANTIC_RULES:
        if not all(_has(tokens, stems) for stems in rule["all"]):
            continue
        if "none" in rule and _has(tokens, rule["none"]):
            continue
        return {
            "category": rule["category"],
            "action": "refuse",
            "match": "semantic:" + rule["category"],
        }
    return None


class SafetyFilter:
    version = VERSION

    def check(self, text: str) -> dict | None:
        value = str(text or "")
        if not value.strip():
            return None
        for category, patterns in _COMPILED.items():
            for pattern in patterns:
                match = pattern.search(value)
                if match:
                    return {
                        "category": category,
                        "action": "refuse",
                        "match": match.group(0)[:80],
                    }
        return _semantic_check(value)

    def refusal(self, category: str) -> str:
        return REFUSALS.get(category, REFUSALS["generic"])

    def filter_reply(self, text: str) -> tuple[str, dict | None]:
        verdict = self.check(text)
        if verdict is None:
            return text, None
        return self.refusal(verdict["category"]), verdict
