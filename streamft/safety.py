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

VERSION = "0.1"

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
        return None

    def refusal(self, category: str) -> str:
        return REFUSALS.get(category, REFUSALS["generic"])

    def filter_reply(self, text: str) -> tuple[str, dict | None]:
        verdict = self.check(text)
        if verdict is None:
            return text, None
        return self.refusal(verdict["category"]), verdict
