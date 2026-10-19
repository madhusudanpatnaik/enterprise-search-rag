"""Text analysis for the chunker, the reranker and the citation gate: one pass turns a sentence
into tokens that know their clause, whether they are negated, and whether they are only
hypothetical — and `compare` checks a claim against a sentence clause by clause.

Normalization handles curly quotes, contractions, citation markers in any bracket style,
meta-preambles ("According to the documents, ..."), units ("15%", "$1.5M") and clock times.
Numbers are values (digits, words, ordinals, scale words; decimals keep their digits) attached to
the word beside them. Negation is a parity per clause, flipped by "not"-type cues and by verbs that
deny what they govern ("reject", "cancel", "block", ...); "if", "unless", "suggested", "might"
mark a clause hypothetical. Sentences split at blank lines and bullet or heading lines, so a
bulleted page never becomes one "sentence" whose words and numbers pool across bullets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

# --------------------------------------------------------------------------- sentences

_SENT_END = re.compile(r"(?:[.!?]+|\u2026)(?=\s|$)")
_LAST_TOKEN = re.compile(r"\S+$")
_PREV_TOKEN = re.compile(r"(\S+)\s+\S+$")
_INITIAL = re.compile(r"[a-z]\.")
_NEVER_END = frozenset(
    "e.g. i.e. vs. mr. mrs. ms. dr. prof. jr. sr. st. approx. dept. fig.".split()
)
_END_IF_CAPITAL = frozenset("a.m. p.m. etc. inc. ltd. co. corp. u.s. u.k. est.".split())
_TIME_FOLLOWERS = frozenset(
    "monday tuesday wednesday thursday friday saturday sunday today tomorrow tonight "
    "pst pdt est edt cst cdt mst mdt utc gmt bst cet ist et pt ct".split()
)
_NUMBERED_NOUNS = frozenset("item issue ticket case room invoice order account policy claim "
                            "model part page section version".split())  # fmt: skip
_FUNCTION_WORDS = frozenset("a an the and or of to by for with from at in on as".split())
# A blank line, or a newline before a list marker or a heading, ends a sentence even without a
# full stop: a bulleted page must not become one "sentence" whose words pool across bullets.
_HARD_BREAK = re.compile(r"\n\s*\n|\n(?=\s*(?:[-*\u2022]|\d+[.)]|#+)\s)")
_LINE_MARKER = re.compile(r"\s*(?:[-*\u2022]|\d+[.)]|#+)\s+")  # match() anchors at pos


def _ends_sentence(text: str, end: int) -> bool:
    token_match = _LAST_TOKEN.search(text[:end])
    raw = token_match.group(0) if token_match else ""
    token = raw.lower().lstrip("(\"'[")
    following = text[end:].lstrip()
    nxt_word = re.match(r"[A-Za-z]+", following)
    nxt = nxt_word.group(0).lower() if nxt_word else ""
    begins = end - len(raw)
    if re.fullmatch(r"\d+[.)]", raw) and (begins == 0 or text[begins - 1] == "\n"):
        return False  # "1." opening a numbered line is a list marker, not a sentence
    if token.endswith(("...", "\u2026")):  # "fifteenth... no, the twentieth" is one thought
        return not following[:1].islower()
    if token in _END_IF_CAPITAL:  # "p.m." looks like initials; it ends only before a new sentence
        return bool(following[:1].isupper()) and nxt not in _TIME_FOLLOWERS
    if token == "no.":
        prev = _PREV_TOKEN.search(text[:end])
        numbered = raw == "No." or (prev is not None and prev.group(1).lower() in _NUMBERED_NOUNS)
        return not (numbered and following[:1].isdigit())
    if token in _NEVER_END:
        return False
    if _INITIAL.fullmatch(token):  # "J. Doe" is a name; "plan B." ends a sentence
        prev = _PREV_TOKEN.search(text[:end])
        before = prev.group(1) if prev else ""
        name_like = not before or before[:1].isupper() or before.lower() in _FUNCTION_WORDS
        return not (name_like and following[:1].isupper())
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Exact (start, end) character offsets of sentences, whitespace and list markers trimmed."""
    spans: list[tuple[int, int]] = []
    blocks, last = [], 0
    for m in _HARD_BREAK.finditer(text):
        blocks.append((last, m.start()))
        last = m.end()
    blocks.append((last, len(text)))
    for lo, hi in blocks:
        start = lo
        for m in _SENT_END.finditer(text, lo, hi):
            if _ends_sentence(text[:hi], m.end()):
                spans.append((start, m.end()))
                start = m.end()
        if text[start:hi].strip():
            spans.append((start, hi))
    trimmed: list[tuple[int, int]] = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        if marker := _LINE_MARKER.match(text, s, e):
            s = marker.end()
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            trimmed.append((s, e))
    return trimmed


def sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


def is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


# --------------------------------------------------------------------------- normalization

_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwon't\b", re.I), "will not"),
    (re.compile(r"\bcan't\b", re.I), "can not"),
    (re.compile(r"\bshan't\b", re.I), "shall not"),
    (re.compile(r"\blet's\b", re.I), "let us"),
    (re.compile(r"n't\b", re.I), " not"),
    (re.compile(r"'ll\b", re.I), " will"),
    (re.compile(r"'re\b", re.I), " are"),
    (re.compile(r"'ve\b", re.I), " have"),
    (re.compile(r"'m\b", re.I), " am"),
    (re.compile(r"'d\b", re.I), " would"),
]
_CONTRACTIONS += [
    (re.compile(r"\bgonna\b", re.I), "going to"),
    (re.compile(r"\bwanna\b", re.I), "want to"),
    (re.compile(r"\bgotta\b", re.I), "got to"),
    (re.compile(r"'s(?=\s+going\s+to\b)", re.I), " is"),  # "Carol's going to" (not a possessive)
]
# Citation markers in any bracket style: [1], [u0004, r12], [u1-u3], (u1), (see u1; u2).
_ID = r"[a-z]{0,2}\d+"
_MARKER = re.compile(rf"[\[(]\s*(?:see\s+)?{_ID}(?:\s*[-–,;]\s*(?:see\s+)?{_ID})*\s*[\])]", re.I)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"t\d{4}\b|[a-z0-9]+(?:\.\d+)?|[,;:—…]")
_BREAK_PUNCT = frozenset([",", ";", ":", "—", "…"])

# "According to the provided context, ..." / "Based on passage 2, ..." — framing, not a claim. It
# must end in a comma, so "Based on the discussion 3 vendors remain" keeps its 3.
_PREAMBLE = re.compile(
    r"\b(?:according to|based on|as (?:stated|noted|mentioned) in|per)\s+(?:the\s+)?"
    r"(?:provided\s+|given\s+|above\s+)?(?:context|documents?|passages?|sources?|policy|policies)"
    r"(?:\s+\d+)?\s*,",
    re.I,
)
_DASH = re.compile(r"\s+[-–]\s+|\s*--\s*|–")  # spaced hyphen, en dash, "--"
_ELLIPSIS = re.compile(r"\.\.\.|…")
_PERCENT = re.compile(r"(\d)\s*%")
_SCALE_SUFFIX = {"k": "thousand", "m": "million", "mm": "million", "b": "billion", "bn": "billion"}
_DOLLARS = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k|mm|m|bn|b)?\b", re.I)
_CLOCK = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\.?(?![a-z])", re.I)
_NOON = re.compile(r"\b(noon|midday|midnight)\b", re.I)


def _dollars(m: re.Match[str]) -> str:
    scale = _SCALE_SUFFIX.get((m.group(2) or "").lower())
    return f" {m.group(1)} {scale + ' ' if scale else ''}dollars "


def _clock(m: re.Match[str]) -> str:
    hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
    if m.group(3).lower() == "p":
        hour += 12
    return f" t{hour:02d}{minute:02d} "


def normalize(text: str) -> str:
    """Curly quotes, contractions, citation markers, meta-preambles, dashes and ellipses as clause
    punctuation, units ("15%" -> "15 percent", "$1.5M" -> "1.5 million dollars") and clock times
    ("10 a.m.", "10am", "10:00 AM" -> one token, t1000)."""
    t = text.translate(_CURLY)
    for pattern, replacement in _CONTRACTIONS:
        t = pattern.sub(replacement, t)
    t = _MARKER.sub(" ", t)
    t = _PREAMBLE.sub(" ", t)
    t = _ELLIPSIS.sub(" … ", t)
    t = _DASH.sub(" — ", t)
    t = _THOUSANDS.sub("", t)
    t = _PERCENT.sub(r"\1 percent", t)
    t = _DOLLARS.sub(_dollars, t)
    t = _CLOCK.sub(_clock, t)
    return _NOON.sub(lambda m: " t0000 " if m.group(1).lower() == "midnight" else " t1200 ", t)


_ACRONYM = re.compile(r"\b[A-Z]{2,5}\b")
_SHORT_WORDS = frozenset("it us am or an as at be do he me my of on so to up we if in is".split())


def _acronym(m: re.Match[str]) -> str:
    """'IT', 'US', 'AM' would lowercase into stopwords; keep them as distinct content words."""
    word = m.group(0).lower()
    return f"x{word}" if word in _SHORT_WORDS else m.group(0)


def tokens(text: str) -> list[str]:
    """Lowercased word/number tokens plus clause punctuation, after normalization."""
    return _TOKEN.findall(_ACRONYM.sub(_acronym, normalize(text)).lower())


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text)


def words(text: str) -> list[str]:
    """Lowercased word/number tokens, no punctuation."""
    return [t for t in tokens(text) if t not in _BREAK_PUNCT]


# --------------------------------------------------------------------------- numbers

_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split())
    if w != "_"
}
_ORDINALS = {
    w: i + 1
    for i, w in enumerate(
        "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
        "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth".split()
    )
} | {"twentieth": 20, "thirtieth": 30}
_SCALES = {"thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
ORDINAL_WORDS = frozenset(_ORDINALS)
_ORDINAL_DIGITS = re.compile(r"(\d+)(?:st|nd|rd|th)")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
_TIME = re.compile(r"t(\d{2})(\d{2})")
_NUMBER_WORDS = frozenset(_UNITS) | frozenset(_TENS) | frozenset(_ORDINALS) | {"hundred"}


def _is_number_token(t: str) -> bool:
    return (
        t in _NUMBER_WORDS or t in _SCALES
        or bool(_DIGITS.fullmatch(t) or _ORDINAL_DIGITS.fullmatch(t) or _TIME.fullmatch(t))
    )  # fmt: skip


@dataclass(frozen=True)
class Num:
    start: int  # token span [start, end)
    end: int
    value: str  # canonical: "15", "1500000", "3.10" (decimals keep their digits), "t19:00"
    ordinal: bool  # "15th" / "fifteenth" — a date or a rank, not a count


def _canonical(value: Decimal) -> str:
    return str(int(value)) if value == value.to_integral_value() else format(value.normalize(), "f")


def _parse_numbers(toks: list[str]) -> list[Num]:
    """Every number in a token list. Number words accumulate the usual way ("two hundred fifty
    thousand" = 250000, "a thousand" = 1000); a scale word after digits multiplies them ("1.5
    million"). Decimals written without a scale keep their digits, so "3.10" is not "3.1"."""
    out: list[Num] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if m := _TIME.fullmatch(t):
            out.append(Num(i, i + 1, f"t{m.group(1)}:{m.group(2)}", False))
            i += 1
            continue
        if _DIGITS.fullmatch(t) or _ORDINAL_DIGITS.fullmatch(t):
            ordinal = not _DIGITS.fullmatch(t)
            digits = _ORDINAL_DIGITS.fullmatch(t).group(1) if ordinal else t  # type: ignore[union-attr]
            j = i + 1
            if not ordinal and j < len(toks) and toks[j] in _SCALES:
                value = _canonical(Decimal(digits) * _SCALES[toks[j]])
                j += 1
            else:
                value = _canonical(Decimal(digits)) if "." not in digits else digits
            out.append(Num(i, j, value, ordinal))
            i = j
            continue
        a_scale = t in ("a", "an") and i + 1 < len(toks) and toks[i + 1] in {"hundred", *_SCALES}
        if t not in _NUMBER_WORDS and not a_scale:
            i += 1
            continue
        total, current, j, ordinal = 0, 0, i, False
        if a_scale:
            current, j = 1, i + 1
        while j < len(toks):
            w = toks[j]
            if w in _UNITS:
                current += _UNITS[w]
            elif w in _TENS:
                current += _TENS[w]
            elif w in _ORDINALS:
                current, ordinal = current + _ORDINALS[w], True
                j += 1
                break  # an ordinal ends the number: "twenty first"
            elif w == "hundred":
                current = (current or 1) * 100
            elif w in _SCALES:
                total, current = total + (current or 1) * _SCALES[w], 0
            elif w == "and" and j + 1 < len(toks) and toks[j + 1] in _NUMBER_WORDS:
                pass  # "two hundred and fifty"
            else:
                break
            j += 1
        out.append(Num(i, j, str(total + current), ordinal))
        i = j
    return out


parse_numbers = _parse_numbers


def numbers(text: str) -> set[str]:
    """Every number the text states, canonical: '1,200' = '1200', 'fifteenth' = '15th' = '15',
    'twenty-five' = '25', 'a thousand' = '1000', '$1.5M' = '1500000', '10 p.m.' = 't22:00'."""
    return {
        n.value for sentence in sentences(text) or [text] for n in _parse_numbers(tokens(sentence))
    }


# --------------------------------------------------------------------------- content words

STOPWORDS = frozenset(
    (
        "a an the of to for from by with as at into about than then so if and or "
        "this that these those it its i you he she we they me him her us them my your his our "
        "their mine yours ours theirs who whom whose which what when where why how many much "
        "is are was were be been being am do does did done doing have has had having "
        "will would shall let also just really very too "
        # meta-vocabulary a model wraps a grounded answer in
        "according based provided context passage passages document documents source sources "
        "cited citation mentioned stated states said says noted per"
    ).split()
)
# Modals stay content: "may" (permission) is not "must" (obligation), "could" is not "will".
NEGATION_CUES = frozenset("not no never none nobody nothing neither nor without cannot".split())
_MULTIWORD_CUES = {
    ("fail", "to"), ("fails", "to"), ("failed", "to"), ("failing", "to"),
    ("hold", "off"), ("holds", "off"), ("held", "off"), ("holding", "off"),
    ("rather", "than"), ("instead", "of"),
}  # fmt: skip
# Verbs and prepositions that deny what they govern. Unlike "not", they are content words too, so
# "cancel the launch" is not supported by "postponed the launch".
_REVERSAL_WORDS = frozenset(
    (
        "against cancel cancels canceled cancelled canceling cancelling postpone postpones "
        "postponed postponing reject rejects rejected rejecting veto vetoes vetoed vetoing scrap "
        "scraps scrapped scrapping decline declines declined declining delay delays delayed "
        "delaying defer defers deferred deferring drop drops dropped dropping abandon abandons "
        "abandoned abandoning object objects objected objecting oppose opposes opposed opposing "
        "block blocks blocked blocking prevent prevents prevented preventing stop stops stopped "
        "stopping halt halts halted halting shelve shelves shelved shelving refuse refuses "
        "refused refusing suspend suspends suspended suspending"
    ).split()
)
_CLAUSE_BREAK = frozenset([",", ";", ":", "\u2014", "\u2026", "but", "although", "though",
                           "however", "whereas", "while", "yet"])  # fmt: skip
_KEEP_SINGLE_LETTERS = frozenset("a i s".split())  # articles, "I", a stray possessive "s"

_IRREGULAR = {
    "went": "go", "gone": "go", "going": "go", "made": "make", "making": "make", "sent": "send",
    "spent": "spend", "built": "build", "held": "hold", "told": "tell", "wrote": "write",
    "written": "write", "chose": "choose", "chosen": "choose", "took": "take", "taken": "take",
    "gave": "give", "given": "give", "ran": "run", "began": "begin", "begun": "begin",
    "brought": "bring", "bought": "buy", "kept": "keep", "met": "meet", "paid": "pay",
}  # fmt: skip
_VOWELS = frozenset("aeiou")


def stem(word: str) -> str:
    """A small, predictable stemmer: plurals and -s, -ed, -ing, a doubled final consonant, trailing
    e's, and a table of common irregular verbs — so fix/fixed/fixes, move/moved/moving,
    use/used, plan/planned and cancel/cancelled meet. It conflates nothing else on purpose: a
    stemmer that merges two different words lets a wrong claim through; one that misses an
    inflection only refuses a right one, the safe direction."""
    w = _IRREGULAR.get(word, word)
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "shes", "ches", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ed"):
        base = w[:-2]
        if len(base) >= 3 and _VOWELS & set(base):
            w = base
        elif len(base) == 2 and base[-1] not in _VOWELS:  # used -> use, aged -> age
            w = w[:-1]
    elif w.endswith("ing"):
        base = w[:-3]
        if len(base) >= 3 and _VOWELS & set(base):
            w = base
        elif len(base) == 2 and base[0] in _VOWELS:  # using -> use
            w = base + "e"
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in _VOWELS:
        w = w[:-1]  # planned -> plann -> plan, cancelled -> cancel, passed -> pas
    while len(w) > 3 and w.endswith("e"):
        w = w[:-1]  # update -> updat, agree -> agr
    return w


def _is_content(t: str) -> bool:
    return (
        (len(t) > 1 or t not in _KEEP_SINGLE_LETTERS)
        and t not in STOPWORDS
        and t not in NEGATION_CUES
        and t not in _CLAUSE_BREAK
        and not _is_number_token(t)
    )


def content_stems(text: str) -> list[str]:
    """Stemmed content words. Numbers (digits or words) are excluded: they are checked exactly."""
    return [stem(t) for t in tokens(text) if _is_content(t)]


# --------------------------------------------------------------------------- clauses and polarity

_SUBJECTS = frozenset("i we you they he she".split())
_COORDINATORS = frozenset("and or so then".split())
_CONDITIONS = frozenset("if unless whether because".split())
_HYPOTHETICAL_OPENERS = frozenset("if unless whether".split())
_REPORTING = frozenset(
    (
        "think thinks thought believe believes believed expect expects expected guess suppose "
        "supposed feel feels felt agree agrees agreed say says said suggest suggests suggested "
        "propose proposes proposed want wants wanted hope hopes hoped know knows knew mean means "
        "meant recommend recommends recommended decide decides decided assume assumes assumed "
        "doubt doubts doubted wish insist insisted sure that"
        # auxiliaries: "can you send ...", "should we keep ..." invert, they do not start a clause
        " can could will would should shall do does did are is was were have has had may might must"
    ).split()
)
_HYPOTHETICAL = frozenset(
    (
        "suggest suggests suggested suggesting propose proposes proposed proposing consider "
        "considers considered considering recommend recommends recommended maybe perhaps might "
        "could possibly"
    ).split()
)
_CLAUSE_NEGATORS = [("unlikely",), ("off", "the", "table"), ("ruled", "out"),
                    ("out", "of", "the", "question")]  # fmt: skip
_CORRECTIONS = frozenset("no sorry actually".split())
# "and" / "or" before a new predicate ("... and are not covered", "... or will ship") starts a new
# scope for negation: parity must not carry "not accrue ... and not covered" back to positive.
# Before a noun ("not ship the app and the docs") the scope continues.
_PREDICATE_STARTS = frozenset(
    "are is was were do does did will would can could should must may might has have had not "
    "never also".split()
)
_ARTICLES = frozenset("the a an".split())


@dataclass(frozen=True)
class Tok:
    text: str
    index: int  # position in the sentence's token list
    clause: int
    negated: bool  # inside an odd number of negations in its clause
    hypothetical: bool  # inside a condition, a suggestion or a possibility

    @property
    def is_content(self) -> bool:
        return _is_content(self.text)


def _phrase_at(toks: list[str], i: int, phrase: tuple[str, ...]) -> bool:
    return tuple(toks[i : i + len(phrase)]) == phrase


def analyze(sentence: str) -> list[Tok]:
    """One sentence -> tokens with their clause, negation parity and hypothetical state."""
    toks = tokens(sentence)
    state: list[list[object]] = []  # per token: [clause, negated, hypothetical]
    clause, neg, hyp, start, last_break = 0, False, False, 0, ""
    cues: list[int] = []  # single-word negation cues in the current clause

    def close() -> None:
        members = [k for k in range(start, len(state)) if state[k][0] == clause]
        for c in cues:  # "employees are not": a cue with nothing after it denies what came before
            if not any(_is_content(toks[k]) or _is_number_token(toks[k]) for k in members if k > c):
                for k in members:
                    if k < c:
                        state[k][1] = not state[k][1]
        phrase = {k + j for k in members for ph in _CLAUSE_NEGATORS if _phrase_at(toks, k, ph)
                  for j in range(len(ph))}  # fmt: skip
        if phrase:
            for k in members:  # "launching on the fifteenth is off the table"
                if k not in phrase:
                    state[k][1] = not state[k][1]

    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        hard = t in _CLAUSE_BREAK or t in _CONDITIONS or (t in _COORDINATORS and nxt in _SUBJECTS)
        soft = (
            not hard and t in _SUBJECTS and i > start
            and next((toks[k] for k in range(i - 1, start - 1, -1) if toks[k] != "that"), "")
            not in _REPORTING
        )  # fmt: skip
        if hard or soft:
            close()
            clause, neg, cues = clause + 1, False, []
            hyp = t in _HYPOTHETICAL_OPENERS
            start, last_break = i, (t if hard else "")
            if hard:
                state.append([clause, False, hyp])
                start, i = i + 1, i + 1
                continue
        if t in _CORRECTIONS and i == start and last_break in ("\u2014", "\u2026"):
            for k in range(len(state)):  # "the fifteenth — no, the twentieth": retract
                if state[k][0] == clause - 1:
                    state[k][1] = not state[k][1]
            state.append([clause, False, hyp])
            i += 1
            continue
        if t in ("and", "or", "nor") and nxt in _PREDICATE_STARTS:
            neg = False
        if (t, nxt) in _MULTIWORD_CUES:
            state += [[clause, neg, hyp], [clause, neg, hyp]]
            neg, i = not neg, i + 2
            continue
        state.append([clause, neg, hyp])
        if t in NEGATION_CUES:
            neg = not neg
            cues.append(i)
        elif t in _REVERSAL_WORDS:
            neg = not neg
        elif t in _HYPOTHETICAL:
            hyp = True
        i += 1
    close()
    return [Tok(t, k, int(st[0]), bool(st[1]), bool(st[2]))  # type: ignore[call-overload]
            for k, (t, st) in enumerate(zip(toks, state, strict=True))]  # fmt: skip


def negated_stems(text: str) -> set[str]:
    """Content stems that are negated in their clause."""
    return {
        stem(t.text) for s in sentences(text) or [text] for t in analyze(s)
        if t.negated and t.is_content
    }  # fmt: skip


def signed_numbers(text: str) -> set[tuple[str, bool]]:
    """Numbers with whether a negation scopes them: "the fifteenth, not the tenth" gives
    {("15", False), ("10", True)} — so it supports "the fifteenth" and refutes "the tenth"."""
    out: set[tuple[str, bool]] = set()
    for sentence in sentences(text) or [text]:
        analyzed = analyze(sentence)
        out |= {(n.value, analyzed[n.start].negated)
                for n in _parse_numbers([t.text for t in analyzed])}  # fmt: skip
    return out


# --------------------------------------------------------------------------- claim vs evidence


@dataclass(frozen=True)
class _Num:
    value: str
    negated: bool
    hypothetical: bool
    left: str | None  # nearest word before it in its clause (articles skipped)
    right: str | None


def _clause_numbers(analyzed: list[Tok], clause: int) -> list[_Num]:
    words = [t.text for t in analyzed]
    out = []
    for n in _parse_numbers(words):
        if analyzed[n.start].clause != clause:
            continue
        before = [t for t in analyzed[: n.start] if t.clause == clause and t.text not in _ARTICLES]
        after = [t for t in analyzed[n.end :] if t.clause == clause and t.text not in _ARTICLES]
        out.append(_Num(n.value, analyzed[n.start].negated, analyzed[n.start].hypothetical,
                        before[-1].text if before else None,
                        after[0].text if after else None))  # fmt: skip
    return out


@dataclass(frozen=True)
class Comparison:
    """How well one evidence sentence supports a claim, clause by clause."""

    coverage: float
    missing: tuple[str, ...]
    flipped: tuple[str, ...]  # present only with the other negation / hypothetical state
    absent_numbers: tuple[str, ...]
    misbound_numbers: tuple[str, ...]  # present, but attached to different words

    def problem(self, min_support: float = 1.0) -> str | None:
        if self.coverage < min_support:
            return (f"no cited sentence says it: best covers {self.coverage:.0%} of its words "
                    f"(missing: {', '.join(self.missing)})")  # fmt: skip
        if self.absent_numbers:
            return f"number(s) {', '.join(self.absent_numbers)} not in the cited sentence"
        if self.flipped:
            return f"negation differs from the cited sentence on: {', '.join(self.flipped)}"
        if self.misbound_numbers:
            return (f"number(s) {', '.join(self.misbound_numbers)} attached to different words in "
                    "the cited sentence")  # fmt: skip
        return None


def _compare_clause(claim: list[Tok], cnums: list[_Num], ev: list[Tok], enums: list[_Num],
                    min_support: float) -> Comparison:  # fmt: skip
    have: dict[str, set[tuple[bool, bool]]] = {}
    for t in ev:
        if t.is_content:
            have.setdefault(stem(t.text), set()).add((t.negated, t.hypothetical))
    want = [(stem(t.text), (t.negated, t.hypothetical)) for t in claim if t.is_content]
    stems = list(dict.fromkeys(s for s, _ in want))
    missing = tuple(s for s in stems if s not in have)
    coverage = 1 - len(missing) / len(stems) if stems else 1.0
    flipped = [s for s, st in want if s in have and st not in have[s]]
    absent, misbound = [], []
    for n in cnums:
        same = [e for e in enums if e.value == n.value]
        if not same:
            absent.append(n.value)
        elif not any((e.negated, e.hypothetical) == (n.negated, n.hypothetical) for e in same):
            flipped.append(n.value)
        elif not any(
            (e.negated, e.hypothetical) == (n.negated, n.hypothetical)
            and (n.left is None or n.left == e.left or (n.right is not None and n.right == e.right))
            for e in same
        ):
            misbound.append(n.value)
    del min_support
    return Comparison(coverage, missing, tuple(dict.fromkeys(flipped)), tuple(absent),
                      tuple(misbound))  # fmt: skip


def compare(claim: str, evidence: str, *, min_support: float = 1.0) -> Comparison:
    """Every clause of the claim must be supported by one clause of the evidence sentence: its
    content words present with the same negation and hypothetical state, its numbers present with
    the same state and attached to the same neighbouring word. Returns the worst claim clause."""
    ev = analyze(evidence)
    ev_clauses = sorted({t.clause for t in ev})
    worst: Comparison | None = None
    for sentence in sentences(claim) or [claim]:
        cl = analyze(sentence)
        for c in sorted({t.clause for t in cl}):
            ctoks = [t for t in cl if t.clause == c]
            cnums = _clause_numbers(cl, c)
            if not any(t.is_content for t in ctoks) and not cnums:
                continue
            best: Comparison | None = None
            for e in ev_clauses:
                r = _compare_clause(ctoks, cnums, [t for t in ev if t.clause == e],
                                    _clause_numbers(ev, e), min_support)  # fmt: skip
                key = (r.coverage, r.problem(min_support) is None)
                if best is None or key > (best.coverage, best.problem(min_support) is None):
                    best = r
            if best is None:
                best = Comparison(
                    0.0, tuple(stem(t.text) for t in ctoks if t.is_content), (), (), ()
                )
            rank = (best.coverage, best.problem(min_support) is None)
            if worst is None or rank < (worst.coverage, worst.problem(min_support) is None):
                worst = best
    return worst or Comparison(1.0, (), (), (), ())


def polarity_conflicts(claim: str, evidence: str) -> set[str]:
    """Claim words (or numbers) the evidence states only with the opposite polarity."""
    return set(compare(claim, evidence).flipped)
