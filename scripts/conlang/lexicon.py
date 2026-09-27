"""Build the conlang lexicon: concepts shared by English, German and Spanish, weighted by frequency.

1. Lemma frequency per language. wordfreq gives WORD-FORM frequencies; UD treebanks give
   P(lemma, class | form). A lemma's frequency is Σ_form P(lemma, class | form) · freq(form), normalised
   to sum to 1 per language, so the three languages weigh equally.
2. Concepts. English lemmas are the pivot. Each claims its most frequent unclaimed German and Spanish
   translation of the same word class (MUSE bilingual dictionaries, both directions). Frequent German or
   Spanish lemmas left over then join a concept whose slot for that language is empty, or found their own.
   A concept's weight is the SUM of its three words' normalised frequencies.
3. Role particles. The Link rule introduces every subject, object and indirect object by a relator
   particle. Their weight is how often those links occur per word in the three treebanks.
4. Forms. Alphabet: 14 consonants `p t k b d g m n l r s f v h` and 5 vowels. A root is C(VC)*; a word
   is root + arity vowel. Root LENGTH follows frequency: the 14 heaviest concepts get one-consonant roots,
   the next 980 get CVC, the rest get CVCVC. With these tier sizes, the most-frequent-gets-shortest
   assignment minimises expected word length (rearrangement inequality). Root SHAPE: within its tier, each
   concept (heaviest first) takes the free root with the highest frequency-weighted similarity to its
   three source words.

  .venv/bin/python scripts/conlang/lexicon.py --out conlang/lexicon.tsv
"""
from __future__ import annotations

import argparse
import itertools
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from wordfreq import word_frequency

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT.parent
TREEBANKS = {
    "en": [CODE / "glossa/corpora/en_ewt/en_ewt-ud-train.conllu", CODE / "satzklar-model/runs/en/train.conllu"],
    "de": [CODE / "germandata/corpora/gsd/de_gsd-ud-train.conllu", CODE / "germandata/corpora/hdt/de_hdt-ud-train-a-1.conllu"],
    "es": [CODE / "satzklar-model/runs/es/train.conllu"],
}
MUSE = ROOT / "data" / "lexicon"
CONS = "ptkbdgmnlrsfvh"
VOWS = "aeiou"
SKIP = {"PUNCT", "SYM", "X", "PROPN"}
CLASSES = ["NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "CCONJ", "SCONJ", "NUM", "PART", "INTJ"]
ROLES = {"nsubj": "SUBJ", "csubj": "SUBJ", "expl": "SUBJ", "obj": "OBJ", "ccomp": "OBJ", "xcomp": "OBJ", "iobj": "DAT"}


def wclass(upos: str) -> str:
    return "VERB" if upos == "AUX" else upos


def read_treebank(lang: str):
    """(form→Counter[(lemma, class)], role-link counts, word count)."""
    forms = defaultdict(Counter)
    roles, words = Counter(), 0
    for path in TREEBANKS[lang]:
        for line in path.open(encoding="utf-8"):
            line = line.lstrip("\ufeff")
            if not line or line[0] == "#" or line == "\n":
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 8 or "-" in c[0] or "." in c[0] or c[3] == "PUNCT":
                continue
            words += 1
            roles[ROLES.get(c[7].split(":")[0], "")] += 1
            if c[3] in SKIP or not c[2].replace("\ufeff", "").isalpha():
                continue
            forms[c[1].lower()][(c[2].replace("\ufeff", "").lower(), wclass(c[3]))] += 1
    return forms, roles, words


def lemma_freqs(lang: str, forms) -> dict[tuple[str, str], float]:
    freq = defaultdict(float)
    for form, dist in forms.items():
        f = word_frequency(form, lang)
        if f == 0:
            continue
        tot = sum(dist.values())
        for key, n in dist.items():
            freq[key] += f * n / tot
    z = sum(freq.values())
    return {k: v / z for k, v in freq.items()}


def cognate_ok(w: str, cl: str) -> bool:
    """Same-spelling matching only for open-class words of 4+ letters: short closed-class strings shared
    across languages (es `to`, `at` inside quoted English) are borrowings, not cognates."""
    return len(w) >= 4 and cl in {"NOUN", "VERB", "ADJ", "ADV"}


def wfreq(Fl: dict, w: str, cl: str) -> float:
    """A word's frequency in its concept's class; core words a treebank tags differently use their top class."""
    return Fl.get((w, cl)) or max((Fl.get((w, k), 0.0) for k in CLASSES), default=0.0)


def muse(a: str, b: str) -> dict[str, list[str]]:
    d = defaultdict(list)
    for p, rev in ((MUSE / f"{a}-{b}.txt", False), (MUSE / f"{b}-{a}.txt", True)):
        for line in p.open(encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2:
                x, y = (parts[1], parts[0]) if rev else parts
                if y not in d[x]:
                    d[x].append(y)
    return d


# ── orthography: source word → conlang letters ───────────────────────────────────────────────────────────
def conlangize(word: str, lang: str) -> str:
    w = word.lower().replace("ß", "ss")
    w = "".join(ch for ch in unicodedata.normalize("NFD", w) if unicodedata.category(ch) != "Mn")
    for a, b in (("sch", "s"), ("sh", "s"), ("ch", "k"), ("th", "t"), ("ph", "f"), ("qu", "k"), ("ck", "k"),
                 ("gue", "ge"), ("gui", "gi"), ("x", "ks"), ("z", "s"), ("w", "v"), ("y", "i")):
        w = w.replace(a, b)
    out = []
    for i, ch in enumerate(w):
        if ch == "c":
            out.append("s" if i + 1 < len(w) and w[i + 1] in "ei" else "k")
        elif ch == "j":
            out.append({"en": "d", "de": "i", "es": "h"}[lang])
        elif ch in CONS or ch in VOWS:
            out.append(ch)
    s = "".join(out)
    dedup = [ch for i, ch in enumerate(s) if i == 0 or ch != s[i - 1] or ch in VOWS]   # tt → t
    return "".join(dedup)


def stem(word: str, lang: str) -> str:
    """Source word as a conlang root candidate: letters mapped, trailing vowels (grammar in the conlang) cut."""
    return conlangize(word, lang).rstrip(VOWS)


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a: str, b: str) -> float:
    return 1 - lev(a, b) / max(len(a), len(b), 1)


def root_space(L: int):
    for cs in itertools.product(CONS, repeat=L):
        for vs in itertools.product(VOWS, repeat=L - 1):
            yield "".join(c + v for c, v in zip(cs, vs + ("",)))


def candidates(stems: list[str], L: int) -> set[str]:
    """Roots of L consonants read off the source stems (order-preserving consonant choices, the vowel that
    follows each chosen consonant), extended with every (vowel, consonant) pair when a stem is too short."""
    out = set()
    for s in stems:
        cpos = [i for i, ch in enumerate(s) if ch in CONS]
        for k in range(min(L, len(cpos)), 0, -1):
            for pick in itertools.combinations(cpos, k):
                parts = []
                for n, i in enumerate(pick):
                    parts.append(s[i])
                    if n < len(pick) - 1:
                        nxt = s[i + 1] if i + 1 < len(s) and s[i + 1] in VOWS else "e"
                        parts.append(nxt)
                base = "".join(parts)
                if k == L:
                    out.add(base)
                else:
                    for ext in itertools.product(VOWS, CONS, repeat=L - k):
                        out.add(base + "".join(ext))
            if len(out) > 400:
                break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=12000, help="lemmas kept per language")
    p.add_argument("--concepts", type=int, default=6000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--force", action="store_true", help="overwrite an existing lexicon (reassigns roots!)")
    a = p.parse_args()
    if a.out.exists() and not a.force:
        # The book's root law: a root is never reassigned. Rebuilding reshuffles every root, so the committed
        # lexicon is frozen; correct senses by editing its rows, and add new words through conlang/book/coin.tsv.
        raise SystemExit(f"{a.out} exists and is frozen (roots are never reassigned); use --force to rebuild anyway")

    F, role_rate = {}, defaultdict(float)
    for lang in ("en", "de", "es"):
        forms, roles, words = read_treebank(lang)
        freq = lemma_freqs(lang, forms)
        F[lang] = dict(sorted(freq.items(), key=lambda kv: -kv[1])[: a.top])
        for r in ("SUBJ", "OBJ", "DAT"):
            role_rate[r] += roles[r] / words / 3
        print(f"{lang}: {len(freq)} lemmas, kept {len(F[lang])}; roles/word "
              + " ".join(f"{r}={roles[r] / words:.3f}" for r in ("SUBJ", "OBJ", "DAT")), flush=True)
    # renormalise over the kept lemmas + role mass so every language sums to 1
    for lang in F:
        z = sum(F[lang].values())
        F[lang] = {k: v / z for k, v in F[lang].items()}

    dicts = {"de": muse("en", "de"), "es": muse("en", "es")}
    rev = {l: defaultdict(list) for l in dicts}
    for l, d in dicts.items():
        for en, outs in d.items():
            for o in outs:
                rev[l][o].append(en)

    # concepts: the curated closed-class core first, then English lemmas by frequency (pivot), each claiming
    # its most frequent unclaimed translation; a translation absent from MUSE falls back to the same spelling
    concepts, owner = [], {}                        # owner[(lang, lemma, class)] -> concept
    for row in (ROOT / "conlang" / "core.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        cl, *cells = row.split("\t")[:4]
        c = {"class": cl, "words": {}}
        for lang, cell in zip(("en", "de", "es"), cells):
            ws = [w for w in cell.split("|") if w and "_" not in w]
            c["words"][lang] = ws
            for w in ws:
                owner.setdefault((lang, w, cl), c)
                owner.setdefault((lang, w, "*"), c)   # core words match whatever class a treebank gives them
        concepts.append(c)
    by_en = {}
    for (lem, cl), f in sorted(F["en"].items(), key=lambda kv: -kv[1]):
        if ("en", lem, "*") in owner or ("en", lem, cl) in owner:
            continue
        c = {"class": cl, "words": {"en": [lem], "de": [], "es": []}}
        owner[("en", lem, cl)] = c
        for l in ("de", "es"):
            cands = dicts[l].get(lem, []) or ([lem] if cognate_ok(lem, cl) else [])
            opts = [(F[l][(t, cl)], t) for t in cands
                    if (t, cl) in F[l] and (l, t, cl) not in owner and (l, t, "*") not in owner]
            if opts:
                t = max(opts)[1]
                c["words"][l].append(t)
                owner[(l, t, cl)] = c
        concepts.append(c)
        by_en[(lem, cl)] = c
    # leftover German/Spanish lemmas join the concept of their most frequent English translation (many-to-one),
    # or the same-spelling English concept, or found a single-language concept
    for l in ("de", "es"):
        for (lem, cl), f in sorted(F[l].items(), key=lambda kv: -kv[1]):
            if (l, lem, cl) in owner or (l, lem, "*") in owner:
                continue
            if word_frequency(lem, "en") > 10 * word_frequency(lem, l):   # English quoted inside de/es text
                continue
            # join only a concept of the SAME class (the core's class-agnostic keys are for lookup, not joining)
            ens = sorted((F["en"].get((e, cl), 0), e) for e in rev[l].get(lem, []) + ([lem] if cognate_ok(lem, cl) else [])
                         if ("en", e, cl) in owner)
            host = owner.get(("en", ens[-1][1], cl)) if ens else None
            if host is None:
                host = {"class": cl, "words": {"en": [], "de": [], "es": []}}
                concepts.append(host)
            if lem not in host["words"][l]:
                host["words"][l].append(lem)
            owner[(l, lem, cl)] = host
    for c in concepts:
        c["freq"] = {l: sum(wfreq(F[l], w, c["class"]) for w in c["words"].get(l, [])) for l in ("en", "de", "es")}
        c["weight"] = sum(c["freq"].values())
        for l in ("en", "de", "es"):
            c[l] = c["words"].get(l, [None])[0] if c["words"].get(l) else None
    for r, name in (("SUBJ", "subject"), ("OBJ", "object"), ("DAT", "to-whom")):
        # a role particle is one word per role link; its weight is that rate on the same per-word scale
        concepts.append({"en": None, "de": None, "es": None, "class": "ROLE", "role": r, "gloss": name,
                         "words": {}, "freq": {}, "weight": 3 * role_rate[r]})
    concepts.sort(key=lambda c: -c["weight"])
    concepts = concepts[: a.concepts]

    # forms: tier by frequency, then best frequency-weighted similarity among free roots
    tiers = [len(CONS), len(CONS) * len(VOWS) * len(CONS)]
    used = set()
    tier_of = lambda rank: 1 if rank < tiers[0] else 2 if rank < tiers[0] + tiers[1] else 3  # noqa: E731
    full = {1: list(root_space(1)), 2: list(root_space(2))}
    for rank, c in enumerate(concepts):
        L = tier_of(rank)
        srcs = [(wfreq(F[l], w, c["class"]) or 1e-9, stem(w, l)) for l in ("en", "de", "es")
                for w in c["words"].get(l, [])]
        wsum = sum(f for f, _ in srcs) or 1.0
        pool = full[L] if L in full else sorted(candidates([s for _, s in srcs], L))
        best, best_score = None, -1.0
        for r in pool:
            if r in used:
                continue
            score = sum(f / wsum * sim(r, s) for f, s in srcs) if srcs else 0.0
            if score > best_score:
                best, best_score = r, score
        if best is None:                                   # every derived candidate taken: first free root
            best = next(r for r in root_space(L) if r not in used)
            best_score = sum(f / wsum * sim(best, s) for f, s in srcs) if srcs else 0.0
        used.add(best)
        c["root"], c["tier"], c["similarity"] = best, L, best_score

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as f:
        f.write("root\tclass\ten\tde\tes\tweight\ttier\tsimilarity\tfreq_en\tfreq_de\tfreq_es\n")
        for c in concepts:
            gloss = c.get("gloss")
            f.write("\t".join([c["root"], c["class"], "|".join(c["words"].get("en", [])) or (f"<{gloss}>" if gloss else ""),
                               "|".join(c["words"].get("de", [])), "|".join(c["words"].get("es", [])),
                               f"{c['weight']:.3e}", str(c["tier"]), f"{c['similarity']:.3f}"]
                              + [f"{c['freq'].get(l, 0):.3e}" for l in ("en", "de", "es")]) + "\n")
    trilingual = sum(1 for c in concepts if all(c["words"].get(l) for l in ("en", "de", "es")))
    stats = {"concepts": len(concepts), "trilingual": trilingual, "tiers": dict(Counter(c["tier"] for c in concepts)),
             "mean_similarity": sum(c["similarity"] for c in concepts) / len(concepts),
             "weighted_mean_root_consonants": sum(c["weight"] * c["tier"] for c in concepts) / sum(c["weight"] for c in concepts)}
    print(json.dumps(stats))
    print("top 40:", ", ".join(f"{c['root']}={c['en'] or c.get('gloss')}/{c['de']}/{c['es']}" for c in concepts[:40]))


if __name__ == "__main__":
    main()
