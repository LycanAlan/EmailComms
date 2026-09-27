"""Pair-level decoy rules, derived from the train labels (see SUBMISSIONS.md reasoning, 2026-09-27).

Each rule looks only at the raw text of one (S1, record) pair and says "this is a decoy":
  R1 legal_swap    : S1 and record carry different corporate forms that the generator uses for sibling decoys
                     (Inc<->Corp, LLC->Co); true copies keep, drop or harmlessly swap (PC/LP->Inc/LLC) their form.
  R2 decoy_word    : the record adds a word from the generator's sibling-decoy list ("X Holdings", "X Midtown")
                     to a name it otherwise shares with the S1.
  R3 fr_number     : (France only) both sides have a house number and the S1's number is not among the record's.
"""
import re

LEG = {"llc": "llc", "l.l.c.": "llc", "l.l.c": "llc", "inc": "inc", "inc.": "inc", "incorporated": "inc",
       "corp": "corp", "corp.": "corp", "corporation": "corp", "co": "co", "co.": "co", "company": "co",
       "ltd": "ltd", "ltd.": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pc": "pc", "p.c.": "pc",
       "p.c": "pc", "pllc": "pllc", "lp": "lp", "l.p.": "lp", "llp": "llp", "pa": "pa", "p.a.": "pa"}
SWAPS = {("inc", "corp"), ("corp", "inc"), ("llc", "co")}
DECOY = set("group holdings industries ventures enterprises exports overseas public solutions trading technologies "
            "infratech midtown downtown uptown east west north south metro harbor lakeside riverside valley summit "
            "central eastgate westgate northside southside highland coastal greater".split())
WORD = re.compile(r"[a-z0-9]+")


def legal(name):
    return frozenset(LEG[t] for t in re.findall(r"[a-z.]+", name.lower()) if t in LEG)


def words(name):
    return {t for t in WORD.findall(name.lower()) if t not in LEG}


def legal_swap(s1_name, rec_name):
    a, b = legal(s1_name), legal(rec_name)
    return len(a) == 1 and len(b) == 1 and (next(iter(a)), next(iter(b))) in SWAPS


def decoy_word(s1_name, rec_name):
    a, b = words(s1_name), words(rec_name)
    return bool(a & b) and bool((b - a) & DECOY)


def _nums(addr):
    return [n.lstrip("0") or "0" for n in re.findall(r"\d+", addr)]


def fr_number(s1_addr, rec_addr):
    a, b = _nums(s1_addr), _nums(rec_addr)
    return bool(a) and bool(b) and a[0] not in b


def demo():
    assert legal_swap("Capital Prairie Best Corp", "capital prairie best inc")
    assert legal_swap("Juniper LLC", "Juniper Co") and not legal_swap("Smith PC", "Smith Inc")
    assert not legal_swap("Acme Inc", "ACME") and not legal_swap("Acme", "Acme LLC")
    assert decoy_word("Acme Plumbing LLC", "Acme Plumbing Holdings") and not decoy_word("Acme Plumbing", "Acme Plumbing Services")
    assert not decoy_word("Acme Group", "Acme Group Inc") and not decoy_word("Blue Sky", "Evodova Holdings")
    assert fr_number("17 Avenue des Mimosas, La Baule", "18 AVENUE DES MIMOSAS") and not fr_number("7 Rue X", "0007 R. X")
    assert not fr_number("7 Rue X", "Rue X, Lille") and not fr_number("125 Rue X", "0014 Cour G, 125 Rue X")
    print("rules.demo OK")


if __name__ == "__main__":
    demo()
