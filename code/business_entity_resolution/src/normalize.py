"""Turn raw business names / addresses into canonical, comparable "views".

The same business is written many ways: "Pvt. Ltd." vs "Private Limited",
Hindi script vs Latin, "St" vs "Street", "(ID: 2231)" junk, a website instead
of a name... Normalising first means every later score compares *meaning*,
not formatting.

Views produced per record (all lower-case Latin):
  core   : the name without legal form / noise / filler words
  alt    : the other half of "X formerly known as Y", "X dba Y" ("" if none)
  legal  : legal-form families found in the name, e.g. "ltd", "llc", "sarl"
  phon   : consonant skeleton of core (robust to typos and transliteration)
  nosp   : core with spaces removed (matches "interfaithcenter.com" to "Interfaith Center")
  addr   : address tokens with abbreviations / state names canonicalised
  nums   : numbers found in the address (house, plot, zip...), leading zeros stripped
  nl     : 1 if the raw name was written in a non-Latin script
"""
import re
from anyascii import anyascii

# ---------------------------------------------------------------- names
# token -> legal family. Includes transliterated Hindi spellings seen in the
# training data ("praivet" = private, "elelpi" = LLP ...).
LEGAL = {
    "private": "ltd", "pvt": "ltd", "praivet": "ltd", "praibhet": "ltd", "piraivet": "ltd",
    "praivrr": "ltd", "pra": "ltd", "limited": "ltd", "ltd": "ltd", "limitet": "ltd",
    "limirrd": "ltd", "li": "ltd", "plc": "ltd", "opc": "ltd",
    "llp": "llp", "elelpi": "llp", "llc": "llc", "pllc": "llc", "lp": "lp",
    "inc": "inc", "incorporated": "inc", "corp": "inc", "corporation": "inc", "pc": "inc",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co", "gmbh": "gmbh",
    "sarl": "sarl", "eurl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "sci": "sci",
    "snc": "snc", "ei": "ei", "scop": "scop", "selarl": "selarl",
}
FILLER = {"the", "and", "of", "mr", "mrs", "ms", "messrs", "de", "du", "des", "la", "le", "les", "l", "d", "et"}
# Prefixes the S2/S3 noise adds to ~8% of India names (~55k records each); only 65 India S1 names
# start with one, so they are stripped at position 0 only ("Dr Smt Rama Traders" -> "rama traders").
HONORIFIC = {"smt", "shri", "sri", "dr", "mr", "mrs", "ms"}
M_S = re.compile(r"\bm\s*/\s*s\b\.?")                            # "M/s ABC Traders" (messrs)
NAME_SYN = {"etablissements": "ets", "etablissement": "ets", "societe": "ste"}
ALIAS = re.compile(r"\b(?:formerly known as|also known as|doing business as|trading as|formerly|"
                   r"f/k/a|fka|nee|a/k/a|aka|d/b/a|dba|t/a)\b")
URL = re.compile(r"((?:https?://)?(?:www\.)?)([a-z0-9-]+)\.(?:co\.in|com|net|org|in|fr|biz|info|us|io|co)\b")
ID_TAG = re.compile(r"\(?\bid\s*[:#]?\s*\d+\)?")
PHONE = re.compile(r"\+?\d[\d\s-]{8,}\d")
ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\b\.?")          # l.l.c. -> llc, s.a.r.l -> sarl
LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"}
INNER_DIGIT = re.compile(r"(?<=[a-z])[013456789](?=[a-z])")   # a digit with letters on both sides
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _deleet(tok):
    # "br0thers", "c0astal" -> letters. One digit at the edge of an otherwise alphabetic word of
    # 4+ chars is leet too: "denta1", "5ervices", "6roup" (~100k S2/S3 tokens). "24hr", "8th" stay.
    tok = INNER_DIGIT.sub(lambda m: LEET[m.group(0)], tok)
    if len(tok) >= 4:
        if tok[0] in LEET and tok[1:].isalpha():
            tok = LEET[tok[0]] + tok[1:]
        elif tok[-1] in LEET and tok[:-1].isalpha():
            tok = tok[:-1] + LEET[tok[-1]]
    return tok


def _name_tokens(s):
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"\(p\)", " pvt ", s)
    s = re.sub(r"\((?:india|france|usa|us)\)", " ", s)
    s = ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    toks = [NAME_SYN.get(t, t) for t in (_deleet(t) for t in NON_ALNUM.sub(" ", s).split())]
    out = []
    for t in toks:                                              # "niniex niniex" -> "niniex"
        if not out or out[-1] != t:
            out.append(t)
    return out


def phonetic(text):
    """Consonant skeleton: 'Classic Tech' and Hindi 'klasik tek' both -> 'klsk tk'."""
    res = []
    for t in text.split():
        if t.isdigit():
            res.append(t)
            continue
        t = t.replace("ph", "f").replace("x", "ks")
        t = t[0] + re.sub(r"h", "", t[1:])                      # sh->s, th->t, kh->k
        t = re.sub(r"m(?=[^aeioubpm])", "n", t)                 # Hindi anusvara: 'kmsltimg' ~ 'consulting'
        t = re.sub(r"j$", "s", t)                               # 'proprtij' ~ 'properties'
        t = t.translate(str.maketrans("cqzwy", "kksvi"))
        t = t[0] + re.sub(r"[aeiou]", "", t[1:])
        res.append(re.sub(r"(.)\1+", r"\1", t))
    return " ".join(res)


def norm_name(raw):
    """-> (core, alt, legal, phon, nosp, nl)"""
    nl = int(any(ord(c) > 0x24F for c in raw))
    s = anyascii(raw).lower()
    s = ID_TAG.sub(" ", s)
    s = PHONE.sub(" ", s)
    s = CARE_OF.sub(" ", M_S.sub(" ", s))                           # "M/s X", "S/O X" left stray "m s" tokens
    stems = [m.group(2) for m in URL.finditer(s)]
    # "Name | www.x.com" is appended noise -> drop it; a bare "metrocomponents.com" IS the name -> keep the stem
    s = URL.sub(lambda m: " " if m.group(1) else " " + m.group(2) + " ", s).replace("|", " ")
    if not NON_ALNUM.sub("", s) and stems:                      # the whole name was a website
        s = stems[0]
    parts = ALIAS.split(s, maxsplit=1)
    if len(parts) > 1 and not NON_ALNUM.sub("", parts[0]):      # "AKA Sushi Bar" is a name, not an alias
        parts = [s]
    main, alt = parts[0], (parts[1] if len(parts) > 1 else "")

    def core_legal(x):
        toks = _name_tokens(x)
        while len(toks) > 1 and toks[0] in HONORIFIC:
            toks = toks[1:]
        legal = {LEGAL[t] for t in toks if t in LEGAL}
        core = [t for t in toks if t not in LEGAL and t not in FILLER]
        return core, legal

    core, legal = core_legal(main)
    acore, alegal = core_legal(alt)
    if not core and acore:                                      # "LLC formerly X" edge case
        core, acore = acore, []
    core_s = " ".join(core)
    return (core_s, " ".join(acore), " ".join(sorted(legal | alegal)),
            phonetic(core_s), core_s.replace(" ", ""), nl)


# ---------------------------------------------------------------- addresses
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
# Latin names + the transliterated native-script names found in the training data.
IN_STATES = {
    "andhra pradesh": "ap", "amdhrprdes": "ap", "arunachal pradesh": "arp", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "goa": "goa", "gujarat": "gj", "gujrat": "gj",
    "haryana": "hr", "hriyana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "krnatk": "ka", "kerala": "kl", "kerlm": "kl", "keralam": "kl", "madhya pradesh": "mp",
    "mdhy prdes": "mp", "maharashtra": "mh", "mharastr": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "od isa": "od",
    "punjab": "pb", "pmjab": "pb", "rajasthan": "rj", "rajsthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "tmilnatu": "tn", "telangana": "tg", "telmgan": "tg", "tripura": "trp",
    "uttar pradesh": "up", "uttr prdes": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "pscimbng": "wb", "delhi": "dl", "dilli": "dl", "jammu and kashmir": "jk",
    "jammu kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
# France: records name either the region or the department, so both map to one region code.
FR_REGIONS = {
    "nouvelle aquitaine": "naq", "gironde": "naq",
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf",
    "pays de la loire": "pdl", "loire atlantique": "pdl",
}
STATES = {**US_STATES, **IN_STATES, **FR_REGIONS}
ABBR = {
    "street": "st", "str": "st", "road": "rd", "lane": "ln", "avenue": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "drive": "dr", "court": "ct", "crt": "ct",
    "place": "pl", "circle": "cir", "trail": "trl", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "square": "sq", "expressway": "expy", "freeway": "fwy", "mount": "mt",
    "fort": "ft", "point": "pt", "heights": "hts", "center": "ctr", "centre": "ctr",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "route": "rte", "saint": "st", "sainte": "ste",
    "suite": "ste", "r": "rue", "allee": "all", "impasse": "imp", "chemin": "ch", "chem": "ch",
    "faubourg": "fbg", "near": "nr", "opposite": "opp", "building": "bldg", "floor": "fl",
    "flr": "fl", "sector": "sec", "extension": "extn", "ext": "extn", "district": "dist",
    "apartment": "apt", "apartments": "apt", "apts": "apt",
    # measured swaps between S1 and its true match (train) / confident test pairs (France)
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bengaluru": "bangalore",
    "bis": "b", "t": "ter", "q": "quai", "crs": "cours", "res": "residence", "psg": "passage",
    "pass": "passage", "appartement": "apt", "appt": "apt", "app": "apt", "alle": "all",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
    "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
}
ADDR_DROP = {"null", "none", "no", "nos", "number", "ndeg", "hno", "house", "plot", "shop",
             "door", "flat", "unit", "pmb",
             # place TYPE words, one-sided in 12.6k true pairs ("cdp" never appears in S1 at all)
             "city", "township", "twp", "county", "cdp", "town", "village", "of"}
CARE_OF = re.compile(r"\b[cswd]\s*/\s*o\b")                    # c/o, s/o, w/o, d/o ("care of", "son of" ...)


def _state(segment):
    """A comma-separated component that IS a state/region name -> its code. Only whole
    components are replaced, so 'Washington Street' or 'Rue du Nord' stay untouched."""
    key = re.sub(r"[^a-z]+", " ", segment).strip()
    if key in STATES:
        return " ".join([STATES[key]] + re.findall(r"\d+", segment))   # keep a trailing PIN/ZIP
    return segment


def norm_addr(raw):
    """-> (addr, nums)"""
    s = anyascii(raw).lower()
    s = CARE_OF.sub(" ", s)
    s = " ".join(_state(seg) for seg in s.split(","))
    s = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", s)            # 13th -> 13
    s = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", s)    # 441514gali -> 441514 gali
    toks, nums = [], []
    for t in NON_ALNUM.sub(" ", s).split():
        t = ABBR.get(t, t)
        if t in ADDR_DROP:
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"                            # 00357 -> 357
            nums.append(t)
        toks.append(t)
    return " ".join(toks), " ".join(nums)


def demo():
    core, alt, legal, phon, nosp, nl = norm_name("Pvt. EFS Print Ventures Ltd.")
    assert (core, legal) == ("efs print ventures", "ltd"), (core, legal)
    assert norm_name("Synarcumbra formerly known as Office of Public Works")[:2] == ("synarcumbra", "office public works")
    assert norm_name("interfaithcenter.com")[4] == "interfaithcenter"
    assert norm_name("Office óf Public Wórks (ID: 23374)")[0] == "office public works"
    assert norm_name("Br0thers International School LLP")[:3] == ("brothers international school", "", "llp")
    assert norm_name("Liberty Family LLC Services - 8918170415")[0] == "liberty family services"
    assert norm_name("S.A.R.L. Joliot & Frères")[2] == "sarl"
    assert norm_name("क्लासिक टेक लिमिटेड")[3] == phonetic("classic tech") == "klsk tk"
    assert phonetic("kmsltimg") == phonetic("consulting")
    assert norm_addr("12029 SHERATON LANE, CINCINNATI, OH") == norm_addr("12029 Sheraton Ln, Cincinnati, Ohio")
    assert norm_addr("South Blooming Grove, New York, 00357 Lake Shore Dr")[1] == "357"
    assert norm_addr("4415/14Gali Lotan, दिल्ली")[0] == "4415 14 gali lotan dl"
    assert norm_addr("5739 THIRTEENTH STREET COUAT")[1] == "5739 13"
    assert norm_addr("63 R. DE DIEPPE, LILLE")[0] == "63 rue de dieppe lille"
    assert norm_addr("5 Rue X, Bordeaux, Nouvelle-Aquitaine") == norm_addr("5 RUE X, BORDEAUX, Gironde")
    assert norm_name("Établissements Classes SARL")[0] == norm_name("ETS CLASSES")[0] == "ets classes"
    # regressions found in code review
    assert norm_name("Mr metrocomponents.com")[0] == "metrocomponents"          # bare domain = the name
    assert norm_name("West Ltd | www.westltd.com")[0] == "west"                 # appended website = noise
    assert norm_name("24hr Locksmith")[0] == "24hr locksmith"                   # edge digits are real
    assert norm_name("Specia1ists C0astal")[0] == "specialists coastal"         # inner digits are leet
    assert norm_name("AKA Sushi Bar")[0] == "aka sushi bar"
    assert norm_addr("45 Washington Street, Methuen, MA")[0] == "45 washington st methuen ma"
    assert norm_addr("5 Rue du Nord, Lille, Nord")[0] == "5 rue du nord lille hdf"
    assert norm_addr("S/O Ramesh Kumar, Delhi")[0] == "ramesh kumar dl"
    assert norm_addr("118 O Street, Salt Lake City, UT")[0] == "118 o st salt lake ut"
    assert norm_addr("Pune, Maharashtra 411001") == ("pune mh 411001", "411001")
    # audit round 2 (measured on train true pairs / confident France test pairs)
    assert norm_name("M/s Smt Denta1 5ervices")[0] == norm_name("Dental Services")[0] == "dental services"
    assert norm_name("Sri")[0] == "sri"                                          # a lone word is the name
    assert norm_addr("SAINT ANNE CITY, IL")[0] == norm_addr("St Anne, Illinois")[0]
    assert norm_addr("5 bis Quai X, Keralam")[0] == norm_addr("5B Q X, Kerala")[0] == "5 b quai x kl"
    assert norm_addr("12 Oak Terrace")[0] == norm_addr("12 OAK TER")[0] == "12 oak ter"
    assert norm_addr("12t Rue X")[0] == norm_addr("12 ter Rue X")[0] == "12 ter rue x"
    print("normalize.demo OK")


if __name__ == "__main__":
    demo()
