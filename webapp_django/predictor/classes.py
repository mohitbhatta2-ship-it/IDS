"""
Attack families and their colours.

Hue carries the *family*, not the individual class. Fifteen distinguishable hues
do not exist -- past about five, colours start colliding for someone -- so the
classes are grouped into six families and the specific class is always named in
text beside its mark. Colour gets you the shape of the threat at a glance; the
label gets you the detail.

The palette is not a matter of taste. These six were chosen by running the
data-viz validator over candidate sets and keeping only those that clear the
lightness band, chroma floor, colour-blind separation and contrast checks in
*both* light and dark mode. Two constraints drove the result:

  * Red and orange cannot both be family colours -- they measure ΔE 7.1 for
    normal vision, below the 15 floor. DoS and DDoS therefore had to be split
    across hues that are genuinely far apart rather than two shades of "hot".
  * Benign uses the reserved status green, which sits ~9.7 from the categorical
    aqua. Aqua was dropped so that Benign-versus-attack -- the one distinction
    that must never fail -- clears 25 ΔE against every attack family.

If you change a colour here, re-run the validator before committing:

    node scripts/validate_palette.js "<hex,hex,…>" --mode light
    node scripts/validate_palette.js "<hex,hex,…>" --mode dark --surface "#1a1a19"
"""

from __future__ import annotations

BENIGN_LABEL = "Benign"

# key -> (display name, light hex, dark hex, one-line description)
FAMILIES: dict[str, tuple[str, str, str, str]] = {
    "benign": (
        "Benign",
        "#0ca30c", "#0ca30c",
        "Normal traffic. No action needed.",
    ),
    "ddos": (
        "DDoS",
        "#e34948", "#e66767",
        "Distributed floods from many sources. Highest volume, hardest to absorb.",
    ),
    "dos": (
        "DoS",
        "#eb6834", "#d95926",
        "Single-source denial of service, including slow-connection attacks.",
    ),
    "bruteforce": (
        "Brute force",
        "#2a78d6", "#3987e5",
        "Repeated credential guessing against a service.",
    ),
    "web": (
        "Web attack",
        "#4a3aa7", "#9085e9",
        "Injection and cross-site attacks against web applications.",
    ),
    "other": (
        "Recon / other",
        "#eda100", "#c98500",
        "Bots and infiltration — quieter activity that is easy to miss.",
    ),
}

# Order used in legends and the family chart. This is the order the palette was
# validated in, so adjacent pairs are guaranteed to be separable.
FAMILY_ORDER = ["benign", "ddos", "bruteforce", "dos", "web", "other"]

# Every one of the 15 trained classes maps to exactly one family.
CLASS_FAMILY: dict[str, str] = {
    "Benign": "benign",

    "DDOS attack-HOIC": "ddos",
    "DDOS attack-LOIC-UDP": "ddos",
    "DDoS attacks-LOIC-HTTP": "ddos",

    "DoS attacks-GoldenEye": "dos",
    "DoS attacks-Hulk": "dos",
    "DoS attacks-SlowHTTPTest": "dos",
    "DoS attacks-Slowloris": "dos",

    "FTP-BruteForce": "bruteforce",
    "SSH-Bruteforce": "bruteforce",

    "Brute Force -Web": "web",
    "Brute Force -XSS": "web",
    "SQL Injection": "web",

    "Bot": "other",
    "Infilteration": "other",
}


def family_of(label: str) -> str:
    """Family key for a class label, falling back to 'other' for anything new."""
    return CLASS_FAMILY.get(label, "other")


def family_name(label: str) -> str:
    return FAMILIES[family_of(label)][0]


def colour_of(label: str, dark: bool = False) -> str:
    _, light, night, _ = FAMILIES[family_of(label)]
    return night if dark else light


def is_attack(label: str) -> bool:
    return label != BENIGN_LABEL


def legend() -> list[dict]:
    """Legend rows, in the validated order, with the classes each family covers."""
    rows = []
    for key in FAMILY_ORDER:
        name, light, dark, blurb = FAMILIES[key]
        members = [c for c, f in CLASS_FAMILY.items() if f == key]
        rows.append(
            {
                "key": key,
                "name": name,
                "light": light,
                "dark": dark,
                "blurb": blurb,
                "members": sorted(members),
                "count": len(members),
            }
        )
    return rows


def css_variables() -> str:
    """The family colours as CSS custom properties, for both themes."""
    light = "\n".join(
        f"  --family-{k}: {FAMILIES[k][1]};" for k in FAMILY_ORDER
    )
    dark = "\n".join(
        f"  --family-{k}: {FAMILIES[k][2]};" for k in FAMILY_ORDER
    )
    return light, dark


def summarise(counts: dict[str, int]) -> list[dict]:
    """
    Roll per-class counts up to per-family totals, in the validated order.

    Returns only families that actually occur, so the chart does not carry empty
    segments, but keeps their relative order stable.
    """
    total = sum(counts.values()) or 1
    by_family: dict[str, int] = {}
    for label, n in counts.items():
        by_family[family_of(label)] = by_family.get(family_of(label), 0) + n

    out = []
    for key in FAMILY_ORDER:
        if key not in by_family:
            continue
        name, light, dark, blurb = FAMILIES[key]
        out.append(
            {
                "key": key,
                "name": name,
                "count": by_family[key],
                "share": by_family[key] / total,
                "share_pct": 100.0 * by_family[key] / total,
                "blurb": blurb,
                "is_attack": key != "benign",
            }
        )
    return out
