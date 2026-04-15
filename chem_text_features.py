from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


VISUAL_LATIN_MAP = str.maketrans(
    {
        "С": "C",
        "с": "c",
        "Н": "H",
        "н": "h",
        "О": "O",
        "о": "o",
        "Р": "P",
        "р": "p",
        "А": "A",
        "а": "a",
        "М": "M",
        "м": "m",
        "К": "K",
        "к": "k",
        "В": "B",
        "в": "b",
        "Е": "E",
        "е": "e",
        "Т": "T",
        "т": "t",
        "Х": "X",
        "х": "x",
        "І": "I",
        "і": "i",
    }
)

MATCH_TRANSLIT_MAP = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "j", "з": "z",
        "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
        "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh",
        "щ": "sh", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
        "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E", "Ж": "J", "З": "Z",
        "И": "I", "Й": "I", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
        "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "Ts", "Ч": "Ch", "Ш": "Sh",
        "Щ": "Sh", "Ъ": "", "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
    }
)

CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
FORMULA_RE = re.compile(r"C\s*(\d+)\s*H\s*(\d+)", re.IGNORECASE)
RATIO_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[:/]\s*(\d+(?:[.,]\d+)?)")
ELEMENT_RE = re.compile(r"\[([A-Z][a-z]?|[cnops])(?:[^\]]*)\]|Br|Cl|Na|Ca|Mg|Zn|Mo|Li|[BCNOPSFIbcnops]")
C_TOKEN_RE = re.compile(r"[Cc](\d+)")


@dataclass
class TextFeatureResult:
    normalized_value: Optional[str]
    numeric_features: Dict[str, float]


def normalize_text(raw: Optional[str]) -> str:
    text = (raw or "").strip()
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_formula_text(raw: Optional[str]) -> str:
    return normalize_text(raw).translate(VISUAL_LATIN_MAP)


def normalize_match_text(raw: Optional[str]) -> str:
    text = normalize_text(raw)
    text = text.translate(MATCH_TRANSLIT_MAP)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def slugify_text(raw: str) -> str:
    text = normalize_text(raw).lower()
    text = text.replace("%", " pct ")
    text = re.sub(r"[^a-z0-9а-я_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80] if text else "unknown"


def is_smiles_like(raw: str) -> bool:
    text = normalize_text(raw)
    if not text:
        return False
    if CAS_RE.fullmatch(text) or (";" not in text and "," in text and all(CAS_RE.fullmatch(part.strip()) for part in text.split(","))):
        return False
    return any(ch in text for ch in "[]=()#/\\") or bool(re.search(r"[cnops]", text))


def parse_formula_descriptors(raw: str) -> Dict[str, float]:
    text = normalize_formula_text(raw).replace(" ", "")
    match = FORMULA_RE.search(text)
    if not match:
        return {}
    c_count = float(match.group(1))
    h_count = float(match.group(2))
    return {
        "formula_c_count": c_count,
        "formula_h_count": h_count,
        "formula_hc_ratio": h_count / c_count if c_count else 0.0,
    }


def parse_ratio_descriptors(raw: str) -> Dict[str, float]:
    text = normalize_text(raw)
    match = RATIO_RE.search(text)
    if not match:
        return {}
    left = float(match.group(1).replace(",", "."))
    right = float(match.group(2).replace(",", "."))
    total = left + right
    return {
        "ratio_left": left,
        "ratio_right": right,
        "ratio_left_fraction": left / total if total else 0.0,
        "ratio_right_fraction": right / total if total else 0.0,
    }


def parse_chain_numbers(raw: str) -> Dict[str, float]:
    text = normalize_formula_text(raw)
    numbers = [float(m.group(1)) for m in C_TOKEN_RE.finditer(text)]
    if not numbers:
        return {}
    return {
        "chain_count_mean": sum(numbers) / len(numbers),
        "chain_count_max": max(numbers),
        "chain_count_n": float(len(numbers)),
    }


def parse_smiles_descriptors(raw: str) -> Dict[str, float]:
    text = normalize_text(raw)
    if not is_smiles_like(text):
        return {}

    tokens = []
    aromatic_atoms = 0
    for match in ELEMENT_RE.finditer(text):
        token = match.group(1) or match.group(0)
        if token in {"c", "n", "o", "s", "p"}:
            aromatic_atoms += 1
            token = token.upper()
        tokens.append(token)

    def count_element(name: str) -> float:
        return float(sum(1 for token in tokens if token == name))

    hetero = count_element("N") + count_element("O") + count_element("S") + count_element("P")
    metal = count_element("Ca") + count_element("Mg") + count_element("Zn") + count_element("Mo") + count_element("Na") + count_element("Li")

    descriptors = {
        "smiles_length": float(len(text)),
        "smiles_branch_count": float(text.count("(")),
        "smiles_ring_digit_count": float(sum(ch.isdigit() for ch in text)),
        "smiles_double_bond_count": float(text.count("=")),
        "smiles_triple_bond_count": float(text.count("#")),
        "smiles_charge_count": float(text.count("+") + text.count("-")),
        "smiles_aromatic_atom_count": float(aromatic_atoms),
        "smiles_c_count": count_element("C"),
        "smiles_n_count": count_element("N"),
        "smiles_o_count": count_element("O"),
        "smiles_s_count": count_element("S"),
        "smiles_p_count": count_element("P"),
        "smiles_ca_count": count_element("Ca"),
        "smiles_mg_count": count_element("Mg"),
        "smiles_mo_count": count_element("Mo"),
        "smiles_hetero_count": hetero,
        "smiles_metal_count": metal,
        "smiles_sulfonate_motif_count": float(text.count("S(=O)(=O)") + text.count("S(=O)=O")),
        "smiles_carboxylate_motif_count": float(text.count("C(=O)O") + text.count("C(=O)[O-]")),
        "smiles_has_ionic_pair": float("[O-]" in text and "+2" in text),
    }

    carbon = descriptors["smiles_c_count"]
    descriptors["smiles_hetero_to_carbon_ratio"] = hetero / carbon if carbon else 0.0
    return descriptors


def extract_text_feature_result(property_name: str, raw_value: Optional[str]) -> TextFeatureResult:
    text = normalize_text(raw_value)
    if not text:
        return TextFeatureResult(None, {})

    text_l = text.lower()
    match_text = normalize_match_text(text)
    numeric: Dict[str, float] = {"has_text_value": 1.0, "text_length": float(len(text))}
    normalized_value: Optional[str] = slugify_text(text)

    if property_name == "Тип АО":
        normalized_value = "diphenylamine" if "difenilamin" in match_text else "phenol" if "fenol" in match_text else normalized_value
        numeric["ao_type_is_diphenylamine"] = 1.0 if normalized_value == "diphenylamine" else 0.0
        numeric["ao_type_is_phenol"] = 1.0 if normalized_value == "phenol" else 0.0

    elif property_name == "Класс субстрата":
        metal = "calcium" if "kaltsi" in match_text else "magnesium" if "magni" in match_text else "other"
        substrate = "sulfonate" if "sulfonat" in match_text else "salicylate" if "salitsilat" in match_text else "phenate" if "fenolyat" in match_text else "other"
        normalized_value = f"{metal}_{substrate}"
        numeric["substrate_is_sulfonate"] = 1.0 if substrate == "sulfonate" else 0.0
        numeric["substrate_is_salicylate"] = 1.0 if substrate == "salicylate" else 0.0
        numeric["substrate_is_phenate"] = 1.0 if substrate == "phenate" else 0.0
        numeric["substrate_has_calcium"] = 1.0 if metal == "calcium" else 0.0
        numeric["substrate_has_magnesium"] = 1.0 if metal == "magnesium" else 0.0

    elif property_name == "Структура УВ-радикала":
        normalized_value = "alkyl_formula"
        numeric.update(parse_formula_descriptors(text))

    elif property_name == "Класс полиамина":
        normalized_value = "tepa" if "tepa" in match_text else slugify_text(text)
        numeric["polyamine_is_tepa"] = 1.0 if normalized_value == "tepa" else 0.0

    elif property_name == "Модификация":
        normalized_value = "boronated" if "bor" in match_text else "non_boronated" if "nebor" in match_text else slugify_text(text)
        numeric["is_boronated"] = 1.0 if normalized_value == "boronated" else 0.0

    elif property_name == "Тип сукцинимида":
        normalized_value = "bis" if "bis" in match_text else "mono" if "mono" in match_text else slugify_text(text)
        numeric["succinimide_is_bis"] = 1.0 if normalized_value == "bis" else 0.0
        numeric["succinimide_is_mono"] = 1.0 if normalized_value == "mono" else 0.0

    elif property_name == "Тип спиртового радикала":
        normalized_value = "mixed" if "smesh" in match_text else "primary" if "pervich" in match_text else "secondary" if "vtorich" in match_text else slugify_text(text)
        numeric["radical_is_primary"] = 1.0 if normalized_value == "primary" else 0.0
        numeric["radical_is_secondary"] = 1.0 if normalized_value == "secondary" else 0.0
        numeric["radical_is_mixed"] = 1.0 if normalized_value == "mixed" else 0.0

    elif property_name == "Разветвленность радикала / радикалов":
        normalized_value = "mixed" if "%" in text or "," in text or "/" in text else "single"
        numeric["branching_is_mixed"] = 1.0 if normalized_value == "mixed" else 0.0
        numeric["branching_has_iso"] = 1.0 if "izo" in match_text or "i " in match_text else 0.0
        numeric["branching_has_ethylhexyl"] = 1.0 if "etilgeks" in match_text or "ethylhex" in match_text else 0.0
        numeric.update(parse_chain_numbers(text))

    elif property_name == "Происхождение":
        normalized_value = "hydrocracking" if "gidrokreking" in match_text else "mineral" if "mineral" in match_text else "synthetic" if "sintet" in match_text else slugify_text(text)
        numeric["origin_is_hydrocracking"] = 1.0 if normalized_value == "hydrocracking" else 0.0
        numeric["origin_is_mineral"] = 1.0 if normalized_value == "mineral" else 0.0
        numeric["origin_is_synthetic"] = 1.0 if normalized_value == "synthetic" else 0.0

    elif property_name == "Группа по API":
        group_match = re.search(r"\b([IVX]+)\b", text.upper())
        normalized_value = f"group_{group_match.group(1).lower()}" if group_match else slugify_text(text)

    elif property_name == "Категория":
        normalized_value = "modtc" if "modtc" in match_text else "modtp" if "modtp" in match_text else "amine_complex" if "amin" in match_text else slugify_text(text)
        numeric["category_is_modtc"] = 1.0 if normalized_value == "modtc" else 0.0
        numeric["category_is_modtp"] = 1.0 if normalized_value == "modtp" else 0.0
        numeric["category_is_amine_complex"] = 1.0 if normalized_value == "amine_complex" else 0.0

    elif property_name == "Тип лиганда":
        normalized_value = (
            "dithiocarbamate" if "ditiokarb" in match_text else
            "dithiophosphate" if "ditiofosfat" in match_text else
            "amide" if "amid" in match_text else
            slugify_text(text)
        )
        numeric["ligand_is_dithiocarbamate"] = 1.0 if normalized_value == "dithiocarbamate" else 0.0
        numeric["ligand_is_dithiophosphate"] = 1.0 if normalized_value == "dithiophosphate" else 0.0
        numeric["ligand_is_amide"] = 1.0 if normalized_value == "amide" else 0.0
        numeric.update(parse_chain_numbers(text))

    elif property_name == "Соотношение мономеров (EO:PO)":
        normalized_value = "eo_po_ratio"
        numeric.update(parse_ratio_descriptors(text))

    elif property_name in {"SMILES для наиболее вероятной (средней) молекулы сульфокислоты", "Номер CAS / SMILES"}:
        if is_smiles_like(text):
            normalized_value = "smiles_like"
            numeric.update(parse_smiles_descriptors(text))
        elif CAS_RE.search(text):
            normalized_value = "cas_like"
            numeric["cas_count"] = float(len(CAS_RE.findall(text)))
        else:
            normalized_value = slugify_text(text)

    elif property_name in {"Номер CAS", "CAS"}:
        normalized_value = "cas_like" if CAS_RE.search(text) else slugify_text(text)
        numeric["cas_count"] = float(len(CAS_RE.findall(text)))

    else:
        if is_smiles_like(text):
            numeric.update(parse_smiles_descriptors(text))

    return TextFeatureResult(normalized_value, numeric)
