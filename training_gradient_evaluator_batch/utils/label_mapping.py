import json
import os
import re
from typing import Dict, List, Tuple


def _normalize_label(s: str) -> str:
    """Normalize a class label for robust matching.

    - Lowercase
    - Replace underscores with spaces
    - Remove non-letter characters
    - Collapse multiple spaces
    """
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _letters_set(s: str) -> set:
    return set(re.findall(r"[a-z]", s.lower()))


def letters_common_ratio(a: str, b: str) -> float:
    """Compute overlap ratio between sets of letters in a and b.

    ratio = |A∩B| / max(|A|, |B|)
    """
    A = _letters_set(a)
    B = _letters_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = max(len(A), len(B))
    return inter / max(denom, 1)


def load_logit_class_list(json_path: str) -> List[str]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"logit class mapping json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Expected JSON to be a list[str] of class names in logit order")
    return data


def load_wnid_to_name_txt(mapping_txt_path: str) -> Dict[str, str]:
    """Load mapping from wnid to human-readable class name from the txt file.

    Expected CSV format per line: wnid,index,class_name
    """
    if not os.path.exists(mapping_txt_path):
        raise FileNotFoundError(f"wnid->name mapping txt not found: {mapping_txt_path}")
    mapping: Dict[str, str] = {}
    with open(mapping_txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            wnid = parts[0].strip()
            # parts[1] is index (1-based or 0-based not used here)
            name = parts[2].strip()
            mapping[wnid] = name
    if not mapping:
        raise ValueError("Parsed empty wnid->name mapping from txt")
    return mapping


def build_wnid_to_logit_index(
    wnid_to_name: Dict[str, str],
    logit_class_list: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build mapping from wnid to index in model logit order using case-insensitive string matching.

    Returns:
      - wnid_to_index: wnid -> index in logits
      - index_to_name: index -> original class name string from logit list
    """
    index_to_name: Dict[int, str] = {i: name for i, name in enumerate(logit_class_list)}
    name_to_index: Dict[str, int] = { name.strip().casefold(): i for i, name in enumerate(logit_class_list) }

    wnid_to_index: Dict[str, int] = {}
    for wnid, raw_name in wnid_to_name.items():
        key = raw_name.strip().casefold()
        if key in name_to_index:
            wnid_to_index[wnid] = name_to_index[key]
        else:
            raise ValueError(
                f"Could not map wnid {wnid} ('{raw_name}') to any logit class by case-insensitive match"
            )

    return wnid_to_index, index_to_name


def is_exact_match(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


