import os
import csv
import json
from typing import Tuple, Dict, List


def default_hierarchy_json_path() -> str:
	return os.path.join('bars', 'imagenet_synset_hierarchy.json')


def load_imagenet_hierarchy(path: str) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, str]]:
	with open(path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	wnid_to_idx: Dict[str, int] = {}
	idx_to_words: Dict[int, str] = {}
	wnid_to_words: Dict[str, str] = {}
	for wnid, obj in data.items():
		idx = int(obj.get('pytorch_class_id'))
		words = str(obj.get('words', '')).strip()
		wnid_to_idx[wnid] = idx
		wnid_to_words[wnid] = words
		if idx not in idx_to_words:
			idx_to_words[idx] = words
	return wnid_to_idx, idx_to_words, wnid_to_words


def read_imagenet_difficulty_order(csv_path: str) -> List[str]:
	with open(csv_path, "r", encoding="utf-8") as f:
		text = f.read()
	text = text.lstrip("\ufeff").strip()
	if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
		text = text[1:-1]
	# Preserve literal placeholder tokens like "None" so they count toward ranking indices.
	parts = [p.strip() for p in text.split(",")]
	# Remove only empty tokens, not the literal word "None" or "null".
	paths = [p for p in parts if p != ""]
	return paths


def resolve_model_index(models_csv_path: str, model_name: str) -> int:
	with open(models_csv_path, 'r', encoding='utf-8') as f:
		reader = csv.reader(f)
		for row in reader:
			for j, name in enumerate(row):
				if name.strip() == model_name:
					return j
	raise ValueError(f"Model '{model_name}' not found in {models_csv_path}")


