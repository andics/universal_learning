import os
import json
import random
from typing import List, Tuple, Dict


class ExampleSelector:
	def __init__(self, output_dir: str, seed: int = 42) -> None:
		self.output_dir = output_dir
		self.seed = seed
		self.persist_path = os.path.join(self.output_dir, "selected_examples.json")

	def select_wrong_examples(
		self,
		wrong_indices: List[int],
		all_paths: List[str],
		difficulty_ordered_paths: List[str],
		root_dir: str | None,
		max_examples: int,
	) -> List[str]:
		def resolve_full(p: str) -> str:
			return os.path.join(root_dir, p) if root_dir and not os.path.isabs(p) else p

		path_to_rank: Dict[str, int] = {p: i for i, p in enumerate(difficulty_ordered_paths)}
		pairs: List[Tuple[str, int]] = []
		for idx in wrong_indices:
			if idx < len(all_paths):
				p = all_paths[idx]
				full = resolve_full(p)
				if os.path.exists(full) and p in path_to_rank:
					pairs.append((p, path_to_rank[p]))

		if os.path.exists(self.persist_path):
			try:
				with open(self.persist_path, 'r', encoding='utf-8') as f:
					persisted = json.load(f)
					if isinstance(persisted, list) and all(isinstance(t, list) and len(t) == 2 for t in persisted):
						pairs = [(str(p), int(r)) for p, r in persisted]
			except Exception:
				pass
		else:
			if len(pairs) > max_examples:
				random.seed(self.seed)
				pairs = random.sample(pairs, max_examples)
			with open(self.persist_path, 'w', encoding='utf-8') as f:
				json.dump([[p, int(r)] for p, r in pairs], f, indent=2)

		pairs.sort(key=lambda x: x[1])
		return [p for p, _ in pairs]

	def load_processed_paths(self, results_csv: str) -> set[str]:
		processed: set[str] = set()
		if not os.path.exists(results_csv):
			return processed
		import csv
		try:
			with open(results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				if header is None:
					return processed
				path_idx = None
				for j, h in enumerate(header):
					if h.strip().lower() == 'path':
						path_idx = j
						break
				if path_idx is not None:
					for row in reader:
						if len(row) > path_idx:
							processed.add(row[path_idx])
		except Exception:
			return processed
		return processed


