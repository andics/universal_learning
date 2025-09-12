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

		# Build list of valid wrong examples with their global index and rank
		path_to_rank: Dict[str, int] = {p: i for i, p in enumerate(difficulty_ordered_paths)}
		valid: List[Tuple[int, str, int]] = []  # (global_index, path, rank)
		for idx in wrong_indices:
			if 0 <= idx < len(all_paths):
				p = all_paths[idx]
				# Treat explicit "None" tokens as non-image placeholders
				if isinstance(p, str) and p.strip().lower() in {"none", "null"}:
					continue
				full = resolve_full(p)
				if os.path.exists(full) and p in path_to_rank:
					valid.append((idx, p, path_to_rank[p]))

		# If we have a persisted selection, reuse it for determinism across runs
		if os.path.exists(self.persist_path):
			try:
				with open(self.persist_path, 'r', encoding='utf-8') as f:
					persisted = json.load(f)
					if isinstance(persisted, list) and all(isinstance(t, list) and len(t) == 2 for t in persisted):
						pairs = [(str(p), int(r)) for p, r in persisted]
						pairs.sort(key=lambda x: x[1])
						return [p for p, _ in pairs]
			except Exception:
				pass

		# Nothing persisted: perform simple stratified sampling across ranking bins
		if not valid:
			return []

		import numpy as np
		# Bin by ranking position with fixed bin size (e.g., 100 ranks per bin)
		bin_size = 100
		N_ranks = max(len(difficulty_ordered_paths), 1)
		num_bins = int(np.ceil(N_ranks / float(bin_size)))
		# Map each valid example to its rank bin
		bins: Dict[int, list[int]] = {b: [] for b in range(num_bins)}
		for idx, _, rank in valid:
			b = int(rank // bin_size)
			b = max(0, min(num_bins - 1, b))
			bins[b].append(idx)

		# Determine how many to sample in total (avoid replacement if possible)
		n_valid = len(valid)
		k_total = min(int(max_examples), n_valid)
		# Initial per-bin targets distributed as evenly as possible
		base = k_total // num_bins
		rem = k_total % num_bins
		target = [base + (1 if b < rem else 0) for b in range(num_bins)]
		capacity = [len(bins[b]) for b in range(num_bins)]
		assign = [min(target[b], capacity[b]) for b in range(num_bins)]
		leftover = k_total - int(sum(assign))
		# Redistribute leftover to bins with remaining capacity (round-robin)
		while leftover > 0:
			progress = False
			for b in range(num_bins):
				if assign[b] < capacity[b]:
					assign[b] += 1
					leftover -= 1
					progress = True
					if leftover == 0:
						break
			if not progress:
				break

		# Sample within each bin without replacement using seed for determinism
		rng = np.random.default_rng(self.seed)
		chosen_global_indices: list[int] = []
		for b in range(num_bins):
			cnt = int(assign[b])
			if cnt <= 0 or capacity[b] <= 0:
				continue
			candidates = np.array(bins[b], dtype=int)
			if cnt >= capacity[b]:
				# Take all candidates in deterministic shuffled order
				order = rng.permutation(capacity[b])
				chosen_global_indices.extend(candidates[order].tolist())
			else:
				pick = rng.choice(candidates, size=cnt, replace=False)
				chosen_global_indices.extend(pick.tolist())

		# Map back to (path, rank) pairs for persistence
		# Build index -> (path, rank)
		idx_to_path_rank = {idx: (p, r) for idx, p, r in valid}
		pairs = [idx_to_path_rank[i] for i in chosen_global_indices]
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


