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
		n_valid = len(valid)
		k_total = min(int(max_examples), n_valid)
		if k_total <= 0:
			return []
		# If requesting more than available, take all sorted by rank
		if k_total >= n_valid:
			pairs = [(p, r) for _, p, r in sorted(valid, key=lambda t: t[2])]
			with open(self.persist_path, 'w', encoding='utf-8') as f:
				json.dump([[p, int(r)] for p, r in pairs], f, indent=2)
			return [p for p, _ in pairs]

		# Prepare arrays of ranks and indices
		valid_sorted = sorted(valid, key=lambda t: t[2])
		indices_sorted = np.array([idx for idx, _, _ in valid_sorted], dtype=int)
		ranks_sorted = np.array([r for _, _, r in valid_sorted], dtype=float)

		# For small selections, pick evenly spaced quantiles across the sorted ranks
		if k_total <= 50:
			positions = np.linspace(0, n_valid - 1, num=k_total)
			chosen_pos = []
			used = set()
			for pos in positions:
				cand = int(round(pos))
				# Ensure uniqueness by moving to nearest unused neighbor
				left = cand
				right = cand
				while left >= 0 or right < n_valid:
					pick = None
					if left >= 0 and left not in used:
						pick = left
					elif right < n_valid and right not in used:
						pick = right
					if pick is not None:
						chosen_pos.append(pick)
						used.add(pick)
						break
					left -= 1
					right += 1
			chosen_pos.sort()
			chosen_global_indices = indices_sorted[chosen_pos].tolist()
		else:
			# Inverse-density weighted sampling for near-uniform coverage across rank
			# Try Gaussian KDE if available; fallback to histogram-based density
			weights = None
			try:
				from scipy.stats import gaussian_kde  # type: ignore
				# Normalize ranks to [0, 1] for numerical stability
				r0 = float(ranks_sorted.min())
				r1 = float(ranks_sorted.max())
				span = max(r1 - r0, 1.0)
				x = (ranks_sorted - r0) / span
				kde = gaussian_kde(x)
				dens = kde(x)
				weights = 1.0 / (dens + 1e-8)
			except Exception:
				# Histogram fallback (Sturges' rule for bin count)
				bins = int(np.ceil(np.log2(n_valid) + 1))
				bins = max(5, bins)
				counts, bin_edges = np.histogram(ranks_sorted, bins=bins)
				# Avoid zero counts
				counts = counts + (counts == 0)
				bin_idx = np.digitize(ranks_sorted, bin_edges[:-1], right=False) - 1
				bin_idx = np.clip(bin_idx, 0, len(counts) - 1)
				weights = 1.0 / counts[bin_idx].astype(float)
			# Sample without replacement according to normalized weights
			w = np.asarray(weights, dtype=float)
			w_sum = w.sum()
			if not np.isfinite(w_sum) or w_sum <= 0:
				w = np.ones_like(w) / len(w)
			else:
				w = w / w_sum
			rng = np.random.default_rng(self.seed)
			pick_pos = rng.choice(np.arange(n_valid), size=k_total, replace=False, p=w)
			chosen_pos = np.sort(pick_pos)
			chosen_global_indices = indices_sorted[chosen_pos].tolist()

		# Map back to (path, rank) pairs for persistence
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



