import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


def _read_first_correct(model_csv: str) -> List[Tuple[str, int]]:
	"""
	Read either a first_correct_summary.csv (path, first_correct_step) or
	per-step example_statistics.csv (step, path, correct) and return
	[(path, first_correct_step>=0)].
	"""
	rows: List[Tuple[str, int]] = []
	with open(model_csv, "r", encoding="utf-8") as f:
		r = csv.reader(f)
		head = next(r, None)
		# Try to detect format by header
		is_summary = False
		if head and len(head) >= 2:
			lhs = head[0].strip().lower()
			rhs = head[1].strip().lower()
			if lhs == "path" and ("first_correct" in rhs):
				is_summary = True
		# If summary: read directly
		if is_summary:
			for line in r:
				if len(line) < 2:
					continue
				path, step_str = line[0], line[1]
				try:
					step = int(step_str)
				except Exception:
					continue
				if step >= 0:
					rows.append((path, step))
			return rows
		# Else assume example_statistics.csv: (step, path, correct)
		first_seen: Dict[str, int] = {}
		# If there was a header, continue from current iterator; else we already consumed first data row
		for line in r:
			if len(line) < 3:
				continue
			try:
				step = int(line[0])
			except Exception:
				# Maybe there was no header and this is first row: try again by reading entire file without header
				continue
			path = line[1]
			try:
				correct = int(line[2])
			except Exception:
				continue
			if correct == 1 and path not in first_seen:
				first_seen[path] = step
	# Convert to list
	for p, s in first_seen.items():
		if s >= 0:
			rows.append((p, s))
	return rows


def _read_imagenet_rank(imagenet_csv: str) -> List[str]:
	# CSV is a single comma-separated line of paths ordered by difficulty (0 easiest)
	with open(imagenet_csv, "r", encoding="utf-8") as f:
		text = f.read()
	text = text.lstrip("\ufeff").strip()
	if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
		text = text[1:-1]
	paths = [p.strip() for p in text.split(",") if p.strip()]
	return paths


def analyze_and_plot(model_csv: str, imagenet_csv: str) -> None:
	first_correct = _read_first_correct(model_csv)
	if not first_correct:
		print("No examples with first_correct_step >= 0. Skipping analysis plot.")
		return
	# Rank by first-correct step ascending (lower step = earlier learned)
	first_correct.sort(key=lambda x: x[1])
	learn_order_paths_all = [p for p, _ in first_correct]

	# Universal order (0 easiest -> larger index harder)
	imagenet_paths_all = _read_imagenet_rank(imagenet_csv)

	# Compute overlap set only of images that eventually became correct AND exist in universal list
	set_universal = set(imagenet_paths_all)
	overlap_paths = [p for p in learn_order_paths_all if p in set_universal]
	if len(overlap_paths) < 2:
		print("Not enough overlapping images to plot.")
		return

	# Local ranks among overlap only
	# Learn-order local rank (0..N-1 in the order of earliest learned first)
	learn_local_rank: Dict[str, int] = {p: i for i, p in enumerate(overlap_paths)}
	# Universal-order local rank: take universal list and filter to overlap, preserving universal order
	universal_overlap_order = [p for p in imagenet_paths_all if p in set(overlap_paths)]
	universal_local_rank: Dict[str, int] = {p: i for i, p in enumerate(universal_overlap_order)}

	# Build aligned arrays (in learn-order sequence)
	X = np.array([learn_local_rank[p] for p in overlap_paths], dtype=float)
	Y = np.array([universal_local_rank[p] for p in overlap_paths], dtype=float)

	# Normalize ranks to 0..1 so scales are comparable
	den = max(float(len(overlap_paths) - 1), 1.0)
	Xn = X / den
	Yn = Y / den

	# Correlation
	if Xn.std() > 0 and Yn.std() > 0:
		corr = float(np.corrcoef(Xn, Yn)[0, 1])
	else:
		corr = float('nan')

	out_dir = os.path.dirname(os.path.abspath(model_csv))
	plot_path = os.path.join(out_dir, "order_correlation.png")

	plt.figure(figsize=(7, 6))
	plt.scatter(Xn, Yn, s=8, alpha=0.5)
	plt.xlabel("Normalized learn-order among overlap (0=earliest, 1=latest)")
	plt.ylabel("Normalized universal-order among overlap (0=easiest, 1=hardest)")
	plt.title(f"Overlap size={len(overlap_paths)}  |  Pearson r={corr:.4f}")
	plt.tight_layout()
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"Saved order correlation plot to {plot_path}")


def _default_paths() -> Tuple[str, str]:
	# Programming root is parent of this file's parent
	root = Path(__file__).resolve().parents[1]
	model_csv = str(root / "training_gradient_evaluator" / "outputs" / "example_statistics.csv")
	imagenet_csv = str(root / "bars" / "imagenet_examples_ammended.csv")
	return model_csv, imagenet_csv


def main() -> None:
	parser = argparse.ArgumentParser(description="Analyze and plot order correlation between training first-correct steps and universal difficulty order.")
	default_model_csv, default_imagenet_csv = _default_paths()
	parser.add_argument("--model_csv", type=str, default=default_model_csv, help="Path to first_correct_summary.csv or example_statistics.csv")
	parser.add_argument("--imagenet_csv", type=str, default=default_imagenet_csv, help="Path to bars/imagenet_examples_ammended.csv")
	args = parser.parse_args()

	analyze_and_plot(args.model_csv, args.imagenet_csv)


if __name__ == "__main__":
	main()


