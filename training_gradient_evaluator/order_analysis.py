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
	# Rank by first-correct step ascending (lower step = easier to learn)
	first_correct.sort(key=lambda x: x[1])
	learn_order_paths = [p for p, _ in first_correct]
	learn_order_ranks = {p: i for i, p in enumerate(learn_order_paths)}

	imagenet_paths = _read_imagenet_rank(imagenet_csv)
	imagenet_ranks = {p: i for i, p in enumerate(imagenet_paths)}

	# Intersect keys
	common = sorted(set(learn_order_ranks.keys()) & set(imagenet_ranks.keys()), key=lambda p: learn_order_ranks[p])
	if len(common) < 2:
		print("Not enough overlapping images to plot.")
		return

	X = np.array([learn_order_ranks[p] for p in common], dtype=float)
	Y = np.array([imagenet_ranks[p] for p in common], dtype=float)

	# Normalize to 0..1 for nicer scale (optional)
	# Xn = X / max(X) if X.max() > 0 else X
	# Yn = Y / max(Y) if Y.max() > 0 else Y

	# Correlation
	if X.std() > 0 and Y.std() > 0:
		corr = float(np.corrcoef(X, Y)[0, 1])
	else:
		corr = float('nan')

	out_dir = os.path.dirname(os.path.abspath(model_csv))
	plot_path = os.path.join(out_dir, "order_correlation.png")

	plt.figure(figsize=(7, 6))
	plt.scatter(X, Y, s=8, alpha=0.5)
	plt.xlabel("Order by training first-correct step (lower = earlier)")
	plt.ylabel("Order by difficulty from imagenet_examples_ammended.csv (lower = easier)")
	plt.title(f"Overlap size={len(common)}  |  Pearson r={corr:.4f}")
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


