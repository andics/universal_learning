import os
import csv
import json
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


class ResultsWriter:
	HEADER = [
		"example_index", "path", "total_steps_to_epsilon", "total_loss_sum", "final_loss",
		"weight_distance", "softmax_wasserstein", "grad_mass_wasserstein", "universal_difficulty_rank",
		"global_linear_cka"
	]

	def __init__(self, model_out_dir: str) -> None:
		self.model_out_dir = model_out_dir
		self.results_csv = os.path.join(model_out_dir, "single_example_results.csv")
		self._ensure_header()

	def _ensure_header(self) -> None:
		os.makedirs(self.model_out_dir, exist_ok=True)
		if not os.path.exists(self.results_csv):
			with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
				writer = csv.writer(f)
				writer.writerow(self.HEADER)
			return
		try:
			with open(self.results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				rows = list(reader)
			if header is None or any(h not in header for h in self.HEADER):
				with open(self.results_csv, 'w', newline='', encoding='utf-8') as wf:
					writer = csv.writer(wf)
					writer.writerow(self.HEADER)
					for row in rows:
						writer.writerow(row)
		except Exception:
			pass

	def append(self, row: List) -> None:
		with open(self.results_csv, 'a', newline='', encoding='utf-8') as f:
			writer = csv.writer(f)
			writer.writerow(row)

	def build_plots(self, epsilon: float) -> None:
		try:
			with open(self.results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				if not header:
					return
				name_to_idx = {name: i for i, name in enumerate(header)}
				rows = [row for row in reader if row]
			x_steps, x_loss, x_weight, x_soft, x_grad, y_rank = [], [], [], [], [], []
			for row in rows:
				try:
					st = int(float(row[name_to_idx["total_steps_to_epsilon"]]))
					rk = int(float(row[name_to_idx["universal_difficulty_rank"]]))
					if st <= 0:
						continue
					x_steps.append(st)
					x_loss.append(float(row[name_to_idx["total_loss_sum"]]))
					x_weight.append(float(row[name_to_idx["weight_distance"]]))
					y_rank.append(rk)
					if "softmax_wasserstein" in name_to_idx and row[name_to_idx["softmax_wasserstein"]] != '':
						x_soft.append(float(row[name_to_idx["softmax_wasserstein"]]))
					if "grad_mass_wasserstein" in name_to_idx and row[name_to_idx["grad_mass_wasserstein"]] != '':
						x_grad.append(float(row[name_to_idx["grad_mass_wasserstein"]]))
				except Exception:
					continue
			if len(y_rank) < 2:
				return
			# Steps vs Difficulty
			plt.figure(figsize=(10, 6))
			plt.scatter(x_steps, y_rank, alpha=0.7, s=50)
			plt.xlabel("Total SGD Steps to Reach Epsilon")
			plt.ylabel("Universal Difficulty Ranking (1=easiest)")
			plt.title(f"Universal Difficulty vs SGD Steps to Reach Epsilon\n({len(y_rank)} examples, ε={epsilon})")
			plt.grid(True, alpha=0.3)
			corr = np.corrcoef(x_steps, y_rank)[0, 1] if len(y_rank) > 1 else 0.0
			plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
			plt.savefig(os.path.join(self.model_out_dir, "steps_vs_difficulty.png"), dpi=150, bbox_inches='tight')
			plt.close()
			# Loss vs Difficulty
			plt.figure(figsize=(10, 6))
			plt.scatter(x_loss, y_rank, alpha=0.7, s=50, color='red')
			plt.xlabel("Total Loss Sum to Reach Epsilon")
			plt.ylabel("Universal Difficulty Ranking (1=easiest)")
			plt.title(f"Universal Difficulty vs Loss Sum to Reach Epsilon\n({len(y_rank)} examples, ε={epsilon})")
			plt.grid(True, alpha=0.3)
			corr = np.corrcoef(x_loss, y_rank)[0, 1] if len(y_rank) > 1 else 0.0
			plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
			plt.savefig(os.path.join(self.model_out_dir, "loss_sum_vs_difficulty.png"), dpi=150, bbox_inches='tight')
			plt.close()
			# Weight vs Difficulty
			plt.figure(figsize=(10, 6))
			plt.scatter(x_weight, y_rank, alpha=0.7, s=50, color='green')
			plt.xlabel("Weight Distance to Reach Epsilon")
			plt.ylabel("Universal Difficulty Ranking (1=easiest)")
			plt.title(f"Universal Difficulty vs Weight Distance to Reach Epsilon\n({len(y_rank)} examples, ε={epsilon})")
			plt.grid(True, alpha=0.3)
			corr = np.corrcoef(x_weight, y_rank)[0, 1] if len(y_rank) > 1 else 0.0
			plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
			plt.savefig(os.path.join(self.model_out_dir, "weight_distance_vs_difficulty.png"), dpi=150, bbox_inches='tight')
			plt.close()
			# Softmax W1 vs Difficulty
			if len(x_soft) == len(y_rank) and len(x_soft) >= 2:
				plt.figure(figsize=(10, 6))
				plt.scatter(x_soft, y_rank, alpha=0.7, s=50, color='purple')
				plt.xlabel("Softmax Distribution W1 (pre vs post)")
				plt.ylabel("Universal Difficulty Ranking (1=easiest)")
				plt.title(f"Universal Difficulty vs Softmax W1\n({len(y_rank)} examples)")
				plt.grid(True, alpha=0.3)
				corr = np.corrcoef(x_soft, y_rank)[0, 1] if len(y_rank) > 1 else 0.0
				plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
				plt.savefig(os.path.join(self.model_out_dir, "softmax_wasserstein_vs_difficulty.png"), dpi=150, bbox_inches='tight')
				plt.close()
			# Grad-mass W1 vs Difficulty
			if len(x_grad) == len(y_rank) and len(x_grad) >= 2:
				plt.figure(figsize=(10, 6))
				plt.scatter(x_grad, y_rank, alpha=0.7, s=50, color='brown')
				plt.xlabel("Gradient-Mass W1 (first vs final)")
				plt.ylabel("Universal Difficulty Ranking (1=easiest)")
				plt.title(f"Universal Difficulty vs Gradient-Mass W1\n({len(y_rank)} examples)")
				plt.grid(True, alpha=0.3)
				corr = np.corrcoef(x_grad, y_rank)[0, 1] if len(y_rank) > 1 else 0.0
				plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
				plt.savefig(os.path.join(self.model_out_dir, "grad_mass_wasserstein_vs_difficulty.png"), dpi=150, bbox_inches='tight')
				plt.close()
		except Exception:
			return

	def compute_correlations(self) -> dict:
		"""Compute the same correlations used in plots and return as a dict."""
		try:
			with open(self.results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				if not header:
					return {}
				name_to_idx = {name: i for i, name in enumerate(header)}
				rows = [row for row in reader if row]
			x_steps, x_loss, x_weight, x_soft, x_grad, y_rank = [], [], [], [], [], []
			for row in rows:
				try:
					st = int(float(row[name_to_idx["total_steps_to_epsilon"]]))
					rk = int(float(row[name_to_idx["universal_difficulty_rank"]]))
					if st <= 0:
						continue
					x_steps.append(st)
					x_loss.append(float(row[name_to_idx["total_loss_sum"]]))
					x_weight.append(float(row[name_to_idx["weight_distance"]]))
					y_rank.append(rk)
					if "softmax_wasserstein" in name_to_idx and row[name_to_idx["softmax_wasserstein"]] != '':
						x_soft.append(float(row[name_to_idx["softmax_wasserstein"]]))
					if "grad_mass_wasserstein" in name_to_idx and row[name_to_idx["grad_mass_wasserstein"]] != '':
						x_grad.append(float(row[name_to_idx["grad_mass_wasserstein"]]))
				except Exception:
					continue
			if len(y_rank) < 2:
				return {}
			corr_steps = float(np.corrcoef(x_steps, y_rank)[0, 1]) if len(y_rank) > 1 else 0.0
			corr_loss = float(np.corrcoef(x_loss, y_rank)[0, 1]) if len(y_rank) > 1 else 0.0
			corr_weight = float(np.corrcoef(x_weight, y_rank)[0, 1]) if len(y_rank) > 1 else 0.0
			corr_soft = None
			corr_grad = None
			if len(x_soft) == len(y_rank) and len(x_soft) >= 2:
				corr_soft = float(np.corrcoef(x_soft, y_rank)[0, 1])
			if len(x_grad) == len(y_rank) and len(x_grad) >= 2:
				corr_grad = float(np.corrcoef(x_grad, y_rank)[0, 1])
			return {
				"num_points": int(len(y_rank)),
				"steps_vs_rank": corr_steps,
				"loss_sum_vs_rank": corr_loss,
				"weight_distance_vs_rank": corr_weight,
				"softmax_wasserstein_vs_rank": corr_soft,
				"grad_mass_wasserstein_vs_rank": corr_grad,
			}
		except Exception:
			return {}

	def write_correlations(self, overwrite: bool = True) -> None:
		"""Write correlations.json with values matching plot correlations."""
		out_path = os.path.join(self.model_out_dir, "correlations.json")
		if (not overwrite) and os.path.exists(out_path):
			return
		data = self.compute_correlations()
		if not data:
			return
		try:
			with open(out_path, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2)
		except Exception:
			return

	def write_summary(self, epsilon: float) -> None:
		summary_path = os.path.join(self.model_out_dir, "training_summary.json")
		try:
			with open(self.results_csv, 'r', encoding='utf-8') as rf:
				reader = csv.reader(rf)
				header = next(reader, None)
				name_to_idx = {name: i for i, name in enumerate(header or [])}
				rows = [row for row in reader if row]
			results_out = []
			success, fail = 0, 0
			for row in rows:
				try:
					st = int(float(row[name_to_idx["total_steps_to_epsilon"]]))
					res = {
						"path": row[name_to_idx["path"]],
						"total_steps": st,
						"total_loss_sum": float(row[name_to_idx["total_loss_sum"]]),
						"final_loss": float(row[name_to_idx["final_loss"]]),
						"weight_distance": float(row[name_to_idx["weight_distance"]]),
						"rank": int(float(row[name_to_idx["universal_difficulty_rank"]]))
					}
					if "softmax_wasserstein" in name_to_idx:
						res["softmax_wasserstein"] = float(row[name_to_idx["softmax_wasserstein"]]) if row[name_to_idx["softmax_wasserstein"]] != '' else None
					if "grad_mass_wasserstein" in name_to_idx:
						res["grad_mass_wasserstein"] = float(row[name_to_idx["grad_mass_wasserstein"]]) if row[name_to_idx["grad_mass_wasserstein"]] != '' else None
					results_out.append(res)
					if st > 0:
						success += 1
					else:
						fail += 1
				except Exception:
					continue
			summary = {
				"epsilon": float(epsilon),
				"total_examples_attempted": len(rows),
				"successful_examples": success,
				"failed_examples": fail,
				"results": results_out,
			}
			random_meta_path = os.path.join(self.model_out_dir, "meta.json")
			if os.path.exists(random_meta_path):
				try:
					import json as _json
					with open(random_meta_path, 'r', encoding='utf-8') as mf:
						meta = _json.load(mf)
					summary.update(meta)
				except Exception:
					pass
			with open(summary_path, "w", encoding="utf-8") as f:
				json.dump(summary, f, indent=2)
		except Exception:
			return


