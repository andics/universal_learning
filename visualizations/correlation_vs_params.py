import os
import json
import csv
import argparse
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


def default_paths() -> Tuple[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, '..'))
    outputs_root = os.path.join(root, 'training_gradient_evaluator_single_loss', 'outputs_15sol')
    mapping_csv = os.path.join(root, 'training_gradient_evaluator_single_loss', 'model_name_mapping.csv')
    return outputs_root, mapping_csv


def load_param_counts(mapping_csv: str) -> Dict[str, float]:
    """Return {model_in_timm: params_in_millions} from mapping CSV."""
    mapping: Dict[str, float] = {}
    with open(mapping_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get('modProj    del_in_timm') or '').strip()
            pcount = (row.get('parameter_count') or '').strip()
            if not name:
                continue
            # Normalize and parse values like '24.4M', '1.6M', or bare '2.38'
            val = pcount.replace(',', '').strip().upper()
            try:
                if val.endswith('M'):
                    val_num = float(val[:-1])
                else:
                    val_num = float(val) if val else float('nan')
                mapping[name] = val_num
            except Exception:
                continue
    return mapping


def read_correlations(path: str) -> Dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def pick_best_corr(corrs: Dict) -> Tuple[str, float]:
    keys = [
        'steps_vs_rank',
        'loss_sum_vs_rank',
        'weight_distance_vs_rank',
        'softmax_wasserstein_vs_rank',
        'grad_mass_wasserstein_vs_rank',
    ]
    best_key = None
    best_val = -float('inf')
    for k in keys:
        v = corrs.get(k, None)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if np.isfinite(vf) and vf > best_val:
            best_val = vf
            best_key = k
    return best_key or 'n/a', (best_val if np.isfinite(best_val) else float('nan'))


def ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def main():
    d_outputs, d_map = default_paths()
    parser = argparse.ArgumentParser(description='Plot and save correlation vs parameter count summary.')
    parser.add_argument('--outputs_root', type=str, default=d_outputs, help='Root outputs directory that contains per-model subfolders')
    parser.add_argument('--mapping_csv', type=str, default=d_map, help='Path to model_name_mapping.csv')
    parser.add_argument('--out_png', type=str, default='correlation_vs_params.png', help='Filename for output plot (saved to outputs_root)')
    parser.add_argument('--out_json', type=str, default='correlation_vs_params.json', help='Filename for output json (saved to outputs_root)')
    args = parser.parse_args()

    outputs_root = args.outputs_root
    mapping_csv = args.mapping_csv
    ensure_dir(outputs_root)

    name_to_params = load_param_counts(mapping_csv)

    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    records: List[Dict] = []

    try:
        subdirs = [d for d in os.listdir(outputs_root) if os.path.isdir(os.path.join(outputs_root, d))]
    except Exception:
        subdirs = []

    for d in sorted(subdirs):
        corr_path = os.path.join(outputs_root, d, 'correlations.json')
        if not os.path.exists(corr_path):
            continue
        corrs = read_correlations(corr_path)
        best_key, best_val = pick_best_corr(corrs)
        # Map model folder name to parameter count via model_in_timm
        params_m = name_to_params.get(d, float('nan'))
        if not np.isfinite(params_m) or not np.isfinite(best_val):
            continue
        xs.append(params_m)
        ys.append(best_val)
        labels.append(d)
        records.append({
            'model': d,
            'params_millions': params_m,
            'best_metric': best_key,
            'best_correlation': best_val,
            'num_points': int(corrs.get('num_points', 0)),
        })

    if xs and ys:
        plt.figure(figsize=(10, 6))
        plt.scatter(xs, ys, alpha=0.8, s=50)
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y), textcoords='offset points', xytext=(4, 4), fontsize=8)
        plt.xlabel('Parameters (millions)')
        plt.ylabel('Best correlation (Pearson r)')
        plt.title(f'Best correlation vs parameter count (n={len(xs)})')
        plt.grid(True, alpha=0.3)
        out_png = os.path.join(outputs_root, args.out_png)
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()

    out_json = os.path.join(outputs_root, args.out_json)
    try:
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'points': records}, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()


