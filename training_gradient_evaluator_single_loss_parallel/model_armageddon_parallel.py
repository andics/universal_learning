import os
import csv
import shlex
import time
import argparse
import subprocess
from datetime import datetime


# Cluster/container execution settings (mirrored from single-worker script)
CONTAINER = 'ops:5000/universal_learning_2.4:1'
QUEUE = 'waic-short'
GPU_SPEC = 'num=1:j_exclusive=yes'
RESOURCES = '-R rusage[mem=256000] -R affinity[thread*24] -R select[hname!=hgn50] -R select[hname!=ibdgx010]'

# Paths
PROG_ROOT = "/home/projects/bagon/andreyg"
SEQ_ARR = "../shared/seq_arr.sh"  # wrapper that repeats a job N times sequentially via -e N
TRAIN_PY = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/training_gradient_evaluator_single_loss_parallel/train_grad_parallel.py"

# Defaults derived from the provided argument sample
DEFAULTS = {
    'bars_npy': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/geq6wrong_21017_geq6correct_1525_imagenet.npy',
    'examples_csv': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/geq6wrong_21017_geq6correct_1525_imagenet_examples_ammended.csv',
    'imagenet_models_csv': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_models.csv',
    'max_examples': 50000,
    'max_steps_per_example': 10000,
    'lr': 0.0001,
    'weight_decay': 0,
    'epsilon': 0.001,
    'device': 'cuda',
    'deterministic': True,
    'cka_layer_fraction': 0.05,
    'num_workers': 8,
    'output_dir': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/training_gradient_evaluator_single_loss_parallel/output_parallel_8_workers',
    'hierarchy_json': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_synset_hierarchy.json',
}

# Default mapping CSV (same structure as single-worker script: columns 'model_in_csv', 'model_in_timm')
DEFAULT_MAPPING_CSV = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_model_name_mapping.csv"


def dict_to_cli(args_dict: dict) -> str:
    parts = []
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {shlex.quote(str(v))}")
    return " ".join(parts)


def build_bsub_command(train_args: str, job_name: str, repeats: int) -> str:
    base = (
        f'bsub -env LSB_CONTAINER_IMAGE="{CONTAINER}" -app docker-gpu '
        f'-gpu {GPU_SPEC} -q {QUEUE} {RESOURCES} '
        f'-o /home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Cluster_runtime/model_training/useCase_out_%J.log '
        f'-e /home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Cluster_runtime/model_training/useCase_err_%J.log '
        f'-J "{job_name}" -H '
        f'python3 {TRAIN_PY} {train_args}'
    )
    return f"{SEQ_ARR} -c \"{base}\" -e {int(repeats)} -d ended"


def load_model_rows(csv_path: str):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mi_csv = (row.get('model_in_csv') or '').strip()
            mi_timm = (row.get('model_in_timm') or '').strip()
            if not mi_csv or not mi_timm:
                continue
            rows.append((mi_csv, mi_timm))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Submit parallel single-example training jobs across workers.")
    parser.add_argument('--mapping_csv', type=str, default=DEFAULT_MAPPING_CSV, help='CSV with columns model_in_csv,model_in_timm')
    parser.add_argument('--max_models', type=int, default=10, help='Number of models (rows) from mapping CSV to submit')
    parser.add_argument('--num_workers', type=int, default=DEFAULTS['num_workers'], help='Number of workers to launch per model')
    parser.add_argument('--repeats_per_worker', type=int, default=14, help='Number of sequential repeats per worker via seq_arr -e')
    # Optional overrides for key defaults
    parser.add_argument('--output_dir', type=str, default=DEFAULTS['output_dir'], help='Base output directory for results')
    parser.add_argument('--bars_npy', type=str, default=DEFAULTS['bars_npy'])
    parser.add_argument('--examples_csv', type=str, default=DEFAULTS['examples_csv'])
    parser.add_argument('--imagenet_models_csv', type=str, default=DEFAULTS['imagenet_models_csv'])
    parser.add_argument('--max_examples', type=int, default=DEFAULTS['max_examples'])
    parser.add_argument('--max_steps_per_example', type=int, default=DEFAULTS['max_steps_per_example'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--weight_decay', type=float, default=DEFAULTS['weight_decay'])
    parser.add_argument('--epsilon', type=float, default=DEFAULTS['epsilon'])
    parser.add_argument('--device', type=str, default=DEFAULTS['device'])
    parser.add_argument('--cka_layer_fraction', type=float, default=DEFAULTS['cka_layer_fraction'])
    parser.add_argument('--hierarchy_json', type=str, default=DEFAULTS['hierarchy_json'])
    parser.add_argument('--deterministic', action='store_true', default=DEFAULTS['deterministic'])
    args = parser.parse_args()

    models = load_model_rows(args.mapping_csv)
    if not models:
        print(f"No valid models found in {args.mapping_csv}")
        return

    # Respect max_models (Y)
    models = models[: max(0, int(args.max_models))]
    if not models:
        print("Nothing to submit: max_models=0")
        return

    print(f"Submitting {len(models)} models x {int(args.num_workers)} workers each; each worker repeats {int(args.repeats_per_worker)} times via seq_arr")

    for midx, (model_in_csv, model_in_timm) in enumerate(models, start=1):
        # Assemble base training args for this model
        base_args = {
            'bars_npy': args.bars_npy,
            'examples_csv': args.examples_csv,
            'model_csv_name': model_in_csv,
            'imagenet_models_csv': args.imagenet_models_csv,
            'max_examples': int(args.max_examples),
            'max_steps_per_example': int(args.max_steps_per_example),
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epsilon': args.epsilon,
            'device': args.device,
            'deterministic': bool(args.deterministic),
            'cka_layer_fraction': float(args.cka_layer_fraction),
            'num_workers': int(args.num_workers),
            'output_dir': args.output_dir,
            'hierarchy_json': args.hierarchy_json,
            'model_name': model_in_timm,
        }

        for worker_id in range(int(args.num_workers)):
            worker_args = dict(base_args)
            worker_args['current_worker'] = int(worker_id)
            cli = dict_to_cli(worker_args)
            job_name = f"train_grad_p_m{midx}_w{worker_id}"
            cmd = build_bsub_command(cli, job_name, repeats=int(args.repeats_per_worker))
            print(f"Submitting (cwd={PROG_ROOT}): {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=False, cwd=PROG_ROOT)
            except Exception as e:
                print(f"Submission failed for model {model_in_csv} / {model_in_timm} worker {worker_id}: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()


