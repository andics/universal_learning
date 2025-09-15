import os
import csv
import shlex
import time
import subprocess
from datetime import datetime


# Fixed environment settings taken from your example command
CONTAINER = 'ops:5000/universal_learning_2.4:1'
QUEUE = 'waic-long'
GPU_SPEC = 'num=1:j_exclusive=yes'
RESOURCES = '-R rusage[mem=256000] -R affinity[thread*24] -R select[hname!=hgn50] -R select[hname!=ibdgx010]'

# Paths (match your example)
PROG_ROOT = "/home/projects/bagon/andreyg"
SEQ_ARR = "../shared/seq_arr.sh"
TRAIN_PY = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/training_gradient_evaluator_single_loss/train_grad.py"
MAPPING_CSV = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_model_name_mapping.csv"

LOG_OUT = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Cluster_runtime/model_training/useCase_out_%J.log"
LOG_ERR = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Cluster_runtime/model_training/useCase_err_%J.log"

# Fixed train args (do not vary these here)
FIXED_ARGS = {
    'bars_npy': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/geq6wrong_21017_geq6correct_1525_imagenet.npy',
    'examples_csv': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/geq6wrong_21017_geq6correct_1525_imagenet_examples_ammended.csv',
    'imagenet_models_csv': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_models.csv',
    'max_examples': 1500,
    'max_steps_per_example': 10000,
    'lr': 0.0001,
    'weight_decay': 0,
    'epsilon': 0.001,
    'device': 'cuda',
    'seed': 1337,
    'zero_aug_train': True,
    'deterministic': True,
    'grad_clip_norm': 1.0,
    'output_dir': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/training_gradient_evaluator_single_loss/outputs_lr_0.0001',
    'hierarchy_json': '/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/bars/imagenet_synset_hierarchy.json',
}


def dict_to_cli(args_dict: dict) -> str:
    parts = []
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {shlex.quote(str(v))}")
    return " ".join(parts)


def build_bsub_command(train_args: str, job_name: str) -> str:
    base = (
        f'bsub -env LSB_CONTAINER_IMAGE="{CONTAINER}" -app docker-gpu '
        f'-gpu {GPU_SPEC} -q {QUEUE} {RESOURCES} '
        f'-o {LOG_OUT} -e {LOG_ERR} -J "{job_name}" -H '
        f'python3 {TRAIN_PY} {train_args}'
    )
    return f"{SEQ_ARR} -c \"{base}\" -e 4 -d ended"


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
    models = load_model_rows(MAPPING_CSV)
    if not models:
        print(f"No valid models found in {MAPPING_CSV}")
        return

    print(f"Submitting {len(models)} jobs from {MAPPING_CSV}")

    for idx, (model_in_csv, model_in_timm) in enumerate(models, start=1):
        args = dict(FIXED_ARGS)
        # Only vary these two:
        args['model_csv_name'] = model_in_csv
        args['model_name'] = model_in_timm

        cli = dict_to_cli(args)
        job_name = f"train_grad_{idx}"
        cmd = build_bsub_command(cli, job_name)
        print(f"Submitting (cwd={PROG_ROOT}): {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=False, cwd=PROG_ROOT)
        except Exception as e:
            print(f"Submission failed for {model_in_csv} / {model_in_timm}: {e}")
        time.sleep(2)


if __name__ == "__main__":
    main()
