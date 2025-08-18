import os
import subprocess
import shlex
import time
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROG_ROOT = os.path.normpath(os.path.join(ROOT_DIR, ".."))  # .../Programming

BASE_TRAIN_PY = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/train.py"
# Always use the literal relative path form expected by the environment
SEQ_ARR = "../shared/seq_arr.sh"
LSF_LOG_OUT = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Cluster_runtime/model_training/useCase_out_from_%J.log"
LSF_LOG_ERR = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Cluster_runtime/model_training/useCase_err_from_%J.log"
CSV_PATH = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/imagenet_examples.csv"
BASE_OUTPUT_DIR = "/home/projects/bagon/andreyg/Projects/BMM_school/Universal_learning/Programming/image_difficulty_classifier/output"


def build_lsf_command(train_args: str, array_count: int = 1) -> str:
    container = 'ops:5000/universal_learning_2.4:1'
    gpu_spec = 'num=1:j_exclusive=yes:gmodel=NVIDIAA40'
    queue = 'waic-short'
    resources = '-R rusage[mem=256000] -R select[hname!=ibdgxa01] -R select[hname!=hgn42] -R select[hname!=hgn43]'
    # Submit one job per configuration; do not repeat as an array unless explicitly requested
    job_name = f'model_train[1-1]'
    base = (
        f'bsub -env LSB_CONTAINER_IMAGE="{container}" -app docker-gpu -gpu {gpu_spec} '
        f'-q {queue} {resources} -o {LSF_LOG_OUT} -e {LSF_LOG_ERR} -J "{job_name}" -H '
        f'python3 {BASE_TRAIN_PY} {train_args}'
    )
    # Always submit via seq array wrapper, preserving the literal '../shared/seq_arr.sh'
    # Only the content inside the single quotes should be relevant to upstream tooling
    return f"{SEQ_ARR} -c \"{base}\" -e 1 -d ended"


def spawn_job(args_dict):
    # Convert dict to CLI string
    parts = []
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {shlex.quote(str(v))}")
    train_args = " ".join(parts)
    cmd = build_lsf_command(train_args)
    print(f"Submitting (cwd={PROG_ROOT}): {cmd}")
    subprocess.run(cmd, shell=True, check=False, cwd=PROG_ROOT)


def main():
    # Parameter grid (kept moderate to avoid explosion). Adjust as needed.
    from itertools import product

    model_names = ['clip_linear', 'clip_mlp']
    clip_backbones = ['ViT-B-32', 'ViT-B-16']
    unfreeze_opts = [False, True]
    batch_sizes = [64, 128]
    lrs = [5e-4, 2e-4]
    weight_decays = [0.01, 0.05]
    num_bins_list = [2, 5]
    min_common_list = [0, 0]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_out = f"{BASE_OUTPUT_DIR}_grid_{timestamp}"

    combos = list(product(model_names, clip_backbones, unfreeze_opts, batch_sizes, lrs, weight_decays, num_bins_list, min_common_list))

    print(f"Total configurations: {len(combos)}")

    # Submit jobs for each combination
    for (model_name, backbone, unfreeze, bs, lr, wd, nb, minc) in combos:
        cfg = {
            'csv': CSV_PATH,
            'output-dir': base_out,
            'model-name': model_name,
            'clip-backbone': backbone,
            'clip-pretrained': 'openai',
            'unfreeze-backbone': unfreeze,
            'batch-size': bs,
            'epochs': 20,
            'lr': lr,
            'weight-decay': wd,
            'num-bins': nb,
            'minimum-images-common': minc,
            'image-size': 224,
            'num-workers': 4,
            'seed': 42,
        }
        spawn_job(cfg)
        # Pause 5 seconds between submissions
        time.sleep(5)


if __name__ == "__main__":
    main()


