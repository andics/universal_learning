import csv
import os
import sys
import subprocess
import time
from pathlib import Path


def main() -> None:
	# Ensure we run from repo root so relative paths in train_grad work
	repo_root = str(Path(__file__).resolve().parents[1])
	os.chdir(repo_root)

	# Inputs
	mapping_csv = os.path.join('bars', 'imagenet_model_name_mapping.csv')
	output_csv = os.path.join('assess_training_difficulty', 'model_wait_times.csv')

	# Fixed hyperparameters from provided log
	args_common = {
		'--bars_npy': os.path.join('bars', 'geq6wrong_21017_geq6correct_1525_imagenet.npy'),
		'--examples_csv': os.path.join('bars', 'geq6wrong_21017_geq6correct_1525_imagenet_examples_ammended.csv'),
		'--hierarchy_json': os.path.join('bars', 'imagenet_synset_hierarchy.json'),
		'--imagenet_models_csv': os.path.join('bars', 'imagenet_models.csv'),
		'--lr': '0.0001',
		'--max_examples': '1',
		'--max_steps_per_example': '10000',
		'--epsilon': '0.001',
		'--device': 'cuda',
		'--grad_clip_norm': '1.0',
		'--seed': '1337',
	}

	# Disable AMP flag default is off unless provided; we keep AMP enabled (no --no_amp) and deterministic true.
	# Also set cka_layer_fraction=0.0 to skip CKA for faster timing.
	cka_disable_flag = ['--cka_layer_fraction', '0.0']
	DeterministicFlag = ['--deterministic']
	ZeroAugFlag = ['--zero_aug_train']

	rows = []
	with open(mapping_csv, 'r', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			model_csv_name = row.get('model_in_csv')
			model_in_timm = row.get('model_in_timm')
			if not model_csv_name or not model_in_timm:
				continue

			cmd = [sys.executable, os.path.join('training_gradient_evaluator_single_loss', 'train_grad.py')]
			# per-model arguments
			per_model_args = [
				'--model_name', str(model_in_timm),
				'--model_csv_name', str(model_csv_name),
				'--output_dir', os.path.join('training_gradient_evaluator_single_loss', 'outputs_lr_0.0001'),
			]
			# common args
			for k, v in args_common.items():
				per_model_args += [k, v]
			# flags
			per_model_args += DeterministicFlag
			per_model_args += ZeroAugFlag
			per_model_args += cka_disable_flag

			start = time.time()
			returncode = 1
			try:
				proc = subprocess.run(cmd + per_model_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
				returncode = proc.returncode
				log_output_path = os.path.join('assess_training_difficulty', 'logs')
				os.makedirs(log_output_path, exist_ok=True)
				# Save stdout for debugging per model
				fname_safe = model_in_timm.replace('/', '_') + '.log'
				with open(os.path.join(log_output_path, fname_safe), 'w', encoding='utf-8') as lf:
					lf.write(proc.stdout)
			finally:
				elapsed = time.time() - start

			rows.append({
				'model_in_csv': model_csv_name,
				'model_in_timm': model_in_timm,
				'wait_time_seconds': f"{elapsed:.3f}",
				'return_code': str(returncode),
			})

	# write CSV
	fieldnames = ['model_in_csv', 'model_in_timm', 'wait_time_seconds', 'return_code']
	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	with open(output_csv, 'w', newline='', encoding='utf-8') as wf:
		writer = csv.DictWriter(wf, fieldnames=fieldnames)
		writer.writeheader()
		for r in rows:
			writer.writerow(r)

	print(f"Wrote wait times for {len(rows)} models to {output_csv}")


if __name__ == '__main__':
	main()


