import os
import shlex
import subprocess
from typing import List, Tuple, Optional


def read_models_csv(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8") as f:
		text = f.read().strip()
	if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
		text = text[1:-1]
	return [x.strip() for x in text.split(",") if x.strip()]


def parse_model(raw: str) -> Tuple[str, Optional[int]]:
	"""
	Translate raw model id from bars/imagenet_models.csv into (timm_like_name, image_size).

	Rules inferred from examples:
	- resnet_18_160_classification_imagenet_1k -> ("resnet_18", 160)
	- mobilenetv3_small_050_224_lamb_imagenet_1k -> ("mobilenetv3_small_050.lamb_in1k", 224)
	"""
	toks = raw.split("_")
	img_size: Optional[int] = None
	img_idx = None
	for i, t in enumerate(toks):
		if t.isdigit() and not t.startswith("0"):
			try:
				val = int(t)
				# Treat typical resolution tokens as image size
				if 32 <= val <= 1024:
					img_size = val
					img_idx = i
					break
			except Exception:
				pass

	base_tokens: List[str]
	suffix_tokens: List[str]
	if img_idx is not None:
		base_tokens = toks[:img_idx]
		suffix_tokens = toks[img_idx + 1 :]
	else:
		# No explicit size token found
		base_tokens = toks
		suffix_tokens = []

	base = "_".join(base_tokens)
	suffix_set = set(suffix_tokens)

	# Mapping rules
	name = base
	if {"lamb", "imagenet", "1k"}.issubset(suffix_set):
		name = f"{base}.lamb_in1k"
	elif {"classification", "imagenet", "1k"}.issubset(suffix_set):
		# Keep base as-is, per example (resnet_18)
		name = base
	elif {"imagenet", "1k"}.issubset(suffix_set):
		# Fallback mapping
		name = base
	return name, img_size


def build_command(model_name: str, image_size: Optional[int]) -> str:
	parts = ["-m", "training_gradient_evaluator.train_grad", f"--model_name {shlex.quote(model_name)}"]
	if image_size is not None:
		parts.append(f"--image_size {int(image_size)}")
	# rely on defaults for bars_npy and examples_csv; they point to bars/* files inside project
	return "python " + " ".join(parts)


def main() -> None:
	models_path = os.path.join("bars", "imagenet_models.csv")
	all_models = read_models_csv(models_path)
	if not all_models:
		raise RuntimeError("No models found in bars/imagenet_models.csv")
	bottom_50 = all_models[-50:]

	print(f"Found {len(all_models)} models. Submitting {len(bottom_50)} bottom models.")
	for raw in bottom_50:
		name, img_size = parse_model(raw)
		cmd = build_command(name, img_size)
		print(f"Launching: {cmd}")
		# Run jobs sequentially; adjust to your scheduler as needed
		subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
	main()


