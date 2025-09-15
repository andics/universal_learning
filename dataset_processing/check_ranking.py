import argparse
import os
from typing import List


def parse_imagenet_examples_csv(csv_path: str) -> List[str]:
	with open(csv_path, "r", encoding="utf-8") as f:
		text = f.read()
	text = text.lstrip("\ufeff").strip()
	if not text:
		return []
	if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
		text = text[1:-1]
	# Preserve placeholders like "None"; remove only empty tokens
	parts = [p.strip() for p in text.split(",")]
	return [p for p in parts if p != ""]


def main() -> None:
	parser = argparse.ArgumentParser(description="Find the index of a path inside an ImageNet examples-style CSV (comma-separated list of paths).")
	parser.add_argument("--csv", default='./bars/imagenet_examples_ammended.csv', help="Path to the imagenet_examples-like CSV file")
	parser.add_argument("--image_path", default='/home/projects/bagon/shared/imagenet512/val/n02877765/ILSVRC2012_val_00031839.JPEG',
	 help="Exact path string to search for (as written in the CSV)")
	args = parser.parse_args()

	paths = parse_imagenet_examples_csv(args.csv)
	query = args.image_path

	# Try exact match first
	try:
		idx = paths.index(query)
	except ValueError:
		# Try normalized separators as fallback
		query_norm_slash = query.replace("\\", "/")
		paths_norm_slash = [p.replace("\\", "/") for p in paths]
		try:
			idx = paths_norm_slash.index(query_norm_slash)
		except ValueError:
			print("Not found")
			return

	# Report both 0-based index and 1-based rank-like position
	print(idx)


if __name__ == "__main__":
	main()


