import logging
import os


def create_logger(output_dir: str, name: str = "train", level: int = logging.INFO) -> logging.Logger:
	os.makedirs(output_dir, exist_ok=True)
	logger = logging.getLogger(f"{name}_v2")
	logger.setLevel(level)
	logger.propagate = False
	for h in list(logger.handlers):
		logger.removeHandler(h)
	ch = logging.StreamHandler()
	ch.setLevel(level)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	fh = logging.FileHandler(os.path.join(output_dir, f"{name}.log"))
	fh.setLevel(level)
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	return logger


