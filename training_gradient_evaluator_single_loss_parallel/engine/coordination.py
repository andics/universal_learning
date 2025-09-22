import os
import json
import time
from typing import Dict


class WorkerCoordinator:
	def __init__(self, model_out_dir: str, num_workers: int, logger) -> None:
		self.model_out_dir = model_out_dir
		self.num_workers = int(num_workers)
		self.logger = logger
		self.status_path = os.path.join(self.model_out_dir, "worker_status.json")

	def _read_status(self) -> Dict[str, dict]:
		if not os.path.exists(self.status_path):
			return {}
		try:
			with open(self.status_path, 'r', encoding='utf-8') as f:
				data = json.load(f) or {}
			if isinstance(data, dict):
				return data
			except Exception:
			return {}

	def update_status(self, worker_id: int, status: str) -> None:
		try:
			data = self._read_status()
			data[str(int(worker_id))] = {"status": str(status), "timestamp": float(time.time())}
			with open(self.status_path, 'w', encoding='utf-8') as f:
				json.dump(data, f, ensure_ascii=False, indent=2)
		except Exception:
			if self.logger:
				self.logger.exception("Failed to update worker_status.json", exc_info=True)

	def all_workers_done(self) -> bool:
		data = self._read_status()
		for i in range(self.num_workers):
			entry = data.get(str(i))
			if not entry or str(entry.get("status")) != "DONE":
				return False
		return True

	def wait_for_all_done(self, timeout_s: int = 24 * 3600) -> None:
		deadline = time.time() + int(timeout_s)
		while time.time() < deadline:
			if self.all_workers_done():
				return
			if self.logger:
				self.logger.info("Waiting for all workers to report DONE in worker_status.json...")
			time.sleep(30)


