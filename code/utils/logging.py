from __future__ import annotations

import os
from datetime import datetime


def log_to_file(log_msg: str, log_path: str = "project_log.txt") -> None:
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        final_log = f"[{time_str}] | {log_msg.strip()}\n"
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(final_log)
    except Exception:
        return




