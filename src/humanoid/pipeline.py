import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class Pipeline:
    def __init__(self, cfg, force: bool = False) -> None:
        self.cfg = cfg
        self.force = force
        self.root = Path(cfg.output_root) / cfg.run_name

    def stage(self, name: str, fn: Callable[[], T], key: str | None = None) -> T:
        directory = self.root / name if key is None else self.root / name / key
        directory.mkdir(parents=True, exist_ok=True)

        done = directory / "done.json"
        result_path = directory / "result.pkl"
        if done.is_file() and not self.force:
            with result_path.open("rb") as f:
                return pickle.load(f)

        result = fn()
        with result_path.open("wb") as f:
            pickle.dump(result, f)
        done.write_text("{}\n")
        return result

    def out(self, relative: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_report(self, report: dict) -> None:
        self.out("report/changes.json").write_text(
            json.dumps(report["changes"], indent=2) + "\n"
        )
        self.out("report/report.md").write_text(report["markdown"])
        