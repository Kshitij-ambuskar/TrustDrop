"""Read-only access to the JSON report artifacts written by the training pipeline."""
import json
from pathlib import Path


class ReportStore:
    def __init__(self, reports_dir: Path):
        self.dir = Path(reports_dir)

    def read(self, rel: str) -> dict | None:
        p = self.dir / rel
        return json.loads(p.read_text()) if p.exists() else None

    def model_performance(self) -> dict | None:
        comp = self.read("model_comparison.json")
        if comp is None:
            return None
        cms = {n: self.read(f"confusion_matrices/{n.lower().replace(' ', '_')}.json") for n in comp["results"]}
        return {**comp, "confusion_matrices": cms}
