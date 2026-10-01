import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage


def test_mnist_training_script_runs_offline_and_writes_metrics(tmp_path):
    """The example trains from a small local Hugging Face DatasetDict."""
    rng = np.random.default_rng(7)
    features = Features({"image": Image(), "label": Value("int64")})

    def make_split(size):
        images = [
            PILImage.fromarray(rng.integers(0, 256, (28, 28), dtype=np.uint8))
            for _ in range(size)
        ]
        return Dataset.from_dict(
            {"image": images, "label": np.arange(size) % 4},
            features=features,
        )

    data_dir = tmp_path / "mnist"
    output_dir = tmp_path / "output"
    DatasetDict({"train": make_split(8), "test": make_split(4)}).save_to_disk(
        data_dir
    )

    script = Path(__file__).parents[1] / "examples" / "train_mnist.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-dir",
            str(data_dir),
            "--train-size",
            "8",
            "--test-size",
            "4",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--width",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert "num params:" in completed.stdout
    assert "memory:" in completed.stdout
    assert "memory breakdown:" in completed.stdout
    assert "saved_activations_bytes=" in completed.stdout
    assert "epoch=1/1" in completed.stdout
    assert "train_time=" in completed.stdout
    assert "total_train_time=" in completed.stdout
    assert metrics["epochs"][0]["test_total"] == 4
    assert np.isfinite(metrics["epochs"][0]["loss"])
    assert metrics["epochs"][0]["training_seconds"] >= 0
    assert metrics["epochs"][0]["evaluation_seconds"] >= 0
    assert metrics["epochs"][0]["epoch_seconds"] >= 0
    assert metrics["total_training_seconds"] >= 0
    assert metrics["total_process_seconds"] >= metrics["total_training_seconds"]
    assert 0 <= metrics["final_accuracy"] <= 1
    memory_estimate = metrics["memory_estimate"]
    breakdown = memory_estimate["breakdown"]
    assert memory_estimate["parameter_count"] > 0
    assert memory_estimate["peak_bytes"] == sum(breakdown.values())
