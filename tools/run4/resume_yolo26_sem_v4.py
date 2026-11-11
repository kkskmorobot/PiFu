"""Resume the interrupted runs4 YOLO26 semantic training from last.pt.

Right-click this file in PyCharm and choose Run after a reboot or interruption.
Ultralytics restores the epoch, optimizer, learning-rate scheduler and early-stop
state from the checkpoint. The most recent incomplete epoch is repeated.
"""

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "04_YOLO26s_Semantic_C2PSA"
    / "results"
    / "semantic"
    / "skin7_yolo26s_sem_balanced"
    / "weights"
    / "last.pt"
)


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {CHECKPOINT}")
    print(f"Resuming runs4 from: {CHECKPOINT}")
    YOLO(str(CHECKPOINT), task="semantic").train(resume=True)


if __name__ == "__main__":
    main()
