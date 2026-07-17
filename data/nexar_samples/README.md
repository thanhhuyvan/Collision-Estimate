# Nexar starter sample set

This folder contains a deliberately small, balanced sample from the public [Nexar Collision Prediction Dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

## Purpose

Use these six clips for manual inspection, pipeline I/O development, annotated-output checks, and early false-warning investigation. They are **not** the final benchmark and must not be used to claim model performance.

## Selection

| Local path | Dataset split | Label | Original path |
|---|---|---:|---|
| `positive/00015.mp4` | train | 1 | `train/positive/00015.mp4` |
| `positive/00026.mp4` | train | 1 | `train/positive/00026.mp4` |
| `positive/00054.mp4` | train | 1 | `train/positive/00054.mp4` |
| `negative/01042.mp4` | train | 0 | `train/negative/01042.mp4` |
| `negative/01079.mp4` | train | 0 | `train/negative/01079.mp4` |
| `negative/01102.mp4` | train | 0 | `train/negative/01102.mp4` |

The dataset documentation specifies 1280x720 video at 30 FPS. Positive clips represent collision or imminent-collision events; negative clips represent normal driving. Event and alert timestamps are available in the full dataset metadata and should be incorporated when constructing the larger labelled evaluation suite.

## License and provenance

The source dataset is distributed under the Nexar Open Data License. Review the upstream `LICENSE` and README before redistributing clips or using them outside the stated research/baseline scope.

Source repository: `nexar-ai/nexar_collision_prediction`, revision downloaded from `main` on 2026-07-17.
