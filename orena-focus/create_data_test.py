from focus import FocusDataset, DatasetSplit, Track
from focus import FO_DEFINITIONS_FILE, FocusConfig, set_config

import torch
import json
import os
import signal
import sys
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from focus.data.video_dataset import FocusVideoDataset, VideoSample
from tqdm import tqdm

CONFIG = {
    "root_dir": "/iopsstor/scratch/cscs/lpanta32/focus",
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dataset_name": "heico",
    "track": Track.PROCEDURE,
    "video_stride": 50,  # keep every 10th frame (2.5 fps at 25 fps source)
    "video_resolution": (224, 224),
    "use_overlay": True,  # use timestamp-overlaid videos
    "num_eval": None,  # set to None to evaluate all test samples
    "output_dir": None,  # set to a path to persist results CSV files
    "checkpoint_every": 10,  # save json to disk after every N completed samples
}

# --- Global setup (must run identically in each forked worker) ---
set_config(FocusConfig(root_dir=CONFIG["root_dir"]))

base_dataset = FocusDataset(
    dataset=CONFIG["dataset_name"],
    split=DatasetSplit.TEST,
    track=CONFIG["track"],
)
dataset = FocusVideoDataset(
    base_dataset,
    stride=CONFIG["video_stride"],
    use_overlay=CONFIG["use_overlay"],
    resolution=CONFIG["video_resolution"],
)

SYSTEM_PROMPT = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
    + FO_DEFINITIONS_FILE.read_text()
)

output_dir = Path(CONFIG["root_dir"])
output_dir.mkdir(parents=True, exist_ok=True)
json_path = output_dir / "heico_test_overlay_long.json"

target_dir = Path(CONFIG["root_dir"] + "/heico_clips_test_overlay_long")
target_dir.mkdir(parents=True, exist_ok=True)


def process_index(i):
    """
    Runs in a worker process. Re-fetches the sample by index (avoids
    pickling issues with decord readers / CUDA contexts / open file
    handles that may live inside `sample` or `dataset`), moves the
    generated clip into target_dir (instead of copying, to avoid a
    redundant full byte-for-byte disk copy), and builds the message dict.
    """
    try:
        sample = dataset[i]

        dst = target_dir / Path(sample.video_path).name

        # move (not copy) — avoids a second full read+write of the file.
        # shutil.move handles cross-filesystem moves gracefully (falls
        # back to copy+delete automatically when needed, e.g. /tmp -> scratch).
        shutil.move(sample.video_path, dst)
        sample.video_path = str(dst)

        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": sample.video_path,
                        "fps": sample.fps,
                        "video_metadata": {
                            "fps": sample.fps,
                            "width": CONFIG["video_resolution"][0],
                            "height": CONFIG["video_resolution"][1],
                        },
                    },
                    {"type": "text", "text": sample.request.question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.reference.answer}],
            },
            {
                "start_time": sample.request.start_time,
                "end_time": sample.request.end_time,
                "procedure_type": sample.request.procedure_type,
            },
        ]
        return sample.request.qID, message, None

    except Exception as e:
        return None, None, f"Skipping index {i}: {e}"


def save_json(dataset_json, path):
    """Atomic-ish write: write to a temp file then replace, so a crash
    mid-write never corrupts the last good checkpoint."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    os.replace(tmp_path, path)


def main():
    n_cpus = len(os.sched_getaffinity(0))
    max_workers = max(1, 80)
    checkpoint_every = CONFIG["checkpoint_every"]

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(process_index, i): i for i in range(len(dataset))}

    dataset_json = {}
    errors = []
    completed_since_checkpoint = 0

    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            qid, message, err = future.result()
            if err is not None:
                errors.append(err)
            else:
                dataset_json[qid] = message

            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                save_json(dataset_json, json_path)
                completed_since_checkpoint = 0
    except KeyboardInterrupt:
        print("\nInterrupted — cancelling pending tasks and shutting down workers...")
        for f in futures:
            f.cancel()  # cancels tasks not yet started; running ones can't be stopped mid-call
        executor.shutdown(wait=False, cancel_futures=True)  # Python 3.9+
        # Save whatever we have so far before exiting
        save_json(dataset_json, json_path)
        sys.exit(1)
    finally:
        executor.shutdown(wait=True)

    # Final save to make sure the last partial batch (< checkpoint_every) is persisted
    save_json(dataset_json, json_path)

    if errors:
        print(f"\n{len(errors)} samples were skipped due to errors.")
        err_log_path = output_dir / "heico_train_overlay_long_errors.log"
        with open(err_log_path, "w") as f:
            f.write("\n".join(errors))
        print(f"Error details written to {err_log_path}")


if __name__ == "__main__":
    main()