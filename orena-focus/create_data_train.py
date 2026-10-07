from focus import FocusDataset, DatasetSplit, Track
from focus import FO_DEFINITIONS_FILE, FocusConfig, set_config

import torch
import json
import os
import signal
import sys
import shutil
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from focus.data.video_dataset import FocusVideoDataset, VideoSample
from tqdm import tqdm

CONFIG = {
    "root_dir": "/iopsstor/scratch/cscs/lpanta32/focus",
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "dataset_name": "lapchole",
    "track": Track.PROCEDURE,
    "video_stride": 50,  # keep every 10th frame (2.5 fps at 25 fps source)
    "video_resolution": (224, 224),
    "use_overlay": True,  # use timestamp-overlaid videos
    "num_eval": None,  # set to None to evaluate all test samples
    "output_dir": None,  # set to a path to persist results CSV files
    "checkpoint_every": 10,  # save json to disk after every N completed samples
    "max_workers": None,  # None -> auto (min(cpu_count, 32)); set explicitly if OOM/crashes persist
    "max_pool_restarts": 5,  # how many times to rebuild the pool after a BrokenProcessPool
}

# --- Global setup (must run identically in each forked worker) ---
set_config(FocusConfig(root_dir=CONFIG["root_dir"]))

base_dataset = FocusDataset(
    dataset=CONFIG["dataset_name"],
    split=DatasetSplit.TRAIN,
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
json_path = output_dir / "lapchole_train_overlay_long.json"

target_dir = Path(CONFIG["root_dir"] + "/lapchole_clips_train_overlay_long")
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
    max_workers = CONFIG["max_workers"] or min(n_cpus, 32)
    checkpoint_every = CONFIG["checkpoint_every"]
    max_pool_restarts = CONFIG["max_pool_restarts"]

    # 'spawn' avoids inheriting a half-initialized CUDA context / open file
    # handles / decord state from the parent process via fork(), which is a
    # common cause of workers dying silently and taking the whole pool down.
    mp_ctx = mp.get_context("spawn")

    dataset_json = {}
    errors = []
    completed_since_checkpoint = 0

    pending_indices = set(range(len(dataset)))
    total = len(pending_indices)
    pool_restarts = 0

    pbar = tqdm(total=total, desc="Processing samples")

    try:
        while pending_indices:
            executor = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx)
            futures = {executor.submit(process_index, i): i for i in pending_indices}

            try:
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        qid, message, err = future.result()
                    except BrokenProcessPool:
                        # Propagate up so we can rebuild the pool and retry
                        # whatever indices are still pending.
                        raise
                    except Exception as e:
                        # A single task raised something unexpected (not
                        # caught inside process_index) — log and move on,
                        # don't let it take down the whole run.
                        err = f"Skipping index {i}: {e}"
                        qid, message = None, None

                    pending_indices.discard(i)

                    if err is not None:
                        errors.append(err)
                    else:
                        dataset_json[qid] = message

                    completed_since_checkpoint += 1
                    pbar.update(1)
                    if completed_since_checkpoint >= checkpoint_every:
                        save_json(dataset_json, json_path)
                        completed_since_checkpoint = 0

                # Completed the while-loop pass without a BrokenProcessPool
                executor.shutdown(wait=True)

            except BrokenProcessPool as e:
                save_json(dataset_json, json_path)  # never lose completed work
                pool_restarts += 1
                print(
                    f"\nWorker pool crashed ({e}). "
                    f"{len(pending_indices)} samples still pending. "
                    f"Restarting pool (attempt {pool_restarts}/{max_pool_restarts})..."
                )
                executor.shutdown(wait=False, cancel_futures=True)
                if pool_restarts > max_pool_restarts:
                    print(
                        "Exceeded max pool restarts — stopping. Re-run the "
                        "script to resume; already-completed samples are saved."
                    )
                    break
                # Loop back around: a fresh executor is created with the
                # remaining pending_indices at the top of the while loop.
                continue

    except KeyboardInterrupt:
        print("\nInterrupted — saving progress and shutting down workers...")
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except NameError:
            pass
        save_json(dataset_json, json_path)
        sys.exit(1)
    finally:
        pbar.close()

    # Final save to make sure the last partial batch (< checkpoint_every) is persisted
    save_json(dataset_json, json_path)

    if errors:
        print(f"\n{len(errors)} samples were skipped due to errors.")
        err_log_path = output_dir / "lapchole_train_overlay_long_errors.log"
        with open(err_log_path, "w") as f:
            f.write("\n".join(errors))
        print(f"Error details written to {err_log_path}")

    if pending_indices:
        print(
            f"\n{len(pending_indices)} samples never completed after "
            f"{max_pool_restarts} pool restarts. Consider lowering "
            f"max_workers in CONFIG (memory pressure is the usual cause)."
        )


if __name__ == "__main__":
    main()