"""End-to-end inference example for the ORena SAVE FOCUS challenge,
using VideoChat-Flash-Qwen2_5-2B_res448 instead of Qwen3-VL.

Pipeline:
  1. Load a pre-split FocusDataset (see data_preparation.py for how to create one)
  2. Wrap it in FocusVideoDataset to produce on-the-fly temporary clips
  3. Run a VideoChat-Flash model over every test sample (single-turn only)
  4. Collect Responses and evaluate with the Evaluator

Prerequisites
-------------
    pip install orena-focus transformers accelerate
    pip install av imageio decord opencv-python
    pip install flash-attn --no-build-isolation   # required by the vision tower

The dataset must have been downloaded and split beforehand — see
``examples/data_preparation.py`` for the full preparation pipeline.
"""

import logging
import time
import warnings
import os
import torch
from progiter import ProgIter
import json
from transformers import AutoModel, AutoTokenizer
from focus import FO_DEFINITIONS_FILE, FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import Response
from focus.enums import DatasetSplit, Track
import random
import json

random.seed(0)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
# All tunable parameters in one place.

CONFIG = {
    "root_dir": "/iopsstor/scratch/cscs/lpanta32/focus/original_json", # json file created for inferencing
    "model_id": "/iopsstor/scratch/cscs/lpanta32/checkpoints/stage3-video_sft/stage3-surgvqa_overlay_refined", # checkpoint path
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "dataset_name": "lapchole",
    "track": Track.SEGMENT,
    "video_stride": 10,  # keep every 10th frame (1 fps at 10 fps source)
    "video_resolution": (224, 224),
    "use_overlay": True,  # use timestamp-overlaid videos
    "num_eval": None,  # set to None to evaluate all test samples
    "output_dir": "/capstor/store/cscs/swissai/a0264/lpanta32/orena-focus",
    "max_num_frames": 400,
}
#
SYSTEM_PROMPT = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise.\n\n"
    + FO_DEFINITIONS_FILE.read_text()
)

# ── Inference engine ──────────────────────────────────────────────────
class VideoChatFlashInferenceEngine:
    """VideoChat-Flash model wrapper for FOCUS challenge inference.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier
        (e.g. ``"OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"``).
    device : str
        Torch device string (``"cpu"`` or ``"cuda"``).
    max_num_frames : int
        Max number of frames the model will sample from each video.

    Notes
    -----
    Call :meth:`load` before :meth:`predict` to initialise model weights.
    Only single-turn conversation is supported here — each call to
    :meth:`predict` is independent, no chat history is carried over.
    """

    def __init__(self, model_id: str, device: str, max_num_frames: int = 512) -> None:
        self.model_id = model_id
        self.device = device
        self.max_num_frames = max_num_frames
        self.model = None
        self.tokenizer = None
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=100,
            top_p=0.1,
            num_beams=1,
        )

    def load(self) -> None:
        """Load the tokenizer and model weights from HuggingFace Hub."""
        logger.info(f"Loading model {self.model_id!r} on device {self.device!r}…")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(self.model_id,  attn_implementation="sdpa", trust_remote_code=True,
        torch_dtype=torch.float16, device_map = "auto",
        ).eval()
        # self.model.config.mm_llm_compress = False
        mm_llm_compress = False  # use the global compress or not
        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = "uniform0_attention"
            self.model.config.llm_compress_layer_list = [4, 18]
            self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False
        logger.info("Model ready.")

    def predict(
        self, question: str, duration: int, video_path: str
    ) -> str:
        """Run single-turn inference on one sample.

        Parameters
        ----------
        question : str
            The surgical question to ask about the clip.
        duration : int
            Duration of the video clip in seconds.
        video_path : str
            Path to the video clip to analyze.

        Returns
        -------
        str
            The raw text generated by the model, or an error message
            prefixed with ``"Inference Error:"`` if generation fails.
        """
        try:
            time_info1 = f"The video lasts for {duration} seconds. Also make use of overlay timestamps in hh:mm:ss format to answer the question."
            full_prompt = f"{SYSTEM_PROMPT}\n\n{time_info1}\n\nQuestion: {question}"
            with torch.no_grad():
                output = self.model.chat(
                    video_path=video_path,
                    tokenizer=self.tokenizer,
                    user_prompt=full_prompt,
                    return_history=False,
                    max_num_frames=self.max_num_frames,
                    generation_config=self.generation_config,
                )
            output = output.replace("Answer:", "").strip()
            return output
        except Exception as exc:
            logger.error(f"[{video_path}] Inference failed: {exc}")
            return f"Inference Error: {str(exc)[:50]}"


# ── Main pipeline ─────────────────────────────────────────────────────


def main() -> None:
    # ── 1. Configure and load dataset ────────────────────────────────
    # Run examples/data_preparation.py first to download and split the dataset.
    set_config(FocusConfig(root_dir=CONFIG["root_dir"]))

    base_dataset = FocusDataset(
        dataset=CONFIG["dataset_name"],
        split=DatasetSplit.TEST,
        track=CONFIG["track"],
    )

    n_total = len(base_dataset)
    n_eval = min(CONFIG["num_eval"], n_total) if CONFIG["num_eval"] else n_total
    logger.info(f"Evaluating {n_eval}/{n_total} test samples.")

    # ── 2. Load model ─────────────────────────────────────────────────
    engine = VideoChatFlashInferenceEngine(
        CONFIG["model_id"], CONFIG["device"], CONFIG["max_num_frames"]
    )
    engine.load()
    with open(
        os.path.join(CONFIG["root_dir"], "lapchole_test_overlay.json"),
        "r",
        encoding="utf-8",
    ) as file:
        dataset_json = json.load(file)
    print(len(list(dataset_json.keys())))
    reports = []
    # ── 3. Inference loop ─────────────────────────────────────────────
    requests, references, responses = [], [], []
    count_index = 0
    for i, sample in enumerate(ProgIter(base_dataset, total=n_eval, desc="Inference")):
        if count_index >= n_eval:
            break
        count_index = count_index + 1

        t0 = time.perf_counter()
        try:
            qid = sample[0].qID
            json_sample = dataset_json[qid]
        except Exception as e:
            print(f"Skipping due to error as {e}")
            continue

        video_path = json_sample[1]["content"][0]["video"]
        start_time = sample[0].start_time
        duration = sample[0].end_time - start_time

        prediction = engine.predict(
            sample[0].question, duration, video_path
        )
        latency = time.perf_counter() - t0

        requests.append(sample[0])
        references.append(sample[1])
        responses.append(
            Response(
                qID=sample[0].qID,
                content=prediction,
                latency=latency,
            )
        )
        reports.append(
            {"qID": sample[0].qID, "content": prediction, "latency": latency}
        )
        print(
            "Index: ",
            i,
            "\n Question: ",
            sample[0].question,
            "\n Prediction: ",
            prediction,
            "\n GT: ",
            sample[1].answer,
            "\n Latency: ",
            latency,
        )
    if not responses:
        warnings.warn("No test samples were processed. Check dataset status.")
        return

    with open("reports_finetuned_lapchole_lm_integrated.json", "w") as f:
        json.dump(reports, f, indent=4)

if __name__ == "__main__":
    main()