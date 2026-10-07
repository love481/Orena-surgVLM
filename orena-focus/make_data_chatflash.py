"""
Convert a QA-sample dataset (dict keyed by qid) into VideoChat-Flash
training format.

Input format (one entry per qid):
[
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "video", "video": <path>, "fps": ..., "video_metadata": {...}},
            {"type": "text", "text": <question>},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": <answer>}],
    },
    {
        "start_time": <float seconds>,
        "end_time": <float seconds>,
        "procedure_type": <str>,
    },
]

Output format:
[
    {
        "id": "<video_id_without_ext>",
        "video": "<video_filename>",
        "conversations": [
            {"from": "human", "value": "<image>\n<full_prompt>"},
            {"from": "gpt", "value": "<answer>"},
        ],
    },
    ...
]
"""

import argparse
import json
import os


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to hh:mm:ss format."""
    seconds = int(round(seconds))
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def build_time_info(start_time: float) -> str:
    convert_time = seconds_to_hhmmss(start_time)
    time_info1 = (
        f" This video clip is extracted from a longer procedure video and starts at "
        f"{convert_time} (hh:mm:ss) within that full video. Any timestamp mentioned in "
        "the question is given relative to the full video, not this clip. To find the "
        "relevant moment in this clip, subtract this start time from the question's "
        "timestamp. When your answer includes a timestamp, add this start time back so "
        "it is expressed relative to the full video, not this clip. "
    )
    return time_info1


def extract_video_and_question(user_msg: dict):
    """Pull the video path and the question text out of the user message content list."""
    video_path = None
    question = None
    for item in user_msg.get("content", []):
        if item.get("type") == "video":
            video_path = item.get("video")
        elif item.get("type") == "text":
            question = item.get("text")
    return video_path, question


def extract_answer(assistant_msg: dict):
    content = assistant_msg.get("content", [])
    if isinstance(content, list) and content:
        return content[0].get("text", "")
    if isinstance(content, str):
        return content
    return ""


def convert_sample(sample_messages: list) -> dict:
    """Convert a single [system, user, assistant, meta] list into one VideoChat-Flash entry."""
    system_msg = sample_messages[0]
    user_msg = sample_messages[1]
    assistant_msg = sample_messages[2]
    meta = sample_messages[3]

    system_prompt = system_msg.get("content", "")
    video_path, question = extract_video_and_question(user_msg)
    answer = extract_answer(assistant_msg)
    start_time = meta.get("start_time", 0)
    end_time = meta.get("end_time", 0)
    duration = end_time - start_time

    # time_info1 = build_time_info(start_time)
    time_info1 = f"The video lasts for {duration} seconds. Make use of overlay timestamps in hh:mm:ss format to answer the question."
    full_prompt = f"{system_prompt}\n\n{time_info1}\n\nQuestion: {question}"

    video_filename = os.path.basename(video_path) if video_path else ""
    video_id = os.path.splitext(video_filename)[0] if video_filename else ""
    return {
        "id": video_id,
        "video": video_filename,
        "conversations": [
            {"from": "human", "value": f"<image>\n{full_prompt}"},
            {"from": "gpt", "value": answer},
        ],
    }


def convert_dataset(data: dict) -> list:
    """data: dict keyed by qid (qid itself is discarded, only   values are used)."""
    output = []
    for sample_messages in data.values():
        try:
            output.append(convert_sample(sample_messages))
        except (IndexError, KeyError, AttributeError, TypeError) as e:
            # Skip malformed entries but keep going; print for visibility.
            print(f"Skipping a malformed sample due to: {e}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert QA dataset to VideoChat-Flash format.")
    parser.add_argument(
        "input",
        nargs="?",
        default="/iopsstor/scratch/cscs/lpanta32/focus/heico_test_overlay_long.json",
        help="Path to input JSON file (dict keyed by qid). Default: input.json",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="/iopsstor/scratch/cscs/lpanta32/focus/video_chatflash/heico_test_overlay_long_flash.json",
        help="Path to output JSON file (VideoChat-Flash list format). Default: output.json",
    )
    args = parser.parse_args()
 
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    converted = convert_dataset(data)
 
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
 
    print(f"Converted {len(converted)} samples -> {args.output}")


if __name__ == "__main__":
    main()