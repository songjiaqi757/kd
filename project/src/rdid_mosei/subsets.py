from __future__ import annotations

from typing import Any

SUBSETS = ("t", "a", "v", "ta", "tv", "av", "tav")

PROMPT_PROFILES = {
    "v1": {
        "system": (
            "You analyze sentiment in short opinion clips. Use only the modalities explicitly "
            "provided by the user and never imagine unavailable cues."
        ),
        "task": (
            "Predict the speaker's sentiment intensity from -3 (strongly negative) to +3 "
            "(strongly positive). Return one concise numeric assessment."
        ),
    },
    "v2_affect": {
        "system": (
            "You estimate the sentiment expressed by a speaker in a short opinion clip. Use only "
            "the supplied modalities. When available, use semantic content, vocal prosody, facial "
            "expression, gaze, gesture, posture, and their changes over time. Never invent an "
            "unavailable cue."
        ),
        "task": (
            "Estimate sentiment intensity from -3 (strongly negative) to +3 (strongly positive) "
            "from the observable evidence. Use 0 only when that evidence is genuinely neutral or "
            "balanced. Return exactly one number and no explanation."
        ),
    },
}
PROMPT_VERSIONS = tuple(PROMPT_PROFILES)


def build_conversation(
    sample: dict[str, Any],
    subset: str,
    prompt_version: str = "v1",
    video_fps: float | None = None,
) -> list[dict[str, Any]]:
    if subset not in SUBSETS:
        raise ValueError(f"invalid subset: {subset}")
    if prompt_version not in PROMPT_PROFILES:
        raise ValueError(f"invalid prompt version: {prompt_version}")
    if video_fps is not None and video_fps <= 0:
        raise ValueError("video_fps must be positive")

    available = []
    content: list[dict[str, Any]] = []
    if "a" in subset:
        available.append("audio")
        content.append({"type": "audio", "audio": sample["audio_segment_path"]})
    if "v" in subset:
        available.append("silent video")
        video_content: dict[str, Any] = {"type": "video", "video": sample["silent_video_path"]}
        if video_fps is not None:
            video_content["fps"] = video_fps
        content.append(video_content)
    if "t" in subset:
        available.append("transcript")

    prompt_lines = [f"Available modalities: {', '.join(available)}."]
    if "t" in subset:
        prompt_lines.append(f"Transcript: {sample['text']}")
    profile = PROMPT_PROFILES[prompt_version]
    prompt_lines.append(profile["task"])
    content.append({"type": "text", "text": "\n".join(prompt_lines)})

    return [
        {"role": "system", "content": [{"type": "text", "text": profile["system"]}]},
        {"role": "user", "content": content},
    ]
