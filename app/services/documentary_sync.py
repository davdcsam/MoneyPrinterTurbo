"""Per-segment audio/video sync ("documentary style") pipeline.

Opt-in alternative to the default flow of synthesizing the whole script as one
audio file and chopping video into fixed-duration blocks. Here the script is
split into segments (merged forward until each reaches a minimum estimated
duration), each segment gets its own measured TTS audio, and downstream each
segment gets its own video clip trimmed to that exact duration. This keeps
camera cuts aligned with sentence boundaries instead of a fixed clock.

Generalizes the segment-synthesis pattern (loop/concat/measure per segment)
to work with whichever TTS provider is configured, via the generic
`voice.tts()` dispatcher.
"""
from __future__ import annotations

import os
import tempfile

from loguru import logger
from pydub import AudioSegment

from app.services import voice
from app.utils import utils

# Hard cap, not user-facing: bounds sequential TTS latency and the number of
# stock-video searches a long script can trigger (one search per segment).
MAX_SEGMENTS = 25

# Small fixed gap between concatenated segment clips so consecutive words
# don't run into each other at the splice point.
_INTER_SEGMENT_GAP_SECONDS = 0.15


def _estimate_segment_seconds(text: str, voice_rate: float) -> float:
    rate = float(voice_rate or 1.0)
    if rate <= 0:
        rate = 1.0
    return voice.estimate_no_voice_duration(text) / rate


def plan_segments(
    video_script: str,
    voice_rate: float = 1.0,
    min_segment_seconds: float = 4.5,
    max_segments: int = MAX_SEGMENTS,
) -> list[str]:
    """Split a script into merged segments, each at least `min_segment_seconds`.

    Merging only ever uses `voice.estimate_no_voice_duration()` (a heuristic),
    never a real measured duration - the real duration is only known after
    per-segment TTS runs, and that's what actually drives video trimming
    downstream. This function only decides where the sentence boundaries go.
    """
    sentences = [
        s.strip()
        for s in utils.split_string_by_punctuations(video_script or "")
        if s.strip()
    ]
    if not sentences:
        return []

    segments: list[str] = []
    buffer = ""
    for sentence in sentences:
        buffer = f"{buffer} {sentence}".strip() if buffer else sentence
        if _estimate_segment_seconds(buffer, voice_rate) >= min_segment_seconds:
            segments.append(buffer)
            buffer = ""

    if buffer:
        if segments:
            # Last chunk is short (script ended before reaching another full
            # segment) - fold it into the previous one rather than emitting a
            # too-short trailing segment.
            segments[-1] = f"{segments[-1]} {buffer}".strip()
        else:
            segments.append(buffer)

    while len(segments) > max_segments:
        # Merge the two adjacent segments with the smallest combined estimate
        # first, so the merge that costs the least "documentary" granularity
        # happens first.
        best_index = 0
        best_estimate = None
        for i in range(len(segments) - 1):
            estimate = _estimate_segment_seconds(
                f"{segments[i]} {segments[i + 1]}", voice_rate
            )
            if best_estimate is None or estimate < best_estimate:
                best_estimate = estimate
                best_index = i
        segments[best_index] = f"{segments[best_index]} {segments[best_index + 1]}".strip()
        del segments[best_index + 1]

    logger.info(
        f"documentary sync: planned {len(segments)} segments from "
        f"{len(sentences)} sentences (min_segment_seconds={min_segment_seconds})"
    )
    return segments


def synthesize_segments_generic(
    segments: list[str],
    voice_name: str,
    voice_rate: float,
    voice_volume: float,
    output_path: str,
) -> list[tuple[str, float, float, float]]:
    """Render each segment via the generic TTS dispatcher, concatenate, measure.

    Calls `voice.tts()` so it works with any configured provider. Returns
    one (text, spoken_duration_seconds,
    pre_pause_seconds, post_pause_seconds) tuple per segment, in order - the
    exact shape `voice.populate_submaker_from_segment_durations()` expects.

    A single segment's TTS failure falls back to silent audio at the
    estimated duration rather than aborting the whole run.
    """
    if not segments:
        raise ValueError("no segments to synthesize")

    ffmpeg_binary = utils.get_ffmpeg_binary()
    if ffmpeg_binary:
        AudioSegment.converter = ffmpeg_binary

    combined = AudioSegment.empty()
    rendered: list[tuple[str, float, float, float]] = []
    post_pause = _INTER_SEGMENT_GAP_SECONDS

    with tempfile.TemporaryDirectory(prefix="documentary_sync_") as tmp_dir:
        for i, text in enumerate(segments):
            seg_path = os.path.join(tmp_dir, f"seg_{i:03d}.mp3")
            sub_maker = voice.tts(
                text=text,
                voice_name=voice_name,
                voice_rate=voice_rate,
                voice_file=seg_path,
                voice_volume=voice_volume,
            )

            if sub_maker is None or not os.path.exists(seg_path):
                logger.warning(
                    f"documentary sync: TTS failed for segment {i}, "
                    "falling back to silent audio at the estimated duration"
                )
                estimated = _estimate_segment_seconds(text, voice_rate)
                if not voice.generate_silent_audio(estimated, seg_path):
                    raise RuntimeError(
                        f"failed to synthesize or fall back for segment {i}"
                    )

            audio = AudioSegment.from_file(seg_path)
            duration_seconds = audio.duration_seconds

            combined += audio
            is_last = i == len(segments) - 1
            gap = 0.0 if is_last else post_pause
            if gap > 0:
                combined += AudioSegment.silent(duration=int(gap * 1000))

            rendered.append((text, duration_seconds, 0.0, gap))
            logger.info(
                f"documentary sync: audio segment {i + 1}/{len(segments)} synthesized"
            )

        voice.ensure_file_path_exists(output_path)
        combined.export(output_path, format="mp3")

    return rendered
