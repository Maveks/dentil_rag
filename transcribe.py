"""
Pipeline: AAC → ElevenLabs Scribe v2 → Claude Sonnet postprocessing → JSON
ElevenLabs приймає AAC напряму (ліміт 2GB), ffmpeg не потрібен.
Usage: python transcribe.py --audio-dir ./audio --out-dir ./transcripts
"""
import argparse
import json
import pathlib
import time

import anthropic
from elevenlabs import ElevenLabs

import config
from glossary import POSTPROCESS_SYSTEM_PROMPT

_MAX_RETRIES = 3


def transcribe_file(client: ElevenLabs, audio_path: pathlib.Path) -> list[dict]:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with open(audio_path, "rb") as f:
                result = client.speech_to_text.convert(
                    file=f,
                    model_id="scribe_v2",
                    language_code=config.TRANSCRIPTION_LANGUAGE,
                    timestamps_granularity="word",
                )
            break
        except Exception as e:
            if attempt == _MAX_RETRIES:
                raise
            wait = 10 * attempt
            print(f"          Помилка (спроба {attempt}/{_MAX_RETRIES}): {e}. Повтор через {wait}с...")
            time.sleep(wait)

    segments = []
    current: dict | None = None

    for word in result.words:
        if word.type != "word":
            continue
        if current is None:
            current = {"start": word.start, "end": word.end, "text": word.text}
        elif word.end - current["start"] < 30:
            current["text"] += " " + word.text
            current["end"] = word.end
        else:
            segments.append(current)
            current = {"start": word.start, "end": word.end, "text": word.text}

    if current:
        segments.append(current)

    return segments


def postprocess_segments(
    anthropic_client: anthropic.Anthropic,
    segments: list[dict],
    batch_size: int = 20,
) -> list[dict]:
    processed = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        raw_text = "\n".join(s["text"] for s in batch)

        response = anthropic_client.messages.create(
            model=config.POSTPROCESS_MODEL,
            max_tokens=4096,
            system=POSTPROCESS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": raw_text}],
        )

        cleaned_lines = response.content[0].text.strip().split("\n")

        for j, segment in enumerate(batch):
            cleaned = cleaned_lines[j].strip() if j < len(cleaned_lines) else segment["text"]
            start_sec = int(segment["start"])
            end_sec = int(segment["end"])
            processed.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "timestamp": _fmt_timestamp(start_sec),
                "text": cleaned,
            })

        time.sleep(0.5)

    return processed


def _fmt_timestamp(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_lesson(
    aac_path: pathlib.Path,
    out_dir: pathlib.Path,
    elevenlabs_client: ElevenLabs,
    anthropic_client: anthropic.Anthropic,
    lesson_num: str,
    course_name: str,
) -> pathlib.Path:
    print(f"  [{lesson_num}] Транскрибація через ElevenLabs Scribe v2: {aac_path.name}")
    segments = transcribe_file(elevenlabs_client, aac_path)
    print(f"          Отримано {len(segments)} сегментів")

    print(f"  [{lesson_num}] Постобробка через Claude Sonnet 4.6...")
    segments = postprocess_segments(anthropic_client, segments)

    result = {
        "course": course_name,
        "lesson": lesson_num,
        "lesson_file": aac_path.name,
        "segments": segments,
    }

    out_path = out_dir / f"lesson_{lesson_num}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"  [{lesson_num}] Збережено: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe dental course audio files")
    parser.add_argument("--audio-dir", default=str(config.AUDIO_DIR), help="Folder with .aac files")
    parser.add_argument("--out-dir", default=str(config.TRANSCRIPTS_DIR), help="Output folder for JSON transcripts")
    parser.add_argument("--course", default=config.COURSE_NAME, help="Course name for metadata")
    args = parser.parse_args()

    audio_dir = pathlib.Path(args.audio_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aac_files = sorted(audio_dir.glob("*.aac"))
    if not aac_files:
        print(f"AAC-файли не знайдені в: {audio_dir}")
        return

    print(f"Знайдено {len(aac_files)} файлів у {audio_dir}")

    elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY, timeout=600.0)
    anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    for idx, aac_path in enumerate(aac_files, start=1):
        lesson_num = f"{idx:02d}"
        out_path = out_dir / f"lesson_{lesson_num}.json"
        if out_path.exists():
            print(f"  [{lesson_num}] Пропускаємо (вже є): {out_path.name}")
            continue
        process_lesson(
            aac_path=aac_path,
            out_dir=out_dir,
            elevenlabs_client=elevenlabs_client,
            anthropic_client=anthropic_client,
            lesson_num=lesson_num,
            course_name=args.course,
        )

    print(f"\nГотово! {len(aac_files)} файлів оброблено → {out_dir}")


if __name__ == "__main__":
    main()
