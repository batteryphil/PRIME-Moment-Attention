"""
PRIME Sovereign Audio Narrator (Kokoro-82M Edition)
===================================================
Autonomous studio-grade audiobook narration engine for the PRIME library.
Features:
- Multi-voice casting (Narrator, Leo Mercer, Captain Astrid Ross / Vex, Ship AI)
- Intelligent dialogue extraction and speaker attribution
- Human breath pauses, paragraph pacing, and dramatic scene-break silences
- ACX / EBU R128 (-23 dB LUFS) professional loudness mastering via FFmpeg
- Export to high-fidelity MP3 and WAV with full ID3 album and chapter metadata
"""

import os
import re
import sys
import time
import shutil
import pathlib
import subprocess
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import soundfile as sf

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOVELS_ROOT = PROJECT_ROOT / "novels"
AUDIO_CACHE_DIR = PROJECT_ROOT / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Voice Casting Map for Kokoro-82M
DEFAULT_VOICE_CAST = {
    "narrator": "am_michael",      # Cinematic, grounded American narrator
    "leo": "am_adam",              # Disciplined, battle-worn military baritone
    "astrid": "af_bella",          # Sharp, agile, slightly husky corsair captain
    "vex": "af_bella",             # Alias for Astrid
    "orlo": "bm_george",           # Deep, deliberate synthetic ship AI
    "default_male": "am_adam",
    "default_female": "af_bella"
}

SAMPLE_RATE = 24000  # Native Kokoro sample rate


class AudioSegment:
    """Represents a discrete line of narration or dialogue with voice metadata."""
    def __init__(self, text: str, speaker: str, voice: str, pause_after_sec: float = 0.4):
        self.text = text.strip()
        self.speaker = speaker
        self.voice = voice
        self.pause_after_sec = pause_after_sec

    def __repr__(self):
        return f"<AudioSegment speaker={self.speaker} voice={self.voice} text='{self.text[:35]}...'>"


class ManuscriptAudioParser:
    """
    Parses novel markdown manuscripts into structured audio segments with
    contextual character speaker attribution.
    """
    def __init__(self, voice_cast: Optional[Dict[str, str]] = None):
        self.voice_cast = voice_cast or DEFAULT_VOICE_CAST

    def parse_chapter(self, text: str) -> List[AudioSegment]:
        segments: List[AudioSegment] = []
        
        # 1. Strip top-level chapter header and prompt metadata
        lines = text.split('\n')
        clean_paragraphs: List[str] = []
        current_p: List[str] = []

        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                if current_p:
                    clean_paragraphs.append(' '.join(current_p))
                    current_p = []
                continue

            # Strip markdown title '# Chapter X: Title'
            if re.match(r'^#\s+Chapter\s+\d+.*$', trimmed, re.IGNORECASE):
                continue
            # Check for scene break
            if re.match(r'^(\*|\-|✦|\s){3,}$', trimmed):
                if current_p:
                    clean_paragraphs.append(' '.join(current_p))
                    current_p = []
                clean_paragraphs.append('___SCENE_BREAK___')
                continue

            current_p.append(trimmed)

        if current_p:
            clean_paragraphs.append(' '.join(current_p))

        # 2. Parse each paragraph into narration vs dialogue
        last_speaker = "astrid"  # For alternating dialogue inference

        for p in clean_paragraphs:
            if p == '___SCENE_BREAK___':
                # Scene break pause: 2.0s silence
                segments.append(AudioSegment(text="", speaker="scene_break", voice=self.voice_cast["narrator"], pause_after_sec=2.0))
                continue

            # Extract dialogue quotes and surrounding prose
            tokens = re.split(r'(["“][^"”]+["”])', p)

            for token in tokens:
                t = token.strip()
                if not t:
                    continue

                if (t.startswith('"') and t.endswith('"')) or (t.startswith('“') and t.endswith('”')):
                    # Dialogue quote
                    clean_dialogue = t[1:-1].strip()
                    speaker = self._attribute_speaker(clean_dialogue, p, last_speaker)
                    voice = self.voice_cast.get(speaker, self.voice_cast["narrator"])
                    segments.append(AudioSegment(text=clean_dialogue, speaker=speaker, voice=voice, pause_after_sec=0.5))
                    last_speaker = speaker
                else:
                    # Clean markdown formatting (*italics*, **bold**)
                    clean_narration = re.sub(r'[*_]{1,3}', '', t).strip()
                    if clean_narration:
                        segments.append(AudioSegment(
                            text=clean_narration,
                            speaker="narrator",
                            voice=self.voice_cast["narrator"],
                            pause_after_sec=0.75  # Paragraph breath pause
                        ))

        return segments

    def _attribute_speaker(self, dialogue: str, context_paragraph: str, last_speaker: str) -> str:
        """Determines whether dialogue belongs to Leo, Astrid/Vex, Orlo, or other."""
        lower_p = context_paragraph.lower()
        
        # Explicit tags for Leo Mercer
        if re.search(r'\b(leo|mercer|apex-one)\s+(said|asked|croaked|growled|snorted|muttered|breathed|warned|whispered|nodded)\b', lower_p):
            return "leo"
        if re.search(r'\b(said|asked|croaked|growled|snorted|muttered|breathed|warned|whispered)\s+leo\b', lower_p):
            return "leo"

        # Explicit tags for Astrid / Vex
        if re.search(r'\b(astrid|vex|ross)\s+(said|asked|hissed|purred|mocked|barked|countered|yelled|demanded|whispered)\b', lower_p):
            return "astrid"
        if re.search(r'\b(said|asked|hissed|purred|mocked|barked|countered|yelled|demanded|whispered)\s+(astrid|vex)\b', lower_p):
            return "astrid"

        # Ship AI / Orlo
        if re.search(r'\b(orlo|transponder|intercom|synthetic voice|ship(?:\'s)? computer)\b', lower_p):
            return "orlo"

        # Pronoun tags
        if re.search(r'\bhe\s+(said|asked|replied|growled|muttered|answered)\b', lower_p):
            return "leo"
        if re.search(r'\bshe\s+(said|asked|hissed|purred|countered|mocked|barked)\b', lower_p):
            return "astrid"

        # If indeterminate, alternate from last speaker
        return "leo" if last_speaker == "astrid" else "astrid"


class SovereignAudioNarrator:
    """
    Synthesizes and masters full audiobooks using Kokoro-82M.
    """
    def __init__(self, device: str = "cpu", voice_cast: Optional[Dict[str, str]] = None):
        from kokoro import KPipeline
        self.device = device
        self.voice_cast = voice_cast or DEFAULT_VOICE_CAST
        self.parser = ManuscriptAudioParser(self.voice_cast)
        
        print(f"[*] Initializing Kokoro-82M Audio Pipeline on device='{self.device}'...")
        t0 = time.time()
        self.pipeline = KPipeline(lang_code="a", device=self.device)
        print(f"[✓] Kokoro-82M pipeline active in {time.time() - t0:.2f}s")

        # Cache pre-loaded voice tensors
        self.loaded_voices: Dict[str, Any] = {}
        for role, v_name in self.voice_cast.items():
            if v_name not in self.loaded_voices:
                try:
                    self.loaded_voices[v_name] = self.pipeline.load_voice(v_name)
                    print(f"  [+] Pre-cached voice: '{v_name}' ({role})")
                except Exception as e:
                    print(f"  [!] Warning loading voice '{v_name}': {e}")

    def narrate_chapter(
        self,
        manuscript_text: str,
        output_wav_path: pathlib.Path,
        progress_callback: Optional[callable] = None
    ) -> float:
        """
        Synthesizes an entire chapter text into a 24kHz WAV file.
        Returns total audio duration in seconds.
        """
        segments = self.parser.parse_chapter(manuscript_text)
        print(f"\n[*] Synthesizing chapter ({len(segments)} narrative/dialogue segments)...")
        t_start = time.time()

        audio_chunks: List[np.ndarray] = []
        total_samples = 0

        for idx, seg in enumerate(segments):
            if seg.speaker == "scene_break":
                # Add silence
                silence_samples = int(seg.pause_after_sec * SAMPLE_RATE)
                audio_chunks.append(np.zeros(silence_samples, dtype=np.float32))
                continue

            if not seg.text:
                continue

            try:
                # Generate voice chunk with Kokoro
                generator = self.pipeline(
                    seg.text,
                    voice=seg.voice,
                    speed=1.0,
                    split_pattern=r'\n+'
                )
                
                seg_audio = []
                for _, _, audio in generator:
                    if isinstance(audio, np.ndarray):
                        seg_audio.append(audio)
                    else:
                        seg_audio.append(audio.cpu().numpy())

                if seg_audio:
                    merged = np.concatenate(seg_audio)
                    audio_chunks.append(merged)
                    total_samples += len(merged)

                    # Append breath/pause silence
                    if seg.pause_after_sec > 0:
                        pause_samples = int(seg.pause_after_sec * SAMPLE_RATE)
                        audio_chunks.append(np.zeros(pause_samples, dtype=np.float32))
                        total_samples += pause_samples

            except Exception as e:
                print(f"  [!] Synthesis error on segment {idx}: {e}")

            if progress_callback:
                progress_callback(idx + 1, len(segments))

            if (idx + 1) % 15 == 0 or (idx + 1) == len(segments):
                elapsed = time.time() - t_start
                curr_audio_sec = total_samples / SAMPLE_RATE
                speed = curr_audio_sec / max(elapsed, 0.001)
                print(f"  [{idx + 1}/{len(segments)}] Synthesized {curr_audio_sec:.1f}s of audio in {elapsed:.1f}s ({speed:.1f}x real-time)")

        # Combine all audio chunks
        print("[*] Concatenating audio stems...")
        final_audio = np.concatenate(audio_chunks)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), final_audio, SAMPLE_RATE)
        
        duration_sec = len(final_audio) / SAMPLE_RATE
        total_time = time.time() - t_start
        print(f"[✓] Chapter WAV complete: {output_wav_path.name} ({duration_sec:.1f}s / {duration_sec/60:.2f} min, rendered in {total_time:.1f}s)")
        return duration_sec

    def master_to_mp3(
        self,
        input_wav: pathlib.Path,
        output_mp3: pathlib.Path,
        title: str = "Chapter 1",
        album: str = "Beyond the Event Horizon",
        artist: str = "PRIME Sovereign Studio",
        track: int = 1
    ) -> bool:
        """
        Applies EBU R128 / ACX-compliant loudness normalization (-23 LUFS, -3 dB Peak)
        and encodes to high-bitrate MP3 with ID3 metadata.
        """
        print(f"[*] Mastering to ACX-compliant MP3: {output_mp3.name}...")
        output_mp3.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_wav),
            "-af", "loudnorm=I=-23:LRA=7:TP=-3.0",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-metadata", f"title={title}",
            "-metadata", f"album={album}",
            "-metadata", f"artist={artist}",
            "-metadata", f"track={track}",
            str(output_mp3)
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print(f"[✓] Mastered MP3 successfully: {output_mp3} ({output_mp3.stat().st_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"[!] FFmpeg mastering failed: {res.stderr}")
            return False


def narrate_chapter_file(
    novel_slug: str,
    chapter_num: int,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Top-level pipeline entry for narrating a single chapter file."""
    ch_file = NOVELS_ROOT / novel_slug / "chapters" / f"chapter_{chapter_num:02d}.md"
    if not ch_file.exists():
        raise FileNotFoundError(f"Manuscript not found: {ch_file}")

    text = ch_file.read_text(encoding="utf-8")
    
    # Target audio directory
    audio_dir = NOVELS_ROOT / novel_slug / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"chapter_{chapter_num:02d}.wav"
    mp3_path = audio_dir / f"chapter_{chapter_num:02d}.mp3"

    narrator = SovereignAudioNarrator(device=device)
    dur = narrator.narrate_chapter(text, wav_path)

    # Master MP3
    novel_title = novel_slug.replace("_", " ").title()
    mastered = narrator.master_to_mp3(
        input_wav=wav_path,
        output_mp3=mp3_path,
        title=f"Chapter {chapter_num}",
        album=novel_title,
        artist="PRIME Sovereign Studio",
        track=chapter_num
    )

    return {
        "slug": novel_slug,
        "chapter": chapter_num,
        "duration_sec": dur,
        "duration_min": dur / 60.0,
        "wav_path": str(wav_path),
        "mp3_path": str(mp3_path),
        "mastered": mastered
    }


def narrate_novel_full(
    novel_slug: str = "beyond_the_event_horizon",
    device: str = "cpu",
    bitrate: str = "128k"
) -> Dict[str, Any]:
    """
    Narrates all chapters of a novel and concatenates them into a single master audiobook file.
    Generates embedded chapter markers for MP3 and M4B audiobook players.
    """
    ch_dir = NOVELS_ROOT / novel_slug / "chapters"
    if not ch_dir.exists():
        raise FileNotFoundError(f"Chapters directory not found: {ch_dir}")

    ch_files = sorted(ch_dir.glob("chapter_*.md"))
    if not ch_files:
        raise FileNotFoundError(f"No chapters found in: {ch_dir}")

    audio_dir = NOVELS_ROOT / novel_slug / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    export_dir = NOVELS_ROOT / novel_slug / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🎙️ [PRIME Sovereign Audio Studio] Narrating Full Novel: '{novel_slug}'")
    print(f"   Chapters to process: {len(ch_files)}")
    print("=" * 80)

    narrator = SovereignAudioNarrator(device=device)

    chapter_entries = []
    current_offset_ms = 0
    concat_list_file = audio_dir / "concat_list.txt"
    concat_lines = []

    for idx, ch_path in enumerate(ch_files, 1):
        ch_num = int(ch_path.stem.split("_")[-1])
        wav_path = audio_dir / f"chapter_{ch_num:02d}.wav"
        
        # Read title from first line of markdown
        text = ch_path.read_text(encoding="utf-8")
        first_line = text.strip().split('\n')[0]
        if first_line.startswith("#"):
            ch_title = first_line.lstrip("#").strip()
        else:
            ch_title = f"Chapter {ch_num}"

        print(f"\n[{idx}/{len(ch_files)}] --- {ch_title} ---")

        # Reuse existing WAV if it exists and has content
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            print(f"  [✓] Found existing WAV: {wav_path.name} ({wav_path.stat().st_size / (1024*1024):.1f} MB)")
            info = sf.info(str(wav_path))
            ch_duration_sec = info.duration
        else:
            ch_duration_sec = narrator.narrate_chapter(text, wav_path)

        duration_ms = int(ch_duration_sec * 1000)
        start_ms = current_offset_ms
        end_ms = current_offset_ms + duration_ms
        current_offset_ms = end_ms

        chapter_entries.append({
            "num": ch_num,
            "title": ch_title,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_sec": ch_duration_sec,
            "wav_path": str(wav_path)
        })

        concat_lines.append(f"file '{wav_path.resolve()}'")

    concat_list_file.write_text("\n".join(concat_lines), encoding="utf-8")

    # Generate FFMPEG chapter metadata
    novel_title = novel_slug.replace("_", " ").title()
    meta_file = audio_dir / "ffmetadata.txt"
    meta_content = [
        ";FFMETADATA1",
        f"title={novel_title} (Unabridged)",
        f"album={novel_title}",
        "artist=PRIME Sovereign Studio",
        "genre=Audiobook",
        ""
    ]
    for ch in chapter_entries:
        meta_content.append("[CHAPTER]")
        meta_content.append("TIMEBASE=1/1000")
        meta_content.append(f"START={ch['start_ms']}")
        meta_content.append(f"END={ch['end_ms']}")
        meta_content.append(f"title={ch['title']}")
        meta_content.append("")

    meta_file.write_text("\n".join(meta_content), encoding="utf-8")

    total_duration_sec = current_offset_ms / 1000.0
    total_duration_hours = total_duration_sec / 3600.0
    print("\n" + "=" * 80)
    print(f"[*] All {len(ch_files)} chapters synthesized!")
    print(f"    Total Runtime: {total_duration_hours:.2f} hours ({total_duration_sec:.1f}s)")
    print("[*] Concatenating and mastering single full-novel audiobook...")

    master_mp3 = export_dir / f"{novel_slug}_Full_Audiobook.mp3"
    master_m4b = export_dir / f"{novel_slug}_Full_Audiobook.m4b"

    # Master to MP3 with chapter metadata and ACX loudness
    cmd_mp3 = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
        "-i", str(meta_file),
        "-map_metadata", "1",
        "-af", "loudnorm=I=-23:LRA=7:TP=-3.0",
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(master_mp3)
    ]
    print(f"[*] Encoding Master MP3 ({bitrate})...")
    subprocess.run(cmd_mp3, check=True)
    print(f"[✓] Master MP3 created: {master_mp3} ({master_mp3.stat().st_size / (1024*1024):.1f} MB)")

    # Master to M4B (AAC with native chapter markers)
    cmd_m4b = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
        "-i", str(meta_file),
        "-map_metadata", "1",
        "-c:a", "aac",
        "-b:a", bitrate,
        str(master_m4b)
    ]
    print(f"[*] Encoding Master M4B Audiobook...")
    subprocess.run(cmd_m4b, check=True)
    print(f"[✓] Master M4B created: {master_m4b} ({master_m4b.stat().st_size / (1024*1024):.1f} MB)")

    return {
        "slug": novel_slug,
        "chapters_count": len(ch_files),
        "total_duration_sec": total_duration_sec,
        "total_duration_hours": total_duration_hours,
        "master_mp3": str(master_mp3),
        "master_m4b": str(master_m4b)
    }


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "beyond_the_event_horizon"
    if len(sys.argv) > 2 and sys.argv[2] == "--full":
        narrate_novel_full(slug)
    else:
        ch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        narrate_chapter_file(slug, ch)

