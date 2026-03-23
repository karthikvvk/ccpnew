"""
Dynamic Chunked Whisper Transcription with Native Language Detection
(GPU Optimized)
"""

import os
import json
import whisper
import numpy as np

from pathlib import Path
from typing import Dict, List, Any

import logging


# ----------------------------
# Logger Setup
# ----------------------------

def setup_logger(name: str):

    logger = logging.getLogger(name)

    if not logger.handlers:

        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger("speech_to_text")


# ----------------------------
# Main Class
# ----------------------------

class SpeechToText:
    """
    Whisper Speech-to-Text with
    - Native language detection
    - Dynamic overlapping chunks
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda"
    ):

        self.model_size = model_size
        self.device = device

        logger.info(
            f"Loading Whisper model: {self.model_size} on {self.device}"
        )

        self.model = whisper.load_model(
            self.model_size,
            device=self.device
        )

        logger.info("Whisper model loaded")


    # ----------------------------
    # Native Language Detection
    # ----------------------------

    def _detect_language(
        self,
        audio: np.ndarray
    ) -> str:

        logger.info("Detecting language from first 30 seconds")

        sample_rate = whisper.audio.SAMPLE_RATE

        max_samples = 30 * sample_rate

        sample = audio[:max_samples]

        sample = whisper.pad_or_trim(sample)

        mel = whisper.log_mel_spectrogram(
            sample
        ).to(self.model.device)

        _, probs = self.model.detect_language(mel)

        lang = max(probs, key=probs.get)

        logger.info(f"Detected language: {lang}")

        return lang


    # ----------------------------
    # Main Chunked Transcription
    # ----------------------------

    def transcribe_chunked(
    self,
    audio_path: Path,
    overlap_ratio: float = 0.25
) -> Dict[str, Any]:

      logger.info(f"Loading audio: {audio_path}")

      audio = whisper.load_audio(str(audio_path))

      sample_rate = whisper.audio.SAMPLE_RATE

      total_samples = len(audio)

      total_duration = total_samples / sample_rate

      logger.info(f"Audio duration: {total_duration:.2f}s")


      # ----------------------------
      # Detect Language Once (optional, for logging)
      # ----------------------------

      lang = self._detect_language(audio)


      # ----------------------------
      # Chunk Parameters
      # ----------------------------

      chunk_duration = total_duration * 0.20

      overlap_duration = chunk_duration * overlap_ratio

      step_duration = chunk_duration - overlap_duration

      chunk_samples = int(chunk_duration * sample_rate)

      step_samples = int(step_duration * sample_rate)

      logger.info(
          f"Chunk: {chunk_duration:.2f}s | "
          f"Overlap: {overlap_duration:.2f}s | "
          f"Step: {step_duration:.2f}s"
      )


      segments: List[Dict[str, Any]] = []

      full_text: List[str] = []


      start = 0

      chunk_id = 0


      # ----------------------------
      # Process Each Chunk with transcribe()
      # ----------------------------

      while start < total_samples:

          end = start + chunk_samples

          chunk_audio = audio[start:end]


          logger.info(f"Transcribing chunk {chunk_id}")


          # IMPORTANT: use transcribe(), not decode()
          result = self.model.transcribe(
              chunk_audio,
              language=lang,
              temperature=0.0,
              fp16=(self.device == "cuda"),
              condition_on_previous_text=False,
              verbose=False
          )


          text = result["text"].strip()


          if text:

              start_time = start / sample_rate

              end_time = min(end, total_samples) / sample_rate


              segments.append({
                  "chunk": chunk_id,
                  "start": round(start_time, 3),
                  "end": round(end_time, 3),
                  "text": text
              })


              full_text.append(text)


          start += step_samples

          chunk_id += 1


      merged_text = self._remove_overlap_duplicates(
          " ".join(full_text)
      )


      logger.info(
          f"Transcription done: {len(segments)} chunks"
      )


      return {
          "text": merged_text,
          "segments": segments,
          "language": lang
      }


    # ----------------------------
    # Overlap Cleanup
    # ----------------------------

    def _remove_overlap_duplicates(
        self,
        text: str,
        window: int = 6
    ) -> str:

        words = text.split()

        result = []


        for word in words:

            if len(result) >= window:

                recent = result[-window:]

                if word in recent:
                    continue


            result.append(word)


        return " ".join(result)


    # ----------------------------
    # Save Output
    # ----------------------------

    def save_result(
        self,
        result: Dict[str, Any],
        json_path: Path,
        txt_path: Path
    ):

        json_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        txt_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )


        with open(
            txt_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(result["text"])


        logger.info(f"Saved: {json_path}")

        logger.info(f"Saved: {txt_path}")


# ----------------------------
# Standalone Runner
# ----------------------------

def transcribe_audio_gpu_chunked(
    audio_path: str,
    output_json: str,
    output_txt: str,
    model_size: str = "medium",
    device: str = "cuda"
) -> Dict[str, Any]:


    stt = SpeechToText(
        model_size=model_size,
        device=device
    )


    result = stt.transcribe_chunked(
        Path(audio_path)
    )


    stt.save_result(
        result,
        Path(output_json),
        Path(output_txt)
    )


    return result


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":

    import sys


    if len(sys.argv) < 4:

        print(
            "Usage:\n"
            "python speech_to_text_chunked.py "
            "<audio> <out.json> <out.txt>"
        )

        sys.exit(1)


    audio = sys.argv[1]

    out_json = sys.argv[2]

    out_txt = sys.argv[3]


    transcribe_audio_gpu_chunked(
        audio,
        out_json,
        out_txt
    )




#!python ./ccpnew/disections/whisper_stt.py  /content/carguys.mp3 out.json out.txt