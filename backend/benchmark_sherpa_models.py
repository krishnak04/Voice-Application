"""
Automated Empirical Benchmark for Sherpa-ONNX English Models.
Tests loading time, RAM usage, first partial latency, final latency, RTF, and WER
using standard test audio and reference transcripts from LibriSpeech.
"""
import os
import sys
import time
import wave
import gc
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import psutil
import sherpa_onnx
from sherpa_models_catalog import OFFICIAL_SHERPA_ENGLISH_MODELS, find_model_files

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (Levenshtein distance on word tokens)."""
    ref_words = reference.upper().strip().split()
    hyp_words = hypothesis.upper().strip().split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )
    return round((d[len(ref_words)][len(hyp_words)] / len(ref_words)) * 100, 2)


def read_wav_16k(wav_path: Path) -> tuple[np.ndarray, float]:
    """Read wave file into float32 mono array normalized to [-1.0, 1.0]."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    duration = n_frames / framerate
    if sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(raw_bytes, dtype=np.float32)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]

    if framerate != 16000:
        # Simple linear resample if needed
        target_len = int(len(samples) * 16000 / framerate)
        samples = np.interp(
            np.linspace(0, len(samples), target_len, endpoint=False),
            np.arange(len(samples)),
            samples
        ).astype(np.float32)

    return samples, duration


def run_benchmark(force_recompute: bool = False):
    root_dir = Path(__file__).resolve().parent.parent
    sherpa_models_dir = root_dir / "sherpa_models"
    cache_path = sherpa_models_dir / "benchmark_cache.json"

    if not force_recompute and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if isinstance(cached, list) and len(cached) > 0:
                    return cached
        except Exception:
            pass
    
    # Read reference ground truths
    test_wavs_dir = root_dir / "sherpa-onnx-streaming-zipformer-en-2023-06-26" / "test_wavs"
    references = {}
    trans_file = test_wavs_dir / "trans.txt"
    if trans_file.exists():
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    references[parts[0]] = parts[1]

    # Models to benchmark
    target_models = [
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "sherpa-onnx-zipformer-small-en-2023-06-26",
        "sherpa-onnx-zipformer-large-en-2023-06-26"
    ]

    benchmark_results = []
    print("=" * 78)
    print("      OFFICIAL SHERPA-ONNX ENGLISH ASR BENCHMARK (MEASURED RESULTS)     ")
    print("=" * 78)

    process = psutil.Process()

    for model_id in target_models:
        meta = OFFICIAL_SHERPA_ENGLISH_MODELS.get(model_id, {})
        # Check both potential directory locations
        dir_candidates = [
            sherpa_models_dir / model_id,
            root_dir / model_id
        ]
        model_dir = next((d for d in dir_candidates if d.exists()), None)

        if not model_dir:
            print(f"[SKIP] Model directory not found for: {model_id}")
            continue

        model_files = find_model_files(model_dir, use_int8=True)
        if not model_files["is_valid"]:
            print(f"[SKIP] Incomplete model files in: {model_dir}")
            continue

        print(f"\n---> Benchmarking: {meta.get('name', model_id)}")
        print(f"     Type: {meta.get('type').upper()} | Size: {meta.get('category')} ({meta.get('params')})")
        print(f"     Directory: {model_dir.name}")

        gc.collect()
        ram_before = process.memory_info().rss / (1024 * 1024)
        t_load_start = time.perf_counter()

        recognizer = None
        is_streaming = meta.get("type") == "streaming"

        if is_streaming:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(model_files["tokens"]),
                encoder=str(model_files["encoder"]),
                decoder=str(model_files["decoder"]),
                joiner=str(model_files["joiner"]),
                num_threads=2,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True
            )
        else:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=str(model_files["tokens"]),
                encoder=str(model_files["encoder"]),
                decoder=str(model_files["decoder"]),
                joiner=str(model_files["joiner"]),
                num_threads=2
            )

        load_time = round(time.perf_counter() - t_load_start, 3)
        ram_after = process.memory_info().rss / (1024 * 1024)
        model_ram_mb = round(ram_after - ram_before, 1)

        # Benchmark on 0.wav (6.63s) and 1.wav (16.72s)
        test_audios = ["0.wav", "1.wav"]
        model_test_runs = []

        for wav_name in test_audios:
            wav_path = test_wavs_dir / wav_name
            if not wav_path.exists():
                continue

            samples, duration = read_wav_16k(wav_path)
            ref_text = references.get(wav_name, "")

            if is_streaming:
                # Streaming chunk-based recognition (100ms / 1600 samples chunks)
                stream = recognizer.create_stream()
                chunk_size = 1600
                first_partial_t = None
                t_decode_start = time.perf_counter()

                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    stream.accept_waveform(16000, chunk)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    
                    partial = recognizer.get_result(stream).strip()
                    if partial and first_partial_t is None:
                        first_partial_t = round((time.perf_counter() - t_decode_start) * 1000, 1)

                stream.input_finished()
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                
                final_text = recognizer.get_result(stream).strip()
                decode_elapsed = time.perf_counter() - t_decode_start

            else:
                # Offline full-buffer batch recognition
                first_partial_t = None  # Offline does not produce streaming partials
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, samples)
                t_decode_start = time.perf_counter()
                recognizer.decode_stream(stream)
                decode_elapsed = time.perf_counter() - t_decode_start
                final_text = stream.result.text.strip()

            rtf = round(decode_elapsed / max(duration, 0.001), 4)
            wer = calculate_wer(ref_text, final_text) if ref_text else None

            model_test_runs.append({
                "wav": wav_name,
                "duration": round(duration, 2),
                "first_partial_ms": first_partial_t,
                "final_latency_ms": round(decode_elapsed * 1000, 1),
                "rtf": rtf,
                "wer": wer,
                "hyp": final_text
            })

            print(f"  • {wav_name} ({duration:.2f}s): Latency={decode_elapsed*1000:.1f}ms | RTF={rtf:.4f} | "
                  f"1st-Partial={'N/A (Offline)' if first_partial_t is None else f'{first_partial_t}ms'} | WER={wer}%")
            print(f"    Transcript: \"{final_text}\"")

        avg_rtf = round(sum(r["rtf"] for r in model_test_runs) / len(model_test_runs), 4) if model_test_runs else 0
        avg_wer = round(sum(r["wer"] for r in model_test_runs if r["wer"] is not None) / len(model_test_runs), 2) if model_test_runs else 0
        first_partial_sample = model_test_runs[0]["first_partial_ms"] if model_test_runs else None

        benchmark_results.append({
            "model_id": model_id,
            "name": meta.get("name"),
            "category": meta.get("category"),
            "type": meta.get("type"),
            "params": meta.get("params"),
            "load_time_sec": load_time,
            "ram_delta_mb": model_ram_mb,
            "avg_rtf": avg_rtf,
            "avg_wer": avg_wer,
            "first_partial_ms": first_partial_sample,
            "runs": model_test_runs
        })

        del recognizer
        gc.collect()

    print("\n" + "=" * 78)
    print("                         SUMMARY BENCHMARK TABLE                         ")
    print("=" * 78)
    print(f"{'Model':<34} | {'Type':<9} | {'Size':<8} | {'Load(s)':<7} | {'RAM(MB)':<7} | {'Avg RTF':<8} | {'WER (%)':<7}")
    print("-" * 78)
    for res in benchmark_results:
        print(f"{res['name'][:34]:<34} | {res['type']:<9} | {res['category']:<8} | {res['load_time_sec']:<7} | {res['ram_delta_mb']:<7} | {res['avg_rtf']:<8} | {res['avg_wer']:<7}")
    print("=" * 78)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write benchmark cache: {e}")

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()
