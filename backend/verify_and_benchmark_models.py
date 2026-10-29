import os
import sys
import time
import io
import wave
import json
import psutil
import subprocess
from pathlib import Path
import numpy as np

# Patch PyAV if installed to prevent 'metadata_errors' TypeError in newer PyAV versions
try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
# Ground truth reference for 0.wav
REF_TEXT = "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS"
TEST_WAV = ROOT_DIR / "sherpa-onnx-streaming-zipformer-en-2023-06-26" / "test_wavs" / "0.wav"

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis strings."""
    ref_words = reference.upper().replace(",", "").replace(".", "").replace(";", "").replace(":", "").replace("-", " ").split()
    hyp_words = hypothesis.upper().replace(",", "").replace(".", "").replace(";", "").replace(":", "").replace("-", " ").split()
    
    # Levenshtein distance on word level
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint32)
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    wer = float(d[len(ref_words), len(hyp_words)]) / float(max(len(ref_words), 1))
    return round(wer * 100, 2)

def read_wav_16k(wav_path: Path):
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples[::n_channels]
        duration = n_frames / framerate
        return samples, framerate, duration

def inspect_hardware():
    print("=" * 70)
    print("HARDWARE & INFERENCE ENVIRONMENT INSPECTION")
    print("=" * 70)
    import torch
    
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory()
    print(f"CPU: {cpu_cores} physical cores, {cpu_threads} logical processors")
    print(f"System RAM: {ram.total / (1024**3):.2f} GB (Available: {ram.available / (1024**3):.2f} GB, {ram.percent}% used)")
    
    # GPU / PyTorch CUDA
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available in PyTorch: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
    else:
        print("  Notice: No active NVIDIA CUDA device detected in PyTorch runtime.")
        
    # Check Windows Video Controller via powershell
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        controllers = [c.strip() for c in res.stdout.strip().splitlines() if c.strip()]
        print(f"Physical Display Adapters (OS): {', '.join(controllers)}")
    except Exception as e:
        print(f"Could not query Win32_VideoController: {e}")
    print("=" * 70)

def download_and_extract_sherpa_model(model_id: str, download_url: str, tar_name: str, dest_dir: Path) -> bool:
    print(f"[*] Downloading {tar_name} from official release...")
    tar_path = dest_dir / tar_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd_dl = ["curl.exe", "-L", "-o", str(tar_path), download_url]
    p_dl = subprocess.run(cmd_dl, cwd=str(dest_dir))
    if p_dl.returncode != 0:
        print(f"[!] curl failed to download {download_url}")
        return False
    print(f"[*] Extracting {tar_name}...")
    cmd_ex = ["tar", "-xjf", str(tar_path)]
    p_ex = subprocess.run(cmd_ex, cwd=str(dest_dir))
    if tar_path.exists():
        try:
            tar_path.unlink()
        except Exception:
            pass
    if p_ex.returncode != 0:
        print(f"[!] tar extraction failed for {tar_name}")
        return False
    print(f"[+] Successfully extracted {model_id}")
    return True

def run_whisper_tests():
    print("\n" + "=" * 70)
    print("WHISPER MODELS VERIFICATION & BENCHMARKING (CPU int8)")
    print("=" * 70)
    from faster_whisper import WhisperModel
    
    models = ["tiny", "base", "small", "medium", "large-v3-turbo", "base.en"]
    results = {}
    
    samples, sr, duration = read_wav_16k(TEST_WAV)
    
    for model_id in models:
        print(f"\n--- Testing Whisper Model: {model_id} ---")
        item = {
            "model_id": model_id,
            "engine": "faster-whisper",
            "type": "offline",
            "status": "pending",
            "load_time_s": None,
            "cold_latency_s": None,
            "warm_latency_s": None,
            "cold_rtf": None,
            "warm_rtf": None,
            "wer": None,
            "transcript": "",
            "cpu_percent": None,
            "ram_mb": None,
            "error": None
        }
        
        # 1. Load Model
        t0 = time.perf_counter()
        proc = psutil.Process()
        ram_before = proc.memory_info().rss
        try:
            model = WhisperModel(model_id, device="cpu", compute_type="int8", cpu_threads=8)
            load_time = round(time.perf_counter() - t0, 3)
            ram_after = proc.memory_info().rss
            item["load_time_s"] = load_time
            item["ram_mb"] = round((ram_after - ram_before) / (1024 * 1024), 1)
            print(f"  Loaded {model_id} in {load_time}s (RAM diff: {item['ram_mb']} MB)")
        except Exception as e:
            item["status"] = "failed"
            item["error"] = f"Model loading failed: {str(e)}"
            print(f"  [FAIL] {item['error']}")
            results[model_id] = item
            continue
            
        # 2. Cold Start Run
        psutil.cpu_percent(interval=None)
        t_cold_start = time.perf_counter()
        try:
            segments, info = model.transcribe(samples, language="en", beam_size=1)
            cold_text = " ".join([s.text.strip() for s in segments]).strip()
            cold_elapsed = round(time.perf_counter() - t_cold_start, 3)
            cold_rtf = round(cold_elapsed / duration, 3)
            item["cold_latency_s"] = cold_elapsed
            item["cold_rtf"] = cold_rtf
            print(f"  Cold Run: {cold_elapsed}s (RTF: {cold_rtf}x)")
        except Exception as e:
            item["status"] = "failed"
            item["error"] = f"Cold transcription failed: {str(e)}"
            print(f"  [FAIL] {item['error']}")
            results[model_id] = item
            continue

        # 3. Warm Run (repeat 2 times for stability)
        warm_times = []
        last_text = ""
        cpu_readings = []
        for _ in range(2):
            t_warm = time.perf_counter()
            segments, info = model.transcribe(samples, language="en", beam_size=1)
            last_text = " ".join([s.text.strip() for s in segments]).strip()
            elapsed = time.perf_counter() - t_warm
            warm_times.append(elapsed)
            cpu_readings.append(psutil.cpu_percent(interval=None))
            
        avg_warm = round(float(np.mean(warm_times)), 3)
        item["warm_latency_s"] = avg_warm
        item["warm_rtf"] = round(avg_warm / duration, 3)
        item["cpu_percent"] = round(float(np.mean(cpu_readings)), 1)
        item["transcript"] = last_text
        item["wer"] = calculate_wer(REF_TEXT, last_text)
        item["status"] = "passed"
        
        print(f"  Warm Latency: {avg_warm}s (RTF: {item['warm_rtf']}x, Speed: {round(duration/avg_warm, 1)}x)")
        print(f"  WER: {item['wer']}% | CPU: {item['cpu_percent']}%")
        print(f"  Transcript: \"{last_text}\"")
        print(f"  [PASS] Whisper {model_id} verified successfully!")
        
        del model
        results[model_id] = item
        
    return results

def run_sherpa_tests():
    print("\n" + "=" * 70)
    print("SHERPA-ONNX MODELS VERIFICATION & BENCHMARKING (CPU)")
    print("=" * 70)
    import sherpa_onnx
    from sherpa_models_catalog import OFFICIAL_SHERPA_ENGLISH_MODELS, find_model_files
    
    sherpa_dir = ROOT_DIR / "sherpa_models"
    sherpa_dir.mkdir(parents=True, exist_ok=True)
    root_dir = ROOT_DIR
    
    samples, sr, duration = read_wav_16k(TEST_WAV)
    results = {}
    
    # 6 Target Sherpa Models
    target_models = [
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "sherpa-onnx-streaming-zipformer-en-2023-06-21",
        "sherpa-onnx-zipformer-small-en-2023-06-26",
        "sherpa-onnx-zipformer-large-en-2023-06-26",
        "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
    ]
    
    for model_id in target_models:
        meta = OFFICIAL_SHERPA_ENGLISH_MODELS.get(model_id)
        print(f"\n--- Testing Sherpa-ONNX Model: {model_id} ({meta['type'].upper()} {meta['category']}) ---")
        item = {
            "model_id": model_id,
            "engine": "sherpa-onnx",
            "type": meta["type"],
            "category": meta["category"],
            "params": meta["params"],
            "status": "pending",
            "load_time_s": None,
            "first_partial_ms": None,
            "cold_latency_s": None,
            "warm_latency_s": None,
            "warm_rtf": None,
            "wer": None,
            "transcript": "",
            "cpu_percent": None,
            "ram_mb": None,
            "error": None
        }
        
        # 1. Locate or download directory
        candidates = [sherpa_dir / model_id, root_dir / model_id]
        m_dir = next((c for c in candidates if c.is_dir()), None)
        
        if not m_dir or not find_model_files(m_dir, use_int8=True)["is_valid"]:
            print(f"[*] Model files missing for {model_id}. Initiating official release download...")
            ok = download_and_extract_sherpa_model(model_id, meta["download_url"], meta["tar_filename"], sherpa_dir)
            if not ok:
                item["status"] = "failed"
                item["error"] = "Failed to download model archive from official release."
                results[model_id] = item
                continue
            m_dir = sherpa_dir / model_id
            
        files = find_model_files(m_dir, use_int8=True)
        if not files["is_valid"]:
            item["status"] = "failed"
            item["error"] = f"Missing required ONNX files in {m_dir}"
            results[model_id] = item
            continue
            
        # 2. Load Recognizer
        proc = psutil.Process()
        ram_before = proc.memory_info().rss
        t0 = time.perf_counter()
        try:
            if meta["type"] == "streaming":
                recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(files["tokens"]),
                    encoder=str(files["encoder"]),
                    decoder=str(files["decoder"]),
                    joiner=str(files["joiner"]),
                    num_threads=6,
                    sample_rate=16000,
                    feature_dim=80,
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=2.4,
                    rule2_min_trailing_silence=0.8,
                    rule3_min_utterance_length=20.0,
                    decoding_method="greedy_search",
                    provider="cpu"
                )
            else:
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    tokens=str(files["tokens"]),
                    encoder=str(files["encoder"]),
                    decoder=str(files["decoder"]),
                    joiner=str(files["joiner"]),
                    num_threads=6,
                    decoding_method="greedy_search"
                )
            load_elapsed = round(time.perf_counter() - t0, 3)
            ram_after = proc.memory_info().rss
            item["load_time_s"] = load_elapsed
            item["ram_mb"] = round((ram_after - ram_before) / (1024 * 1024), 1)
            print(f"  Loaded {model_id} in {load_elapsed}s (RAM: {item['ram_mb']} MB)")
        except Exception as e:
            item["status"] = "failed"
            item["error"] = f"Failed to instantiate Recognizer: {str(e)}"
            print(f"  [FAIL] {item['error']}")
            results[model_id] = item
            continue
            
        # 3. Test Transcription & Timing
        def run_inference():
            if meta["type"] == "streaming":
                stream = recognizer.create_stream()
                chunk_size = 1600  # 100ms
                first_partial_t = None
                t_start = time.perf_counter()
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    stream.accept_waveform(16000, chunk)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    partial = recognizer.get_result(stream).strip()
                    if partial and first_partial_t is None:
                        first_partial_t = round((time.perf_counter() - t_start) * 1000, 2)
                        
                stream.input_finished()
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                text = recognizer.get_result(stream).strip()
                elapsed = time.perf_counter() - t_start
                return text, elapsed, first_partial_t
            else:
                t_start = time.perf_counter()
                stream = recognizer.create_stream()
                stream.accept_waveform(16000, samples)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                elapsed = time.perf_counter() - t_start
                return text, elapsed, None
                
        # Cold start run
        try:
            cold_text, cold_time, cold_fp = run_inference()
            item["cold_latency_s"] = round(cold_time, 3)
            item["first_partial_ms"] = cold_fp
            print(f"  Cold Run: {cold_time:.3f}s (1st Partial: {cold_fp}ms)" if cold_fp else f"  Cold Run: {cold_time:.3f}s")
        except Exception as e:
            item["status"] = "failed"
            item["error"] = f"Inference execution failed: {str(e)}"
            print(f"  [FAIL] {item['error']}")
            results[model_id] = item
            continue
            
        # Warm runs
        warm_times = []
        warm_fps = []
        last_text = ""
        cpu_readings = []
        psutil.cpu_percent(interval=None)
        for _ in range(2):
            text, w_time, w_fp = run_inference()
            warm_times.append(w_time)
            if w_fp is not None:
                warm_fps.append(w_fp)
            last_text = text
            cpu_readings.append(psutil.cpu_percent(interval=None))
            
        avg_warm = round(float(np.mean(warm_times)), 3)
        item["warm_latency_s"] = avg_warm
        item["warm_rtf"] = round(avg_warm / duration, 3)
        if warm_fps:
            item["first_partial_ms"] = round(float(np.mean(warm_fps)), 1)
        item["cpu_percent"] = round(float(np.mean(cpu_readings)), 1)
        item["transcript"] = last_text
        item["wer"] = calculate_wer(REF_TEXT, last_text)
        item["status"] = "passed"
        
        print(f"  Warm Latency: {avg_warm}s (RTF: {item['warm_rtf']}x, Speed: {round(duration/avg_warm, 1)}x real-time)")
        if item["first_partial_ms"]:
            print(f"  1st Partial Latency: {item['first_partial_ms']}ms")
        print(f"  WER: {item['wer']}% | CPU: {item['cpu_percent']}%")
        print(f"  Transcript: \"{last_text}\"")
        print(f"  [PASS] Sherpa-ONNX {model_id} verified successfully!")
        
        del recognizer
        results[model_id] = item
        
    return results

def main():
    inspect_hardware()
    whisper_results = run_whisper_tests()
    sherpa_results = run_sherpa_tests()
    
    out_file = Path("benchmark_all_models_report.json")
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu": "Intel(R) Arc(TM) Graphics (CPU INT8 / Float32 inference)"
        },
        "whisper": whisper_results,
        "sherpa": sherpa_results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved full empirical benchmark results to {out_file}")

if __name__ == "__main__":
    main()
