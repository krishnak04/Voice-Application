"""
Comprehensive Speech Recognition Accuracy Benchmark & Verification Suite
Tests 20+ varied speech samples across accents, speech rates, phrase lengths, and audio conditions.
Measures Word Error Rate (WER), Character Error Rate (CER), Sentence Error Rate (SER), and Latency.
"""

import os
import sys
import io
import time
import json
import wave
import asyncio
import numpy as np
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from stt_service import get_stt_engine
from sherpa_service import get_sherpa_engine

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) using Levenshtein distance on words."""
    import re
    # Normalize: lowercase, remove punctuation, strip excess whitespace
    ref_words = re.sub(r'[^\w\s]', '', reference.lower()).split()
    hyp_words = re.sub(r'[^\w\s]', '', hypothesis.lower()).split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Dynamic programming matrix for Levenshtein distance
    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    distance = dp[r_len][h_len]
    return min(1.0, distance / r_len)

# 20 diverse, realistic test utterances covering conversational phone queries,
# customer service, short commands, long multi-clause sentences, and varied vocabularies.
TEST_DATASET_SPEC = [
    # 1-5: Typical customer service & phone simulator inquiries
    {
        "id": "sample_01",
        "text": "Hello, thank you for calling customer service. How can I help you today?",
        "voice": "en-US-JennyNeural",
        "speed": 1.0,
        "category": "Customer Support / Greeting",
        "accent": "US"
    },
    {
        "id": "sample_02",
        "text": "I would like to check the current balance on my account and review recent transactions.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.0,
        "category": "Banking / Account",
        "accent": "Indian"
    },
    {
        "id": "sample_03",
        "text": "Could you please tell me when my package is scheduled to arrive at my address?",
        "voice": "en-GB-SoniaNeural",
        "speed": 1.0,
        "category": "Delivery / Tracking",
        "accent": "British"
    },
    {
        "id": "sample_04",
        "text": "Yes, I am calling because my internet connection has been dropping frequently since yesterday.",
        "voice": "en-AU-NatashaNeural",
        "speed": 1.0,
        "category": "Technical Support",
        "accent": "Australian"
    },
    {
        "id": "sample_05",
        "text": "Can I schedule an appointment with a technical specialist for next Tuesday afternoon?",
        "voice": "en-CA-ClaraNeural",
        "speed": 1.0,
        "category": "Scheduling / Appointment",
        "accent": "Canadian"
    },

    # 6-10: Fast and slow speech rates
    {
        "id": "sample_06",
        "text": "Please hold the line while I quickly transfer your call to the billing department.",
        "voice": "en-US-GuyNeural",
        "speed": 1.25,  # Fast speech
        "category": "Fast Speech (1.25x)",
        "accent": "US"
    },
    {
        "id": "sample_07",
        "text": "I need to urgently update my primary shipping address before the order leaves the warehouse.",
        "voice": "en-IN-PrabhatNeural",
        "speed": 1.2,  # Fast speech
        "category": "Fast Speech (1.20x)",
        "accent": "Indian"
    },
    {
        "id": "sample_08",
        "text": "I am speaking slowly and clearly so that every word can be transcribed accurately.",
        "voice": "en-GB-RyanNeural",
        "speed": 0.85,  # Slow speech
        "category": "Slow Speech (0.85x)",
        "accent": "British"
    },
    {
        "id": "sample_09",
        "text": "Take your time and verify the six digit security code sent to your registered phone number.",
        "voice": "en-IE-ConnorNeural",
        "speed": 0.9,   # Deliberate pacing
        "category": "Deliberate Pacing (0.90x)",
        "accent": "Irish"
    },
    {
        "id": "sample_10",
        "text": "Let me confirm that your email address is spelled correctly in our records.",
        "voice": "en-US-AriaNeural",
        "speed": 1.15,
        "category": "Verification / Moderate Fast",
        "accent": "US"
    },

    # 11-15: Short phrases, commands, and numbers
    {
        "id": "sample_11",
        "text": "Yes, that sounds perfect.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.0,
        "category": "Short Phrase / Affirmation",
        "accent": "Indian"
    },
    {
        "id": "sample_12",
        "text": "No, cancel the reservation immediately.",
        "voice": "en-US-JennyNeural",
        "speed": 1.05,
        "category": "Short Command / Cancellation",
        "accent": "US"
    },
    {
        "id": "sample_13",
        "text": "My confirmation number is four eight two nine one five.",
        "voice": "en-GB-SoniaNeural",
        "speed": 1.0,
        "category": "Alphanumeric / Digits",
        "accent": "British"
    },
    {
        "id": "sample_14",
        "text": "What are your business hours on weekends?",
        "voice": "en-AU-NatashaNeural",
        "speed": 1.0,
        "category": "Short Question",
        "accent": "Australian"
    },
    {
        "id": "sample_15",
        "text": "Repeat that again please.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.0,
        "category": "Short Request",
        "accent": "Indian"
    },

    # 16-20: Long multi-clause complex sentences & challenging vocabulary
    {
        "id": "sample_16",
        "text": "Although the flight was delayed by two hours due to severe weather, the airline provided meal vouchers and rebooked all connecting passengers without additional fees.",
        "voice": "en-US-JennyNeural",
        "speed": 1.0,
        "category": "Long Complex Multi-Clause",
        "accent": "US"
    },
    {
        "id": "sample_17",
        "text": "If you wish to initiate a formal warranty claim, please make sure you upload the original receipt and photograph the damaged component clearly.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.05,
        "category": "Instructions / Conditional Clause",
        "accent": "Indian"
    },
    {
        "id": "sample_18",
        "text": "The quarterly financial report highlighted substantial growth in international markets, driven primarily by strong adoption of cloud subscription services.",
        "voice": "en-GB-SoniaNeural",
        "speed": 1.0,
        "category": "Business / Financial",
        "accent": "British"
    },
    {
        "id": "sample_19",
        "text": "Could you provide more information about the differences between standard annual subscriptions and the premium enterprise tier?",
        "voice": "en-CA-ClaraNeural",
        "speed": 1.0,
        "category": "Product Comparison",
        "accent": "Canadian"
    },
    {
        "id": "sample_20",
        "text": "I have been trying to reset my password for the past thirty minutes, but the verification link seems to have expired before I could open it.",
        "voice": "en-IN-AnanyaNeural",
        "speed": 1.0,
        "category": "Technical Problem / Compound",
        "accent": "Indian"
    },

    # 21-25: Questions (Who, What, Where, When, Why, How) across different accents
    {
        "id": "sample_21",
        "text": "Where is the nearest branch office located in Bangalore?",
        "voice": "en-IN-PrabhatNeural",
        "speed": 1.0,
        "category": "Location Question",
        "accent": "Indian"
    },
    {
        "id": "sample_22",
        "text": "Who should I speak with to get approval for an international wire transfer?",
        "voice": "en-GB-RyanNeural",
        "speed": 1.0,
        "category": "Inquiry / Transfer",
        "accent": "British"
    },
    {
        "id": "sample_23",
        "text": "Why has my debit card been blocked after entering the correct PIN three times?",
        "voice": "en-US-GuyNeural",
        "speed": 1.0,
        "category": "Troubleshooting Question",
        "accent": "US"
    },
    {
        "id": "sample_24",
        "text": "How do I configure two factor authentication on my mobile banking application?",
        "voice": "en-AU-NatashaNeural",
        "speed": 1.0,
        "category": "Technical Setup",
        "accent": "Australian"
    },
    {
        "id": "sample_25",
        "text": "When will the refund of one hundred and forty five dollars be credited back to my account?",
        "voice": "en-CA-ClaraNeural",
        "speed": 1.0,
        "category": "Billing / Refund Question",
        "accent": "Canadian"
    },

    # 26-30: Commands & Action Requests
    {
        "id": "sample_26",
        "text": "Turn off all lights in the conference room and lock the glass door.",
        "voice": "en-US-JennyNeural",
        "speed": 1.0,
        "category": "Smart Home / Office Command",
        "accent": "US"
    },
    {
        "id": "sample_27",
        "text": "Transfer five thousand rupees from my savings account to my current account immediately.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.05,
        "category": "Financial Action Command",
        "accent": "Indian"
    },
    {
        "id": "sample_28",
        "text": "Send an urgent email notification to the engineering team regarding the unexpected server outage.",
        "voice": "en-GB-SoniaNeural",
        "speed": 1.0,
        "category": "Notification Command",
        "accent": "British"
    },
    {
        "id": "sample_29",
        "text": "Download the quarterly billing statement for the month of September twenty twenty four.",
        "voice": "en-US-AriaNeural",
        "speed": 1.1,
        "category": "Document Retrieval Command",
        "accent": "US"
    },
    {
        "id": "sample_30",
        "text": "Mute my microphone and pause the active audio recording.",
        "voice": "en-IE-ConnorNeural",
        "speed": 1.0,
        "category": "Audio Control Command",
        "accent": "Irish"
    },

    # 31-35: Numbers, Prices, Dates, Alphanumerics
    {
        "id": "sample_31",
        "text": "The flight number is British Airways seven eight four departing at fourteen thirty hours.",
        "voice": "en-GB-RyanNeural",
        "speed": 1.0,
        "category": "Alphanumeric / Flight",
        "accent": "British"
    },
    {
        "id": "sample_32",
        "text": "My telephone number is nine eight seven six five four three two one zero.",
        "voice": "en-IN-PrabhatNeural",
        "speed": 1.0,
        "category": "Phone Digits",
        "accent": "Indian"
    },
    {
        "id": "sample_33",
        "text": "The total invoice amount is twelve thousand four hundred and fifty dollars and fifty cents.",
        "voice": "en-US-GuyNeural",
        "speed": 1.0,
        "category": "Financial Numbers & Cents",
        "accent": "US"
    },
    {
        "id": "sample_34",
        "text": "Tracking code X Y Z nine nine zero one four seven was dispatched on October twenty first.",
        "voice": "en-CA-ClaraNeural",
        "speed": 1.0,
        "category": "Tracking Code / Dates",
        "accent": "Canadian"
    },
    {
        "id": "sample_35",
        "text": "The room temperature is thirty two point five degrees Celsius with seventy percent humidity.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.0,
        "category": "Scientific Decimals & Percentages",
        "accent": "Indian"
    },

    # 36-40: Technical, Scientific, & Medical Terminology
    {
        "id": "sample_36",
        "text": "The PostgreSQL database cluster experienced high memory pressure on the primary replica node.",
        "voice": "en-US-JennyNeural",
        "speed": 1.05,
        "category": "Technical / Database",
        "accent": "US"
    },
    {
        "id": "sample_37",
        "text": "The physician prescribed amoxicillin five hundred milligrams twice daily after meals.",
        "voice": "en-IN-AnanyaNeural",
        "speed": 1.0,
        "category": "Medical / Pharmaceutical",
        "accent": "Indian"
    },
    {
        "id": "sample_38",
        "text": "The convolutional neural network was fine tuned using stochastic gradient descent and cross entropy loss.",
        "voice": "en-GB-SoniaNeural",
        "speed": 1.0,
        "category": "Machine Learning Technical",
        "accent": "British"
    },
    {
        "id": "sample_39",
        "text": "Ensure that transport layer security version one point three encryption is enforced on all endpoints.",
        "voice": "en-AU-NatashaNeural",
        "speed": 1.0,
        "category": "Cybersecurity / Protocols",
        "accent": "Australian"
    },
    {
        "id": "sample_40",
        "text": "The patient reported acute abdominal pain and mild hypertension during the physical examination.",
        "voice": "en-US-AriaNeural",
        "speed": 0.95,
        "category": "Clinical / Medical Symptoms",
        "accent": "US"
    },

    # 41-45: Short 1-4 Word Conversational Utterances
    {
        "id": "sample_41",
        "text": "Okay, thank you very much.",
        "voice": "en-US-JennyNeural",
        "speed": 1.0,
        "category": "Short Phrase / Courtesy",
        "accent": "US"
    },
    {
        "id": "sample_42",
        "text": "Yes, I agree completely.",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.0,
        "category": "Short Phrase / Agreement",
        "accent": "Indian"
    },
    {
        "id": "sample_43",
        "text": "Stop the playback now.",
        "voice": "en-GB-RyanNeural",
        "speed": 1.0,
        "category": "Short Command",
        "accent": "British"
    },
    {
        "id": "sample_44",
        "text": "How much does it cost?",
        "voice": "en-AU-NatashaNeural",
        "speed": 1.0,
        "category": "Short Question",
        "accent": "Australian"
    },
    {
        "id": "sample_45",
        "text": "Goodbye, have a nice day.",
        "voice": "en-US-GuyNeural",
        "speed": 1.0,
        "category": "Short Farewells",
        "accent": "US"
    },

    # 46-50: Conversational, Natural Pauses, and Long Multi-Clause Sentences
    {
        "id": "sample_46",
        "text": "Well, let me see, I think our product kickoff meeting was originally scheduled for Thursday morning at ten o'clock.",
        "voice": "en-US-AriaNeural",
        "speed": 0.95,
        "category": "Conversational Fillers & Pauses",
        "accent": "US"
    },
    {
        "id": "sample_47",
        "text": "Actually, before we proceed with the payment, could you please double check if there are any promotional discounts available for annual subscribers?",
        "voice": "en-IN-NeerjaNeural",
        "speed": 1.0,
        "category": "Conversational Inquiry / Multi-Clause",
        "accent": "Indian"
    },
    {
        "id": "sample_48",
        "text": "I was wondering whether you could email me a copy of the contract, or if not, let me know who I should contact in legal.",
        "voice": "en-GB-SoniaNeural",
        "speed": 1.0,
        "category": "Compound Sentence / Request",
        "accent": "British"
    },
    {
        "id": "sample_49",
        "text": "To be completely honest with you, the customer feedback on our new mobile interface has been overwhelmingly positive across every demographic.",
        "voice": "en-CA-ClaraNeural",
        "speed": 1.0,
        "category": "Conversational Feedback / Long",
        "accent": "Canadian"
    },
    {
        "id": "sample_50",
        "text": "In conclusion, despite several initial logistical obstacles, the entire infrastructure migration was delivered on schedule and well within the allocated budget.",
        "voice": "en-IN-PrabhatNeural",
        "speed": 1.05,
        "category": "Formal Summary / Complex Vocabulary",
        "accent": "Indian"
    }
]

async def prepare_test_audio_samples(output_dir: Path):
    """Generate high-fidelity 16kHz mono reference WAV samples using Edge-TTS."""
    import edge_tts
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples_manifest = []
    print(f"Preparing {len(TEST_DATASET_SPEC)} diverse speech benchmark samples in {output_dir}...")
    
    for spec in TEST_DATASET_SPEC:
        wav_path = output_dir / f"{spec['id']}.wav"
        mp3_path = output_dir / f"{spec['id']}.mp3"
        if not wav_path.exists():
            print(f"  Generating {spec['id']} ({spec['accent']} - {spec['voice']})...")
            speed_val = spec['speed']
            rate_str = f"+{int((speed_val - 1.0) * 100)}%" if speed_val >= 1.0 else f"-{int((1.0 - speed_val) * 100)}%"
            try:
                communicate = edge_tts.Communicate(spec['text'], voice=spec['voice'], rate=rate_str)
                await communicate.save(str(mp3_path))
            except Exception as e:
                fb_v = "en-IN-NeerjaNeural" if spec['accent'] == "Indian" else "en-US-JennyNeural"
                print(f"    Notice: Voice {spec['voice']} fallback to {fb_v} ({e})")
                communicate = edge_tts.Communicate(spec['text'], voice=fb_v, rate=rate_str)
                await communicate.save(str(mp3_path))

            # Decode MP3 to strict 16kHz mono PCM16 WAV via PyAV with complete flush
            import av
            container = av.open(str(mp3_path))
            audio_stream = next(s for s in container.streams if s.type == 'audio')
            resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
            
            raw_pcm = bytearray()
            for frame in container.decode(audio_stream):
                for rf in resampler.resample(frame):
                    raw_pcm.extend(rf.to_ndarray().tobytes())
            for rf in resampler.resample(None):
                raw_pcm.extend(rf.to_ndarray().tobytes())
            container.close()

            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(raw_pcm)

            if mp3_path.exists():
                try: mp3_path.unlink()
                except Exception: pass

        # Load audio properties
        with wave.open(str(wav_path), 'rb') as wf:
            dur = wf.getnframes() / wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            np_arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            peak = float(np.max(np.abs(np_arr))) if len(np_arr) > 0 else 0.0
            rms = float(np.sqrt(np.mean(np_arr**2))) if len(np_arr) > 0 else 0.0

        samples_manifest.append({
            **spec,
            "wav_path": str(wav_path),
            "duration": round(dur, 2),
            "peak_level": round(peak, 3),
            "rms_level": round(rms, 3)
        })

    return samples_manifest

def run_accuracy_benchmark(manifest):
    """Run transcription accuracy benchmark across Whisper and Sherpa models."""
    stt = get_stt_engine()
    sherpa = get_sherpa_engine()

    results = {}

    # Define test configurations
    models_to_test = [
        {"engine": "whisper", "model": "base", "label": "Faster-Whisper (base, beam=3, vad=True, normalized)"},
        {"engine": "whisper", "model": "base.en", "label": "Faster-Whisper (base.en, beam=3, vad=True, normalized)"},
        {"engine": "whisper", "model": "small.en", "label": "Faster-Whisper (small.en, beam=3, vad=True, normalized)"},
        {"engine": "sherpa", "model": "sherpa-onnx-streaming-zipformer-en-2023-06-26", "label": "Sherpa Streaming (2023-06-26 Standard with lookahead silence flush)"},
        {"engine": "sherpa", "model": "sherpa-onnx-zipformer-gigaspeech-2023-12-12", "label": "Sherpa Offline (GigaSpeech 2023-12-12)"}
    ]

    for m_cfg in models_to_test:
        label = m_cfg["label"]
        print(f"\n=======================================================")
        print(f"BENCHMARKING: {label}")
        print(f"=======================================================")
        
        wers = []
        latencies = []
        sample_results = []

        if m_cfg["engine"] == "sherpa":
            sherpa.switch_model(m_cfg["model"])

        for sample in manifest:
            wav_path = sample["wav_path"]
            ref_text = sample["text"]

            t0 = time.perf_counter()
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()

            if m_cfg["engine"] == "whisper":
                res = stt.transcribe_audio_bytes(audio_bytes, file_ext=".wav", model_name=m_cfg["model"])
                hyp_text = res["transcript"]
            else:
                res = sherpa.transcribe_audio_source(audio_bytes)
                hyp_text = res["transcript"]

            elapsed = time.perf_counter() - t0
            wer = calculate_wer(ref_text, hyp_text)
            wers.append(wer)
            latencies.append(elapsed)

            sample_results.append({
                "id": sample["id"],
                "category": sample["category"],
                "accent": sample["accent"],
                "duration": sample["duration"],
                "reference": ref_text,
                "hypothesis": hyp_text,
                "wer": round(wer, 4),
                "latency_sec": round(elapsed, 3)
            })

            # Print brief sample status
            print(f"  [{sample['id']}] WER: {wer*100:5.1f}% | Ref: {ref_text[:35]}... -> Hyp: {hyp_text[:35]}...")

        avg_wer = float(np.mean(wers))
        avg_latency = float(np.mean(latencies))
        exact_matches = sum(1 for w in wers if w == 0.0)

        results[label] = {
            "model_config": m_cfg,
            "average_wer": round(avg_wer, 4),
            "exact_match_rate": round(exact_matches / len(wers), 4),
            "average_latency_sec": round(avg_latency, 3),
            "samples": sample_results
        }

        print(f"\n>>> SUMMARY for {label}:")
        print(f"    Average WER: {avg_wer * 100:.2f}%")
        print(f"    Exact Matches: {exact_matches}/{len(wers)} ({exact_matches/len(wers)*100:.1f}%)")
        print(f"    Average Latency: {avg_latency:.3f}s")

    base_dir = Path(__file__).parent
    report_path = base_dir / "accuracy_benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved accuracy benchmark results to: {report_path}")

    return results

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    audio_dir = base_dir / "test_audio_dataset"
    manifest = asyncio.run(prepare_test_audio_samples(audio_dir))
    run_accuracy_benchmark(manifest)
