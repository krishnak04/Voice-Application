import sys
import time
import asyncio
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from groq_service import get_groq_engine
from tts_service import get_tts_engine

async def main():
    groq = get_groq_engine()
    tts = get_tts_engine()
    
    print("=" * 60)
    print("PROFILING GROQ LLM (STREAMING TTFT & COMPLETION)")
    print("=" * 60)
    
    for i in range(3):
        t0 = time.perf_counter()
        first_token_t = None
        full_text = ""
        async for chunk in groq.async_stream_reply("Give me a one sentence greeting.", is_phone_call=True):
            delta = chunk.get("delta", "")
            if delta:
                if first_token_t is None:
                    first_token_t = time.perf_counter() - t0
                full_text += delta
        total_t = time.perf_counter() - t0
        print(f"Run {i+1}: TTFT = {first_token_t*1000:.1f}ms | Total = {total_t*1000:.1f}ms | Reply = \"{full_text.strip()}\"")

    print("\n" + "=" * 60)
    print("PROFILING EDGE-TTS (TIME TO SYNTHESIZE CLAUSES & SENTENCES)")
    print("=" * 60)
    
    phrases = [
        "Hello!",
        "How are you doing today?",
        "The weather is clear and pleasant outside this afternoon."
    ]
    for p in phrases:
        t0 = time.perf_counter()
        res = await tts.async_synthesize(p, voice="en-US-JennyNeural", engine="edge_tts")
        elapsed = time.perf_counter() - t0
        audio_len = len(res["audio_bytes"])
        print(f"Phrase (chars={len(p)}): \"{p}\" -> Latency = {elapsed*1000:.1f}ms | Size = {audio_len} bytes")

if __name__ == "__main__":
    asyncio.run(main())
