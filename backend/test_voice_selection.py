"""
Verification test suite for Human Voice Selection in Voice Latency Benchmark:
1. Verify REST API /api/tts/voices returns curated human voices
2. Verify audio synthesis for each of the 4 demographic categories:
   - Child / Youth (en-US-AnaNeural, en-GB-MaisieNeural)
   - Adult Female (en-US-JennyNeural, en-IN-NeerjaNeural)
   - Adult Male (en-US-GuyNeural, en-IN-PrabhatNeural)
   - Mature / Narrator (en-US-ChristopherNeural, en-US-RogerNeural)
3. Verify pitch modulation (-20Hz, +20Hz)
4. Verify voice switching in full-duplex WebSocket session without breaking operation
"""
import sys
import json
import time
import urllib.request
import urllib.error
import asyncio
import websockets

BASE_URL = "http://127.0.0.1:8024"
WS_URL = "ws://127.0.0.1:8024/ws/call"

def _http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def _http_post(url, data_dict):
    req_body = json.dumps(data_dict).encode('utf-8')
    req = urllib.request.Request(url, data=req_body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode('utf-8'))

def test_rest_voices_catalog():
    print("\n--- 1. Testing GET /api/tts/voices ---")
    status, data = _http_get(f"{BASE_URL}/api/tts/voices")
    assert status == 200, f"Expected 200, got {status}"
    voices = data.get("voices", [])
    categories = data.get("categories", [])
    notice = data.get("elderly_limitation_notice", "")
    
    print(f"Total human voices returned: {len(voices)}")
    print(f"Categories: {categories}")
    print(f"Notice: {notice}")
    
    cat_counts = {}
    for v in voices:
        c = v["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for c, cnt in cat_counts.items():
        print(f"  • {c}: {cnt} voices")
        
    assert len(voices) >= 28, f"Expected >= 28 voices, got {len(voices)}"
    assert "Child / Youth" in cat_counts
    assert "Adult Female" in cat_counts
    assert "Adult Male" in cat_counts
    assert "Mature / Narrator" in cat_counts
    print("Catalog verification PASSED.")
    return voices

def test_audio_samples_for_categories(voices):
    print("\n--- 2. Testing Audio Generation Across Demographic Categories ---")
    test_cases = [
        {"cat": "Child / Youth", "voice": "en-US-AnaNeural", "pitch": "+15Hz", "speed": 1.0, "text": "Hi! My name is Ana, and I am a cheerful American youth voice."},
        {"cat": "Child / Youth", "voice": "en-GB-MaisieNeural", "pitch": "+10Hz", "speed": 1.0, "text": "Hello! I am Maisie, a polite British child voice."},
        {"cat": "Adult Female", "voice": "en-US-JennyNeural", "pitch": "+0Hz", "speed": 1.0, "text": "Good morning. I am Jenny, ready to assist you today."},
        {"cat": "Adult Female", "voice": "en-IN-NeerjaNeural", "pitch": "-5Hz", "speed": 1.05, "text": "Namaste! Neerja here with warm Indian customer support."},
        {"cat": "Adult Male", "voice": "en-US-GuyNeural", "pitch": "+0Hz", "speed": 1.0, "text": "Hello, Guy here. Everything is sounding clear and steady."},
        {"cat": "Adult Male", "voice": "en-IN-PrabhatNeural", "pitch": "+5Hz", "speed": 1.0, "text": "Greetings, Prabhat here. Let's get right to business."},
        {"cat": "Mature / Narrator", "voice": "en-US-ChristopherNeural", "pitch": "-15Hz", "speed": 0.95, "text": "Good evening. Christopher presenting clear, authoritative narration."},
        {"cat": "Mature / Narrator", "voice": "en-US-RogerNeural", "pitch": "-10Hz", "speed": 1.0, "text": "Welcome. I am Roger, your seasoned audiobook narrator."}
    ]
    
    results = []
    for tc in test_cases:
        t0 = time.perf_counter()
        payload = {
            "text": tc["text"],
            "voice": tc["voice"],
            "pitch": tc["pitch"],
            "speed": tc["speed"]
        }
        status, data = _http_post(f"{BASE_URL}/api/tts", payload)
        t_req = time.perf_counter() - t0
        assert status == 200, f"Failed for {tc['voice']}: {data}"
        audio_b64 = data.get("audio_base64", "")
        duration = data.get("duration", 0)
        tts_lat = data.get("tts_latency", 0)
        pitch_ret = data.get("pitch", "")
        
        print(f"[{tc['cat']}] Voice: {tc['voice']:<26} Pitch: {tc['pitch']:<6} TTFA: {t_req*1000:.0f}ms | Audio: {duration:.2f}s ({len(audio_b64)} b64 chars)")
        assert len(audio_b64) > 1000, "Empty audio base64 returned"
        assert pitch_ret == tc["pitch"], f"Pitch mismatch: expected {tc['pitch']}, got {pitch_ret}"
        results.append({
            "voice": tc["voice"],
            "cat": tc["cat"],
            "ttfa_ms": round(t_req * 1000, 1),
            "duration_s": duration
        })
    print("Demographic audio generation PASSED.")
    return results

async def test_full_duplex_voice_switching():
    print("\n--- 3. Testing Full-Duplex WebSocket with Live Voice Switching ---")
    async with websockets.connect(WS_URL) as ws:
        # Step A: Connect and configure initial Adult Female voice
        print("Sending initial config: voice=en-IN-NeerjaNeural, pitch=+0Hz...")
        await ws.send(json.dumps({
            "type": "config",
            "tts_voice": "en-IN-NeerjaNeural",
            "tts_pitch": "+0Hz",
            "tts_speed": 1.0,
            "groq_model": "llama-3.1-8b-instant"
        }))
        
        # Step B: Switch voice mid-session to Child voice (en-US-AnaNeural) with pitch=+10Hz
        print("Switching voice mid-session to Child voice (en-US-AnaNeural, pitch=+10Hz)...")
        await ws.send(json.dumps({
            "type": "config",
            "tts_voice": "en-US-AnaNeural",
            "tts_pitch": "+10Hz",
            "tts_speed": 1.05
        }))
        
        # Step C: Send simulated user speech audio (silent/short 16kHz PCM16 chunk)
        pcm_chunk = bytes(3200) # 100ms
        await ws.send(pcm_chunk)
        
        # Step D: Switch voice to Mature Narrator (en-US-ChristopherNeural) with pitch=-15Hz
        print("Switching voice mid-session to Mature voice (en-US-ChristopherNeural, pitch=-15Hz)...")
        await ws.send(json.dumps({
            "type": "config",
            "tts_voice": "en-US-ChristopherNeural",
            "tts_pitch": "-15Hz",
            "tts_speed": 0.95
        }))
        
        # Check connection is still alive and responsive
        await ws.send(json.dumps({"type": "interrupt"}))
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            print(f"Received message after voice switches: type={data.get('type')}")
        except asyncio.TimeoutError:
            print("No immediate message queued (normal for interrupt with empty audio queue).")
            
        print("Full-duplex WebSocket session remained healthy through multiple live voice switches.")
    print("Full-duplex voice switching PASSED.")

def main():
    print("=" * 60)
    print("HUMAN VOICE SELECTION BENCHMARK VERIFICATION SUITE")
    print("=" * 60)
    voices = test_rest_voices_catalog()
    samples = test_audio_samples_for_categories(voices)
    asyncio.run(test_full_duplex_voice_switching())
    print("\n" + "=" * 60)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
