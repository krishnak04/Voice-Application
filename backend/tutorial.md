# Voice-to-Text (STT) External API Integration Tutorial
## Multi-Language & Whisper Model Selection (Including 22 Official Indian Languages)

This guide explains how to integrate the **Faster-Whisper Voice-to-Text** service into any external application (Web, Mobile, Desktop, or Backend microservice), with full support for dynamic Whisper model selection and all **22 official Indian languages**.

---

## 1. Overview & Architecture

* **Decoupled Architecture**: Speech-to-Text (STT) is completely separated from Groq LLM. You get pure, ultra-fast transcription (<100ms on GPU) without paying LLM token costs or latency overhead.
* **Pre-installed Dev PC Model**: The dev server is pre-configured with **`MODEL_NAME = "base"`** — a fast multilingual model capable of transcribing 99+ languages.
* **Dynamic Model Switching**: External clients can specify any model on-the-fly (`base`, `small`, `medium`, `large-v3-turbo`, etc.). The server dynamically loads and caches models in GPU VRAM without requiring a restart.
* **Standard HTTP REST**: No WebSockets or complex SDKs required. Audio is uploaded as standard `multipart/form-data`.
* **22 Official Indian Languages**: Whisper supports the 22 scheduled languages of India with native codes (`hi`, `bn`, `ta`, `te`, `mr`, `gu`, `kn`, etc.) or automatic language detection.

---

## 2. API Endpoints Reference

| Endpoint | Method | Content-Type | Description |
| :--- | :---: | :---: | :--- |
| **`/api/transcribe`** | `POST` | `multipart/form-data` | **Core Voice-to-Text**: Transcribes audio directly with Faster-Whisper. Accepts `language` and `model`. |
| **`/api/chat`** | `POST` | `application/json` | **Standalone Groq LLM**: Generates AI conversational text replies (optional). |
| **`/api/health`** | `GET` | N/A | Returns server health, active device (CUDA/CPU), default model, and list of loaded models in memory. |
| **`/api/voice`** | `POST` | `multipart/form-data` | Legacy combined endpoint (STT + Groq LLM) for latency benchmarking. |

---

## 3. Core Voice-to-Text API (`POST /api/transcribe`)

### Request Details
* **URL**: `http://<server-ip>:8024/api/transcribe`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

### Form Parameters
| Parameter | Type | Required | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `audio` | File | **Yes** | — | Audio file binary (`.wav`, `.webm`, `.mp3`, `.m4a`, `.ogg`, `.flac`). |
| `language` | String | No | `auto` | Language code (e.g. `hi`, `ta`, `te`, `bn`, `en`), language number (`1`–`22`), or `auto` for automatic language detection. |
| `model` | String | No | `base` | Whisper model to use (`base`, `small`, `medium`, `large-v3-turbo`, `tiny`, `base.en`). |

### Success Response (`200 OK`)
```json
{
  "transcript": "नमस्ते, आप कैसे हैं? आज का मौसम बहुत अच्छा है।",
  "language": "hi",
  "language_probability": 0.982,
  "duration": 3.42,
  "stt_latency": 0.089,
  "rtf": 0.026,
  "speed_factor": 38.42,
  "device": "cuda",
  "model_name": "base"
}
```

### Response Field Descriptions
* **`transcript`**: The transcribed text in the original script (e.g. Devanagari, Tamil script, Telugu script, Latin, etc.).
* **`language`**: Detected or specified language code (e.g. `hi`, `ta`, `te`, `bn`, `en`).
* **`language_probability`**: Confidence score of the detected language (`0.0` to `1.0`).
* **`duration`**: Duration of the audio file in seconds.
* **`stt_latency`**: Execution time of the Faster-Whisper transcription in seconds.
* **`rtf`**: Real-Time Factor (`stt_latency / duration`). Values < 0.1 indicate >10x faster than real-time.
* **`speed_factor`**: Processing speed multiplier relative to real-time (e.g. `38.42x` real-time).
* **`device`**: Processing accelerator used (`cuda` GPU FP16 or `cpu` int8 fallback).
* **`model_name`**: The actual Whisper model used for inference (`base`, `small`, etc.).

---

## 4. 22 Official Indian Languages Reference Table

The server supports all 22 official scheduled languages of India. You can specify the language via its **Whisper Language Code**, its **Number (1–22)**, its **Language Name**, or omit it for **Auto-Detect**.

| No. | Indian Language | Whisper Language Code | Native Script | Recommended Model |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Hindi** | `hi` | हिन्दी (Devanagari) | `base` / `small` |
| **2** | **Bengali** | `bn` | বাংলা (Bengali) | `base` / `small` |
| **3** | **Tamil** | `ta` | தமிழ் (Tamil) | `base` / `small` |
| **4** | **Telugu** | `te` | తెలుగు (Telugu) | `base` / `small` |
| **5** | **Marathi** | `mr` | मराठी (Devanagari) | `base` / `small` |
| **6** | **Gujarati** | `gu` | ગુજરાતી (Gujarati) | `base` / `small` |
| **7** | **Kannada** | `kn` | ಕನ್ನಡ (Kannada) | `base` / `small` |
| **8** | **Malayalam** | `ml` | മലയാളം (Malayalam) | `base` / `small` |
| **9** | **Punjabi** | `pa` | ਪੰਜਾਬੀ (Gurmukhi) | `base` / `small` |
| **10** | **Urdu** | `ur` | اردو (Perso-Arabic) | `base` / `small` |
| **11** | **Assamese** | `as` | অসমীয়া (Assamese) | `base` / `small` |
| **12** | **Odia (Oriya)** | `or` | ଓଡ଼ିଆ (Odia) | `base` / `small` |
| **13** | **Nepali** | `ne` | नेपाली (Devanagari) | `base` / `small` |
| **14** | **Sanskrit** | `sa` | संस्कृतम् (Devanagari) | `base` / `small` |
| **15** | **Sindhi** | `sd` | سنڌي (Arabic / Devanagari) | `base` / `small` |
| **16** | **Kashmiri** | `ks` | کٲشُر (Perso-Arabic) | `base` / `small` |
| **17** | **Konkani** | `kok` | कोंकणी (Devanagari / Romi) | `base` / `small` |
| **18** | **Manipuri (Meitei)** | `mni` | মৈতৈলোন্ (Meitei Mayek) | `base` / `small` |
| **19** | **Bodo** | `brx` | बर' (Devanagari) | `base` / `small` |
| **20** | **Dogri** | `doi` | डोगरी (Devanagari) | `base` / `small` |
| **21** | **Santali** | `sat` | ᱥᱟᱱᱛᱟᱲᱤ (Ol Chiki) | `base` / `small` |
| **22** | **Maithili** | `mai` | मैथिली (Devanagari) | `base` / `small` |

> [!TIP]
> **Auto-Detection (`language="auto"` or omitted):** When transcribing an audio clip without specifying a language, Whisper automatically detects both the language and dialect, returning both the language code and confidence score in the response.

---

## 5. Whisper Model Selection Guide

### Available Models & Performance Characteristics

| Model Name | Type | VRAM / RAM | Speed on GPU | Best For |
| :--- | :---: | :---: | :---: | :--- |
| **`base`** *(Installed on Dev PC)* | Multilingual | ~140 MB | **Ultra Fast (~60–90ms)** | **Default recommendation.** Great balance of speed and accuracy for Indian languages and English. |
| **`tiny`** | Multilingual | ~75 MB | Extremely Fast (~30–50ms) | Ultra-lightweight setups with limited compute. |
| **`small`** | Multilingual | ~460 MB | Fast (~120–200ms) | Higher accuracy for regional accents, fast speech, or noisy environments. |
| **`medium`** | Multilingual | ~1.4 GB | Moderate (~350–600ms) | Production grade for complex Indic vocabulary and code-switching (Hinglish/Tanglish). |
| **`large-v3-turbo`** | Multilingual | ~1.5 GB | Fast (~250–400ms) | State-of-the-art multilingual accuracy with optimized inference speed. |
| **`base.en`** | English-Only | ~140 MB | Ultra Fast (~50–80ms) | Use **only** when transcribing pure English audio. |

> [!WARNING]
> **English-Only vs. Multilingual Models:**
> Models ending in `.en` (such as `base.en`) only support English. If an application requests an Indian language (e.g. `language="hi"`) while `model="base.en"`, the server **automatically adapts** and uses the multilingual `base` model to ensure the request succeeds.

> [!NOTE]
> **Zero-Downtime Dynamic Model Caching:**
> You do **not** need to restart the backend to switch models. When a client passes `model="small"`, the backend downloads/loads `small` into GPU memory on the first request and caches it for all subsequent calls.

---

## 6. Standalone CLI & Interactive Tool (`multi_language.py`)

A ready-to-run Python script is available at `backend/multi_language.py` on the dev server for testing models and Indian languages directly from the terminal.

### 1. View Available Indian Languages & Models
```bash
# Print the 22 official Indian languages table
python backend/multi_language.py --list-languages

# Print available Whisper models
python backend/multi_language.py --list-models
```

### 2. Interactive Selection Wizard
Launch an interactive menu to choose an audio file, Whisper model, and Indian language:
```bash
python backend/multi_language.py --interactive
# Or shorthand:
python backend/multi_language.py -i
```

### 3. Transcribe Specifying Language Code
```bash
# Transcribe Hindi with default model ('base')
python backend/multi_language.py sample.wav --language hi

# Transcribe Tamil with shorthand flag
python backend/multi_language.py speech.wav -l ta

# Transcribe Telugu with the 'small' model
python backend/multi_language.py telugu_audio.wav -m small -l te
```

### 4. Transcribe Specifying Language by Number (1–22)
```bash
# Transcribe using number 1 (Hindi)
python backend/multi_language.py speech.wav -l 1

# Transcribe using number 3 (Tamil) with 'medium' model
python backend/multi_language.py speech.wav -m medium -l 3

# Transcribe using number 4 (Telugu)
python backend/multi_language.py speech.wav -l 4
```

### 5. Quick Positional Shorthand
```bash
# Format: python multi_language.py <audio_file> [language] [model]
python backend/multi_language.py sample.wav hi base
python backend/multi_language.py sample.wav bn small
python backend/multi_language.py sample.wav mr large-v3-turbo
```

---

## 7. Ready-to-Use Integration Examples

### Example 1: JavaScript / Web (Browser, React, Vue, Next.js)

```javascript
/**
 * Transcribe an audio Blob or File with language and model selection
 * @param {Blob|File} audioBlob - Audio recording from MediaRecorder or input file
 * @param {string} [language="hi"] - Whisper language code (e.g. 'hi', 'ta', 'te', 'bn') or "auto"
 * @param {string} [model="base"] - Whisper model ('base', 'small', 'medium', 'large-v3-turbo')
 */
async function transcribeAudio(audioBlob, language = "hi", model = "base") {
  const SERVER_URL = "http://61.2.142.91:8024"; // Replace with your dev server IP
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  // Pass language code (e.g. 'hi' for Hindi, 'ta' for Tamil) or 'auto'
  if (language && language !== "auto") {
    formData.append("language", language);
  }
  // Pass model name ('base', 'small', 'medium', etc.)
  if (model) {
    formData.append("model", model);
  }

  const response = await fetch(`${SERVER_URL}/api/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `STT Error: ${response.statusText}`);
  }

  const data = await response.json();
  console.log("Transcript:", data.transcript);
  console.log("Detected Language:", data.language, `(${data.language_probability * 100}% confidence)`);
  console.log("Latency:", data.stt_latency + "s (" + data.speed_factor + "x real-time)");
  return data.transcript;
}
```

---

### Example 2: Python (`requests`)

```python
import requests

SERVER_URL = "http://61.2.142.91:8024"

def transcribe(audio_path, language="hi", model="base"):
    url = f"{SERVER_URL}/api/transcribe"
    
    with open(audio_path, "rb") as f:
        files = {"audio": (audio_path, f, "audio/wav")}
        data = {
            "language": language,  # e.g., 'hi' (Hindi), 'ta' (Tamil), 'te' (Telugu)
            "model": model        # e.g., 'base', 'small', 'medium'
        }
        response = requests.post(url, files=files, data=data)

    response.raise_for_status()
    result = response.json()
    
    print(f"Transcript:   {result['transcript']}")
    print(f"Language:     {result['language']} ({result['language_probability']:.1%})")
    print(f"Latency:      {result['stt_latency']}s | Speed: {result['speed_factor']}x real-time")
    print(f"Model used:   {result['model_name']} on {result['device']}")
    return result["transcript"]

if __name__ == "__main__":
    # Example 1: Transcribe Hindi with base model
    transcribe("sample.wav", language="hi", model="base")

    # Example 2: Transcribe Tamil with small model
    # transcribe("sample.wav", language="ta", model="small")
```

---

### Example 3: Node.js (Backend / Express / Microservice)

```javascript
import fs from 'node:fs';
import FormData from 'form-data';
import fetch from 'node-fetch'; // or global fetch in Node.js 18+

async function transcribeFile(filePath, language = 'hi', model = 'base') {
  const form = new FormData();
  form.append('audio', fs.createReadStream(filePath));
  form.append('language', language); // 'hi', 'ta', 'te', 'bn', etc.
  form.append('model', model);       // 'base', 'small', 'medium'

  const res = await fetch('http://61.2.142.91:8024/api/transcribe', {
    method: 'POST',
    body: form,
    headers: form.getHeaders(),
  });

  if (!res.ok) {
    throw new Error(`Failed to transcribe: ${res.status}`);
  }

  const json = await res.json();
  console.log(`[${json.language}] ${json.transcript}`);
  return json.transcript;
}

// Example usage:
// transcribeFile('speech.wav', 'hi', 'base');
```

---

### Example 4: cURL (Command Line)

```bash
# 1. Transcribe with automatic language detection (default model: base)
curl -X POST "http://61.2.142.91:8024/api/transcribe" \
     -F "audio=@recording.wav"

# 2. Transcribe Hindi (hi) with base model
curl -X POST "http://61.2.142.91:8024/api/transcribe" \
     -F "audio=@hindi_audio.wav" \
     -F "language=hi" \
     -F "model=base"

# 3. Transcribe Tamil (ta) with small model for higher accuracy
curl -X POST "http://61.2.142.91:8024/api/transcribe" \
     -F "audio=@tamil_audio.wav" \
     -F "language=ta" \
     -F "model=small"

# 4. Transcribe Bengali (bn) with large-v3-turbo model
curl -X POST "http://61.2.142.91:8024/api/transcribe" \
     -F "audio=@bengali_audio.wav" \
     -F "language=bn" \
     -F "model=large-v3-turbo"
```

---

### Example 5: Mobile / Flutter (Dart)

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<String> transcribeAudioFile(
  String filePath, {
  String language = 'hi',
  String model = 'base'
}) async {
  var uri = Uri.parse('http://<YOUR-SERVER-IP>:8024/api/transcribe');
  var request = http.MultipartRequest('POST', uri);

  request.files.add(await http.MultipartFile.fromPath('audio', filePath));
  request.fields['language'] = language; // 'hi', 'ta', 'te', etc.
  request.fields['model'] = model;       // 'base', 'small', etc.

  var streamedResponse = await request.send();
  var response = await http.Response.fromStream(streamedResponse);

  if (response.statusCode == 200) {
    var data = jsonDecode(response.body);
    return data['transcript'];
  } else {
    throw Exception('Failed to transcribe audio: ${response.statusCode}');
  }
}
```

---

## 8. Optional: Standalone Groq LLM API (`POST /api/chat`)

If your external app also needs AI conversational replies (e.g. an AI customer support bot or voice assistant), use the separate `/api/chat` endpoint:

### Request
* **URL**: `http://<server-ip>:8024/api/chat`
* **Method**: `POST`
* **Content-Type**: `application/json`

```json
{
  "message": "भारत की राजधानी क्या है?",
  "groq_api_key": "gsk_...",
  "groq_model": "openai/gpt-oss-20b"
}
```
*(Both `groq_api_key` and `groq_model` are optional if already configured in the server's `.env`).*

### Response
```json
{
  "reply": "भारत की राजधानी नई दिल्ली है।",
  "groq_latency": 0.285,
  "model": "openai/gpt-oss-20b"
}
```

---

## 9. Performance & Optimization Tips

1. **Audio Format**:
   * **Web/Mobile**: `audio/webm;codecs=opus` recorded directly from the browser `MediaRecorder` works natively and compresses efficiently over the network.
   * **Local Audio Files**: 16 kHz 16-bit Mono WAV produces the fastest decoding time with zero transcoding overhead.
2. **GPU Acceleration (CUDA FP16)**:
   * On NVIDIA GPU (`WHISPER_DEVICE=cuda`, `float16`), a 3-second audio clip transcribes in **~60–90ms** on the `base` model.
   * If CUDA is temporarily unavailable on a test environment, the server automatically falls back to CPU `int8` without crashing.
3. **Accuracy vs. Latency Tradeoff**:
   * For **Interactive Voice Calling** (<150ms roundtrip required): Use **`model="base"`**.
   * For **Document Dictation / Noisy Audio**: Use **`model="small"`** or **`model="large-v3-turbo"`**.
4. **CORS Configuration**:
   * CORS in `config.py` is configured with `*` so any web app or frontend can call `/api/transcribe` directly from the browser.

---

## 10. Standalone Text-to-Speech (TTS) API (`POST /api/tts`)

Synthesizes arbitrary text into natural human speech with ultra-low VRAM usage.

### Request Details
* **URL**: `http://<server-ip>:8024/api/tts`
* **Method**: `POST`
* **Content-Type**: `application/json`

```json
{
  "text": "Hello! Thanks for calling our automated assistant. How can I help you today?",
  "voice": "af_heart",
  "speed": 1.0,
  "engine": "kokoro"
}
```

### Success Response (`200 OK`)
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "format": "wav",
  "media_type": "audio/wav",
  "tts_latency": 0.185,
  "duration": 3.45,
  "engine": "kokoro-82m",
  "voice": "af_heart"
}
```

### Direct Audio Streaming Endpoint
* **URL**: `GET http://<server-ip>:8024/api/tts/stream?text=Hello+world&voice=af_heart`
* Directly streams `.wav` or `.mp3` bytes suitable for standard HTML5 `<audio>` tags or SIP telephony media gateways.

---

## 11. Full Voice Calling API (`POST /api/call/turn`)

Single-turn composite endpoint simulating a live phone call:
1. **Faster-Whisper STT**: Transcribes caller's microphone audio.
2. **Groq Conversational LLM**: Generates concise, colloquial phone reply using multi-turn conversation memory.
3. **Low-VRAM TTS**: Converts the AI response to spoken audio.

### Request Details
* **URL**: `http://<server-ip>:8024/api/call/turn`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description |
| :--- | :---: | :---: | :--- |
| `audio` | File | **Yes** | Recorded audio chunk from user microphone. |
| `history` | String | No | JSON array of past conversation messages: `[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]`. |
| `language` | String | No | Whisper STT language code (`en`, `hi`, `auto`, etc.). |
| `whisper_model` | String | No | Whisper model (`base`, `small`). |
| `tts_voice` | String | No | Voice ID (`af_heart`, `am_adam`, `en-US-JennyNeural`). |

### Success Response (`200 OK`)
```json
{
  "transcript": "What are your support hours?",
  "reply": "We are open Monday through Friday from 9 AM to 6 PM Eastern Time.",
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "audio_format": "wav",
  "latency": {
    "audio_duration": 2.1,
    "stt_latency": 0.082,
    "groq_latency": 0.194,
    "tts_latency": 0.145,
    "total_server_time": 0.421
  },
  "device": "cuda",
  "whisper_model": "base",
  "tts_engine": "kokoro-82m",
  "tts_voice": "af_heart"
}
```
*(Total server time is under 450ms — broadcast standard for natural conversational phone calls).*
