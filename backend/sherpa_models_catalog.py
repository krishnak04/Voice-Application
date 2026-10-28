"""
Official English Sherpa-ONNX ASR Model Registry and Catalog.
Verified from official documentation: https://k2-fsa.github.io/sherpa/onnx/pretrained_models/

Contains metadata, architecture configurations, file patterns, and download definitions.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import re

OFFICIAL_SHERPA_ENGLISH_MODELS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # 1. STREAMING ZIPFORMER MODELS (Casual chunk-based, OnlineRecognizer)
    # -------------------------------------------------------------
    "sherpa-onnx-streaming-zipformer-en-2023-06-26": {
        "id": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "name": "Sherpa-ONNX Zipformer Streaming Standard (Baseline)",
        "language": "English",
        "lang_code": "en",
        "type": "streaming",  # 'streaming' | 'offline'
        "category": "Standard",  # 'Small' | 'Standard' | 'Large'
        "params": "~65.6M",
        "dataset": "LibriSpeech + GigaSpeech",
        "streaming_supported": True,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
        "folder_name": "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        "is_default": True,
        "license": "Apache-2.0",
        "description": "Official standard streaming Zipformer transducer trained on LibriSpeech & GigaSpeech. Delivers sub-100ms first partial latency for live conversational voice calling."
    },
    "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17": {
        "id": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "name": "Sherpa-ONNX Zipformer Streaming Small (Lightweight)",
        "language": "English",
        "lang_code": "en",
        "type": "streaming",
        "category": "Small",
        "params": "~20M",
        "dataset": "LibriSpeech (960 hours)",
        "streaming_supported": True,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2",
        "folder_name": "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
        "is_default": False,
        "license": "Apache-2.0",
        "description": "Lightweight 20M-parameter streaming Zipformer model. Optimized for ultra-low memory footprint (~42MB INT8 encoder) and ultra-fast real-time inference on edge devices."
    },
    "sherpa-onnx-streaming-zipformer-en-2023-06-21": {
        "id": "sherpa-onnx-streaming-zipformer-en-2023-06-21",
        "name": "Sherpa-ONNX Zipformer Streaming LibriSpeech Checkpoint",
        "language": "English",
        "lang_code": "en",
        "type": "streaming",
        "category": "Standard",
        "params": "~65.6M",
        "dataset": "LibriSpeech",
        "streaming_supported": True,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
        "folder_name": "sherpa-onnx-streaming-zipformer-en-2023-06-21",
        "is_default": False,
        "license": "Apache-2.0",
        "description": "Standard streaming checkpoint trained on LibriSpeech without GigaSpeech fine-tuning. For comparative streaming analysis."
    },
    "sherpa-onnx-streaming-zipformer-en-2023-02-21": {
        "id": "sherpa-onnx-streaming-zipformer-en-2023-02-21",
        "name": "Sherpa-ONNX Zipformer Streaming LibriSpeech 2023-02-21",
        "language": "English",
        "lang_code": "en",
        "type": "streaming",
        "category": "Standard",
        "params": "~65.6M",
        "dataset": "LibriSpeech",
        "streaming_supported": True,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-02-21.tar.bz2",
        "folder_name": "sherpa-onnx-streaming-zipformer-en-2023-02-21",
        "is_default": False,
        "license": "Apache-2.0",
        "description": "Official streaming Zipformer English transducer release from 2023-02-21 for causal chunk recognition."
    },

    # -------------------------------------------------------------
    # 2. OFFLINE ZIPFORMER MODELS (Full-waveform attention, OfflineRecognizer)
    # -------------------------------------------------------------
    "sherpa-onnx-zipformer-large-en-2023-06-26": {
        "id": "sherpa-onnx-zipformer-large-en-2023-06-26",
        "name": "Sherpa-ONNX Zipformer Offline Large (High Accuracy)",
        "language": "English",
        "lang_code": "en",
        "type": "offline",
        "category": "Large",
        "params": "~148M",
        "dataset": "LibriSpeech",
        "streaming_supported": False,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-zipformer-large-en-2023-06-26.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-large-en-2023-06-26.tar.bz2",
        "folder_name": "sherpa-onnx-zipformer-large-en-2023-06-26",
        "is_default": False,
        "license": "Apache-2.0",
        "description": "High-capacity 148M-parameter offline Zipformer transducer with non-causal bidirectional attention. Maximum transcription accuracy for batch & audio file processing."
    },
    "sherpa-onnx-zipformer-small-en-2023-06-26": {
        "id": "sherpa-onnx-zipformer-small-en-2023-06-26",
        "name": "Sherpa-ONNX Zipformer Offline Small",
        "language": "English",
        "lang_code": "en",
        "type": "offline",
        "category": "Small",
        "params": "~30M",
        "dataset": "LibriSpeech",
        "streaming_supported": False,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2",
        "folder_name": "sherpa-onnx-zipformer-small-en-2023-06-26",
        "is_default": False,
        "license": "Apache-2.0",
        "description": "Compact 30M-parameter offline transducer for low-latency non-streaming batch audio transcription with minimal memory requirements."
    },
    "sherpa-onnx-zipformer-gigaspeech-2023-12-12": {
        "id": "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        "name": "Sherpa-ONNX Zipformer Offline GigaSpeech",
        "language": "English",
        "lang_code": "en",
        "type": "offline",
        "category": "Standard",
        "params": "~65.6M",
        "dataset": "GigaSpeech (10,000 hours)",
        "streaming_supported": False,
        "offline_supported": True,
        "tar_filename": "sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2",
        "download_url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2",
        "folder_name": "sherpa-onnx-zipformer-gigaspeech-2023-12-12",
        "is_default": False,
        "license": "Apache-2.0",
        "description": "Robust offline Zipformer trained on 10,000 hours of conversational multi-speaker GigaSpeech podcasts, audiobooks, and YouTube recordings."
    }
}


def find_model_files(model_dir: Path, use_int8: bool = True) -> Dict[str, Optional[Path]]:
    """
    Robustly locate tokens, encoder, decoder, and joiner inside a model directory,
    handling variable naming schemes across Sherpa-ONNX releases.
    """
    if not model_dir.is_dir():
        return {"tokens": None, "encoder": None, "decoder": None, "joiner": None, "is_valid": False}

    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        tokens = model_dir / "bpe.model"
    if not tokens.exists():
        # Check subdirectories
        for f in model_dir.glob("**/tokens.txt"):
            tokens = f
            break

    # Discover encoder, decoder, joiner
    all_onnx = list(model_dir.glob("*.onnx")) + list(model_dir.glob("*/*.onnx"))
    
    encoders = [f for f in all_onnx if "encoder" in f.name.lower()]
    decoders = [f for f in all_onnx if "decoder" in f.name.lower()]
    joiners = [f for f in all_onnx if "joiner" in f.name.lower()]

    def select_best(files: List[Path], prefer_int8: bool) -> Optional[Path]:
        if not files:
            return None
        if prefer_int8:
            int8_files = [f for f in files if ".int8." in f.name.lower() or f.name.lower().endswith(".int8.onnx")]
            if int8_files:
                return sorted(int8_files, key=lambda x: len(x.name))[0]
        # Fallback to standard onnx
        std_files = [f for f in files if ".int8." not in f.name.lower()]
        if std_files:
            return sorted(std_files, key=lambda x: len(x.name))[0]
        return files[0]

    encoder = select_best(encoders, use_int8)
    decoder = select_best(decoders, use_int8)
    joiner = select_best(joiners, use_int8)

    is_valid = bool(tokens and tokens.exists() and encoder and encoder.exists() and decoder and decoder.exists() and joiner and joiner.exists())

    return {
        "tokens": tokens if tokens and tokens.exists() else None,
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "is_valid": is_valid
    }
