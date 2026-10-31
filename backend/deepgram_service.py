"""Session-isolated Deepgram Nova streaming client.

The browser never receives the Deepgram credential: PCM16 frames are relayed by
the FastAPI process over this server-to-server WebSocket.
"""
import asyncio
import json
import logging
import time
from urllib.parse import urlencode

import websockets

import config

logger = logging.getLogger("voice_api.deepgram")


class DeepgramStream:
    """One persistent Deepgram connection and result queue for one call."""

    def __init__(self, session_id: str, sample_rate: int = 16000):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self._socket = None
        self._receiver = None
        self._send_lock = asyncio.Lock()
        self._results: asyncio.Queue[dict] = asyncio.Queue()
        self._closed = False
        self._reconnects = 0
        self.connected_at = None

    @property
    def configured(self) -> bool:
        return bool(config.DEEPGRAM_API_KEY)

    async def connect(self) -> None:
        if self._closed:
            raise RuntimeError("Deepgram stream is closed")
        if not self.configured:
            raise RuntimeError("Deepgram is selected but DEEPGRAM_API_KEY is not configured")
        if self._socket and not self._socket.closed:
            return
        query = urlencode({
            "model": config.DEEPGRAM_MODEL,
            "language": config.DEEPGRAM_LANGUAGE,
            "encoding": "linear16",
            "sample_rate": self.sample_rate,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "endpointing": "300",
            "utterance_end_ms": "1000",
            "vad_events": "true",
        })
        url = f"wss://api.deepgram.com/v1/listen?{query}"
        try:
            self._socket = await websockets.connect(
                url, extra_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
                open_timeout=config.DEEPGRAM_CONNECT_TIMEOUT_SECONDS,
                ping_interval=20, ping_timeout=20, max_size=2**20,
            )
        except TypeError:  # websockets >= 14 renamed extra_headers
            self._socket = await websockets.connect(
                url, additional_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"},
                open_timeout=config.DEEPGRAM_CONNECT_TIMEOUT_SECONDS,
                ping_interval=20, ping_timeout=20, max_size=2**20,
            )
        self.connected_at = time.perf_counter()
        self._receiver = asyncio.create_task(self._receive(), name=f"deepgram-{self.session_id}")
        logger.info("[%s] Deepgram streaming connected", self.session_id)

    async def _receive(self) -> None:
        try:
            async for raw in self._socket:
                if not isinstance(raw, str):
                    continue
                payload = json.loads(raw)
                if payload.get("type") != "Results":
                    continue
                alternatives = payload.get("channel", {}).get("alternatives", [])
                text = alternatives[0].get("transcript", "").strip() if alternatives else ""
                if text:
                    await self._results.put({"text": text, "is_final": bool(payload.get("is_final"))})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                logger.warning("[%s] Deepgram receive connection ended: %s", self.session_id, type(exc).__name__)

    async def send_audio(self, audio: bytes) -> None:
        if not audio or self._closed:
            return
        async with self._send_lock:
            try:
                await self.connect()
                await self._socket.send(audio)
            except Exception as exc:
                # A single reconnect is deliberately bounded and never resends a frame.
                if self._reconnects >= config.DEEPGRAM_MAX_RECONNECTS:
                    raise RuntimeError("Deepgram streaming connection unavailable") from exc
                self._reconnects += 1
                await self._close_socket()
                await self.connect()
                # Do not replay this frame: replay risks duplicate paid transcription.

    def drain_results(self) -> list[dict]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except asyncio.QueueEmpty:
                return results

    async def _close_socket(self) -> None:
        if self._receiver:
            self._receiver.cancel()
            try:
                await self._receiver
            except asyncio.CancelledError:
                pass
            self._receiver = None
        if self._socket:
            try:
                await self._socket.close()
            except Exception:
                pass
            self._socket = None

    async def close(self) -> None:
        self._closed = True
        await self._close_socket()

