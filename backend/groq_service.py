import time
import re
import logging
from typing import Optional, List, Dict, Any
from groq import Groq, AsyncGroq
import config

logger = logging.getLogger("groq_service")

# Default voice assistant prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a fast, intelligent voice assistant. "
    "The user is speaking to you via microphone. "
    "Give concise, clear, and direct conversational responses (under 3 sentences) "
    "unless a longer answer is explicitly requested."
)

# Optimized telephony system prompt for real-time phone calls
TELEPHONY_SYSTEM_PROMPT = (
    "You are an AI assistant speaking live on a telephone call. "
    "Keep every response strictly concise (under 25 words, 1 direct sentence) unless the caller explicitly asks for detail. "
    "Be natural, direct, warm, and conversational. "
    "Always remember and respect details the caller has shared, including their name. "
    "NEVER use markdown formatting, asterisks (*), bullet points, hashes (#), or URLs, "
    "because your reply will be converted directly into spoken audio on the caller's speaker."
)

def extract_caller_info(history: Optional[List[Dict[str, str]]]) -> Dict[str, str]:
    """
    Extract key caller profile facts (such as caller name) across the entire conversation history.
    This ensures that even across long calls, crucial details like the caller's name are never forgotten.
    """
    info = {}
    if not history:
        return info

    user_name_patterns = [
        re.compile(r"\b(?:my name is|call me|this is)\s+([A-Za-z]+)", re.IGNORECASE),
        re.compile(r"\b(?:i am|i'm)\s+([A-Za-z]+)", re.IGNORECASE),
    ]

    assistant_confirm_patterns = [
        re.compile(r"\b(?:nice to meet you|pleasure to meet you|good to meet you)\s*,\s*([A-Za-z]+)", re.IGNORECASE),
        re.compile(r"\byour name is\s+([A-Za-z]+)", re.IGNORECASE),
        re.compile(r"\byou told me\s*,\s*([A-Za-z]+)", re.IGNORECASE),
    ]

    ignore_words = {
        "a", "an", "the", "ready", "here", "fine", "good", "great", "okay", "ok",
        "sorry", "just", "happy", "speaking", "calling", "interested", "trying",
        "looking", "going", "doing", "chatgpt", "ai", "assistant", "not", "sure",
        "new", "back", "all", "done", "listening", "online", "busy", "well", "turning"
    }

    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue

        if role == "user":
            for pat in user_name_patterns:
                match = pat.search(content)
                if match:
                    candidate = match.group(1).strip()
                    if candidate.lower() not in ignore_words and len(candidate) >= 2:
                        info["name"] = candidate.capitalize()
        elif role == "assistant":
            for pat in assistant_confirm_patterns:
                match = pat.search(content)
                if match:
                    candidate = match.group(1).strip()
                    if candidate.lower() not in ignore_words and len(candidate) >= 2:
                        info["name"] = candidate.capitalize()

    return info

def clean_speech_text(text: str) -> str:
    """Strip markdown formatting that interferes with realistic TTS pronunciation."""
    # Remove markdown bold/italics (* and _)
    cleaned = re.sub(r'[*_~`#]', '', text)
    # Remove links [text](url) -> text
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
    # Normalize extra whitespaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class GroqEngine:
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        self.model = config.GROQ_MODEL
        self._init_client()

    def _init_client(self, api_key: str = None):
        key = api_key or self.api_key
        if key:
            self.client = Groq(api_key=key)
            self.async_client = AsyncGroq(api_key=key)
        else:
            self.client = None
            self.async_client = None

    def generate_reply(
        self,
        user_text: str,
        api_key: str = None,
        model_name: str = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        is_phone_call: bool = False
    ) -> dict:
        active_key = api_key or self.api_key
        active_model = model_name or self.model

        if not active_key:
            return {
                "reply": "[Groq API key not provided. Set GROQ_API_KEY in .env or frontend settings]",
                "groq_latency": 0.0,
                "model": active_model
            }

        client = Groq(api_key=active_key) if api_key else self.client
        if client is None:
            client = Groq(api_key=active_key)

        # Base system prompt
        base_prompt = system_prompt or (TELEPHONY_SYSTEM_PROMPT if is_phone_call else DEFAULT_SYSTEM_PROMPT)

        # Extract persistent caller profile facts (e.g. caller name) across ALL history turns
        caller_info = extract_caller_info(history)
        if caller_info.get("name"):
            caller_name = caller_info["name"]
            active_prompt = (
                f"{base_prompt}\n\n"
                f"[Caller Context]\n"
                f"The caller's name is {caller_name}. Always remember this and address them by name when appropriate."
            )
        else:
            active_prompt = base_prompt

        messages = [{"role": "system", "content": active_prompt}]

        # Append previous conversation history if provided (multi-turn phone call context)
        if history:
            max_history = getattr(config, "GROQ_MAX_HISTORY_MESSAGES", 50)
            clean_history = [
                {"role": msg.get("role"), "content": msg.get("content", "")}
                for msg in history
                if msg.get("role") in ("user", "assistant") and msg.get("content")
            ]

            if len(clean_history) <= max_history:
                selected_history = clean_history
            else:
                # Retain initial conversation anchor (first 4 messages: greetings & introductions)
                # plus the most recent turns for fluid dialogue continuity
                anchor_count = min(4, len(clean_history))
                recent_count = max_history - anchor_count
                selected_history = clean_history[:anchor_count] + clean_history[-recent_count:]

            for msg in selected_history:
                messages.append(msg)

        # Append current user prompt
        messages.append({"role": "user", "content": user_text})

        start_time = time.perf_counter()
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=active_model,
                temperature=0.4 if is_phone_call else 0.7,
                max_tokens=256
            )
            elapsed = time.perf_counter() - start_time
            raw_reply = chat_completion.choices[0].message.content.strip()
            reply = clean_speech_text(raw_reply) if is_phone_call else raw_reply

            return {
                "reply": reply,
                "groq_latency": round(elapsed, 3),
                "model": active_model
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Groq API error: {e}")
            return {
                "reply": f"[Groq Error: {str(e)}]",
                "groq_latency": round(elapsed, 3),
                "model": active_model
            }

    def stream_reply(
        self,
        user_text: str,
        api_key: str = None,
        model_name: str = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        is_phone_call: bool = False
    ):
        """
        Stream tokens from Groq LLM incrementally.
        Yields dict with {"delta": str, "time_to_first_token": float, "is_final": bool, "full_text": str}.
        """
        active_key = api_key or self.api_key
        active_model = model_name or self.model

        if not active_key:
            yield {
                "delta": "Groq API key not provided.",
                "time_to_first_token": 0.0,
                "is_final": True,
                "full_text": "Groq API key not provided."
            }
            return

        client = Groq(api_key=active_key) if api_key else self.client
        if client is None:
            client = Groq(api_key=active_key)

        base_prompt = system_prompt or (TELEPHONY_SYSTEM_PROMPT if is_phone_call else DEFAULT_SYSTEM_PROMPT)
        caller_info = extract_caller_info(history)
        if caller_info.get("name"):
            active_prompt = (
                f"{base_prompt}\n\n"
                f"[Caller Context]\n"
                f"The caller's name is {caller_info['name']}. Always address them by name when appropriate."
            )
        else:
            active_prompt = base_prompt

        messages = [{"role": "system", "content": active_prompt}]

        if history:
            max_history = getattr(config, "GROQ_MAX_HISTORY_MESSAGES", 50)
            clean_history = [
                {"role": msg.get("role"), "content": msg.get("content", "")}
                for msg in history
                if msg.get("role") in ("user", "assistant") and msg.get("content")
            ]
            if len(clean_history) <= max_history:
                selected_history = clean_history
            else:
                anchor_count = min(4, len(clean_history))
                recent_count = max_history - anchor_count
                selected_history = clean_history[:anchor_count] + clean_history[-recent_count:]

            for msg in selected_history:
                messages.append(msg)

        messages.append({"role": "user", "content": user_text})

        start_time = time.perf_counter()
        first_token_time = None
        accumulated_text = ""

        try:
            stream = client.chat.completions.create(
                messages=messages,
                model=active_model,
                temperature=0.4 if is_phone_call else 0.7,
                max_tokens=256,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
                if delta:
                    if first_token_time is None:
                        first_token_time = round(time.perf_counter() - start_time, 3)
                    accumulated_text += delta
                    yield {
                        "delta": delta,
                        "time_to_first_token": first_token_time,
                        "is_final": False,
                        "full_text": accumulated_text
                    }

            final_text = clean_speech_text(accumulated_text) if is_phone_call else accumulated_text
            yield {
                "delta": "",
                "time_to_first_token": first_token_time or round(time.perf_counter() - start_time, 3),
                "is_final": True,
                "full_text": final_text
            }
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            yield {
                "delta": f" [Error: {str(e)}]",
                "time_to_first_token": 0.0,
                "is_final": True,
                "full_text": f"Error: {str(e)}"
            }

    async def async_stream_reply(
        self,
        user_text: str,
        api_key: str = None,
        model_name: str = None,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        is_phone_call: bool = False
    ):
        """
        Native asynchronous token streaming using AsyncGroq.
        Eliminates threadpool context-switching and produces sub-200ms first token latency.
        """
        active_key = api_key or self.api_key
        active_model = model_name or self.model

        if not active_key:
            yield {
                "delta": "Groq API key not provided.",
                "time_to_first_token": 0.0,
                "is_final": True,
                "full_text": "Groq API key not provided."
            }
            return

        client = AsyncGroq(api_key=active_key) if api_key else self.async_client
        if client is None:
            client = AsyncGroq(api_key=active_key)

        base_prompt = system_prompt or (TELEPHONY_SYSTEM_PROMPT if is_phone_call else DEFAULT_SYSTEM_PROMPT)
        caller_info = extract_caller_info(history)
        if caller_info.get("name"):
            active_prompt = (
                f"{base_prompt}\n\n"
                f"[Caller Context]\n"
                f"The caller's name is {caller_info['name']}. Always address them by name when appropriate."
            )
        else:
            active_prompt = base_prompt

        messages = [{"role": "system", "content": active_prompt}]

        if history:
            max_history = getattr(config, "GROQ_MAX_HISTORY_MESSAGES", 50)
            clean_history = [
                {"role": msg.get("role"), "content": msg.get("content", "")}
                for msg in history
                if msg.get("role") in ("user", "assistant") and msg.get("content")
            ]
            if len(clean_history) <= max_history:
                selected_history = clean_history
            else:
                anchor_count = min(4, len(clean_history))
                recent_count = max_history - anchor_count
                selected_history = clean_history[:anchor_count] + clean_history[-recent_count:]

            for msg in selected_history:
                messages.append(msg)

        messages.append({"role": "user", "content": user_text})

        start_time = time.perf_counter()
        first_token_time = None
        accumulated_text = ""

        try:
            stream = await client.chat.completions.create(
                messages=messages,
                model=active_model,
                temperature=0.3 if is_phone_call else 0.7,
                max_tokens=256,
                stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
                if delta:
                    if first_token_time is None:
                        first_token_time = round(time.perf_counter() - start_time, 3)
                    accumulated_text += delta
                    yield {
                        "delta": delta,
                        "time_to_first_token": first_token_time,
                        "is_final": False,
                        "full_text": accumulated_text
                    }

            final_text = clean_speech_text(accumulated_text) if is_phone_call else accumulated_text
            yield {
                "delta": "",
                "time_to_first_token": first_token_time or round(time.perf_counter() - start_time, 3),
                "is_final": True,
                "full_text": final_text
            }
        except Exception as e:
            logger.error(f"AsyncGroq streaming error: {e}")
            yield {
                "delta": f" [Error: {str(e)}]",
                "time_to_first_token": 0.0,
                "is_final": True,
                "full_text": f"Error: {str(e)}"
            }


groq_engine = None

def get_groq_engine() -> GroqEngine:
    global groq_engine
    if groq_engine is None:
        groq_engine = GroqEngine()
    return groq_engine
