from typing import Callable, Dict, Any, List
import httpx
from src.components.base import BaseComponent
from src.utils.config import load_config
from src.utils.metrics import MetricSpan


class SpeakerComponent(BaseComponent):
    """
    Provides the agent with the ability to speak text out loud over physical room speakers
    by calling the lertspeak ApiSpeak HTTP endpoint (/api/v1/say).
    """

    def __init__(self, api_url: str = None, voice_id: str = None, timeout_sec: float = 15.0):
        super().__init__()
        self.config = load_config()
        speaker_cfg = self.config.get("speaker", {})
        self.api_url = (api_url if api_url is not None else speaker_cfg.get("api_url", "")).strip()
        self.voice_id = (voice_id if voice_id is not None else speaker_cfg.get("voice_id", "")).strip()
        self.timeout_sec = float(timeout_sec)

        if not self.api_url:
            print("[Speaker] Warning: No api_url found under [speaker] in config.toml. Speaker tool will be disabled.")

    def get_system_prompt_addition(self) -> str:
        if not self.api_url:
            return ""
        return (
            "--- PHYSICAL ROOM SPEAKER ---\n"
            "You have access to a physical speaker system via the `speak` tool, which says text out loud "
            "on speakers in rooms where humans might be. Use it thoughtfully and concisely when you want "
            "to say something out loud in the physical room."
        )

    def get_tools(self) -> List[Callable]:
        if not self.api_url:
            return []
        return [self.speak]

    async def speak(self, message: str) -> str:
        """
        Says the provided text out loud on speakers in rooms where humans might be.
        """
        clean_msg = (message or "").strip()
        if not clean_msg:
            return "Error: message cannot be empty."
        if not self.api_url:
            return "Error: Speaker api_url is not configured."

        agent_obj = getattr(self, "agent", None)
        agent_name = getattr(agent_obj, "name", None) or self.config.get("agent", {}).get("name", "Thunk")
        thread_id = getattr(agent_obj, "thread_id", None) or "main"

        payload: Dict[str, Any] = {"msg": clean_msg}
        if self.voice_id:
            payload["voice_id"] = self.voice_id

        print(f"[Speaker] Speaking out loud ({self.voice_id or 'default'}): '{clean_msg}'")

        with MetricSpan(
            operation="speaker_call",
            agent=agent_name,
            thread=thread_id,
            component="speaker",
        ) as span:
            span.inc("request_bytes", len(clean_msg.encode("utf-8")))
            try:
                async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                    response = await client.post(self.api_url, json=payload)

                span.inc("response_bytes", len(response.content or b""))
                if response.status_code != 200:
                    span.fail()
                    err_msg = (
                        f"Error: Speaker endpoint returned HTTP {response.status_code}: "
                        f"{(response.text or '').strip()}"
                    )
                    print(f"[Speaker] {err_msg}")
                    return err_msg

                return f"Spoke out loud on room speakers: \"{clean_msg}\""
            except Exception as e:
                span.fail()
                err_msg = f"Failed to speak via room speakers: {str(e)}"
                print(f"[Speaker] {err_msg}")
                return err_msg
