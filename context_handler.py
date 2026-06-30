
from typing import Any
from pathlib import Path
import datetime
import json

from base_prompt import BASE_PROMPT
from client import CLIENT, MODEL_NAME

MAX_CONTEXT_TOKENS = 32768 
TRIGGER_SUMMARY_TOKENS = int(MAX_CONTEXT_TOKENS * 0.8)  

class ContextHandler:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": BASE_PROMPT}
        ]
        self.consumed_tokens = 0
        self.current_messages_tokens = 0

    def estimate_tokens(self, message: dict) -> int:
        """
        Estimates tokens for a message payload. Falls back to a reliable 
        character-to-token ratio if a dedicated tokenizer isn't imported,
        ensuring standard tool payloads are cleanly parsed.
        """
        content = message.get("content") or ""
        if isinstance(content, list):  # Handle vision/structured tool arrays
            content = str(content)
            
        # Account for tool call definitions structural overhead
        tool_overhead = 0
        if message.get("tool_calls"):
            tool_overhead += len(str(message["tool_calls"])) // 4
            
        return (len(content) // 4) + 10 + tool_overhead

    def get_total_tokens(self) -> int:
        return sum(self.estimate_tokens(msg) for msg in self.messages)
    
    def get_consumed_tokens(self) -> int:
        return self.consumed_tokens 

    def append_messages(self, message: Any):
        """Standardizes input messages into clean dictionaries before storage."""
        if hasattr(message, "model_dump"):
            # Converts OpenAI object structures (and tool_calls) into raw dicts
            msg_dict = message.model_dump(exclude_none=True)
        elif isinstance(message, dict):
            msg_dict = message.copy()
        else:
            msg_dict = {"role": "assistant", "content": str(message)}
            
        self.messages.append(msg_dict)

        tokens = self.estimate_tokens(msg_dict)
        self.current_messages_tokens += tokens
        self.consumed_tokens += tokens
        
        # Check if we need to compress history
        if self.current_messages_tokens > TRIGGER_SUMMARY_TOKENS:
            self.distill_older_context()
            self.current_messages_tokens = self.get_total_tokens()

    def clear(self):
        self.messages.clear()
        self.messages.append({"role": "system", "content": BASE_PROMPT})
        print("[SYSTEM] Context cleared")

    def distill_older_context(self):
        print("\n[SYSTEM] Context threshold crossed. Compressing historical log...")

        preserve_count = 5
        if len(self.messages) <= preserve_count + 2:
            return

        # Walk forward from the raw tail until we hit a clean turn boundary —
        # avoids leaving an orphaned tool-result message with no preceding tool_calls.
        raw_retained = self.messages[-preserve_count:]
        retained_slice = raw_retained  # fallback
        for i, msg in enumerate(raw_retained):
            role = msg.get("role")
            if role == "user" or (role == "assistant" and not msg.get("tool_calls")):
                retained_slice = raw_retained[i:]
                break

        cut = len(self.messages) - len(retained_slice)

        # Detect a pinned summary block from a prior compression pass
        has_prior_summary = (
            len(self.messages) > 1
            and self.messages[1].get("role") == "system"
            and "[HISTORICAL ARCHIVE SUMMARY]" in (self.messages[1].get("content") or "")
        )

        if has_prior_summary:
            prior_summary = self.messages[1].get("content", "")
            compress_slice = self.messages[2:cut]
            summary_instruction = (
                "You are an internal system routine. A prior summary exists. "
                "Integrate the new dialogue into it, producing one updated ultra-dense summary. "
                "Preserve all facts from the prior summary unless directly contradicted. "
                "Stay under 300 words.\n"
                "YOU are the one who has to read it in the end."
            )
            compress_input = f"[PRIOR SUMMARY]\n{prior_summary}\n\n[NEW DIALOGUE]\n" + json.dumps(compress_slice)
        else:
            compress_slice = self.messages[1:cut]
            summary_instruction = (
                "You are an internal system routine. Summarize the following dialogue sequence. "
                "Extract critical user data, constants, completed goals, and persistent settings. "
                "Be ultra-dense and clear; omit pleasantries. Stay under 300 words.\n"
                "YOU are the one who has to read it in the end."
            )
            compress_input = json.dumps(compress_slice)

        if not compress_slice:
            return

        try:
            summary_response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": summary_instruction},
                    {"role": "user", "content": compress_input}
                ],
                temperature=0.1
            )
            summary_text = summary_response.choices[0].message.content
            summary_message = {
                "role": "system",
                "content": f"[HISTORICAL ARCHIVE SUMMARY]:\n{summary_text}"
            }
            self.messages = [self.messages[0], summary_message] + retained_slice
            print(f"[SYSTEM] Compression complete. New context size: {self.get_total_tokens()} tokens.\n")
        except Exception as e:
            print(f"[SYSTEM WARNING] Historical distillation routine failed: {e}")

    def get_messages_for_inference(self, ephemeral_context: str = "") -> list:
        if not ephemeral_context:
            return self.messages
        msgs = [m.copy() for m in self.messages]
        for i in reversed(range(len(msgs))):
            if msgs[i].get("role") == "user":
                msgs[i] = {**msgs[i], "content": msgs[i]["content"] + ephemeral_context}
                break
        return msgs

    def dump_chat_log(self):
        now = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        log_dir = Path("Logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # This will now never crash because all items are explicitly sanitized into dicts
        with open(log_dir / f"{now}_log.txt", 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.messages, indent=2, ensure_ascii=False))
        print(f"[SYSTEM] Log written to {log_dir / 'chatlog.txt'}")
