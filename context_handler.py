
from typing import Any

from base_prompt import BASE_PROMPT


class ContextHandler:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": BASE_PROMPT}
        ]

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
        
        # Check if we need to compress history
        if self.get_total_tokens() > TRIGGER_SUMMARY_TOKENS:
            self.distill_older_context()

    def clear(self):
        self.messages.clear()
        self.messages.append({"role": "system", "content": BASE_PROMPT})
        print("[SYSTEM] Context cleared")

    def distill_older_context(self):
        """
        Compresses the oldest 40% of conversational history into an 
        orderly 'System Summary' directive block to preserve context space.
        """
        print("\n[SYSTEM] Context threshold crossed. Compressing historical log...")
        
        # Keep system prompt [0] and preserve the most recent 5 messages intact
        preserve_count = 5
        if len(self.messages) <= preserve_count + 2:
            return
            
        compress_slice = self.messages[1:-preserve_count]
        retained_slice = self.messages[-preserve_count:]
        
        summary_instruction = (
            "You are an internal system routine. Summarize the following dialogue sequence. "
            "Extract critical user data, constants, completed goals, and persistent settings. "
            "Be ultra-dense, clear, and omit conversational pleasantries."
            "YOU are the one who has to read it in the end."
        )
        
        try:
            # Execute summary pass using your active engine
            summary_response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": summary_instruction},
                    {"role": "user", "content": json.dumps(compress_slice)}
                ],
                temperature=0.1 # Low variance for exact details extraction
            )
            
            summary_text = summary_response.choices[0].message.content
            
            # Reconstruct history: Base System -> Summary Block -> Retained Context
            summary_message = {
                "role": "system", 
                "content": f"[HISTORICAL ARCHIVE SUMMARY]:\n{summary_text}"
            }
            
            self.messages = [self.messages[0], summary_message] + retained_slice
            print(f"[SYSTEM] Compression successful. New Context Size: {self.get_total_tokens()} tokens.\n")
            
        except Exception as e:
            print(f"[SYSTEM WARNING] Historical distillation routine failed: {e}")

    def get_messages_for_inference(self):
        return self.messages

    def dump_chat_log(self):
        now = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        log_dir = Path("Logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # This will now never crash because all items are explicitly sanitized into dicts
        with open(log_dir / f"{now}_log.txt", 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.messages, indent=2, ensure_ascii=False))
        print(f"[SYSTEM] Log written to {log_dir / 'chatlog.txt'}")
