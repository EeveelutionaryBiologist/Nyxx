import json
from pathlib import Path
from typing import Any
from datetime import datetime

from tool_headers import TOOL_LIST, AVAILABLE_ACTIONS
from base_prompt import BASE_PROMPT
from client import CLIENT, MODEL_NAME
from RAG import db_retrieve


MAX_CONTEXT_TOKENS = 16384
TRIGGER_SUMMARY_TOKENS = int(MAX_CONTEXT_TOKENS * 0.8)  # ~6550 tokens
TEMPERATURE = 0.5 # Global temp setting, Not used for now

# Safety throttle: Prevent a model from getting stuck in an infinite tool-calling loop
MAX_LOOP_CYCLES = 8


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


def parse_system_prompt(user_input: str, context_handler: ContextHandler) -> tuple[bool, bool]:
    cmd = user_input.strip('/').lower()
    terminate = False
    ok = True 

    match cmd:
        case 'exit' | 'quit' | 'close' | 'bye':
            terminate = True
        case 'tokens':
            n_tokens = context_handler.get_total_tokens()
            print(f"[SYSTEM] Estimated current token usage: {n_tokens}")
        case 'clear':
            context_handler.clear()
        case 'log':
            context_handler.dump_chat_log()
        case _:
            ok = False
    
    return ok, terminate


def run_agentic_chat():
    context_handler = ContextHandler()
    tools = TOOL_LIST
    available_actions = AVAILABLE_ACTIONS

    print(f"--- Autonomous Agent Interface Ready ({MODEL_NAME}) ---")

    # ========================================================
    # CORE LOOP STARTS HERE
    # ========================================================
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input[0] == "/":
            ok, terminate = parse_system_prompt(user_input, context_handler)
            if not ok:
                print("[SYSTEM] Unknown system prompt (?)")
                continue
            if terminate: 
                print("[SYSTEM] Shutdown signal received. \nGoodbye!")
                break
            continue

        # Passive CPU RAG Injection
        try:
            raw_memories = db_retrieve(query=user_input, top_n=3)
            if raw_memories:
                memory_context = f"\n\n[LOCAL MEMORY CONTEXT]\n{json.dumps(raw_memories, indent=2)}"
            else:
                memory_context = ""
        except Exception as e:
            print(f"[SYSTEM WARNING] Failed background memory pre-fetch: {e}")
            memory_context = ""

        enriched_content = f"{user_input}{memory_context}"
        context_handler.append_messages({"role": "user", "content": enriched_content})
        cycle_count = 0
        
        while cycle_count < MAX_LOOP_CYCLES:
            cycle_count += 1
            
            # Request inference with current message history state
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=context_handler.get_messages_for_inference(),
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # Base Case: If the model has no tool calls, it's done reasoning and talking to the user
            if not response_message.tool_calls:
                print(f"AI: {response_message.content}\n")
                context_handler.append_messages(response_message)
                break  # Exit the recursion loop, pass control back to the human 'input'
                
            # Recursive Case: Model wants to execute one or many tools
            context_handler.append_messages(response_message)
            print(f"\n[THINKING RUN {cycle_count}/{MAX_LOOP_CYCLES}] Model requested tool execution...")

            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                
                if func_name in available_actions:
                    func_executor, pydantic_schema = available_actions[func_name]
                    
                    try:
                        raw_args = tool_call.function.arguments
                        parsed_args = pydantic_schema.model_validate_json(raw_args)
                        
                        print(f"  └─► Executing '{func_name}' with args: {raw_args}")
                        action_result = func_executor(parsed_args)
                    except Exception as parse_err:
                        action_result = f"[SYSTEM ERROR] Failed parsing/executing tool arguments: {str(parse_err)}"
                    
                    context_handler.append_messages({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": action_result
                    })
                else:
                    # Inform the model natively if it hallucinates an invalid action name
                    context_handler.append_messages({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": f"[SYSTEM] ERROR: Tool '{func_name}' is not registered in AVAILABLE_ACTIONS."
                    })
                    
        else:
            # Executes only if the loop hits MAX_LOOP_CYCLES without breaking cleanly
            warning_msg = "[SYSTEM] Warning - Throttled! Agent exceeded max autonomous thinking steps safety cap."
            print(warning_msg)
            context_handler.append_messages({"role": "system", "content": warning_msg})
