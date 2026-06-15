
import os
import json
import requests
from pathlib import Path
from datetime import datetime

from toolbox_worker import ToolWorkerInterface
from context_handler import ContextHandler
from tool_headers import TOOL_LIST, AVAILABLE_ACTIONS
from base_prompt import BASE_PROMPT
from client import CLIENT, MODEL_NAME


TEMPERATURE = 0.5 # Global temp setting, Not used for now

# Safety throttle: Prevent a model from getting stuck in an infinite tool-calling loop
MAX_LOOP_CYCLES = 8

CURRENT_DIR = Path(os.getcwd())

# The URL where our new Memory Microservice is hosted
MEMORY_SERVER_URL = "http://localhost:8000"

# ======================================================================
# Launch worker process from which tool calls are executed
# ======================================================================
worker = ToolWorkerInterface(CURRENT_DIR)
worker.start()

# ======================================================================
# Main functions
# ======================================================================
def parse_system_prompt(user_input: str, context_handler: ContextHandler) -> tuple[bool, bool]:
    cmd = user_input.strip('/').lower()
    terminate = False
    ok = True 

    match cmd:
        case 'exit' | 'quit' | 'close' | 'bye':
            terminate = True
            worker.shutdown()
        case 'clear':
            context_handler.clear()
        case 'log':
            context_handler.dump_chat_log()
        case 'forgetall':
            confirm = input("[SYSTEM] This action is irreversible. Continue? [y/n]").strip()
            if confirm in ['y', 'Y']:
                try:
                    response = requests.delete(f"{MEMORY_SERVER_URL}/memory/clear")
                    response.raise_for_status()
                    print("[SYSTEM] Database completely wiped.")
                except Exception as e:
                    print(f"[SYSTEM ERROR] Failed to wipe memory: {e}")
                    
        case 'dumpmemory' | 'writememory':
            try:
                response = requests.get(f"{MEMORY_SERVER_URL}/memory/all")
                response.raise_for_status()
                memories = response.json().get("results", [])
                
                print("ID\tHITS\tCREATED_AT\tTEXT")
                print("-" * 60)
                for entry in memories:
                    print(f"{entry['id'][:8]}...\t{entry['hit_count']}\t{entry['created_at'][:10]}\t{entry['text']}")
                print(f"[SYSTEM] Total records: {len(memories)}")
            except Exception as e:
                print(f"[SYSTEM ERROR] Could not retrieve memories: {e}")
        case _:
            ok = False
    
    return ok, terminate


def run_agentic_chat():
    context_handler = ContextHandler()
    tools = TOOL_LIST

    print(f"--- Autonomous Agent Interface Ready ({MODEL_NAME}) ---")

    # ========================================================
    # CORE LOOP STARTS HERE
    # ========================================================
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        # ========================================================
        # Handle System-level input
        # ========================================================
        if user_input[0] == "/":
            ok, terminate = parse_system_prompt(user_input, context_handler)
            if not ok:
                print("[SYSTEM] Unknown system prompt (?)")
                continue
            if terminate: 
                print("[SYSTEM] System shutting down. \nGoodbye!")
                break
            continue

        # ========================================================
        # Passive Microservice RAG injection -> retrieve relevant memory chunks
        # ========================================================
        try:
            response = requests.post(f"{MEMORY_SERVER_URL}/memory/search", json={"query": user_input, "top_k": 5})
            if response.status_code == 200:
                raw_memories = response.json().get("results", [])
                if raw_memories:
                    # We inject the text AND the hit_counter to give the model context on how important this is
                    formatted_mems = [f"- {m['text']} (Hits: {m['hit_count']})" for m in raw_memories]
                    memory_context = f"\n\n[LOCAL MEMORY CONTEXT]\n" + "\n".join(formatted_mems)
                else:
                    memory_context = ""
            else:
                memory_context = ""
        except requests.exceptions.ConnectionError:
            # Don't crash the chat if the memory server is offline, just warn the user once
            print("[SYSTEM WARNING] Memory Server unreachable. RAG is currently disabled.")
            memory_context = ""
        except Exception as e:
            print(f"[SYSTEM WARNING] Failed background memory pre-fetch: {e}")
            memory_context = ""

        enriched_content = f"{user_input}{memory_context}"
        context_handler.append_messages({"role": "user", "content": enriched_content})

        # ========================================================
        # Dispatch messages to Black Box model, handle response
        # ========================================================
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
                print(f"─► Requesting remote execution for '{func_name}'")

                if not func_name in AVAILABLE_ACTIONS:
                    context_handler.append_messages({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": f"[SYSTEM] ERROR: Tool '{func_name}' is not registered in AVAILABLE_ACTIONS."
                    })
                    continue

                response_payload = worker.dispatch(func_name, tool_call.function.arguments)

                if response_payload["status"] == "success":
                    action_result = response_payload["data"]
                elif response_payload["status"] == "REQUEST_PARENT_WRITE":
                    # Adapt the parent write to use the HTTP Server instead of ChromaDB directly
                    try:
                        fact_to_write = response_payload["payload"]
                        res = requests.post(f"{MEMORY_SERVER_URL}/memory/add", json={"text": fact_to_write})
                        res.raise_for_status()
                        action_result = f"[SUCCESS] Parent process committed fact to Memory Microservice."
                    except Exception as e:
                        action_result = f"[SYSTEM ERROR] Parent failed database write: {e}"
                else:
                    action_result = f"[PROCESS ISOLATION ERROR]: {response_payload['message']}"

                context_handler.append_messages({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": action_result
                    })
        else:
            # Executes only if the loop hits MAX_LOOP_CYCLES without breaking cleanly
            warning_msg = "[SYSTEM] Warning - Throttled! Agent exceeded max autonomous thinking steps safety cap."
            print(warning_msg)
            context_handler.append_messages({"role": "system", "content": warning_msg})
