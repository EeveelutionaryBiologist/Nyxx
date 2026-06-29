import os
import json
import uuid
import requests
from pathlib import Path
from datetime import datetime

from toolbox_worker import ToolWorkerInterface
from context_handler import ContextHandler
from tool_headers import TOOL_LIST, AVAILABLE_ACTIONS
from base_prompt import BASE_PROMPT
from client import CLIENT, MODEL_NAME
from RAG import ChromaDBInterface
from conv_memory import ConversationalMemory

TEMPERATURE = 0.5 # Global temp setting, Not used for now

# Safety throttle: Prevent a model from getting stuck in an infinite tool-calling loop
MAX_LOOP_CYCLES = 8

CURRENT_DIR = Path(os.getcwd())

# The URL where Erebus memory microservice is hosted
MEMORY_SERVER_URL = "http://localhost:8000"

# If this is ON, don't store messages in long term conversational memory
INCOGNITO_MODE = False

# Unique ID for this running session — used to exclude current-session chunks from retrieval
SESSION_ID = str(uuid.uuid4())

# ======================================================================
# Launch worker process from which tool calls are executed
# ======================================================================
worker = ToolWorkerInterface(CURRENT_DIR)
worker.start()

# ======================================================================
# Initialize local conversational memory (ChromaDB, CPU embedding)
# ======================================================================
_chroma_db = ChromaDBInterface()
conv_mem = ConversationalMemory(_chroma_db, CURRENT_DIR / "Memory" / "metadata.db")

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
            confirm = input("[SYSTEM] This action is irreversible. Continue? [y/n] ").strip()
            if confirm in ['y', 'Y']:
                try:
                    response = requests.delete(f"{MEMORY_SERVER_URL}/memory/clear")
                    response.raise_for_status()
                    print("[SYSTEM] Database and Knowledge Graph completely wiped.")
                except Exception as e:
                    print(f"[SYSTEM ERROR] Failed to wipe memory: \n{e}")
                    
        case 'consolidate':
            try:
                print("[SYSTEM] Running memory consolidation (this may take a while)...")
                response = requests.post(f"{MEMORY_SERVER_URL}/memory/consolidate")
                response.raise_for_status()
                result = response.json().get("report", {})
                print(
                    f"[SYSTEM] Consolidation complete. "
                    f"Pruned: {result.get('pruned', 0)}, "
                    f"Merged: {result.get('merged', 0)}, "
                    f"Split: {result.get('split', 0)}"
                )
            except Exception as e:
                print(f"[SYSTEM ERROR] Consolidation failed: \n{e}")

        case 'dumpmemory' | 'writememory':
            try:
                response = requests.get(f"{MEMORY_SERVER_URL}/memory/all")
                response.raise_for_status()
                memories = response.json().get("results", [])
                
                print("ID\tHITS\tTYPE\tCREATED_AT\tTEXT")
                print("-" * 72)
                for entry in memories:
                    print(f"{entry['id'][:8]}...\t{entry['hit_count']}\t{entry.get('record_type','?')}\t{entry['created_at'][:10]}\t{entry['text'][:60]}")
                print(f"[SYSTEM] Total records: {len(memories)}")
            except Exception as e:
                print(f"[SYSTEM ERROR] Could not retrieve memories: \n{e}")
        case 'tokens':
            try:
                tokens = context_handler.get_total_tokens()
                print("[SYSTEM] Estimated tokens used: %s" % tokens)
            except Exception as e:
                print(f"[SYSTEM ERROR] Could not get token usage: \n{e}")
        case 'incognito' | 'anonymous' | 'anon':
            global INCOGNITO_MODE
            if INCOGNITO_MODE:
                INCOGNITO_MODE = False
                print("[SYSTEM] Incognito mode disengaged.")
            else:
                INCOGNITO_MODE = True
                print("[SYSTEM] Messages will not be logged.")
        case _: 
            ok = False
    
    return ok, terminate


def run_agentic_chat():
    context_handler = ContextHandler()
    tools = TOOL_LIST

    print(f"--- Autonomous Agent Interface Ready ({MODEL_NAME}) ---")

    # Check if memory backend is running:
    try:
        _ = requests.post(f"{MEMORY_SERVER_URL}/memory/ping")
        print(f"[SYSTEM] Connected to Erebus on {MEMORY_SERVER_URL}")
    except Exception as e:
        print("[SYSTEM WARNING] Memory Server unreachable. Is Erebus running?")
    

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
        # Passive Microservice RAG injection (Hybrid Graph + Vector)
        # ========================================================
        memory_blocks = []

        try:
            response = requests.post(f"{MEMORY_SERVER_URL}/memory/context", json={"query": user_input, "top_k": 5})
            if response.status_code == 200:
                data = response.json()
                raw_memories = data.get("results", [])
                relational_context = data.get("relational_context", "")

                if raw_memories:
                    formatted_mems = [f"- {m['text']} (Hits: {m['hit_count']})" for m in raw_memories]
                    memory_blocks.append("[SEMANTIC MEMORIES]\n" + "\n".join(formatted_mems))

                if relational_context:
                    memory_blocks.append("[RELATIONAL KNOWLEDGE]\n" + relational_context.strip())
        except requests.exceptions.ConnectionError:
            print("[SYSTEM WARNING] Memory Server unreachable. Is Erebus running?")
        except Exception as e:
            print(f"[SYSTEM WARNING] Failed background memory pre-fetch: {e}")

        # ========================================================
        # Local conversational memory retrieval (independent of Erebus)
        # ========================================================
        if not INCOGNITO_MODE:
            try:
                conv_results = conv_mem.retrieve_similar(user_input, top_k=5, current_session_id=SESSION_ID)
                if conv_results:
                    lines = [
                        f"- [{r['source']}, {r['created_at'][:10]}, topic: {r['topic']}] \"{r['text'][:200]}\""
                        for r in conv_results
                    ]
                    memory_blocks.append("[PAST CONVERSATION EXCERPTS]\n" + "\n".join(lines))
            except Exception as e:
                print(f"[SYSTEM WARNING] Conv memory retrieval failed: {e}")

        # Combine all recalled context into a single ephemeral block
        ephemeral_context = (
            "\n\n=== RECALLED CONTEXT ===\n" + "\n\n".join(memory_blocks) + "\n========================"
        ) if memory_blocks else ""

        # Store only the clean user input in context history (not the enriched version)
        context_handler.append_messages({"role": "user", "content": user_input})

        # Persist user turn to local conversational memory
        if not INCOGNITO_MODE:
            try:
                conv_mem.store_chunk(user_input, "user", SESSION_ID)
            except Exception as e:
                print(f"[SYSTEM WARNING] Conv memory write failed: {e}")

        # ========================================================
        # Dispatch messages to Black Box model, handle response
        # ========================================================
        cycle_count = 0

        while cycle_count < MAX_LOOP_CYCLES:
            cycle_count += 1
            
            # Request inference with current message history state
            # ephemeral_context is injected only for this call and not stored in history
            response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=context_handler.get_messages_for_inference(ephemeral_context=ephemeral_context),
                tools=tools,
                tool_choice="auto"
            )
            ephemeral_context = ""  # Only inject on the first cycle; tool results handle subsequent context
            
            response_message = response.choices[0].message
            
            # Base Case: If the model has no tool calls, it's done reasoning and talking to the user
            if not response_message.tool_calls:
                print(f"AI: {response_message.content}\n")
                context_handler.append_messages(response_message)
                if not INCOGNITO_MODE and response_message.content:
                    try:
                        conv_mem.store_chunk(response_message.content, "assistant", SESSION_ID)
                    except Exception as e:
                        print(f"[SYSTEM WARNING] Conv memory write failed: {e}")
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
                    cycle_count -= 1  # Memory addition ops shouldn't bloat the safety loop counter
                    try:
                        fact_to_write = response_payload["payload"]
                        res = requests.post(f"{MEMORY_SERVER_URL}/memory/add", json={"text": fact_to_write})
                        res.raise_for_status()
                        action_result = f"[SUCCESS] Fact successfully committed to Memory Microservice."
                    except Exception as e:
                        action_result = f"[SYSTEM ERROR] Failed database write: {e}"
                elif response_payload["status"] == "REQUEST_PARENT_CONV_SEARCH":
                    payload = response_payload["payload"]
                    try:
                        action_result = conv_mem.retrieve_by_time(
                            date_hint=payload.get("date_hint", ""),
                            keyword=payload.get("keyword", "")
                        )
                    except Exception as e:
                        action_result = f"[ERROR] Conversational history search failed: {e}"
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
            