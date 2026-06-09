import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Pipe

import toolbox_worker
from context_handler import ContextHandler
from tool_headers import TOOL_LIST, AVAILABLE_ACTIONS
from base_prompt import BASE_PROMPT
from client import CLIENT, MODEL_NAME
from RAG import db_retrieve


MAX_CONTEXT_TOKENS = 16384
TRIGGER_SUMMARY_TOKENS = int(MAX_CONTEXT_TOKENS * 0.8)  
TEMPERATURE = 0.5 # Global temp setting, Not used for now

# Safety throttle: Prevent a model from getting stuck in an infinite tool-calling loop
MAX_LOOP_CYCLES = 8

# ======================================================================
# Launch worker process from which tool calls are executed
# ======================================================================
parent_conn, child_conn = Pipe()
worker_process = Process(target=toolbox_worker.listen_and_execute, args=(child_conn,))
worker_process.start()
print("[SYSTEM] Tool dispatch worker initialized.")


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
                print(f"─► Requesting remote execution for '{func_name}'")

                if not func_name in AVAILABLE_ACTIONS:
                    # Inform the model natively if it hallucinates an invalid action name
                    context_handler.append_messages({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": f"[SYSTEM] ERROR: Tool '{func_name}' is not registered in AVAILABLE_ACTIONS."
                    })
                    continue

                # Pack and send the payload across the pipeline
                payload = json.dumps({"name": func_name, "arguments": tool_call.function.arguments})
                parent_conn.send(payload)

                # Block-read the answer
                response_payload = json.loads(parent_conn.recv())

                if response_payload["status"] == "success":
                    action_result = response_payload["data"]
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
