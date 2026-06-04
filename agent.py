from tool_headers import TOOL_LIST, AVAILABLE_ACTIONS
from base_prompt import BASE_PROMPT

from client import CLIENT, MODEL_NAME


def parse_system_prompt(user_input: str, messages: list) -> tuple[bool, bool]:
    cmd = user_input.strip('/').lower()
    terminate = False
    ok = True 

    match cmd:
        case 'exit' | 'quit' | 'close' | 'bye':
            terminate = True
        case 'clear':
            messages.clear()
            messages.append({"role": "system", "content": BASE_PROMPT})
            print("[SYSTEM] Context cleared")
        case _:
            ok = False
    
    return ok, terminate


def run_agentic_chat():
    messages = [
        {"role": "system", "content": BASE_PROMPT}
    ]

    # Formally declare tools conforming to OpenAI standard schema definitions
    tools = TOOL_LIST
    available_actions = AVAILABLE_ACTIONS

    print(f"--- Unified Agent Interface Ready ({MODEL_NAME}) ---")
    terminate = False

    # 
    # !!! Start core loop !!!
    # 
    while True:
        user_input = input("You: ").strip()
        ok = True

        if not user_input:
            continue

        # Check if it's a system command
        if user_input[0] == "/":
            ok, terminate = parse_system_prompt(user_input, messages)
            
            if not ok:
                print("[SYSTEM] Unknown system prompt (?)")
                continue
            
            if terminate: 
                print("[SYSTEM] Shutdown signal received. \nGoodbye!")
                break

            continue

        # Standard interaction path
        messages.append({"role": "user", "content": user_input})

        # Call local model over unified OpenAI protocol
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        
        # Handle tool calls 
        if response_message.tool_calls:
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                
                if func_name in available_actions:
                    func_executor, pydantic_schema = available_actions[func_name]
                    
                    # Validate and parse arguments cleanly via Pydantic model parsing
                    raw_args = tool_call.function.arguments
                    parsed_args = pydantic_schema.model_validate_json(raw_args)
                    
                    print(f"\n[SYSTEM] Agent executing action '{func_name}'")
                    action_result = func_executor(parsed_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": action_result
                    })
            
            # Request final synthesized answer
            final_response = CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=messages
            )
            print(f"AI: {final_response.choices[0].message.content}\n")
            messages.append(final_response.choices[0].message)
        else:
            print(f"AI: {response_message.content}\n")
            messages.append(response_message)
