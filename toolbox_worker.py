
import sys
import json
from tool_headers import AVAILABLE_ACTIONS


def listen_and_execute(pipe_connection):
    # =============================================================
    # Optional: Drop system permissions right here for security
    # =============================================================
    # os.setuid(sandbox_user_id)
    
    while True:
        try:
            # Wait for the main agent process to send a tool task
            request_raw = pipe_connection.recv()  # Block-waits for incoming data
            request = json.loads(request_raw)
            
            func_name = request["name"]
            arguments_str = request["arguments"]
            
            if func_name in AVAILABLE_ACTIONS:
                func_executor, pydantic_schema = AVAILABLE_ACTIONS[func_name]
                
                # Parse and run inside the isolated process space
                parsed_args = pydantic_schema.model_validate_json(arguments_str)
                result = func_executor(parsed_args)
                
                pipe_connection.send(json.dumps({"status": "success", "data": result}))
            else:
                pipe_connection.send(json.dumps({"status": "error", "message": f"Unknown tool: {func_name}"}))
                
        except Exception as e:
            # If a tool crashes, send back the error report
            pipe_connection.send(json.dumps({"status": "error", "message": str(e)}))
