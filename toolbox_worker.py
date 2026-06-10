
import os
import sys
import time
import json
from multiprocessing import Process, Pipe

from tool_headers import AVAILABLE_ACTIONS


class ToolWorkerInterface:
    def __init__(self, directory):
        self.parent_conn, self.child_conn = Pipe()
        self.worker_process = Process(target=listen_and_execute, args=(self.child_conn,))
        self.current_directory = directory
        print("[SYSTEM] Tool dispatch worker initialized.")

    def start(self):
        self.worker_process.start()

    def dispatch(self, func_name, arguments):
        # Pack and send the payload across the pipeline
        payload = json.dumps({"name": func_name, "arguments": arguments})
        self.parent_conn.send(payload)

        # Block-read the answer
        response_payload = json.loads(self.parent_conn.recv())
        return response_payload

    def shutdown(self):
        print("[SYSTEM] Signalling worker process to halt...")
        # Option A: Send a poison pill message that the worker loop checks for
        try:
            self.parent_conn.send(json.dumps({"name": "SIG_SHUTDOWN", "arguments": "{}"}))
        except Exception as e:
            print(e)
        # Wait for worker process to clean up -
        time.sleep(2)
        
        # Option B: Forcefully kill it from the OS layer if it ignores the signal
        if self.worker_process.is_alive():
            self.worker_process.terminate() # Sends SIGTERM natively
            self.worker_process.join()      # Wait for OS resource cleanup to finish


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
            
            # Handling of termination signal
            if request.get("name") == "SIG_SHUTDOWN":
                print("[WORKER] Exit signal received. Cleaning up...", flush=True)
                
                # TODO: Run any database close routines or file flushes here
                break
            
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
