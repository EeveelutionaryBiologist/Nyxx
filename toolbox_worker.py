
import os
import sys
import time
import json
from multiprocessing import Process, Pipe

from tool_headers import AVAILABLE_ACTIONS
from tools import MemoryInputArgs


class ToolWorkerInterface:
    def __init__(self, directory):
        self.parent_conn, self.child_conn = Pipe()
        
        # Pass memory_interface to args mapping tuple
        self.worker_process = Process(
            target=listen_and_execute, 
            args=(self.child_conn,)
        )
        self.current_directory = directory
        # self.memory_interface = memory_interface
        print("[SYSTEM] Tool dispatch worker initialized.")

    def start(self):
        self.worker_process.start()

    def dispatch(self, func_name, arguments):
        # Pack and send the payload across the pipeline
        payload = json.dumps({"name": func_name, "arguments": arguments})
        self.parent_conn.send(payload)
        print(payload)

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
            request_raw = pipe_connection.recv()
            request = json.loads(request_raw)
            
            if request.get("name") == "SIG_SHUTDOWN":
                print("[WORKER] Exit signal received. Cleaning up...", flush=True)
                break
            
            func_name = request["name"]
            arguments_str = request["arguments"]

            if func_name in AVAILABLE_ACTIONS:

                # If the model wants to write to persitent DB, intercept here
                if func_name == "add_persistent_memory":
                    # Decode what the model wanted to save
                    parsed_args = MemoryInputArgs.model_validate_json(arguments_str)
                    
                    # Signal the parent process to do the write operation
                    pipe_connection.send(json.dumps({
                        "status": "REQUEST_PARENT_WRITE", 
                        "payload": parsed_args.string
                    }))
                    continue

                func_executor, pydantic_schema = AVAILABLE_ACTIONS[func_name]
                
                # Unpack arguments and send off to function
                parsed_args = pydantic_schema.model_validate_json(arguments_str)
                result = func_executor(parsed_args)
                
                pipe_connection.send(json.dumps({"status": "success", "data": result}))
            else:
                pipe_connection.send(json.dumps({"status": "error", "message": f"Unknown tool: {func_name}"}))
                
        except Exception as e:
            pipe_connection.send(json.dumps({"status": "error", "message": str(e)}))
            