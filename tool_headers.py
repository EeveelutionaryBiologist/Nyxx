
from tools import *


AVAILABLE_ACTIONS = {
    "evaluate_arithmetic": (tool_evaluate_arithmetic, ArithmeticArgs),
    "read_file": (tool_read_file, FileReadArgs),
    "write_to_file": (tool_write_file, FileWriterArgs),
    "change_directory": (tool_change_working_directory, ChangeDirArgs),
    "extract_image_info": (tool_extract_image_info, ImageAnalyzerArgs),
    "bash_command": (tool_execute_bash_command, BashCommandArgs),
    "search_web": (tool_web_search, WebSearchArgs),
    "retrieve_RAG": (tool_retrieve_memory, MemoryQueryArgs),
    "add_memory_RAG": (tool_commit_to_memory, MemoryInputArgs)
}


TOOL_LIST = [
        {
            "type": "function",
            "function": {
                "name": "evaluate_arithmetic",
                "description": "Reads and evaluates an arithmetic expression on the CPU. Examples include 2**8, 23144+1412, etc.",
                "parameters": ArithmeticArgs.model_json_schema() 
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads and retrieves the content of a local text file.",
                "parameters": FileReadArgs.model_json_schema() 
            }
        },
        {
            "type": "function",
            "function": {
                "name": "change_directory",
                "description": "Changes to a different directory on the local system.",
                "parameters": ChangeDirArgs.model_json_schema() 
            }
        },
        {
            "type": "function",
            "function": {
                "name": "extract_image_info",
                "description": "Analyzes a local image file and extracts text, data, or charts.",
                "parameters": ImageAnalyzerArgs.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_to_file",
                "description": "Write content in text form to output file.",
                "parameters": FileWriterArgs.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash_command",
                "description": "Execute shell commands via subprocess.",
                "parameters": BashCommandArgs.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information.",
                "parameters": WebSearchArgs.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_RAG",
                "description": "Local RAG query retrieving relevant memories and learned facts.",
                "parameters": MemoryQueryArgs.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_memory_RAG",
                "description": "Add a string to permament memory using the RAG backend. IMPORTANT: Should be phrased in a concise way. Convoluted facts may be broken up into multiple simpler facts/ tool calls.",
                "parameters": MemoryInputArgs.model_json_schema()
            }
        }
    ]
