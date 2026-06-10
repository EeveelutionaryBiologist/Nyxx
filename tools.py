
import os
import sys
import json
import base64
import mimetypes
import subprocess
import ast
import operator as op

from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Any
from duckduckgo_search import DDGS

# from RAG import db_retrieve, add_chunk_to_db

from client import CLIENT, MODEL_NAME
from base_prompt import BASE_PROMPT


# Setup a tmp directory - needed for some tool uses
THIS_DIR = Path(__file__).resolve().parent
TMP_DIR = THIS_DIR / "tmp"
ERROR_DIR = THIS_DIR / "errors"

if not TMP_DIR.exists():
    TMP_DIR.mkdir()

if not ERROR_DIR.exists():
    ERROR_DIR.mkdir()

# Using Pydantic to define tool schemas 
class FileReadArgs(BaseModel):
    filepath: str = Field(description="The relative or absolute file path to the target text file.")

class ChangeDirArgs(BaseModel):
    direcory: str = Field(description="Relative or absolute directory to move to.")

class ArithmeticArgs(BaseModel):
    expression: str = Field(description="String formatted arithmetic expression that will be evaluated in the backend (power operator is '**').")

class ImageAnalyzerArgs(BaseModel):
    image_path: str = Field(description="The relative or absolute path to the local image file (e.g. 'invoice.png').")
    visual_instruction: str = Field(description="Explicit context or extraction instructions for the image.")

class FileWriterArgs(BaseModel):
    filepath: str = Field(description="The relative or absolute file path where the file should be created or overwritten.")
    content: str = Field(description="The raw string content to write into the file.")

class BashCommandArgs(BaseModel):
    command: str = Field(description="The exact terminal/shell command to execute locally (e.g. 'ls -la').")

class WebSearchArgs(BaseModel):
    query: str = Field(description="A specific, optimized search query to lookup on the internet.")

class MemoryInputArgs(BaseModel):
    string: str = Field(description="Information to be retained in permanent memory for later context enrichment (RAG). Should be phrased in a concise way. NOTE: Convoluted facts may be broken up into multiple simpler facts/ tool calls.")

class MemoryQueryArgs(BaseModel):
    query: str = Field(description="Query to search in local RAG data base.")

#
# Core tool functions
#
def tool_evaluate_arithmetic(args: ArithmeticArgs) -> str:
    # Source - https://stackoverflow.com/a/9558001
    # Posted by jfs, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-05-28, License - CC BY-SA 4.0

    # supported operators
    operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
                ast.USub: op.neg}

    def eval_(node):
        match node:
            case ast.Constant(value) if isinstance(value, int):
                return value  # integer
            case ast.BinOp(left, op, right):
                return operators[type(op)](eval_(left), eval_(right))
            case ast.UnaryOp(op, operand):  # e.g., -1
                return operators[type(op)](eval_(operand))
            case _:
                raise TypeError(node)

    def eval_expr(expr):
        return eval_(ast.parse(expr, mode='eval').body)

    try: 
        return f"{eval_expr(args.expression)}"
    except Exception as e:
        return f"Error handling expression {args.expression}. \nError: {e}"


def tool_read_file(args: FileReadArgs) -> str:
    if not os.path.exists(args.filepath):
        return f"Error: File '{args.filepath}' not found."
    try:
        with open(args.filepath, 'r', encoding='utf-8') as f:
            return f"Content of {args.filepath}:\n\n{f.read()}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def tool_write_file(args: FileWriterArgs) -> str:
    # if not os.path.exists(os.pardir(args.filepath)):
    #     return f"Error: Parent directory not available."
    try: 
        with open(args.filepath, 'w') as f:
            f.write(args.content)
            return f"File written to {args.filepath}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"


def tool_change_working_directory(args: ChangeDirArgs) -> str:
    if not os.path.exists(args.direcory): 
        return f"Error: Target directory does not exist."
    try: 
        os.chdir(args.direcory)
        return f"System directory changed to {args.direcory}"
    except Exception as e:
        return f"Error changing directory."


def tool_extract_image_info(args: ImageAnalyzerArgs) -> str:
    if not os.path.exists(args.image_path):
        return f"Error: Image '{args.image_path}' not found."
    
    try:
        print(f"\n[SYSTEM] Vision engine encoding image '{args.image_path}' for Gemini...")
        
        # 1. Determine correct MIME type (e.g., image/png, image/jpeg)
        mime_type, _ = mimetypes.guess_type(args.image_path)
        if not mime_type:
            mime_type = "image/jpeg"  # Fallback

        # 2. Read the local binary file and parse it to base64
        with open(args.image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        # 3. Construct an OpenAI-compliant Data URL
        image_data_url = f"data:{mime_type};base64,{base64_string}"

        # 4. Re-instantiate a dedicated local sub-client tracking your configuration
        # Make sure 'client' or your API parameters are within scope here
        sub_response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": args.visual_instruction
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
        )
        
        analysis_result = sub_response.choices[0].message.content
        if not analysis_result:
            return "[SYSTEM ERROR] Gemini processed the input frame but returned an empty response."
            
        return f"Analysis results for image ({args.image_path}):\n\n{analysis_result}"

    except Exception as e:
        return f"Error processing image through vision pipeline: {str(e)}"
    

def tool_execute_bash_command(args: BashCommandArgs) -> str:
    try:
        result = subprocess.run(
            args.command, shell=True, text=True, capture_output=True, timeout=10
        )
        print(f"[SYSTEM] Running command: {args.command}")
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 10 seconds."


def tool_web_search(args: WebSearchArgs) -> str:
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(args.query, max_results=5)]
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error executing web search: {str(e)}"


def tool_retrieve_memory(args: MemoryQueryArgs) -> str:
    try:
        top_n_hits = memory_interface.db_retrieve(query=args.query)
        return json.dumps(top_n_hits, indent=2)
    except Exception as e:
        return f"Error executing memory search: {str(e)}"


def tool_commit_to_memory(args: MemoryInputArgs) -> str:
    try:
        memory_interface.add_chunk_to_db(chunk=args.string, source="agent")
        return f"[SUCCESS] Fact successfully committed to local persistent storage memory. Current total keys: {memory_interface.db_length()}"
    except Exception as e:
        return f"[SYSTEM ERROR] Failed committing fact to ChromaDB: {str(e)}"
