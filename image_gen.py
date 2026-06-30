
import torch
from diffusers import StableDiffusionPipeline
import os


# Global variable to keep the pipeline in VRAM so we don't 
# re-load from disk every single time (crucial for low latency!)
_PIPELINE = None

def to_device(pipe):
    """Moves the pipeline to the best available hardware."""
    if torch.cuda.is_available():
        print("[INFO] Deploying to NVIDIA GPU (CUDA)!")
        return pipe.to("cuda") 
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[INFO] Deploying to Apple Silicon (MPS)!")
        return pipe.to("mps")
    else:
        print("[WARN] CUDA/MPS not found. Falling back to CPU. Prepare for slow processing... ┐(´∇｀)┌")
        return pipe.to("cpu")

def get_pipeline(model_id="gsdf/Counterfeit-V2.5"):
    """
    Loads the pipeline into memory if it's not already there.
    This prevents the massive overhead of reloading weights every call.
    """
    global _PIPELINE
    if _PIPELINE is None:
        print(f"[INFO] Initializing pipeline with model: {model_id}...")
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            _PIPELINE = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            _PIPELINE = to_device(_PIPELINE)
            print("[INFO] Pipeline loaded and ready for deployment!")
        except Exception as e:
            print(f"[ERROR] Failed to load model architecture: {e}")
            raise e
    return _PIPELINE

def generate_image(prompt, output_filename="generated_output.png"):
    """
    The main entry point for our future Telegram service.
    Takes a string 'prompt' and saves the resulting image to disk.
    """
    global _PIPELINE
    print(f"[EXEC] Starting inference for prompt: '{prompt}'")
    
    try:
        pipe = get_pipeline()
        image = pipe(
            prompt=prompt, 
            num_inference_steps=15, 
            guidance_scale=7.5
        ).images[0]
        
        image.save(output_filename)
        print(f"[SUCCESS] Image cached successfully at: {output_filename}")
        return True, output_filename

    except Exception as e:
        print(f"[ERROR] Critical failure during image generation: {str(e)}")
        return False, str(e)

if __name__ == "__main__":
    # This block is just for testing locally before we wire up the Telegram API.
    test_prompt = "A high-tech cyberpunk hacker working on four monitors, neon lighting, detailed digital art"
    success, result = generate_image(test_prompt, "test_internal.png")
    
    if success:
        print("System Test: PASS")
    else:
        print(f"System Test: FAIL -> {result}")
