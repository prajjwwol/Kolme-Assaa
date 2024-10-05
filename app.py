from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import requests
from PIL import Image
from transformers import LlamaForConditionalGeneration, AutoProcessor
import torch

# Initialize FastAPI app
app = FastAPI()

# Serve static files from the 'static' directory using an absolute path
app.mount("/static", StaticFiles(directory=os.path.abspath("static")), name="static")

# Serve the index.html at the root URL "/"
@app.get("/")
def serve_index():
    file_path = os.path.join(os.path.abspath("static"), "index.html")
    print(f"Serving index.html from: {file_path}")
    return FileResponse(file_path)

# Access the Hugging Face API token from Codespaces secrets (using the secret name 'HUGGING')
huggingface_api_token = os.getenv("HUGGING")
if not huggingface_api_token:
    raise EnvironmentError("HUGGING environment variable is not set or invalid")

# Set up the LLaMA 3.2 Vision-Instruct model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = LlamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Define the data model for the incoming requirement prioritization request
class RequirementInput(BaseModel):
    requirement: str
    importance: int
    complexity: int
    business_value: int

# Route to handle the requirement prioritization
@app.post("/prioritize")
async def prioritize_requirement(input_data: RequirementInput):
    prompt = f"""
    Requirement: {input_data.requirement}
    Stakeholder Importance: {input_data.importance}/10
    Implementation Complexity: {input_data.complexity}/10
    Business Value: {input_data.business_value}/10

    Please prioritize this requirement and explain the reasoning.
    """

    try:
        # Load a sample image (replace with actual image processing as needed)
        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Prepare the input text and image
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors="pt").to(model.device)

        # Generate output from the model
        output = model.generate(**inputs, max_new_tokens=200)
        generated_text = processor.decode(output[0], skip_special_tokens=True)

        # Return the generated response
        return {"response": generated_text}

    except Exception as e:
        print(f"Error calling LLaMA 3.2 Vision-Instruct: {str(e)}")  # Log the error for debugging
        return {"response": f"Error: {str(e)}"}
