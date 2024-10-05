from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from huggingface_hub import InferenceClient  # Use InferenceClient instead of InferenceApi
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI()

# Serve static files from the 'static' directory with an absolute path
app.mount("/static", StaticFiles(directory=os.path.abspath("static")), name="static")

# Serve the index.html file at the root URL
@app.get("/")
def serve_index():
    # Get the absolute path of index.html
    file_path = os.path.join(os.path.abspath("static"), "index.html")
    print(f"Serving index.html from: {file_path}")
    return FileResponse(file_path)

# Access the Hugging Face API token from Codespaces secrets (using the secret name 'HUGGING')
huggingface_api_token = os.getenv("HUGGING")

# Ensure the token is available
if not huggingface_api_token:
    raise EnvironmentError("HUGGING environment variable is not set")

# Set up Hugging Face Inference Client (replacing deprecated InferenceApi)
hf_client = InferenceClient(model="gpt2", token=huggingface_api_token)

# Define the data model for the incoming requirement prioritization request
class RequirementInput(BaseModel):
    requirement: str
    importance: int
    complexity: int
    business_value: int

# Route to handle the requirement prioritization
@app.post("/prioritize")
async def prioritize_requirement(input_data: RequirementInput):
    # Build the prompt to send to the model based on the input
    prompt = f"""
    Requirement: {input_data.requirement}
    Stakeholder Importance: {input_data.importance}/10
    Implementation Complexity: {input_data.complexity}/10
    Business Value: {input_data.business_value}/10

    Please prioritize this requirement and explain the reasoning.
    """

    try:
        # Call Hugging Face Inference API to generate the response using the InferenceClient
        response = hf_client.text_generation(prompt, max_tokens=200)
        generated_text = response["generated_text"]

        # Return the generated text as the response
        return {"response": generated_text}

    except Exception as e:
        # Handle API errors and provide useful feedback
        print(f"Error calling Hugging Face API: {str(e)}")  # Log the error for debugging
        return {"response": f"Error: {str(e)}"}
