from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
import os

# api initialisation
app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Hugging Face Inference Client with LLaMA 3.2
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf_client = InferenceClient(model="meta-llama/Llama-3.2B", token=huggingface_api_token)

# model for the incoming requirement prioritization request
class RequirementInput(BaseModel):
    requirement: str
    importance: int
    complexity: int
    business_value: int

# Route to handle the requirement prioritization
@app.post("/prioritize")
async def prioritize_requirement(input_data: RequirementInput):
    # Prompts
    prompt = f"""
    You are an expert in software requirement prioritization for a financial project.

    Requirement: {input_data.requirement}

    For each of the following, provide a clear, concise answer:

    1. **Importance**: Explain why this requirement is rated {input_data.importance}/10 in terms of importance. What is the significance of this feature to the project and stakeholders?
    2. **Complexity**: Explain why the complexity is rated {input_data.complexity}/10. What are the challenges in implementing this requirement?
    3. **Business Value**: Explain why the business value is rated {input_data.business_value}/10. How will this requirement impact the project’s goals (e.g., security, compliance, user trust)?

    Please keep your response directly focused on the factors listed above.
    """

    try:
        # Calls Hugging Face Inference API to generate the response using LLaMA 3.2
        response = hf_client.text_generation(prompt, max_tokens=200)
        generated_text = response["generated_text"]

        return {"response": generated_text}

    except Exception as e:
        # Handles API errors and provide useful feedback
        print(f"Error calling Hugging Face API: {str(e)}")
        return {"response": f"Error: {str(e)}"}
