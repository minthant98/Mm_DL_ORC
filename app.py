import os
import pydantic
from google import genai
from google.genai import types

# --- Configuration ---
# Set your API Key as an environment variable (best practice)
os.environ["GEMINI_API_KEY"] = "AIzaSyByIZjACIwYmamMfye3rSVDJH3r6iYluVQ"
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure your GEMINI_API_KEY is set correctly.")
    exit()

# Define the model to use. Gemini 2.5 Flash is highly capable for vision and fast.
MODEL_NAME = "gemini-2.5-flash"
IMAGE_FILE_PATH = "img1.jpg" # Use the noisy image for a real test!

# --- 3. Define the Structured Output Schema (The most important step) ---

class DriverLicenseData(pydantic.BaseModel):
    """Schema for extracting structured data from a driver's license."""
    No: str = pydantic.Field(description="The license number/ID from the card.")
    Name: str = pydantic.Field(description="The full name of the license holder, exactly as written.")
    NRC_no: str = pydantic.Field(description="The National Registration Card number (e.g., 12/ABC(N)XXXXXX).")
    DOB: str = pydantic.Field(description="The Date of Birth in DD-MM-YYYY format.")
    Blood_Type: str = pydantic.Field(description="The Blood Type (e.g., A, B, O, AB).")
    Valid_up_to: str = pydantic.Field(description="The expiry date of the license in DD-MM-YYYY format.")

# --- 4. Main Extraction Function ---

def extract_license_data(image_path: str) -> dict:
    """Uploads the image and prompts Gemini for structured data extraction."""
    
    # Upload the image file to the Gemini service
    print(f"Uploading file: {image_path}...")
    image_file = client.files.upload(file=image_path)
    
    # Define the instruction prompt
    prompt = (
        "Analyze the attached Myanmar Driving License image, which may be blurry or at an angle. "
        "Extract ONLY the specified information into the provided JSON schema. "
        "Pay close attention to the Burmese labels for 'No.', 'Name', 'NRC no', 'DOB', 'Blood Type', and 'Valid up to'."
    )
    
    # Configure the request to use the Pydantic schema
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DriverLicenseData,
    )
    
    print("Sending request to Gemini...")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_file],  # Multimodal prompt: text + image
            config=config,
        )
        
        # The response text will be a guaranteed JSON string adhering to the schema
        extracted_json = response.text
        
        # Optional: Parse the JSON string back into a Python object for easy use
        return DriverLicenseData.model_validate_json(extracted_json).model_dump()
        
    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return {"Error": "Extraction failed"}
    finally:
        # Clean up the uploaded file to save storage space
        client.files.delete(name=image_file.name)
        print(f"Cleaned up uploaded file: {image_file.name}")


# --- Execution ---
extracted_data = extract_license_data(IMAGE_FILE_PATH)

print("\n### RESULTS FROM GEMINI API ###")
if "Error" in extracted_data:
    print(extracted_data["Error"])
else:
    for key, value in extracted_data.items():
        print(f"**{key}**: {value}")