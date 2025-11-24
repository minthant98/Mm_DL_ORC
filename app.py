import os
import io
import tempfile
import streamlit as st
import pydantic
from google import genai
from google.genai import types
from PIL import Image

# --- Configuration ---
# Set your API Key as an environment variable (best practice)
os.environ["GEMINI_API_KEY"] = "AIzaSyByIZjACIwYmamMfye3rSVDJH3r6iYluVQ"
try:
    client = genai.Client()
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop() # Stop the app if initialization fails

MODEL_NAME = "gemini-2.5-flash"

# --- 2. Define the Structured Output Schema ---

class DriverLicenseData(pydantic.BaseModel):
    """Schema for extracting structured data from a driver's license."""
    No: str = pydantic.Field(description="The license number/ID from the card.")
    Name: str = pydantic.Field(description="The full name of the license holder, exactly as written.")
    NRC_no: str = pydantic.Field(description="The National Registration Card number (e.g., 12/ABC(N)XXXXXX).")
    DOB: str = pydantic.Field(description="The Date of Birth in DD-MM-YYYY format.")
    Blood_Type: str = pydantic.Field(description="The Blood Type (e.g., A, B, O, AB).")
    Valid_up_to: str = pydantic.Field(description="The expiry date of the license in DD-MM-YYYY format.")

# --- 3. Main Extraction Function ---

# Streamlit caching: prevents the heavy function from re-running unnecessarily
@st.cache_data(show_spinner="Running Gemini extraction on image...")
def extract_license_data(image_bytes: bytes, filename: str) -> dict:
    """
    Uploads the image and prompts Gemini for structured data extraction.
    This function handles the file upload and cleanup.
    """
    
    # Write the bytes to a temporary file, as the client.files.upload requires a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        temp_path = tmp.name
    
    image_file = None
    try:
        # Upload the image file to the Gemini service
        image_file = client.files.upload(file=temp_path)
        
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
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_file], 
            config=config,
        )
        
        # Parse the guaranteed JSON string
        return DriverLicenseData.model_validate_json(response.text).model_dump()
        
    except Exception as e:
        return {"Error": f"Extraction failed: {e}"}
    finally:
        # Clean up both the temporary local file and the file uploaded to the service
        if image_file:
            client.files.delete(name=image_file.name)
        os.unlink(temp_path)


# --- 4. Streamlit UI Implementation ---

st.title("🇲🇲 Myanmar Driving License Extractor")
st.markdown("Use the Gemini API to extract structured data from a photo of a license. Supports both uploads and live camera photos.")

# Create tabs for the two input methods
tab_upload, tab_camera = st.tabs(["Upload Image", "Take Live Photo"])

uploaded_file = None
source_label = ""

with tab_upload:
    st.markdown("### ⬆️ Upload File")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image of the license", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_label = "Uploaded Image"

with tab_camera:
    st.markdown("### 📸 Live Camera")
    # Streamlit's built-in camera input for live photo capture
    camera_file = st.camera_input("Take a photo of the license")
    if camera_file:
        uploaded_file = camera_file
        source_label = "Live Photo"


# --- Execution Logic ---
if uploaded_file is not None:
    # Read the file-like object into bytes
    image_bytes = uploaded_file.getvalue()
    
    st.sidebar.subheader(f"Processing: {source_label}")
    st.sidebar.image(Image.open(io.BytesIO(image_bytes)), use_column_width=True)
    
    # Display a button to run the extraction
    if st.button("✨ Run Structured Extraction"):
        st.info("Sending image to Gemini for analysis. This may take a moment...")
        
        # Call the extraction function with the image bytes and filename
        extracted_data = extract_license_data(image_bytes, uploaded_file.name)
        
        st.subheader("✅ Extracted Data")
        
        if "Error" in extracted_data:
            st.error(extracted_data["Error"])
        else:
            # Display results beautifully
            col1, col2 = st.columns(2)
            
            # Display the data in a structured way
            for i, (key, value) in enumerate(extracted_data.items()):
                if i % 2 == 0:
                    with col1:
                        st.metric(label=key, value=value)
                else:
                    with col2:
                        st.metric(label=key, value=value)
                        
            st.subheader("JSON Output (Raw)")
            st.json(extracted_data)