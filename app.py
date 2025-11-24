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
    """Schema for extracting structured data from a Myanmar driver's license, including Burmese script."""
    No: str = pydantic.Field(description="The license number/ID from the card.")
    Name: str = pydantic.Field(description="The full name of the license holder, exactly as written (English/Transliteration).")
    # --- NEW FIELDS FOR BURMESE TEXT ---
    Name_Burmese: str = pydantic.Field(description="The full name of the license holder as written in Burmese script (e.g., ကျော်စွာမင်း).")
    NRC_no: str = pydantic.Field(description="The National Registration Card number (e.g., 12/ABC(N)XXXXXX).")
    NRC_no_Burmese: str = pydantic.Field(description="The numeric/code portion of the NRC number written in Burmese (e.g., ၁၂/ဒဂန(နိုင်)၀၃၅၃၄၄).")
    # ------------------------------------
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
            
        #Ground Truth input and accuracy display
        st.subheader("📝 Manual Ground Truth Verification")
        st.info("Enter the correct values from the license to calculate the Exact Match Rate.")
        
        # Initialize a dictionary to hold manual inputs (Ground Truth)
        ground_truth = {}
        
        # Create input fields for manual verification
        # The key names must match your extracted_data keys
        
        gt_col1, gt_col2 = st.columns(2)
        
        # Use st.session_state to persist data if necessary, 
        # but for simplicity, we'll use a local dictionary for this run
        
        with gt_col1:
            ground_truth['No'] = st.text_input("Correct License No. (Ground Truth)", "")
            ground_truth['Name_Burmese'] = st.text_input("Correct Name (Burmese)", "")
            ground_truth['DOB'] = st.text_input("Correct Date of Birth (DD-MM-YYYY)", "")
            ground_truth['Valid_up_to'] = st.text_input("Correct Valid Up To (DD-MM-YYYY)", "")
            
        with gt_col2:
            ground_truth['Name'] = st.text_input("Correct Name (English/Latin)", "")
            ground_truth['NRC_no'] = st.text_input("Correct NRC No. (e.g., 12/ABC(N)XXXXXX)", "")
            ground_truth['NRC_no_Burmese'] = st.text_input("Correct NRC No. (Burmese Script)", "")
            ground_truth['Blood_Type'] = st.text_input("Correct Blood Type", "")


        if st.button("Calculate Accuracy"):
            total_fields = 0
            correct_fields = 0
            
            # Compare extracted data against non-empty ground truth fields
            for key, extracted_value in extracted_data.items():
                gt_value = ground_truth.get(key)
                
                # Only evaluate fields where a ground truth value was provided
                if gt_value:
                    total_fields += 1
                    # Strip whitespace to ensure a fair comparison
                    if str(extracted_value).strip() == str(gt_value).strip():
                        correct_fields += 1
            
            if total_fields > 0:
                emr = (correct_fields / total_fields) * 100
                st.metric(
                    label="Field Exact Match Rate (EMR)", 
                    value=f"{emr:.2f}%", 
                    help=f"{correct_fields} out of {total_fields} fields matched the Ground Truth exactly."
                )
            else:
                st.warning("Please enter Ground Truth values in the text boxes above to calculate accuracy.")
        