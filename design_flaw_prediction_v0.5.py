import streamlit as st
import os
from dotenv import load_dotenv
import requests
import io
import pandas as pd
import json
from google import genai
from google.genai import types
import logging
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO

log_filename = datetime.now().strftime("app_log_%Y-%m-%d.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logging.info("Application started.")

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

logo_url = "https://drive.google.com/file/d/1HFz2fWiESfWwNctASYw5aUOggugUHSH-/view?usp=drive_link"
logo_img_id = logo_url.split('/d/')[1].split('/')[0]
logo_direct_url = f"https://drive.google.com/uc?export=download&id={logo_img_id}"
logo_response = requests.get(logo_direct_url)
logo_img = Image.open(BytesIO(logo_response.content))

# Set Streamlit page config for a professional looking interface
st.set_page_config(page_title="Design Flaw Prediction Scenarios", layout="wide")
#st.image(logo_img, width=150)
#st.title("Design Flaw Prediction Scenarios")
# st.markdown(
#     """
#     <h3 style='text-align: center;'>Design Flaw Prediction Scenarios</h3>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown("This application classifies design flaw related query and retrieves response backed up model-based predictions.")

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("<h1 style='text-align: center;'>Design Flaw Prediction Scenarios</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This application classifies design flaw related query and retrieves response backed up model-based predictions.</p>", unsafe_allow_html=True)

with col3:
    st.image(logo_img, width=150)

# Function to load a dataset from a given Google Drive URL
@st.cache_data
def load_dataset(url):
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    logging.debug(f"Loading dataset from URL: {download_url}")
    response = requests.get(download_url)
    response.raise_for_status()
    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data)
    logging.info("Dataset loaded successfully.")
    return df, df.to_string(index=False)

# Sidebar header for data loading info
st.sidebar.header("Data Loading")

# Use st.spinner() (instead of st.sidebar.spinner()) to show progress while loading data
with st.spinner("Loading datasets..."):
    # Carbon Emissions Data
    carbon_emission_url = "https://drive.google.com/file/d/1CT7BoH4GViir01wk-jbAlk0IAC8JOtCF/view?usp=drive_link"
    df_carbon_emission, context_carbon_emission = load_dataset(carbon_emission_url)
    
    # Engine Condition Data
    engine_condition_url = "https://drive.google.com/file/d/1DF70jCVv6wAuMF8WxsHy_GoVOJSiBKmV/view?usp=drive_link"
    df_engine_condition, context_engine_condition = load_dataset(engine_condition_url)
    
    # Parts Failure Data
    part_failure_url = "https://drive.google.com/file/d/1bdQ6G3dQYTbVvoNdF2fCula0ApSiNDFJ/view?usp=drive_link"
    df_part_failure, context_part_failure = load_dataset(part_failure_url)
    
    # FMEA Analysis Data
    fmea_analysis_url = "https://drive.google.com/file/d/1U9pkpYF2ejUuMKEK-YyE_icRSgRuRSqx/view?usp=drive_link"
    df_fmea_analysis, context_fmea_analysis = load_dataset(fmea_analysis_url)
    
    # Warranty Claim Data
    warranty_claim_url = "https://drive.google.com/file/d/12Q3rTROSFAIyiColNMLvGHIWEMk5uAz1/view?usp=drive_link"
    df_warranty_claim, context_warranty_claim = load_dataset(warranty_claim_url)
st.sidebar.success("Datasets loaded successfully!")
logging.info("All datasets loaded.")

prompt_template = "Answer the following question based on the data, also wherever appropriate provide examples from data to support the response:"

# Initialize the Gemini client
client = genai.Client(api_key=google_api_key)

# Define prediction functions
def carbon_emission(query):
    try:
        logging.info("Processing carbon emission query.")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= prompt_template + f' {context_carbon_emission}\n\nQuestion: {query}'
        )
        logging.info("Carbon emission query processed successfully.")
        return response.text
    except Exception as e:
        logging.error("Error in carbon_emission(): " + str(e))
        return str(e)

def engine_condition(query):
    try:
        logging.info("Processing engine condition query.")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= prompt_template + f' {context_engine_condition}\n\nQuestion: {query}'
        )
        logging.info("Engine condition query processed successfully.")
        return response.text
    except Exception as e:
        logging.error("Error in engine_condition(): " + str(e))
        return str(e)

def part_failure(query):
    try:
        logging.info("Processing part failure query.")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= prompt_template + f' {context_part_failure}\n\nQuestion: {query}'
        )
        logging.info("Part failure query processed successfully.")
        return response.text
    except Exception as e:
        logging.error("Error in part_failure(): " + str(e))
        return str(e)

def fmea_analysis(query):
    try:
        logging.info("Processing FMEA analysis query.")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= prompt_template + f' {context_fmea_analysis}\n\nQuestion: {query}'
        )
        logging.info("FMEA analysis query processed successfully.")
        return response.text
    except Exception as e:
        logging.error("Error in fmea_analysis(): " + str(e))
        return str(e)

def warranty_claim(query):
    try:
        logging.info("Processing warranty claim query.")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= prompt_template + f' {context_warranty_claim}\n\nQuestion: {query}'
        )
        logging.info("Warranty claim query processed successfully.")
        return response.text
    except Exception as e:
        logging.error("Error in warranty_claim(): " + str(e))
        return str(e)

def identify_design_flaws(query):
    try:
        logging.info("Processing design flaw identification query.")
        context = "\n".join([
            context_carbon_emission,
            context_engine_condition,
            context_fmea_analysis,
            context_part_failure,
            context_warranty_claim
        ])
        prompt = (
            "Analyze the entire data related to carbon emissions, engine failure conditions, FMEA analysis, part failure prediction and warranty claim predictions. "
            "Provide evidence for identified design flaw and detailed step-by-step reasoning."
        )
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= f' {context}\n\nQuestion: {prompt} {query}'
        )
        logging.info("Design flaw identification query processed successfully.")
        return response.text
    except Exception as e:
        logging.error("Error in identify_design_flaws(): " + str(e))
        return str(e)

# Classification settings and function
classification_keywords = ["carbon_emission", "engine_condition", "part_failure", "fmea_analysis", "warranty", "others"]

def classify_query(query):
    classification_prompt = (
        f"Classify the following query into one of these categories: {', '.join(classification_keywords)}. "
        "Return 'carbon_emission' if query is related to carbon emissions. "
        "Return 'part_failure' if query is related to part failure - failure types and related warranty. "
        "Return 'engine_condition' if query is related to engine condition. Engine condition 1 means engine has failed while 0 means it is working as expected. "
        "Return 'fmea_analysis' if query is related to failure modes, effects, RPN, or potential failures. "
        "Return 'warranty' if query is related to warranty claims or defect issues. "
        "Return 'design_flaw' if query is related to design flaws. "
        "Return only the classification keyword that best fits the query.\n\n"
        f"Query: {query}\n\nClassification:"
    )
    try:
        logging.info("Classifying query.")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents= classification_prompt
        )
        classification = response.candidates[0].content.parts[0].text.strip()
        logging.info(f"Query classified as: {classification}")
        return classification
    except Exception as e:
        logging.error("Error in classify_query(): " + str(e))
        return str(e)

# Main interface for query input and processing
st.subheader("Enter Your Query")
query_input = st.text_area("Query", value="Enter your query here...", height=70)

if st.button("Classify and Process Query"):
    if query_input.strip() == "":
        st.error("Please enter a valid query.")
        logging.warning("No query was entered by the user.")
    else:
        with st.spinner("Classifying query..."):
            classification = classify_query(query_input)
        st.subheader("Query Classification")
        st.markdown(f"**The query was classified as:** {classification}")
        
        final_response = ""
        with st.spinner("Processing query..."):
            if classification == "carbon_emission":
                retVal = carbon_emission(query_input)
                final_response = "Query is related to **Carbon Emissions prediction.**\n\n" + retVal
            elif classification == "engine_condition":
                retVal = engine_condition(query_input)
                final_response = "Query is related to **Engine Condition prediction.**\n\n" + retVal
            elif classification == "part_failure":
                retVal = part_failure(query_input)
                final_response = "Query is related to **Part Failure prediction.**\n\n" + retVal
            elif classification == "fmea_analysis":
                retVal = fmea_analysis(query_input)
                final_response = "Query is related to **Failure Mode Effect Analysis.**\n\n" + retVal
            elif classification == "warranty":
                retVal = warranty_claim(query_input)
                final_response = "Query is related to **Warranty Claim prediction.**\n\n" + retVal
            elif classification == "design_flaw":
                retVal = identify_design_flaws(query_input)
                final_response = "Query is related to **Design Flaw Prediction.**\n\n" + retVal
            elif classification == "others":
                final_response = "Query could not be processed."
            else:
                final_response = "Unrecognized classification result."
                
        st.subheader("Final Response")
        logging.info("Query Input is: " + query_input)
        logging.info("Query Classification is: " + classification)
        logging.info("Response to Query is: " + final_response)
        st.text_area("Response", value=final_response, height=500)
        logging.info("Query processing completed.")
        logging.info("***************************************************************")

