from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import PyPDF2

# Load the model and tokenizer
model_name = "ruslanmv/Medical-Llama3-8B"
device_map = 'auto'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set the pad token to eos if necessary
tokenizer.pad_token = tokenizer.eos_token

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function to analyze the blood report
def analyze_blood_report(report_text):
    # System message specific to analyzing a medical blood report
    sys_message = '''
    You are an AI Medical Assistant. Analyze the following blood report and provide a section-wise analysis. 
    Focus on identifying abnormalities in the Lipid Profile, RBC Count, HbA1c, Thyroid Panel, and other notable areas. 
    Provide possible causes for abnormalities and recommend consulting a healthcare professional for diagnosis.
    '''
    
    # Constructing the full prompt with the blood report text
    full_prompt = sys_message + "\nBlood Report:\n" + report_text + "\nAnalysis:"
    
    # Tokenize the input
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    # Generate the analysis
    outputs = model.generate(**inputs, max_new_tokens=300, use_cache=True)
    
    # Decode the generated output
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return the generated analysis
    return response_text.split("Analysis:")[-1].strip()

# Path to the PDF file
pdf_path = '/path/to/your/blood_report.pdf'

# Extract the blood report text from the PDF
blood_report_text = extract_text_from_pdf(pdf_path)

# Call the function with the extracted blood report text
analysis = analyze_blood_report(blood_report_text)

# Print the analysis
print(analysis)
