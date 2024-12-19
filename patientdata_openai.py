from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Text
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field

#For Testing one Can put OPENAI API key here also directly instead of putting in .env file
OPENAI_KEY=""
#Getting the OPENAI_KEY Environment Variable from .env file.
# load_dotenv()
# OPENAI_KEY = os.getenv("OPENAI_KEY")

class Patient(BaseModel):
    name: str = Field(default="Unknown", description="Name of the patient")
    gender: str = Field(default="Unknown", description="Gender of the patient")
    age: int = Field(default=0, description="Age of the patient")
    weight: float = Field(default=0, description="Weight of the patient in pounds")
    height: float = Field(default=0, description="Height of the patient in inches")
    bmi: float = Field(default=None, description="BMI (Body Mass Index) of the patient")
    chief_med_complaint: str = Field(default=None, description="Chief Medical complaint")

def convert_and_calculate_bmi(height_in_inches, weight_in_pounds):
    # Conversion factors
    inches_to_meters = 0.0254
    pounds_to_kilograms = 0.453592
    
    # Convert height and weight
    height_in_meters = height_in_inches * inches_to_meters
    weight_in_kilograms = weight_in_pounds * pounds_to_kilograms
    
    # Calculate BMI
    bmi = round(weight_in_kilograms / (height_in_meters ** 2), 2)
    return bmi


def create_prompt(data):
    """Creates a structured prompt template for OpenAI."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are expert in extracting patient details from text. Find the details of patient based on text related to patient"),
        ("user", f"Following is the patient description:\n {data}"),

    ])
    return prompt.format_messages()

def generate_output(data):
    """Generates an output using OpenAI API."""
    try:
        model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0,
            openai_api_key=OPENAI_KEY
        )
        messages = create_prompt(data)
        model_with_structure = model.with_structured_output(Patient)
        response = model_with_structure.invoke(messages)
        response_dict=dict(response)
        height=response_dict["height"]
        weight=response_dict["weight"]
        bmi=convert_and_calculate_bmi(height, weight)
        response_dict["bmi"]=bmi
        return response_dict
    
    except Exception as e:
        return f"Error generating output: {str(e)}"


patient_1="""
John R. Whitaker, a 52-year-old male, stands 5'10" (70 inches) tall and weighs 198
lbs. Mr. Whitaker has a history of hypertension and type 2 diabetes, both diagnosed in his
mid-40s, and recently began experiencing worsening peripheral neuropathy in his lower
extremities. He also reports chronic lower back pain, which he attributes to years of heavy
lifting in his previous occupation as a construction worker. Over the past six months, John
has developed shortness of breath during mild exertion, prompting concerns about potential
early-stage congestive heart failure. Additionally, he struggles with obesity-related sleep
apnea, contributing to fatigue and cognitive fog throughout the day. Despite his conditions,
Mr. Whitaker maintains a generally positive outlook but admits to inconsistent medication
adherence and difficulty following a healthy diet.
"""
patient_2="""
Patient 2: For the past six months, Emily J. Rivera has been dealing with persistent chest
tightness and shortness of breath, especially during moderate activity. She works as a
schoolteacher and describes her symptoms as worsening under stress, which she initially
dismissed as anxiety. A thorough evaluation revealed mild asthma, along with borderline high
cholesterol levels. Emily is 41 years old, 5'5" (65 inches) tall, and weighs 172 lbs. She also
complains of intermittent joint stiffness in her hands, particularly in the mornings, which her
physician suspects could be early osteoarthritis. Emily’s sedentary lifestyle and inconsistent
exercise routine have contributed to her struggles with maintaining a healthy weight, though
she remains committed to improving her overall health with proper guidance and treatment.
"""
patient_3="""
Karen L. Thompson, a 38-year-old female, is 5'4" (64 inches) tall and weighs 162
lbs. She has a history of irritable bowel syndrome (IBS) and recurrent migraines, both of
which have intensified over the past year. Karen also experiences chronic fatigue and joint
pain, leading her physician to investigate possible early-stage rheumatoid arthritis. She
reports frequent episodes of dizziness and occasional heart palpitations, which have been
attributed to mild anemia and elevated stress levels. Karen’s symptoms are exacerbated by
her demanding job as a paralegal, where long hours and poor posture have contributed to
persistent neck and shoulder tension. Recently, she has begun experiencing intermittent
insomnia, further impacting her energy levels and overall well-being.
"""

#Change the patient variable if required to test data extraction of other patients
print("======EXTRACTED PATIENT DATA========")
response=generate_output(patient_3)
print(response)
