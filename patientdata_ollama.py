from ollama import chat
from pydantic import BaseModel
import json

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


class Patient(BaseModel):
    name: str 
    gender: str 
    age: int 
    weight: float
    height: float
    # bmi: float
    chief_med_complaint: str

response = chat(
    messages=[
        {
            'role': 'user', 
            'content': f"""
            You are highly expert in extracting patient information.
            Extract the name, age, gender, weight, height, chief_med_complaint from the given text. Only consider height in inch. There should be no nested dictionary.
            Consider He or she to determine male or female. Only include numbers not units. Only give the dictonary output nothing else.
            text:
            {patient_1}
            Provide the information in JSON format with fields: name, age, gender, weight, height, chief_med_complaint
            
            """}
    ],
    model='llama3.2'
)

# Parse the response
try:
    content = response['message']['content']
    # Extract the JSON part from the response
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    json_str = content[json_start:json_end]
    
    # Parse the JSON and create a Country object
    patient_data = json.loads(json_str)
    patient = Patient(**patient_data)
    res_dict=dict(patient)
    height=res_dict["height"]
    weight=res_dict["weight"]
    res_dict["bmi"]=convert_and_calculate_bmi(height,weight)
    print("=========Dictionary Output:===========")
    print(res_dict)
except json.JSONDecodeError:
    print("Failed to parse JSON from the response")
except ValueError as e:
    print(f"Invalid data format: {e}")
