from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import instructor

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
    name: str = Field(default="Unknown", description="Name of the patient")
    gender: str = Field(default="Unknown", description="Gender of the patient")
    age: int = Field(default=0, description="Age of the patient")
    weight: float = Field(default=0, description="Weight of the patient in pounds lbs")
    height: float = Field(default=0, description="Height of the patient in inches")
    bmi: float = Field(default=None, description="BMI (Body Mass Index) of the patient")
    chief_med_complaint: str = Field(default=None, description="Chief Medical complaint")

# class Patient(BaseModel):
#     name: str 
#     gender: str 
#     age: int 
#     weight: float
#     height: float
#     # bmi: float
#     chief_med_complaint: str

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)

resp = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {
            'role': 'user', 
            'content': f"""
            You are highly expert in extracting patient information.
            Extract the name, age, gender, weight, height, chief_med_complaint form the given text. Only consider height in inch.
            determine gender as male or female. Only include numbers not units. Only give the dictonary output nothing else.
            text:
            {patient_3}
            Provide the information in JSON format with fields: name, age, gender, weight, height, chief_med_complaint
            
            """}
    ],
    response_model=Patient,
)


print(dict(resp))