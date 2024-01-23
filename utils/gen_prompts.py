from enum import Enum


class Prompts(Enum):
    DEMOGRAPHIC = """Provide the Continuity of Care Document (CCD) one  patient demographic details only in below format including 
    Sex: 
    Age:
    Race:
    Ethnicity:
    Marital Status:
    Zip Code:
    .Focusing only on the information related to the patient named [Patient].Do not give me an answer if it is not mentioned in the prompt as a fact""" 
    DEMOGRAPHIC2 = """Generate a Continuity of Care Document (CCD) summarizing the patient demographics, including Sex, Age, Race, Ethnicity, Marital Status, and Zip Code. The document should provide a comprehensive overview of the patient's essential information, ensuring accuracy and compliance with healthcare standards. The goal is to facilitate seamless and accurate information transfer across healthcare providers, promoting effective continuity of care for the patient. Please ensure the generated document is clear, concise, and adheres to the relevant data privacy regulations."""
    PROBLEM = """Retrieve comprehensive information on the patient's health over the past 12 months, specifically focusing on existing problems, current encounter details, diagnostic codes, result values, out-of-range lab results, and abnormal vital signs. Organize the data by zip codes, diagnosis codes, and education levels. Emphasize any health issues that fall outside the normal range, providing context on their significance and potential impact on the patient's well-being. Ensure the generated report is thorough, well-organized, and adheres to privacy regulations."""
    
    
#print(Prompts.DEMOGRAPICS.value)
#print(type(Prompts.DEMOGRAPICS))
#print(repr(Prompts.DEMOGRAPICS))
#print(list(Prompts.DEMOGRAPICS))    

