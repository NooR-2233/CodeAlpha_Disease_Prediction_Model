# Import necessary libraries
# streamlit: A Python library for building web applications
import streamlit as st
# pandas: A Python library for data manipulation and analysis
import pandas as pd
# numpy: A Python library for numerical computing
import numpy as np
# scikit-learn: A Python library for machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# pickle: A Python library for serializing and deserializing objects
import pickle
# matplotlib: A Python library for creating static, animated, and interactive visualizations
import matplotlib.pyplot as plt
# random: A Python library for generating random numbers
import random
# wikipedia: A Python library for accessing Wikipedia data
import wikipedia
# warnings: A Python library for handling warnings
import warnings
# BeautifulSoup: A Python library for parsing HTML and XML documents
from bs4 import BeautifulSoup

# Function to load the dataset
@st.cache_data
def load_data():
    """
    Loads the dataset from a CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    data = pd.read_csv("Medical_Dataset.csv")
    return data

# Load the dataset
data = load_data()

# Prepare the data by splitting it into features (X) and target (y)
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Use LabelEncoder to convert the target variable into numerical values
le = LabelEncoder()
y = le.fit_transform(y)

# Function to train the model
@st.cache_resource
def train_model():
    """
    Trains a RandomForestClassifier model on the dataset.
    
    Returns:
        RandomForestClassifier: The trained model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    return model

# Train the model
model = train_model()

# Function to get a Wikipedia summary for a given disease
def get_wikipedia_summary(disease, sentences=2):
    """
    Retrieves a Wikipedia summary for a given disease.
    
    Args:
        disease (str): The name of the disease.
        sentences (int, optional): The number of sentences to include in the summary. Defaults to 2.
    
    Returns:
        str: The Wikipedia summary for the disease.
    """
    # Custom summaries for certain diseases
    custom_summaries = {
        "Allergy": "Allergy occurs when a person's immune system reacts to substances in the environment that are harmless to most people. These substances are known as allergens and are found in dust mites, pets, pollen, insects, ticks, moulds, foods, and drugs (medications). Atopy is the genetic tendency to develop allergic diseases.",
        "GERD": "Gastroesophageal reflux disease (GERD) is a condition in which the stomach contents leak backward from the stomach into the esophagus (food pipe). Food travels from your mouth to the stomach through your esophagus. GERD can irritate the food pipe and cause heartburn and other symptoms.",
        "Bronchial Asthma": "It is caused by inflammation and muscle tightening around the airways, which makes it harder to breathe. Symptoms can include coughing, wheezing, shortness of breath and chest tightness. These symptoms can be mild or severe and can come and go over time.",
        "Common Cold": "The common cold is an illness affecting your nose and throat. Most often, it's harmless, but it might not feel that way. Germs called viruses cause a common cold. Often, adults may have two or three colds each year.",
        "Dimorphic hemmorhoids(piles)": "Hemorrhoids, or piles, are a common issue. These swollen veins inside of your rectum or outside of your anus can cause pain, anal itching and rectal bleeding. Symptoms often improve with at-home treatments, but on occasion, people need medical procedures. Eating more fiber can help prevent hemorrhoids.",
        "Acne": "Acne is a common skin condition that happens when hair follicles under the skin become clogged. Sebum—oil that helps keep skin from drying out—and dead skin cells plug the pores, which leads to outbreaks of lesions, commonly called pimples or zits.",
        "Paroymsal Positional Vertigo": "Benign paroxysmal positional vertigo (BPPV) is a common inner ear disorder that causes vertigo, dizziness, and other symptoms. It's caused by calcium crystals (otoconia) becoming dislodged and moving into the inner ear's balance system, which can send incorrect messages to the brain.",
        "Chicken pox": "With chickenpox an itchy rash breaks out mostly on the face, scalp, chest, back with some spots on the arms and legs. The spots quickly fill with a clear fluid, break open and then turn crusty. Chickenpox is an illness caused by the varicella-zoster virus. It brings on an itchy rash with small, fluid-filled bl isters."
        
    }
    
    # Check if a custom summary is available
    if disease in custom_summaries:
        return custom_summaries[disease]
    
    try:
        # Retrieve the Wikipedia summary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return wikipedia.summary(disease, sentences=sentences)
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation errors
        return f"Multiple entries found for {disease}. Please be more specific."
    except wikipedia.exceptions.PageError:
        # Handle page errors
        return f"No detailed information available for {disease}. It may be a specific medical condition requiring professional diagnosis."
    except Exception as e:
        # Handle other exceptions
        return f"An error occurred while fetching information for {disease}: {str(e)}"
    # Function to generate disease information
def generate_disease_info(disease):
    """
    Generates disease information, including symptoms, treatments, and precautions.
    
    Args:
        disease (str): The name of the disease.
    
    Returns:
        str: The generated disease information.
    """
    # Define symptoms, treatments, and precautions
    symptoms = {
    "fatigue": "A state of extreme tiredness, often described as a lack of energy or motivation.",
    "fever": "An elevated body temperature, usually indicating the body's response to infection or illness.",
    "headache": "Pain or discomfort in the head, scalp, or neck, which can vary in intensity and duration.",
    "nausea": "An unpleasant sensation in the stomach that often comes with an urge to vomit.",
    "muscle pain": "Soreness or aching in the muscles, which can be caused by various factors including strain or illness.",
    "cough": "A sudden, often repetitive reflex that helps clear the breathing passages of irritants.",
    "sore throat": "Pain, scratchiness, or irritation of the throat that often worsens when swallowing.",
    "loss of appetite": "A decreased desire to eat, often accompanying various illnesses.",
    "dizziness": "A sensation of lightheadedness, unsteadiness, or loss of balance.",
    "shortness of breath": "Difficulty breathing or a feeling of not getting enough air, also known as dyspnea.",
    "itching": "An irritating sensation that causes the desire to scratch the skin.",
    "skin rash": "An area of irritated or swollen skin, which can appear as red, itchy, or inflamed patches.",
    "nodal skin eruptions": "Raised areas on the skin that may be inflamed or filled with fluid.",
    "continuous sneezing": "Repetitive involuntary expulsion of air from the nose and mouth due to irritation in the nasal passage.",
    "shivering": "Involuntary muscular contractions causing shaking, often due to cold or fever.",
    "chills": "A feeling of coldness accompanied by shivering, often linked to fever.",
    "joint pain": "Discomfort or pain in one or more joints, often caused by inflammation or injury.",
    "stomach pain": "Discomfort or pain in the abdominal area, which can have various causes.",
    "acidity": "A condition caused by excessive stomach acid, often leading to heartburn or discomfort.",
    "ulcers on tongue": "Sores that develop on the tongue, often painful and can be caused by various factors.",
    "muscle wasting": "A decrease in muscle mass and strength, often associated with various diseases.",
    "vomiting": "The act of expelling the contents of the stomach through the mouth.",
    "burning micturition": "A burning sensation during urination, often indicating a urinary tract infection.",
    "spotting urination": "Light bleeding during urination, which can be a sign of various conditions.",
    "weight gain": "An increase in body weight, often due to an increase in body fat.",
    "anxiety": "A feeling of worry, nervousness, or unease about something with an uncertain outcome.",
    "cold hands and feet": "A sensation of coldness in the extremities, often due to poor circulation.",
    "mood swings": "Frequent and intense changes in emotional state.",
    "weight loss": "A decrease in body weight, often due to loss of fat, muscle, or fluid.",
    "restlessness": "An inability to rest or relax, often associated with anxiety or discomfort.",
    "lethargy": "A state of sluggishness, inactivity, and a lack of energy.",
    "patches in throat": "Irritated areas in the throat that may be inflamed or infected.",
    "irregular sugar level": "Fluctuations in blood sugar levels, which can indicate diabetes or other conditions.",
    "high fever": "An elevated body temperature, typically above 100.4°F (38°C), often indicating infection.",
    "sunken eyes": "Eyes that appear deeper in the sockets, often due to dehydration or illness.",
    "breathlessness": "A feeling of not being able to get enough air, which can be caused by various conditions.",
    "sweating": "The body's way of regulating temperature, often increasing during physical activity or in heat.",
    "dehydration": "A condition that occurs when the body loses more fluids than it takes in.",
    "indigestion": "Discomfort in the upper abdomen often associated with difficulty in digesting food.",
    "yellowish skin": "A condition where the skin takes on a yellow tint, often indicating liver problems.",
    "dark urine": "Urine that appears darker than normal, often a sign of dehydration or liver disease.",
    "loss of smell": "A decrease or complete inability to detect odors, often linked to nasal or neurological issues.",
    "pain behind the eyes": "Discomfort or pain felt in the area behind the eyes, which can be due to various factors.",
    "back pain": "Discomfort or pain in the back, often due to strain, injury, or underlying health issues.",
    "constipation": "Difficulty in passing stools or infrequent bowel movements.",
    "abdominal pain": "Pain or discomfort in the stomach area, often due to various gastrointestinal issues.",
    "diarrhea": "Frequent, loose, or watery bowel movements, often indicating infection or digestive issues.",
    "mild fever": "A slightly elevated body temperature, often indicating a mild infection or illness.",
    "yellow urine": "Urine that appears more yellow than usual, often indicating hydration levels.",
    "yellowing of eyes": "A yellow tint to the whites of the eyes, often indicating liver issues.",
    "acute liver failure": "A rapid loss of liver function, which can be life-threatening.",
    "fluid overload": "A condition where there is an excess of fluid in the body, often affecting various organs.",
    "swelling of stomach": "Bloating or distention of the abdominal area, often due to fluid or gas accumulation.",
    "swelled lymph nodes": "Enlarged lymph nodes, often indicating an infection or illness.",
    "malaise": "A general feeling of discomfort, illness, or unease.",
    "blurred and distorted vision": "Reduced sharpness of vision, often indicating eye problems or neurological issues.",
    "phlegm": "Mucus produced in the respiratory tract, often associated with respiratory infections.",
    "throat irritation": "Discomfort or scratchiness in the throat, often caused by infection or allergens.",
    "redness of eyes": "Inflammation or irritation of the eyes, often caused by allergies or infection.",
    "sinus pressure": "A feeling of pressure or pain in the sinus cavities, often due to infection or allergies.",
    "runny nose": "Excess nasal mucus, often caused by allergies or infections.",
    "congestion": "Blockage or narrowing of the nasal passages, often due to swelling or mucus.",
    "chest pain": "Discomfort or pain in the chest area, which can have various causes, including heart issues.",
    "weakness in limbs": "A decrease in strength or control in the arms or legs.",
    "fast heart rate": "A condition where the heart beats faster than normal, often caused by stress or illness.",
    "pain during bowel movements": "Discomfort or pain experienced while passing stools.",
    "pain in anal region": "Discomfort or pain in the area surrounding the anus, which can have various causes.",
    "bloody stool": "Stools that contain blood, often indicating a serious gastrointestinal issue.",
    "irritation in anus": "Discomfort or inflammation in the anal area, often due to infection or irritation.",
    "neck pain": "Discomfort or pain in the neck area, often due to strain or injury.",
    "cramps": "Involuntary contractions of muscles that can cause pain and discomfort.",
    "bruising": "Discoloration of the skin caused by bleeding underneath, often due to injury.",
    "obesity": "A condition characterized by excessive body fat, often associated with various health risks.",
    "swollen legs": "Enlargement of the legs, often due to fluid retention or other medical conditions.",
    "swollen blood vessels": "Enlarged blood vessels that can indicate various circulatory problems.",
    "puffy face and eyes": "Swelling in the facial area, often due to fluid retention or allergies.",
    "enlarged thyroid": "A condition where the thyroid gland becomes larger, often indicating thyroid disorders.",
    "brittle nails": "Nails that are weak and easily break or chip, often indicating nutritional deficiencies.",
    "swollen extremities": "Enlargement of the arms or legs, often due to fluid retention.",
    "excessive hunger": "An increased desire to eat, often associated with metabolic or hormonal issues.",
    "extra marital contacts": "Involvement in romantic or sexual relationships outside of marriage.",
    "drying and tingling lips": "Lips that feel dry and may have a tingling sensation, often indicating dehydration.",
    "slurred speech": "Speech that is difficult to understand, often caused by neurological issues or intoxication.",
    "knee pain": "Discomfort or pain in the knee joint, which can result from injury or medical conditions.",
    "hip joint pain": "Discomfort in the hip joint area, often due to injury or degenerative conditions.",
    "muscle weakness": "A decrease in muscle strength, which can result from various medical conditions.",
    "stiff neck": "Reduced range of motion in the neck, often due to strain or injury.",
    "swelling joints": "Enlargement of joints, often indicating inflammation or injury.",
    "movement stiffness": "Reduced flexibility and ease of movement, often due to muscular or joint issues.",
    "spinning movements": "A sensation of the surroundings spinning, often associated with dizziness.",
    "loss of balance": "Inability to maintain an upright position, often due to neurological or inner ear issues.",
    "unsteadiness": "A feeling of instability or lack of balance.",
    "weakness of one body side": "Loss of strength or control on one side of the body, often indicating neurological issues.",
    "bladder discomfort": "Pain or discomfort in the bladder area, often indicating infection or inflammation.",
    "foul smell of urine": "Unpleasant odor in urine, often indicating infection or other medical conditions.",
    "continuous feel of urine": "A persistent sensation of needing to urinate, which can indicate various issues.",
    "passage of gases": "The release of gas from the digestive system through the rectum.",
    "internal itching": "An irritating sensation occurring within the body, often indicating allergies or infections.",
    "toxic look (typhos)": "A facial appearance associated with severe illness, often seen in cases of typhoid or other infections.",
    "depression": "A mood disorder characterized by persistent feelings of sadness, hopelessness, and loss of interest.",
    "irritability": "Increased sensitivity and emotional reactivity, often linked to stress or mood disorders.",
    "altered sensorium": "Changes in awareness or perception, often due to neurological or medical conditions.",
    "red spots over body": "Small red marks on the skin, which can indicate various skin conditions or allergic reactions.",
    "belly pain": "Discomfort or pain in the abdominal area, often related to digestive issues.",
    "abnormal menstruation": "Irregularities in the menstrual cycle, which can indicate hormonal or health issues.",
    "dischromic patches": "Skin discoloration that appears as patches, often due to various skin conditions.",
    "watering from eyes": "Excessive tearing or fluid production from the eyes, often indicating irritation or allergies.",
    "increased appetite": "An increased desire to eat, often linked to metabolic changes or psychological factors.",
    "polyuria": "Excessive urination, often indicating diabetes or other health conditions.",
    "family history": "A record of health issues within a family that may increase the risk for similar conditions.",
    "mucoid sputum": "Mucus produced in the respiratory tract, often thick and associated with respiratory conditions.",
    "rusty sputum": "Sputum that appears brown or rust-colored, often indicating infection or blood.",
    "lack of concentration": "Difficulty focusing or maintaining attention, often linked to mental health issues.",
    "visual disturbances": "Problems with vision, including blurriness or seeing spots, often linked to various conditions.",
    "receiving blood transfusion": "The process of receiving blood from a donor, which can carry risks of reactions.",
    "receiving unsterile injections": "Injections given with non-sterile equipment, which can lead to infections.",
    "coma": "A state of deep unconsciousness, often due to severe medical conditions.",
    "stomach bleeding": "Internal bleeding in the stomach, which can indicate serious medical conditions.",
    "distention of abdomen": "Abdominal swelling or enlargement, often due to fluid or gas accumulation.",
    "history of alcohol consumption": "A record of drinking alcohol, which can influence health and risk for various conditions.",
    "blood in sputum": "Presence of blood in the mucus coughed up from the respiratory tract, often indicating serious conditions.",
    "prominent veins on calf": "Visible veins in the calf area, often indicating circulatory issues.",
    "palpitations": "The sensation of having a fast-beating, fluttering, or pounding heart.",
    "painful walking ": "Discomfort or pain experienced while walking, often due to injury or joint issues.",
    "pus-filled pimples": "Inflamed skin lesions filled with pus, often due to infection or acne.",
    "blackheads": "Small, dark spots on the skin caused by clogged hair follicles.",
    "scurring": "A term that may refer to rapid movements or behavior, often indicating anxiety or restlessness.",
    "skin peeling": "The shedding of the outer layer of skin, often due to irritation or skin conditions.",
    "silver-like dusting": "A term that may refer to a skin condition characterized by shiny or silvery patches.",
    "small dents in nails": "Indentations or pits in the nails, often indicating health issues.",
    "inflammatory nails": "Nails that are inflamed or swollen, often associated with health conditions.",
    "blister": "A small pocket of fluid that forms on the skin, often due to friction or injury.",
    "red sore around nose": "Irritation or inflammation around the nasal area, often due to infection or allergies.",
    "yellow crust ooze": "A term describing a yellow discharge that can form crusts on the skin, often due to infection."
}
    treatments = {
        "rest and hydration": "Allowing the body to recover through adequate sleep and fluid intake.",
        "over-the-counter pain relievers": "Non-prescription medications to alleviate pain and reduce fever.",
        "prescription medications": "Drugs prescribed by a healthcare professional to treat specific conditions.",
        "lifestyle changes": "Modifications to daily habits that can improve overall health and manage symptoms.",
        "dietary adjustments": "Changes to eating habits that may help manage symptoms or improve overall health.",
        "physical therapy": "Exercises and treatments designed to restore movement and function.",
        "surgery in severe cases": "Medical procedures that may be necessary for advanced or complicated conditions.",
        "counseling or therapy": "Mental health support and interventions to address emotional or psychological issues.",
        "antibiotics or antivirals": "Medications that target and eliminate bacterial or viral infections.",
        "immunotherapy": "Treatment that boosts the body's immune response to fight diseases.",
        "chemotherapy or radiation": "Cancer treatments that target and destroy cancer cells.",
        "alternative or complementary therapies": "Non-conventional treatments or practices used in conjunction with traditional medicine.",
        "home remedies": "Natural or DIY treatments that may help alleviate symptoms or support healing.",
        "medical procedures": "Interventions or surgeries performed by healthcare professionals to diagnose or treat conditions.",
        "self-care practices": "Personal habits and routines that promote health and well-being.",
        "medication management": "Monitoring and adjusting medication use to ensure safety and effectiveness.",
        "physical rehabilitation": "Therapies and exercises to restore physical function and mobility.",
        "stress management techniques": "Strategies to reduce and cope with stress for improved mental and physical health.",
        "allergy testing and treatment": "Tests and interventions to identify and manage allergic reactions.",
        "hydration and electrolyte balance": "Maintaining proper fluid levels and electrolyte balance for optimal health.",
        "wound care and infection control": "Proper treatment and prevention of wounds to avoid complications or infections.",
        "respiratory support": "Assistance with breathing for individuals with respiratory conditions or difficulties.",
        "nutritional counseling": "Guidance on healthy eating habits and dietary choices for improved well-being.",
        "pain management strategies": "Approaches to alleviate and manage pain for better quality of life.",
        "physical activity recommendations": "Guidelines and suggestions for safe and effective exercise routines.",
        "medication adherence": "Following prescribed medication regimens to ensure treatment effectiveness.",
        "monitoring and tracking symptoms": "Keeping records of symptoms to aid in diagnosis and treatment planning.",
        "sleep hygiene practices": "Establishing healthy sleep habits and routines for better rest and recovery.",
        "infection prevention measures": "Precautions and practices to reduce the risk of infections and illnesses.",
        "mental health support": "Therapies and interventions to address emotional and psychological well-being.",
        "lifestyle modifications": "Changes to daily habits and routines to promote health and well-being.",
        "preventive care strategies": "Approaches to prevent or reduce the risk of developing certain conditions.",
        "medical consultation and follow-up": "Seeking professional medical advice and continuing care for optimal health.",
        "symptom management techniques": "Strategies to alleviate and cope with specific symptoms or discomforts.",
        "immune system support": "Boosting and maintaining a healthy immune system for overall wellness.",
        "diagnostic testing and monitoring": "Tests and evaluations to diagnose conditions and track progress.",
        "treatment adherence and compliance": "Following treatment plans and recommendations for optimal outcomes.",
        "emotional and social support": "Assistance and resources for addressing emotional and social needs.",
        "health education and information": "Knowledge and guidance on health-related topics for informed decision-making.",
        "rehabilitation and recovery programs": "Structured plans and therapies to aid in recovery and healing.",
        "medication management and side effects": "Monitoring and addressing medication effects and interactions.",
        "self-care and wellness practices": "Personal habits and activities that promote health and well-being.",
        "stress reduction techniques": "Strategies and activities to manage and reduce stress levels.",
        "dietary and nutritional guidance": "Recommendations and information on healthy eating habits and nutrition.",
        "exercise and physical activity plans": "Structured routines and programs to improve fitness and overall health.",
        "sleep and relaxation techniques": "Practices and strategies to improve sleep quality and promote relaxation.",
        "medical treatment options": "Available interventions and therapies for managing and treating specific conditions.",
        "preventive health measures": "Actions and behaviors to prevent illness and maintain well-being.",
        "symptom relief strategies": "Approaches and remedies to alleviate specific symptoms or discomforts.",
        "health monitoring and management": "Tracking and managing health indicators for optimal well-being.",
        "mental and emotional well-being": "Support and resources for addressing mental and emotional health needs."
        
    }
    
    precautions = {
        "maintain good hygiene": "Regular handwashing and cleanliness to prevent spread of infections.",
        "get adequate sleep": "Ensuring sufficient rest to support the body's natural healing processes.",
        "eat a balanced diet": "Consuming a variety of nutrients to support overall health and immune function.",
        "exercise regularly": "Engaging in physical activity to improve overall health and boost immunity.",
        "avoid stress": "Managing stress levels through relaxation techniques or lifestyle changes.",
        "follow doctor's advice": "Adhering to medical recommendations for optimal treatment outcomes.",
        "attend regular check-ups": "Scheduling routine medical visits for preventive care and early detection.",
        "avoid exposure to triggers": "Identifying and avoiding factors that may worsen symptoms or cause flare-ups.",
        "stay hydrated": "Drinking enough fluids to maintain proper hydration and bodily functions.",
        "practice self-care": "Engaging in activities that promote mental, emotional, and physical well-being.",
        "limit alcohol consumption": "Moderating alcohol intake to reduce health risks and complications.",
        "avoid smoking or tobacco": "Quitting or avoiding tobacco products for better health and well-being.",
        "protect against infections": "Taking precautions to prevent exposure to infectious agents.",
        "manage chronic conditions": "Following treatment plans and lifestyle modifications for ongoing health management.",
        "seek medical attention": "Consulting a healthcare professional for symptoms that persist or worsen.",
        "maintain a healthy weight": "Managing weight through diet and exercise for overall health.",
        "practice good oral hygiene": "Taking care of teeth and gums to prevent dental issues and infections.",
        "limit exposure to allergens": "Reducing contact with substances that trigger allergic reactions.",
        "wear protective gear": "Using appropriate equipment to prevent injuries or exposure to hazards.",
        "avoid sharing personal items": "Preventing the spread of infections by refraining from sharing personal belongings."

    }
    # Select a random subset of symptoms, treatments, and precautions
    num_symptoms = min(4, len(symptoms))
    num_treatments = min(3, len(treatments))
    num_precautions = min(3, len(precautions))

    selected_symptoms = random.sample(list(symptoms.items()), num_symptoms)
    selected_treatments = random.sample(list(treatments.items()), num_treatments)
    selected_precautions = random.sample(list(precautions.items()), num_precautions)
    
    # Generate the disease information
    info = f"""
    ### Common Symptoms:

    {'\n\n '.join([f"**{symptom}**: {description}" for symptom, description in selected_symptoms])}

    ### Possible Treatments:

    {'\n\n '.join([f"**{treatment}**: {description}" for treatment, description in selected_treatments])}

    ### Precautions:

    {'\n\n '.join([f"**{precaution}**: {description}" for precaution, description in selected_precautions])}
    """
    return info

# Streamlit app
st.title('Disease Prediction System')

# Input symptoms
symptoms = st.multiselect('Select Symptoms:', data.columns.tolist()[:-1])

# Predict disease
if st.button('Predict'):
    # Create a DataFrame with the input symptoms
    input_data = pd.DataFrame(np.zeros((1, len(data.columns) - 1)), columns=data.columns[:-1])
    for symptom in symptoms:
        input_data[symptom] = 1
    
    # Predict the disease
    predicted_disease = le.inverse_transform([model.predict(input_data)[0]])
    proba = model.predict_proba(input_data)[0]
    top_5_idx = np.argsort(proba)[::-1][:5]
    
    # Display results
    st.header("Most Likely Disease")
        
    # Primary Prediction
    st.markdown(f"## **{predicted_disease[0]}**")
    st.markdown(get_wikipedia_summary(predicted_disease[0], sentences=4))

    # Top 5 Possible Diseases
    st.markdown("## Expected More Possible Diseases:")
    for idx in top_5_idx:
        disease = le.inverse_transform([idx])[0]
        probability = proba[idx]
        st.markdown(f"### {disease}")
        st.markdown(get_wikipedia_summary(disease, sentences=2))
        st.write(f"**Probability:** {probability:.2%}")
    
    # Create a bar chart
    fig, ax = plt.subplots()
    diseases = [le.inverse_transform([idx])[0] for idx in top_5_idx]
    probabilities = [proba[idx] for idx in top_5_idx]
    ax.barh(diseases, probabilities)
    ax.set_xlabel('Probability')
    ax.set_title('Expected More Possible Diseases')
    st.pyplot(fig)

    # Generate and display disease information
    disease_info = generate_disease_info(predicted_disease[0])
    st.markdown(disease_info)

    st.caption("Note: This prediction is based on the symptoms provided and should not be considered as a definitive diagnosis. The information provided is generated for educational purposes and may not be completely accurate or comprehensive. Always consult with a qualified healthcare professional for proper medical advice and treatment.")
    st.caption("Please consult a qualified healthcare professional for personalized advice and treatment.")