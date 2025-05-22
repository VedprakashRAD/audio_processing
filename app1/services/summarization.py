import os
import json
from groq import Groq
from dotenv import load_dotenv
from app1.utils.logging_config import logging

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

def generate_summary_with_groq(extracted_info, raw_output=False):

    try:
        prompt = f"""
                You are a smart and helpful assistant. Return only a valid JSON object.
                Strictly do not include any commentary, markdown, or extra text — only the JSON output.
                
                Analyze the conversation carefully and do the following:
                
                1. Identify speaker roles based on behavior and conversation flow, not just names.
                   - The Customer is the one requesting a service, asking for help, or reporting an issue.
                   - The Executive is the one providing assistance, offering a solution, or collecting service details.
                   - Even if names are mentioned, you must determine roles from context, not the name itself.
                   - Use diarized segments to see who speaks when, and what they say, to infer roles.
                   - Label as: "Speaker X: Customer(Name or Unknown)" or "Speaker Y: Executive(Name or Unknown)"
                   - If names are not known or unclear, label them as "Customer" or "Executive" without names based on the context.
                   - If there are more than 2 speakers, assign roles to the relevant Customer and Executive participating in the actual service conversation and also add name if available.
                
                
                2. Extract the following important details from the conversation:
                   - Phone number: Look for any mention of mobile or telephone numbers.
                   - Vehicle information: This includes vehicle number (license plate number, e.g., KA02AB3214, ensuring only alphanumeric characters without hyphens), model (e.g., Creta, Swift), brand (e.g., Hyundai, Maruti), variant (e.g., petrol/diesel), year (if any), color (e.g., white, red), tyre size (e.g., 18-inch alloys)**. Extract as many as possible even if partial.
                   - Service-related details: Include the type of service requested (e.g., tyre change, oil replacement), reported issues (e.g., damage, engine failure), any booking or appointment dates/times, or confirmation of service.
                   - Place of service: If a location or delivery address is mentioned, extract it.

                3. Generate a brief and clear summary of the conversation.
                
                Follow this exact JSON format:
                {{
                    "speaker_roles": {{
                        "Speaker 0": "Customer(...)",
                        "Speaker 1": "Executive(...)"
                    }},
                    "important_details": {{
                        "Phone_number": "...",
                        "Place_of_service": "...",
                        "Vehicle_information": {{
                            "Vehicle_number": "... e.g., KA02AB3214",
                            "Model": "...",
                            "Year": "...",
                            "Color": "..." // etc.
                        }},
                        "Service_related_details": "...",
                        "other_details": "related to vehicle"
                    }},
                    "summary": "..."
                }}
                
                Conversation data:
                - Transcription: {extracted_info.get('transcription')}
                - Segments: {extracted_info.get('segments')}
                """

        # Generate structured response using Groq API
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts key details and summarizes conversations. Provide a JSON response only with speaker roles and names correctly, phone numbers, vehicle details like number plate, model of the vehicle, color, important details, and a short summary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500  
        )

        structured_response = response.choices[0].message.content.strip()
        print(f"Structured Response: {structured_response}")
        
        return json.loads(structured_response) 

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return {
            "speaker_roles": {},
            "important_details": {},
            "summary": "Error generating summary."
        }
    
    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return {
            "speaker_roles": {},
            "important_details": {},
            "summary": "Error generating summary due to connection issues."
        }

