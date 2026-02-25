import os
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Clinic AI Backend")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- MODELS ---
class ClinicDataInput(BaseModel): clinics: list
class ChatRequest(BaseModel): message: str; history: list = []
class BookRequest(BaseModel): slot_id: str; time: str; patient_name: str; phone_number: str

# --- DATABASE HELPERS ---
def load_clinic_data():
    file_path = "clinic_data.json"
    if not os.path.exists(file_path): raise HTTPException(status_code=500, detail="Database file not found.")
    with open(file_path, "r") as file: return json.load(file)

def save_clinic_data(data):
    with open("clinic_data.json", "w") as file: json.dump(data, file, indent=4)

def get_10_min_slots(shift):
    slots = []
    fmt = "%H:%M"
    start_str = shift.get("startTime", shift.get("start_time"))
    end_str = shift.get("endTime", shift.get("end_time"))
    if not start_str or not end_str: return slots

    try:
        curr = datetime.strptime(start_str, fmt)
        end = datetime.strptime(end_str, fmt)
        b_start_str = shift.get("breakStart", "0:00")
        b_end_str = shift.get("breakEnd", "0:00")
        b_start = datetime.strptime(b_start_str, fmt) if b_start_str and b_start_str not in ["0:00", "00:00"] else None
        b_end = datetime.strptime(b_end_str, fmt) if b_end_str and b_end_str not in ["0:00", "00:00"] else None
    except ValueError: return slots

    booked_apps = shift.get("booked_appointments", [])
    booked_times = [app.get("time") for app in booked_apps]

    while curr + timedelta(minutes=10) <= end:
        t_str = curr.strftime(fmt)
        is_break = bool(b_start and b_end and (b_start <= curr < b_end))
        if not is_break and t_str not in booked_times: slots.append(t_str)
        curr += timedelta(minutes=10)
    return slots

# --- ADMIN ENDPOINT ---
@app.post("/api/admin/update-db")
async def update_clinic_database(data: ClinicDataInput):
    try:
        save_clinic_data(data.dict())
        return {"status": "success", "message": f"Database updated successfully."}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# --- AI CHAT FLOW ---
@app.post("/api/chat")
async def chat_with_bot(request: ChatRequest):
    user_message = request.message
    
    # NEW: Get the real-world current day and tomorrow's day!
    current_day = datetime.now().strftime("%A").upper()
    tomorrow_day = (datetime.now() + timedelta(days=1)).strftime("%A").upper()
    
    # AI Prompt: Now dynamically injects the real-time day!
    system_prompt = f"""
    You are an AI assistant for a clinic booking system. 
    Look closely at the chat history to carry over context (city, day, time, service) if it's missing from the newest message.
    
    CRITICAL TIME CONTEXT:
    - Today is {current_day}.
    - Tomorrow is {tomorrow_day}.
    
    IMPORTANT EXTRACTION RULES FOR THE NEWEST MESSAGE:
    - intent: "search" or "book"
    - city: Extract the city name. Leave as "" if totally unknown. DO NOT write "Unknown".
    - day: If they ask "which days", "show days", "all days", or "what days", YOU MUST OUTPUT "ALL". 
           If they say "today", output "{current_day}".
           If they say "tomorrow", output "{tomorrow_day}".
           If they specify a single day (e.g., "MONDAY"), output that day. Otherwise "UNKNOWN".
    - time: If they specify a time, extract it in 24-hour HH:MM format (e.g., "08:10", "14:00"). If they ask "all times", use "ALL". Otherwise "UNKNOWN".
    - service: Extract specific doctor types if mentioned (e.g. "Dentist", "General"). Otherwise "unknown".
    - asking_for_doctors: Set to true ONLY IF the NEWEST user message asks "which doctors are available?". Otherwise false.
    - asking_for_cities: Set to true ONLY IF the NEWEST user message asks "which city", "what cities", "where is", or asks for available locations. Otherwise false.
    - patient_name: e.g., "Ahmed" (default "")
    - phone_number: e.g., "03001234567" (default "")

    Return ONLY a valid JSON object matching these keys.
    """
    
    messages_for_ai = [{"role": "system", "content": system_prompt}]
    for msg in request.history:
        role = "assistant" if msg.get("sender") == "bot" else "user"
        text = msg.get("text", "")
        if len(text) > 200: text = text[:200] + "..."
        messages_for_ai.append({"role": role, "content": text})
    messages_for_ai.append({"role": "user", "content": user_message})
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages_for_ai, model="llama-3.1-8b-instant", response_format={"type": "json_object"}, temperature=0
        )
        ai_response = json.loads(chat_completion.choices[0].message.content)
        
        target_city = ai_response.get("city", "").strip().lower()
        if target_city == "unknown" or target_city == "none": target_city = ""
            
        target_day = ai_response.get("day", "UNKNOWN").strip().upper()
        target_time = ai_response.get("time", "UNKNOWN").strip()
        target_service = ai_response.get("service", "UNKNOWN").strip().lower()
        asking_for_doctors = ai_response.get("asking_for_doctors", False)
        asking_for_cities = ai_response.get("asking_for_cities", False)
        p_name = ai_response.get("patient_name", "").strip()
        p_phone = ai_response.get("phone_number", "").strip()
        
        if target_time not in ["UNKNOWN", "ALL"]:
            target_time = target_time.replace(".", ":")
            if len(target_time) == 4 and target_time[1] == ':': 
                target_time = "0" + target_time
                
    except Exception as e:
        return {"status": "error", "message": "My AI brain is temporarily offline.", "slots": []}

    db = load_clinic_data()

    if asking_for_cities:
        available_cities = set()
        for c in db.get("clinics", []):
            if target_service != "unknown" and target_service != "":
                if target_service in c.get("service", "").lower() or c.get("service", "").lower() in target_service:
                    available_cities.add(c.get("city", "").title())
            else:
                available_cities.add(c.get("city", "").title())
        
        if available_cities:
            cities_str = ", ".join(sorted(list(available_cities)))
            s_name = target_service.title() if target_service not in ["unknown", ""] else "clinics"
            return {"status": "success", "message": f"We have {s_name} available in **{cities_str}**. Which city would you like to choose?", "slots": []}
        else:
            return {"status": "success", "message": "I'm sorry, we don't have that service available in any of our cities.", "slots": []}

    if not target_city:
        return {"status": "success", "message": "I'd be happy to help you book an appointment! Which city are you looking for?", "slots": []}

    city_clinics = [c for c in db.get("clinics", []) if c.get("city", "").lower() == target_city]
    
    if not city_clinics:
        return {"status": "success", "message": f"I apologize, but we currently do not have any available appointments in {target_city.title()}. Please try another location.", "slots": []}

    available_services = list(set([c.get("service", "General Doctor") for c in city_clinics]))

    if asking_for_doctors and target_day != "ALL":
        services_str = ", ".join(available_services)
        if len(available_services) == 1:
            return {"status": "success", "message": f"In {target_city.title()}, we only have **{services_str}** available. Which day and time slot do you prefer?", "slots": []}
        else:
            return {"status": "success", "message": f"In {target_city.title()}, we have the following available: **{services_str}**. Which one would you like to book?", "slots": []}

    if len(available_services) > 1 and target_service == "unknown" and target_day != "ALL":
        services_str = ", ".join(available_services)
        return {"status": "success", "message": f"We have multiple doctors in {target_city.title()}: **{services_str}**. Which doctor would you like to book?", "slots": []}

    active_clinics = city_clinics
    if target_service != "unknown":
        filtered = [c for c in city_clinics if target_service in c.get("service", "").lower() or c.get("service", "").lower() in target_service]
        if filtered: active_clinics = filtered

    all_city_slots = []
    day_slots = []
    
    for clinic in active_clinics:
        for shift in clinic.get("slots", []):
            if shift.get("available") == False or shift.get("status") == "booked": continue
            
            times = get_10_min_slots(shift)
            for t in times:
                slot_info = {
                    "clinic_name": clinic.get("clinic_name"),
                    "address": clinic.get("streetAddress", clinic.get("address", "")),
                    "service": clinic.get("service", "General"),
                    "slot_id": shift.get("slot_id"),
                    "start_time": t,
                    "day": shift.get("day").upper()
                }
                all_city_slots.append(slot_info)
                if target_day == "ALL" or target_day == "UNKNOWN" or shift.get("day").upper() == target_day:
                    day_slots.append(slot_info)

    if not all_city_slots:
        return {"status": "success", "message": f"I apologize, but all slots are currently booked for that selection.", "slots": []}

    if target_day == "ALL" and target_time != "ALL":
        day_order = {"MONDAY": 1, "TUESDAY": 2, "WEDNESDAY": 3, "THURSDAY": 4, "FRIDAY": 5, "SATURDAY": 6, "SUNDAY": 7}
        unique_days = sorted(list(set([s["day"] for s in all_city_slots])), key=lambda x: day_order.get(x, 8))
        days_str = ", ".join(unique_days) if unique_days else "various days"
        return {
            "status": "success", 
            "message": f"We have appointments available on **{days_str}**. Which day and time slot do you prefer?", 
            "slots": []
        }

    if target_day == "ALL" and target_time == "ALL":
        return {"status": "success", "message": "Here are all the available time slots:", "slots": day_slots}

    if target_day == "UNKNOWN":
        return {"status": "success", "message": "Great! Which day and time slot do you prefer?", "slots": []}

    if target_day != "ALL" and not day_slots:
        return {"status": "success", "message": f"I'm sorry, but we don't have any available slots on {target_day}. Here are all available slots for other days:", "slots": all_city_slots}

    if target_time == "UNKNOWN":
        return {"status": "success", "message": f"Got it, {target_day}. Which time slot do you prefer?", "slots": []}

    if target_time == "ALL":
        return {"status": "success", "message": f"Here are all the available time slots for {target_day}:", "slots": day_slots}

    matched_slot = next((s for s in day_slots if s["start_time"] == target_time), None)

    if not matched_slot:
        return {
            "status": "success",
            "message": f"I'm sorry, the {target_time} slot is not available on {target_day}. However, here are the other available times for that day:",
            "slots": day_slots
        }

    if not p_name or not p_phone:
        return {
            "status": "success",
            "message": f"The {target_time} slot on {target_day} is available! Please tell me your full name and phone number to proceed.",
            "slots": []
        }

    return {
        "status": "requires_confirmation",
        "message": "Perfect! Please review your details and confirm below:",
        "booking_details": {
            "slot_id": matched_slot["slot_id"],
            "time": target_time,
            "day": target_day,
            "patient_name": p_name,
            "phone_number": p_phone,
            "clinic_name": matched_slot["clinic_name"],
            "address": matched_slot["address"]
        }
    }

# --- FINAL CONFIRMATION ENDPOINT ---
@app.post("/api/book")
async def book_slot(request: BookRequest):
    db = load_clinic_data()
    for clinic in db.get("clinics", []):
        for shift in clinic.get("slots", []):
            if shift.get("slot_id") == request.slot_id:
                if "booked_appointments" not in shift: shift["booked_appointments"] = []
                
                if any(b.get("time") == request.time for b in shift["booked_appointments"]):
                    return {"status": "error", "message": "This specific time was just taken! Please select another."}
                
                shift["booked_appointments"].append({
                    "time": request.time, "patient_name": request.patient_name, "phone_number": request.phone_number
                })
                
                try: save_clinic_data(db)
                except Exception: return {"status": "error", "message": "Database write error."}
                    
                return {
                    "status": "success", 
                    "message": f"Your slot of {request.time} in {clinic.get('clinic_name')} location has been successfully booked."
                }
                
    return {"status": "error", "message": "Invalid Slot ID."}

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    temp_file_path = f"temp_{audio.filename}"
    try:
        content = await audio.read()
        if len(content) < 100: return {"status": "error", "message": "Audio data too small."}
        with open(temp_file_path, "wb") as buffer: buffer.write(content)
        with open(temp_file_path, "rb") as file_obj:
            transcription = groq_client.audio.transcriptions.create(file=(temp_file_path, file_obj), model="whisper-large-v3")
        return {"status": "success", "text": transcription.text}
    except Exception as e: return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)