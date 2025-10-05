from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import os
from typing import Optional
from datetime import datetime
import logging
import traceback
from dotenv import load_dotenv
import httpx

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Shyampari Edutech Chatbot", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Grok API with better error handling
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
grok_available = False

def initialize_grok():
    global grok_available
    if GROK_API_KEY:
        logger.info("Grok API key found, initializing...")
        grok_available = True
        return True
    else:
        logger.warning("GROK_API_KEY not found")
        return False

# Initialize Grok on startup
grok_available = initialize_grok()

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    using_ai: bool

# Load knowledge base
def load_knowledge_base():
    try:
        with open('data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info("Knowledge base loaded successfully")
            return data
    except FileNotFoundError:
        logger.error("data.json file not found")
        return create_default_knowledge_base()
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing data.json: {e}")
        return create_default_knowledge_base()

def create_default_knowledge_base():
    """Create a minimal knowledge base if file is not found"""
    return {
        "company_info": {
            "name": "Shyampari Edutech Pvt. Ltd.",
            "location": "Pune, Maharashtra",
            "established": "2017",
            "website": "https://www.shyampariedutech.com",
            "email": "contact@shyampariedutech.com",
            "boards_supported": ["ICSE", "IGCSE", "CBSE", "IB", "A-Level"]
        },
        "services": [
            "One-on-one personalized tutoring",
            "Small group batches (3-5 students)",
            "24/7 coordinator support",
            "Monthly progress reports",
            "Demo classes",
            "Tutor replacement guarantee"
        ],
        "demo_fee": "₹500 (negotiable)",
        "contact_info": {
            "email": "contact@shyampariedutech.com",
            "website": "https://www.shyampariedutech.com"
        }
    }

knowledge_base = load_knowledge_base()

def create_enhanced_prompt(user_message: str) -> str:
    """Create a more effective prompt for Grok with better context"""
    
    # Create a more focused knowledge summary
    knowledge_summary = f"""
COMPANY: Shyampari Edutech Pvt. Ltd. (Est. 2017, Pune)
CONTACT: contact@shyampariedutech.com | https://www.shyampariedutech.com

SERVICES:
• One-on-one & group tutoring (3-5 students)
• Boards: ICSE, IGCSE, CBSE, IB, A-Level
• Online & offline modes
• 24/7 coordinator support
• Monthly progress reports
• Demo classes: ₹500 (negotiable), 20-30 mins

TUTORS:
• STEM backgrounds (Engineering, Pharmacy, BSc-MSc)
• JEE/GATE qualified
• Within 5km radius
• Home tuition/coaching experience

DEMO PROCESS:
1. Coordinator call
2. WhatsApp communication  
3. ₹500 demo fee
4. Teacher profile (2-3 days)
5. Demo class & feedback
6. Enrollment (1/3/6 month packages)

UNIQUE FEATURES:
• Tutor replacement within 2 weeks
• Counselling support
• Assessment-based planning
• Digital resource library
"""

    prompt = f"""You are an AI assistant for Shyampari Edutech, a premium tutoring service in Pune. 

{knowledge_summary}

RESPONSE GUIDELINES:
- Be conversational, helpful, and professional
- Give specific information when possible
- For detailed pricing: mention demo fee but suggest contacting for full structure
- For bookings: encourage contacting coordinator
- Stay focused on educational services
- If unsure about specifics: suggest direct contact
- Keep responses concise but informative

USER QUESTION: "{user_message}"

Provide a helpful, accurate response about Shyampari Edutech's services:"""

    return prompt

async def get_grok_response(user_message: str) -> Optional[str]:
    """Get response from Grok API"""
    if not GROK_API_KEY:
        logger.warning("No Grok API key available")
        return None
    
    try:
        prompt = create_enhanced_prompt(user_message)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROK_API_KEY}"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for Shyampari Edutech, providing information about their tutoring services."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROK_API_URL,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("choices") and len(result["choices"]) > 0:
                    ai_response = result["choices"][0]["message"]["content"].strip()
                    logger.info("Response generated using Grok AI")
                    return ai_response
                else:
                    logger.warning("Empty response from Grok API")
                    return None
            else:
                logger.error(f"Grok API error: {response.status_code} - {response.text}")
                return None
                
    except httpx.TimeoutException:
        logger.error("Grok API request timed out")
        return None
    except Exception as e:
        logger.error(f"Grok API error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def get_smart_fallback_response(message: str) -> str:
    """Enhanced fallback with better pattern matching"""
    message_lower = message.lower()
    
    # Greeting responses
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'namaste', 'good morning', 'good evening', 'good afternoon']):
        return "Hello! 👋 Welcome to Shyampari Edutech. I'm here to help you learn about our premium tutoring services for ICSE, IGCSE, CBSE, IB, and A-Level boards. What would you like to know?"
    
    # Services inquiry
    elif any(phrase in message_lower for phrase in ['service', 'what do you offer', 'what you provide', 'tell me about']):
        return """We offer comprehensive tutoring services:

🎯 **Personalized Learning:**
• One-on-one tutoring with qualified educators
• Small group classes (3-5 students)
• Assessment-based customized learning plans

📚 **Boards Supported:** ICSE, IGCSE, CBSE, IB, A-Level

🌟 **Key Features:**
• 24/7 coordinator support
• Monthly progress reports  
• Demo classes available (₹500)
• Tutor replacement guarantee within 2 weeks

Would you like to know about our demo process or fee structure?"""
    
    # Demo/trial inquiries
    elif any(word in message_lower for word in ['demo', 'trial', 'test', 'sample', 'try']):
        return """📝 **Demo Class Process:**

1️⃣ Initial coordinator call to discuss your needs
2️⃣ WhatsApp communication and location sharing
3️⃣ Demo registration with ₹500 fee (negotiable)
4️⃣ Teacher profile shared within 2-3 days
5️⃣ 20-30 minute demo class
6️⃣ Feedback collection and teacher finalization

Demo fee: **₹500** (subject to negotiation)

Ready to book a demo? Contact our coordinator at contact@shyampariedutech.com"""
    
    # Fee/pricing inquiries
    elif any(word in message_lower for word in ['fee', 'cost', 'price', 'charge', 'expensive', 'affordable', 'money']):
        return """💰 **Fee Structure:**

• **Demo fee:** ₹500 (negotiable)
• **Payment packages:** 1, 3, or 6 months
• **Pricing varies** based on:
  - Subject and grade level
  - Tutoring mode (online/offline)
  - Individual or group sessions

For detailed fee structure tailored to your requirements, please contact:
📧 contact@shyampariedutech.com
🌐 https://www.shyampariedutech.com"""
    
    # Board/curriculum questions
    elif any(word in message_lower for word in ['board', 'curriculum', 'syllabus', 'icse', 'igcse', 'cbse', 'ib', 'a-level']):
        return """📖 **Educational Boards We Support:**

✅ **ICSE** - Indian Certificate of Secondary Education
✅ **IGCSE** - International General Certificate of Secondary Education  
✅ **CBSE** - Central Board of Secondary Education
✅ **IB** - International Baccalaureate
✅ **A-Level** - Advanced Level qualifications

Our tutors provide **board-specific lessons** aligned with exam patterns and focus on concept-driven teaching for strong foundational understanding."""
    
    # Tutor/teacher questions
    elif any(word in message_lower for word in ['tutor', 'teacher', 'educator', 'faculty', 'instructor']):
        return """👨‍🏫 **Our Expert Tutors:**

🎓 **Qualifications:**
• STEM backgrounds (Engineering, Pharmacy, BSc-MSc)
• Many are JEE/GATE qualified
• Proven teaching experience

📍 **Location:** Within 5km of your location for better continuity

💼 **Experience:**
• Home tuitions
• Coaching institutes  
• School teaching

🔄 **Guarantee:** Tutor replacement within 2 weeks if needed

Want to meet our tutors? Book a demo class!"""
    
    # Contact inquiries
    elif any(word in message_lower for word in ['contact', 'reach', 'call', 'phone', 'address', 'location']):
        return """📞 **Get In Touch:**

📧 **Email:** contact@shyampariedutech.com
🌐 **Website:** https://www.shyampariedutech.com
📍 **Location:** Pune, Maharashtra

⏰ **Coordinator Support:** Available 24/7 to assist you with:
• Demo bookings
• Fee inquiries  
• Tutor assignments
• Schedule management

Ready to start your learning journey? Contact us today!"""
    
    # Timing/schedule questions
    elif any(word in message_lower for word in ['time', 'timing', 'schedule', 'when', 'available', 'hours']):
        return """⏰ **Flexible Scheduling:**

🕐 **Class Timings:** Completely flexible according to your convenience
📅 **Available:** 7 days a week
🌅 **Morning, afternoon, or evening** slots available
⚡ **Online & Offline** modes for maximum flexibility

📋 **Scheduling Process:**
• Discuss preferred timings during coordinator call
• Finalize schedule with your assigned tutor
• 24/7 coordinator support for any changes

Want to discuss your preferred schedule? Contact us for a demo!"""
    
    # Location questions
    elif any(word in message_lower for word in ['where', 'location', 'pune', 'maharashtra', 'area', 'nearby']):
        return """📍 **Location & Coverage:**

🏢 **Head Office:** Pune, Maharashtra
🚀 **Service Area:** Pune and surrounding areas
📏 **Tutor Distance:** Within 5km of your location for offline classes
🌐 **Online Classes:** Available anywhere with internet connection

🏠 **Home Tuitions:** We come to your location
💻 **Online Platform:** High-quality virtual classes with interactive tools

Based in Pune? Perfect! We can provide both online and offline services."""
    
    # Positive responses to questions like "really?"
    elif any(word in message_lower for word in ['really', 'sure', 'certain', 'true', 'confirm']):
        return """Absolutely! 🎯 

Shyampari Edutech has been providing quality education services since 2017. Here's what makes us reliable:

✅ **Established Company:** 7+ years of experience
✅ **Qualified Tutors:** STEM graduates, JEE/GATE qualified  
✅ **Proven Results:** Track record of student success
✅ **Professional Support:** 24/7 coordinator assistance
✅ **Quality Assurance:** Demo classes and tutor replacement guarantee

Want to experience it yourself? Book a demo class for just ₹500!

Any specific concerns you'd like me to address?"""
    
    # Generic questions
    else:
        return f"""Thank you for your interest in Shyampari Edutech! 🙏

For specific information about:
• **Tutoring services & subjects**
• **Demo class bookings** 
• **Detailed fee structure**
• **Tutor assignments**

Please contact our coordinator:
📧 **Email:** contact@shyampariedutech.com
🌐 **Website:** https://www.shyampariedutech.com

Our team is available **24/7** to help you with personalized guidance!

Is there anything specific about our services you'd like to know?"""

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    try:
        user_message = chat_message.message.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logger.info(f"User message: {user_message}")
        
        # Try Grok AI first
        if grok_available and GROK_API_KEY:
            ai_response = await get_grok_response(user_message)
            if ai_response:
                return ChatResponse(
                    response=ai_response,
                    timestamp=datetime.now().isoformat(),
                    using_ai=True
                )
        
        # Use enhanced fallback response
        fallback_response = get_smart_fallback_response(user_message)
        logger.info("Using enhanced fallback response")
        
        return ChatResponse(
            response=fallback_response,
            timestamp=datetime.now().isoformat(),
            using_ai=False
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "grok_available": grok_available,
        "knowledge_base_loaded": bool(knowledge_base),
        "grok_model": "grok-beta" if grok_available else None,
        "api_key_present": bool(GROK_API_KEY),
        "version": "1.0.0"
    }

@app.get("/test-grok")
async def test_grok():
    """Test endpoint to verify Grok integration"""
    if not GROK_API_KEY:
        return {"error": "No API key found", "status": "failed"}
    
    try:
        test_response = await get_grok_response("Say 'Grok is working correctly for Shyampari Edutech!' in a friendly way.")
        if test_response:
            return {
                "status": "success",
                "response": test_response,
                "model": "grok-beta"
            }
        else:
            return {
                "status": "error",
                "error": "No response from Grok API"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }

@app.get("/")
async def root():
    return {
        "message": "Shyampari Edutech Chatbot API",
        "version": "1.0.0",
        "ai_status": "grok-active" if grok_available else "fallback",
        "endpoints": {
            "/chat": "POST - Send chat messages",
            "/health": "GET - Health check",
            "/test-grok": "GET - Test Grok integration",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
