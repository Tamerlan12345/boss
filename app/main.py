from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import os
import requests
import asyncio
import logging
from dotenv import load_dotenv
from app.gemini import GeminiClient
from app.speaker_id import SpeakerIdentifier
from resemblyzer import VoiceEncoder

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SessionRequest(BaseModel):
    quality: str = "medium"
    avatar_name: str
    voice_id: str

class StartSessionRequest(BaseModel):
    session_id: str
    sdp: Dict[str, Any]

class IceCandidateRequest(BaseModel):
    session_id: str
    candidate: Dict[str, Any]

class TaskRequest(BaseModel):
    session_id: str
    text: str

class StopSessionRequest(BaseModel):
    session_id: str

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

global_encoder = None
global_speakers = {}

@app.on_event("startup")
async def startup_event():
    global global_encoder, global_speakers
    try:
        global_encoder = VoiceEncoder()
        logger.info("Global VoiceEncoder loaded")

        # Load speakers
        temp_identifier = SpeakerIdentifier(encoder=global_encoder)
        temp_identifier.load_speakers_from_folder("app/static/samples")
        global_speakers = temp_identifier.speakers
        logger.info(f"Loaded {len(global_speakers)} speakers")

    except Exception as e:
        logger.error(f"Failed to load VoiceEncoder: {e}")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/token")
def get_heygen_token():
    api_key = os.getenv("HEYGEN_API_KEY")
    if not api_key:
        return {"error": "HEYGEN_API_KEY not found"}

    try:
        response = requests.post(
            "https://api.heygen.com/v1/streaming.create_token",
            headers={"x-api-key": api_key}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get HeyGen token: {e}")
        return {"error": str(e)}

@app.get("/avatars")
def get_heygen_avatars():
    api_key = os.getenv("HEYGEN_API_KEY")
    if not api_key:
        return {"error": "HEYGEN_API_KEY not found"}

    try:
        response = requests.get(
            "https://api.heygen.com/v2/avatars",
            headers={"x-api-key": api_key}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get HeyGen avatars: {e}")
        return {"error": str(e)}

@app.get("/voices")
def get_heygen_voices():
    api_key = os.getenv("HEYGEN_API_KEY")
    if not api_key:
        return {"error": "HEYGEN_API_KEY not found"}

    try:
        response = requests.get(
            "https://api.heygen.com/v2/voices",
            headers={"x-api-key": api_key}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get HeyGen voices: {e}")
        return {"error": str(e)}

# Вспомогательная функция для заголовков
def get_auth_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

@app.post("/heygen/session/create")
async def proxy_create_session(request: Request):
    data = await request.json()
    # Клиент должен прислать token в теле или мы получаем его тут
    token = data.get("token")

    try:
        resp = requests.post(
            "https://api.heygen.com/v2/streaming/new",
            headers=get_auth_headers(token),
            json={
                "quality": data.get("quality", "medium"),
                "avatar_name": data.get("avatar_name"),
                "voice": {"voice_id": data.get("voice_id")}
            }
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"HeyGen Create Error: {e}")
        return {"error": str(e)}

@app.post("/heygen/session/start")
async def proxy_start_session(request: Request):
    data = await request.json()
    token = data.get("token")
    try:
        resp = requests.post(
            "https://api.heygen.com/v2/streaming/start",
            headers=get_auth_headers(token),
            json={
                "session_id": data.get("session_id"),
                "sdp": data.get("sdp")
            }
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"HeyGen Start Error: {e}")
        return {"error": str(e)}

@app.post("/heygen/ice")
async def proxy_ice(request: Request):
    data = await request.json()
    token = data.get("token")
    try:
        resp = requests.post(
            "https://api.heygen.com/v2/streaming/ice",
            headers=get_auth_headers(token),
            json={
                "session_id": data.get("session_id"),
                "candidate": data.get("candidate")
            }
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        # ICE ошибки часто не критичны, но логируем
        logger.error(f"HeyGen ICE Error: {e}")
        return {"error": str(e)}

@app.post("/heygen/task")
async def proxy_task(request: Request):
    data = await request.json()
    token = data.get("token")
    try:
        resp = requests.post(
            "https://api.heygen.com/v2/streaming/task",
            headers=get_auth_headers(token),
            json={
                "session_id": data.get("session_id"),
                "text": data.get("text")
            }
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"HeyGen Task Error: {e}")
        return {"error": str(e)}

@app.post("/heygen/session/stop")
async def proxy_stop(request: Request):
    data = await request.json()
    token = data.get("token")
    try:
        resp = requests.post(
            "https://api.heygen.com/v2/streaming/stop",
            headers=get_auth_headers(token),
            json={
                "session_id": data.get("session_id")
            }
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    gemini_client = GeminiClient()
    identifier = SpeakerIdentifier(encoder=global_encoder, speakers=global_speakers)
    current_speaker = None
    loop = asyncio.get_running_loop()

    try:
        # Dos System Instruction
        system_instruction = (
            "You are Dos, an expert and friendly AI assistant. "
            "You are listening to a conversation of a group of up to 8 people. "
            "You must remain absolutely silent unless the input starts explicitly with the name 'Dos'. "
            "Ignore general commands like 'Answer me', 'Are you listening?' if the name 'Dos' is missing. "
            "Allowed patterns: 'Dos, [question]', 'Dos, answer', 'Dos, your opinion'. "
            "If asked 'Who are you?' or 'Introduce yourself', reply exactly: 'Я ИИ спикер Dos'. "
            "When you answer, address the interlocutor by name if you can determine it (e.g., 'Yes, Tamerlan...'). "
            "Be brief, professional, but with a warm tone."
        )
        await gemini_client.connect(system_instruction=system_instruction)

        # Trigger Welcome Message
        # The assistant must initiate the dialogue.
        await gemini_client.send_text('Please say exactly: "Я ИИ спикер Dos. Сегодня я буду вместе с вами разбирать и участвовать в теме обсуждения, которую вы зададите."')

        async def receive_from_client():
            nonlocal current_speaker
            try:
                while True:
                    # Expecting raw PCM bytes
                    data = await websocket.receive_bytes()

                    # Speaker ID (Run in executor to avoid blocking)
                    speaker = await loop.run_in_executor(None, identifier.process_chunk, data)

                    if speaker and speaker != current_speaker:
                        current_speaker = speaker
                        logger.info(f"Speaker changed to: {speaker}")
                        await gemini_client.send_text(f"[User Changed: {speaker}]")

                    await gemini_client.send_audio(data)
            except WebSocketDisconnect:
                logger.info("Client disconnected")
            except Exception as e:
                logger.error(f"Error receiving from client: {e}")

        async def send_to_client():
            try:
                async for chunk in gemini_client.receive():
                    if isinstance(chunk, str):
                        await websocket.send_text(chunk)
                    elif isinstance(chunk, bytes):
                        await websocket.send_bytes(chunk)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")

        await asyncio.gather(receive_from_client(), send_to_client())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await gemini_client.close()
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
