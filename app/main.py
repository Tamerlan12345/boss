from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import os
import asyncio
import logging
from dotenv import load_dotenv
from app.gemini import GeminiClient
from app.speaker_id import SpeakerIdentifier
from resemblyzer import VoiceEncoder

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            "You are Dos, an expert and friendly assistant. "
            "You are listening to a conversation of 4 people. "
            "Your task is to analyze the context but remain silent until you are addressed with phrases "
            "'Dos, answer', 'Dos, are you listening', or similar triggers. "
            "When you answer, address the interlocutor by name if you can determine it. "
            "Be brief, professional, but with a warm tone."
        )
        await gemini_client.connect(system_instruction=system_instruction)

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
                async for audio_chunk in gemini_client.receive():
                    await websocket.send_bytes(audio_chunk)
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
