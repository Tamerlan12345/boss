import pytest
import asyncio
import json

# Replicating the logic from app/main.py send_to_client loop for testing
async def process_stream(mock_gemini_stream):
    audio_buffer = bytearray()
    MIN_CHUNK_SIZE = 4096
    is_silenced = False

    sent_audio = []
    sent_logs = []

    async for chunk in mock_gemini_stream:
        if isinstance(chunk, bytes):
            # If silenced, we ignore audio chunks
            if is_silenced:
                continue

            # Mock resampling (pass through)
            resampled_audio = chunk
            if resampled_audio:
                audio_buffer.extend(resampled_audio)

                if len(audio_buffer) >= MIN_CHUNK_SIZE:
                    sent_audio.append(bytes(audio_buffer))
                    audio_buffer.clear()

        elif isinstance(chunk, str):
            # Check for [SILENCE] token
            if "[SILENCE]" in chunk:
                is_silenced = True
                audio_buffer.clear()
            else:
                if chunk.strip():
                    # Received valid text, assume we are not silenced (or new turn)
                    is_silenced = False

                # Send text log to frontend
                log_msg = {"type": "log", "role": "ai", "text": chunk}
                sent_logs.append(log_msg)

    # Flush remaining audio
    if len(audio_buffer) > 0 and not is_silenced:
        sent_audio.append(bytes(audio_buffer))

    return sent_audio, sent_logs

@pytest.mark.asyncio
async def test_silence_token_logic():
    # Scenario 1: Audio BEFORE Silence (Edge case: model generates audio then realizes it should be silent?)
    # If model is perfectly compliant, it sends NO audio.
    # But if it sends audio then [SILENCE], we test if we stop sending subsequent audio.
    async def stream_1():
        yield b'a' * 5000 # Audio > 4096, will be sent immediately
        yield "[SILENCE]"      # Silence command
        yield b'b' * 5000 # More audio (should be ignored)

    audio, logs = await process_stream(stream_1())

    # First audio chunk is sent because we didn't know yet.
    assert len(audio) == 1
    # Second audio chunk is ignored.

    assert len(logs) == 0 # [SILENCE] is not logged

@pytest.mark.asyncio
async def test_normal_speech():
    # Scenario 2: Normal speech
    async def stream_2():
        yield "Hello"
        yield b'a' * 5000
        yield " World"

    audio, logs = await process_stream(stream_2())
    assert len(logs) == 2
    assert logs[0]['text'] == "Hello"
    assert logs[1]['text'] == " World"
    assert len(audio) == 1

@pytest.mark.asyncio
async def test_recovery_from_silence():
    # Scenario 3: Silence then new turn
    async def stream_3():
        yield "[SILENCE]"
        yield b'ignored' * 5000
        yield "New Turn"
        yield b'valid' * 5000

    audio, logs = await process_stream(stream_3())

    assert len(logs) == 1
    assert logs[0]['text'] == "New Turn"
    assert len(audio) == 1 # Only valid audio
    assert audio[0] == b'valid' * 5000
