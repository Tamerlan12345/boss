import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Logic to be tested (extracted/simulated from app/main.py)
async def process_audio_stream(mode, state, gemini_stream):
    """
    Simulates the audio processing loop in app/main.py send_to_client
    """
    output_audio = []
    output_logs = []

    is_silenced = False

    async for chunk in gemini_stream:
        if isinstance(chunk, bytes):
            # NEW LOGIC TO TEST
            if mode == "speaker":
                # Check if active or processing summary
                if not state.get("speaker_active", False) and not state.get("processing_summary", False):
                    continue
            elif mode == "panel":
                if not state.get("speaking_enabled", True):
                    continue

            if is_silenced:
                continue

            # Assume resampling is mocked/skipped
            output_audio.append(chunk)

        elif isinstance(chunk, str):
            if "[SILENCE]" in chunk:
                is_silenced = True
                # NEW LOGIC TO TEST: Reset summary processing
                if mode == "speaker":
                    state["processing_summary"] = False
            else:
                if chunk.strip():
                    is_silenced = False
                output_logs.append(chunk)

    return output_audio, output_logs

def test_speaker_passive_mode_silence():
    async def run():
        # Setup
        mode = "speaker"
        state = {"speaker_active": False, "processing_summary": False}

        # Mock stream: Audio chunks
        async def mock_stream():
            yield b'audio_chunk_1'
            yield b'audio_chunk_2'

        audio, logs = await process_audio_stream(mode, state, mock_stream())

        # Expectation: No audio passed through because speaker is inactive and not summarizing
        assert len(audio) == 0
    asyncio.run(run())

def test_speaker_active_mode_audio():
    async def run():
        # Setup
        mode = "speaker"
        state = {"speaker_active": True, "processing_summary": False}

        async def mock_stream():
            yield b'audio_chunk_1'

        audio, logs = await process_audio_stream(mode, state, mock_stream())

        # Expectation: Audio passed through
        assert len(audio) == 1
        assert audio[0] == b'audio_chunk_1'
    asyncio.run(run())

def test_speaker_passive_mode_summary():
    async def run():
        # Setup
        mode = "speaker"
        state = {"speaker_active": False, "processing_summary": True}

        async def mock_stream():
            yield b'summary_audio_1'
            yield "[SILENCE]" # Should reset processing_summary
            yield b'hallucinated_audio' # Should be blocked now

        audio, logs = await process_audio_stream(mode, state, mock_stream())

        # Expectation:
        # 1. Summary audio passed
        # 2. Silence received -> state updated
        # 3. Hallucinated audio blocked

        assert len(audio) == 1
        assert audio[0] == b'summary_audio_1'
        assert state["processing_summary"] is False
    asyncio.run(run())
