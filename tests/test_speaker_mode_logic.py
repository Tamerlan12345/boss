import asyncio

# Logic to be tested (extracted/simulated from app/main.py)
async def process_audio_stream(mode, state, gemini_stream):
    """
    Simulates the audio processing loop in app/main.py send_to_client
    (Updated to match new non-blocking logic for commands)
    """
    output_audio = []
    output_logs = []

    is_silenced = False

    # Simulate the loop
    async for chunk in gemini_stream:
        if isinstance(chunk, bytes):
            is_silenced = False
            should_send = True

            if mode == "speaker":
                # Strict blocking unless active OR command executing
                # Note: Logic must match app/main.py exactly
                if not state.get("speaker_active", False) and \
                   not state.get("intro_active", False) and \
                   not state.get("processing_summary", False):
                    should_send = False

            elif mode == "panel":
                if not state.get("speaking_enabled", True):
                    should_send = False

            if is_silenced:
                should_send = False

            if should_send:
                output_audio.append(chunk)

        elif isinstance(chunk, str):
            if "[SILENCE]" in chunk:
                is_silenced = True

                if mode == "speaker":
                    if state.get("processing_summary"):
                        state["processing_summary"] = False
                    if state.get("intro_active"):
                        state["intro_active"] = False

            else:
                if chunk.strip():
                    is_silenced = False
                output_logs.append(chunk)

    return output_audio, output_logs

def test_speaker_passive_mode_silence():
    async def run():
        mode = "speaker"
        state = {"speaker_active": False, "processing_summary": False, "intro_active": False}
        async def mock_stream():
            yield b'audio_chunk_1'
            yield "[SILENCE]"

        audio, _ = await process_audio_stream(mode, state, mock_stream())
        assert len(audio) == 0, f"Expected 0 audio chunks, got {len(audio)}"
        print("PASS: test_speaker_passive_mode_silence")
    asyncio.run(run())

def test_speaker_active_mode_audio():
    async def run():
        mode = "speaker"
        state = {"speaker_active": True, "processing_summary": False, "intro_active": False}
        async def mock_stream():
            yield b'audio_chunk_1'
            yield "[SILENCE]"

        audio, _ = await process_audio_stream(mode, state, mock_stream())
        assert len(audio) == 1, f"Expected 1 audio chunk, got {len(audio)}"
        print("PASS: test_speaker_active_mode_audio")
    asyncio.run(run())

def test_speaker_passive_intro_command():
    async def run():
        mode = "speaker"
        # Passive but Intro command active
        state = {"speaker_active": False, "processing_summary": False, "intro_active": True}
        async def mock_stream():
            yield b'intro_audio'
            yield "[SILENCE]"
            yield b'ignored_audio' # Should be ignored after silence resets intro_active
            yield "[SILENCE]"

        audio, _ = await process_audio_stream(mode, state, mock_stream())

        # Expect intro audio to pass, but subsequent audio to be blocked
        assert len(audio) == 1, f"Expected 1 audio chunk (intro), got {len(audio)}"
        assert audio[0] == b'intro_audio'
        assert state["intro_active"] == False, "intro_active should be reset"
        print("PASS: test_speaker_passive_intro_command")
    asyncio.run(run())

def test_speaker_passive_summary_command():
    async def run():
        mode = "speaker"
        # Passive but Summary command active
        state = {"speaker_active": False, "processing_summary": True, "intro_active": False}
        async def mock_stream():
            yield b'summary_audio'
            yield "[SILENCE]"
            yield b'ignored_audio'
            yield "[SILENCE]"

        audio, _ = await process_audio_stream(mode, state, mock_stream())

        assert len(audio) == 1, f"Expected 1 audio chunk, got {len(audio)}"
        assert audio[0] == b'summary_audio'
        assert state["processing_summary"] == False, "processing_summary should be reset"
        print("PASS: test_speaker_passive_summary_command")
    asyncio.run(run())

if __name__ == "__main__":
    test_speaker_passive_mode_silence()
    test_speaker_active_mode_audio()
    test_speaker_passive_intro_command()
    test_speaker_passive_summary_command()
