import asyncio
import json

# Logic to be tested (extracted/simulated from app/main.py send_to_client logic)
async def process_audio_stream(mode, state, gemini_stream, summary_buffer=None):
    """
    Simulates the audio processing loop in app/main.py send_to_client
    (Updated to match new buffering logic)
    """
    if summary_buffer is None:
        summary_buffer = bytearray()

    output_audio = []
    output_logs = []

    is_silenced = False

    # Simulate the loop
    async for chunk in gemini_stream:
        if isinstance(chunk, bytes):
            is_silenced = False
            should_send = True

            # Simulate resampling (pass-through for test)
            resampled_audio = chunk

            # Logic Output Gating
            if mode == "panel":
                if not state.get("speaking_enabled", True):
                    should_send = False

            elif mode == "speaker":
                if state.get("processing_summary"):
                    # Accumulate audio in buffer
                    if resampled_audio:
                        summary_buffer.extend(resampled_audio)
                    should_send = False
                elif not state.get("speaker_active", False) and \
                     not state.get("intro_active", False):
                    should_send = False

            if is_silenced:
                should_send = False

            if should_send and resampled_audio:
                output_audio.append(resampled_audio)

        elif isinstance(chunk, str):
            if "[SILENCE]" in chunk:
                is_silenced = True

                if mode == "speaker":
                    if state.get("processing_summary"):
                        state["processing_summary"] = False
                        output_logs.append(json.dumps({"type": "summary_done"}))

                        # Edge case: If active mode was enabled during generation (simulated check)
                        if state.get("speaker_active") and len(summary_buffer) > 0:
                            output_audio.append(bytes(summary_buffer))
                            summary_buffer.clear()

                    if state.get("intro_active"):
                        state["intro_active"] = False

            else:
                if chunk.strip():
                    is_silenced = False
                output_logs.append(chunk)

    return output_audio, output_logs, summary_buffer

def test_speaker_passive_mode_silence():
    async def run():
        mode = "speaker"
        state = {"speaker_active": False, "processing_summary": False, "intro_active": False}
        async def mock_stream():
            yield b'audio_chunk_1'
            yield "[SILENCE]"

        audio, logs, _ = await process_audio_stream(mode, state, mock_stream())
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

        audio, logs, _ = await process_audio_stream(mode, state, mock_stream())
        assert len(audio) == 1, f"Expected 1 audio chunk, got {len(audio)}"
        print("PASS: test_speaker_active_mode_audio")
    asyncio.run(run())

def test_speaker_passive_intro_command():
    async def run():
        mode = "speaker"
        state = {"speaker_active": False, "processing_summary": False, "intro_active": True}
        async def mock_stream():
            yield b'intro_audio'
            yield "[SILENCE]"
            yield b'ignored_audio' # Should be ignored after silence resets intro_active
            yield "[SILENCE]"

        audio, logs, _ = await process_audio_stream(mode, state, mock_stream())

        assert len(audio) == 1, f"Expected 1 audio chunk (intro), got {len(audio)}"
        assert audio[0] == b'intro_audio'
        assert state["intro_active"] == False, "intro_active should be reset"
        print("PASS: test_speaker_passive_intro_command")
    asyncio.run(run())

def test_speaker_summary_buffering():
    async def run():
        mode = "speaker"
        # Processing summary
        state = {"speaker_active": False, "processing_summary": True, "intro_active": False}
        async def mock_stream():
            yield b'summary_part_1'
            yield b'summary_part_2'
            yield "[SILENCE]"

        summary_buffer = bytearray()
        audio, logs, buf = await process_audio_stream(mode, state, mock_stream(), summary_buffer)

        # Should NOT send audio to client (output_audio empty)
        assert len(audio) == 0, f"Expected 0 sent audio chunks, got {len(audio)}"

        # Buffer should contain data
        assert buf == b'summary_part_1summary_part_2', f"Buffer mismatch: {buf}"

        # processing_summary should be reset
        assert state["processing_summary"] == False

        # logs should contain summary_done
        assert any("summary_done" in log for log in logs), "Missing summary_done message"

        print("PASS: test_speaker_summary_buffering")
    asyncio.run(run())

def test_speaker_summary_flush_on_silence_if_active():
    async def run():
        mode = "speaker"
        # Simulate user activating mode DURING generation (before silence)
        state = {"speaker_active": True, "processing_summary": True, "intro_active": False}
        async def mock_stream():
            yield b'summary_audio'
            yield "[SILENCE]"

        summary_buffer = bytearray()
        audio, logs, buf = await process_audio_stream(mode, state, mock_stream(), summary_buffer)

        # Logic: Buffers audio, then sees SILENCE. Since speaker_active=True, it flushes buffer to audio.
        assert len(audio) == 1, f"Expected 1 flushed audio chunk, got {len(audio)}"
        assert audio[0] == b'summary_audio'
        assert len(buf) == 0, "Buffer should be cleared"

        print("PASS: test_speaker_summary_flush_on_silence_if_active")
    asyncio.run(run())

if __name__ == "__main__":
    test_speaker_passive_mode_silence()
    test_speaker_active_mode_audio()
    test_speaker_passive_intro_command()
    test_speaker_summary_buffering()
    test_speaker_summary_flush_on_silence_if_active()
