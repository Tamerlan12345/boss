import pytest
import asyncio

# Logic to be tested (extracted/simulated from app/main.py)
async def process_audio_stream(mode, state, gemini_stream):
    """
    Simulates the audio processing loop in app/main.py send_to_client
    """
    output_audio = []
    output_logs = []

    # NEW: Summary buffer
    summary_audio_buffer = bytearray()

    is_silenced = False

    # Simulate the loop
    async for chunk in gemini_stream:
        if isinstance(chunk, bytes):
            # FIX: Reset silence on new audio
            is_silenced = False

            # --- Logic from app/main.py ---
            should_buffer = False
            should_send = True

            if mode == "speaker":
                if not state.get("speaker_active", False):
                    if state.get("processing_summary", False):
                        should_buffer = True
                        should_send = False
                    else:
                        # Discard
                        should_send = False

            elif mode == "panel":
                if not state.get("speaking_enabled", True):
                    should_send = False

            if is_silenced:
                should_send = False

            # Assume resampling is mocked (just use chunk directly)
            resampled_audio = chunk

            if should_buffer:
                summary_audio_buffer.extend(resampled_audio)
                continue

            if should_send:
                output_audio.append(resampled_audio)
            # -----------------------------

        elif isinstance(chunk, str):
            if "[SILENCE]" in chunk:
                is_silenced = True

                # --- Logic from app/main.py ---
                if mode == "speaker":
                    # Mark summary generation as done
                    if state.get("processing_summary"):
                        state["processing_summary"] = False

                    # If Active + Summary Buffer -> Play it
                    # Note: We play buffer even if silenced is True (because we are silencing *incoming* stream)
                    if state.get("speaker_active") and len(summary_audio_buffer) > 0:
                        output_audio.append(bytes(summary_audio_buffer))
                        summary_audio_buffer.clear()
                # -----------------------------

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

def test_summary_buffer_playback_after_active_intro():
    async def run():
        # Setup
        mode = "speaker"
        # Start in Passive, Generating Summary
        state = {"speaker_active": False, "processing_summary": True}

        async def mock_stream():
            # 1. Summary Audio (Passive)
            yield b'summary_part1'
            yield b'summary_part2'
            yield "[SILENCE]" # End of summary generation

            # Simulate User clicking "Active" externally
            # In real code, this happens via receive loop modifying shared state
            state["speaker_active"] = True

            # 2. Intro Audio (Active) -> Triggered by [CMD: INTRODUCE]
            yield b'intro_audio'
            yield "[SILENCE]" # End of Intro

            # 3. Normal conversation
            yield b'response_audio'

        audio, logs = await process_audio_stream(mode, state, mock_stream())

        # Expectation:
        # 1. Summary parts NOT emitted directly (buffered).
        # 2. Intro emitted directly (because active=True).
        # 3. Summary buffer emitted AFTER Intro [SILENCE].
        # 4. Response emitted directly.

        # Expected sequence:
        # - b'intro_audio' (from step 2, sent immediately)
        # - b'summary_part1summary_part2' (from step 2 [SILENCE], buffer playback)
        # - b'response_audio' (from step 3)

        assert len(audio) == 3
        assert audio[0] == b'intro_audio'
        assert audio[1] == b'summary_part1summary_part2'
        assert audio[2] == b'response_audio'

        # Verify state reset
        assert state["processing_summary"] is False
    asyncio.run(run())
