import sys
import unittest
from unittest.mock import MagicMock, patch

class TestAudioResampling(unittest.TestCase):
    def setUp(self):
        # We need to mock dependencies for every test or setup class
        # because we need to import app.main inside the mocked environment

        # Create mocks
        self.mock_numpy = MagicMock()
        self.mock_numpy.int16 = "int16"
        self.mock_numpy.float32 = "float32"

        self.mock_torch = MagicMock()
        self.mock_torchaudio = MagicMock()
        self.mock_resemblyzer = MagicMock()
        self.mock_fastapi = MagicMock()
        self.mock_pydantic = MagicMock()
        self.mock_uvicorn = MagicMock()
        self.mock_dotenv = MagicMock()
        self.mock_app_gemini = MagicMock()
        self.mock_app_speaker_id = MagicMock()

        # Patch sys.modules to inject mocks
        self.modules_patcher = patch.dict(sys.modules, {
            "numpy": self.mock_numpy,
            "torch": self.mock_torch,
            "torchaudio": self.mock_torchaudio,
            "torchaudio.transforms": MagicMock(),
            "resemblyzer": self.mock_resemblyzer,
            "fastapi": self.mock_fastapi,
            "fastapi.staticfiles": MagicMock(),
            "fastapi.templating": MagicMock(),
            "fastapi.requests": MagicMock(),
            "pydantic": self.mock_pydantic,
            "uvicorn": self.mock_uvicorn,
            "dotenv": self.mock_dotenv,
            "app.gemini": self.mock_app_gemini,
            "app.speaker_id": self.mock_app_speaker_id
        })
        self.modules_patcher.start()

        # Ensure app.main is imported (or re-imported) within the patched environment
        if 'app.main' in sys.modules:
            del sys.modules['app.main']
        import app.main
        self.app_main = app.main

        # Reset global_resampler
        self.app_main.global_resampler = None

    def tearDown(self):
        # Stop patcher to restore sys.modules
        self.modules_patcher.stop()

        # Clean up app.main from sys.modules to avoid pollution
        if 'app.main' in sys.modules:
            del sys.modules['app.main']

    def test_empty_input(self):
        """Test that empty input returns empty bytes."""
        result = self.app_main.resample_audio_sync(b"")
        self.assertEqual(result, b"")

    def test_none_input(self):
        """Test that None input acts like empty input."""
        result = self.app_main.resample_audio_sync(None)
        self.assertEqual(result, b"")

    def test_no_resampler_initialized(self):
        """Test that if global_resampler is None, original bytes are returned."""
        self.app_main.global_resampler = None
        input_bytes = b"some_audio_data"
        result = self.app_main.resample_audio_sync(input_bytes)
        self.assertEqual(result, input_bytes)

    def test_resampling_success(self):
        """Test successful resampling flow."""
        input_bytes = b"raw_audio"
        expected_output = b"resampled_audio"

        # Mock objects
        mock_waveform_np = MagicMock()
        mock_waveform_float = MagicMock()
        mock_waveform_tensor = MagicMock()
        mock_waveform_unsqueezed = MagicMock()
        mock_resampled_tensor = MagicMock()
        mock_resampled_squeezed = MagicMock()
        mock_resampled_np = MagicMock()
        mock_resampled_int16 = MagicMock()

        # Chain configuration
        self.mock_numpy.frombuffer.return_value = mock_waveform_np
        mock_waveform_np.astype.return_value = mock_waveform_float

        self.mock_torch.from_numpy.return_value = mock_waveform_tensor
        mock_waveform_tensor.unsqueeze.return_value = mock_waveform_unsqueezed

        # Mock global_resampler
        mock_resampler = MagicMock(return_value=mock_resampled_tensor)
        self.app_main.global_resampler = mock_resampler

        mock_resampled_tensor.squeeze.return_value = mock_resampled_squeezed
        mock_resampled_squeezed.numpy.return_value = mock_resampled_np
        mock_resampled_np.astype.return_value = mock_resampled_int16
        mock_resampled_int16.tobytes.return_value = expected_output

        # Execute
        result = self.app_main.resample_audio_sync(input_bytes)

        # Verify result
        self.assertEqual(result, expected_output)

        # Verify calls
        self.mock_numpy.frombuffer.assert_called_with(input_bytes, dtype=self.mock_numpy.int16)
        mock_waveform_np.astype.assert_called_with(self.mock_numpy.float32)
        self.mock_torch.from_numpy.assert_called_with(mock_waveform_float)
        mock_waveform_tensor.unsqueeze.assert_called_with(0)
        mock_resampler.assert_called_with(mock_waveform_unsqueezed)
        mock_resampled_tensor.squeeze.assert_called_with(0)
        mock_resampled_squeezed.numpy.assert_called()
        mock_resampled_np.astype.assert_called_with(self.mock_numpy.int16)
        mock_resampled_int16.tobytes.assert_called()

    def test_resampling_exception(self):
        """Test that exceptions during resampling are caught and original bytes returned."""
        input_bytes = b"raw_audio"

        # Setup mock resampler to raise an exception
        mock_resampler = MagicMock(side_effect=RuntimeError("Resampling failed"))
        self.app_main.global_resampler = mock_resampler

        # Ensure earlier steps don't fail
        self.mock_numpy.frombuffer.return_value = MagicMock()
        self.mock_torch.from_numpy.return_value = MagicMock()

        # Execute
        result = self.app_main.resample_audio_sync(input_bytes)

        # Verify fallback to original input
        self.assertEqual(result, input_bytes)

if __name__ == "__main__":
    unittest.main()
