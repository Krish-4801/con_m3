import unittest
import torch
import numpy as np
from PIL import Image
from conclave.perception.vision.scene import SceneProcessor
from conclave.perception.audio.voice import VoiceProcessor
from unittest.mock import MagicMock, patch

class TestModelMigration(unittest.TestCase):
    def setUp(self):
        self.config = {"min_duration_for_audio": 0.5}

    @patch("conclave.perception.vision.scene.Florence2ForConditionalGeneration")
    @patch("conclave.perception.vision.scene.AutoProcessor")
    @patch("conclave.perception.vision.scene.SiglipVisionModel")
    @patch("conclave.perception.vision.scene.SiglipProcessor")
    @patch("conclave.perception.vision.scene.YOLO")
    @patch("conclave.perception.vision.scene.easyocr.Reader")
    def test_scene_processor_init(self, mock_ocr, mock_yolo, mock_siglip_proc, mock_siglip_model, mock_vlm_proc, mock_vlm_model):
        processor = SceneProcessor(self.config)
        self.assertTrue(hasattr(processor, "vlm_model"))
        self.assertTrue(hasattr(processor, "_run_vlm_caption_sequential"))
        print("\n[Passed] SceneProcessor initialized with Florence-2.")

    @patch("conclave.perception.audio.voice.torch.hub.load")
    @patch("conclave.perception.audio.voice.EncoderClassifier")
    @patch("whisper.load_model")
    def test_voice_processor_init(self, mock_whisper, mock_sb, mock_vad):
        # Mock torch.hub.load to return a tuple of (model, utils)
        mock_model = MagicMock()
        mock_utils = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
        mock_vad.return_value = (mock_model, mock_utils)
        
        processor = VoiceProcessor(self.config)
        mock_whisper.assert_called_with("tiny", device="cpu")
        print("\n[Passed] VoiceProcessor initialized with Whisper Tiny.")

if __name__ == "__main__":
    unittest.main()
