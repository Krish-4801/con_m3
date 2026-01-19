import unittest
from unittest.mock import MagicMock, patch
import logging
from typing import List, Dict, Any

# Adjust path if needed to import from local modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conclave.core.schemas import (
    FaceObservation,
    VoiceObservation,
    VisualObservation,
    MemoryType,
    MemoryNode
)
from conclave.agent.reasoning import ReasoningAgent

class TestReasoningAgent(unittest.TestCase):
    def setUp(self):
        self.config = {"openai_api_key": "fake-key"}
        self.agent = ReasoningAgent(self.config)
        
        # Disable logging for tests
        logging.getLogger("Conclave.ReasoningAgent").setLevel(logging.CRITICAL)

    def test_prepare_m3_context(self):
        # Setup observations
        visuals = [
            VisualObservation(
                video_id="vid1", clip_id=1, ts_ms=1000,
                clip_embedding=[0.1], ocr_tokens=[], detected_objects=[],
                spatial_metadata={"dense_description": "A sunny day in the park."}
            )
        ]
        faces = [
            FaceObservation(
                video_id="vid1", clip_id=1, ts_ms=1000,
                embedding=[0.1], bbox=[0,0,10,10], base64_img="", 
                detection_score=0.9, quality_score=0.9, entity_id="face_abc"
            )
        ]
        voices = [
            VoiceObservation(
                video_id="vid1", clip_id=1, ts_ms=1000,
                embedding=[0.1], asr_text="Hello world", start_sec=1.0, end_sec=2.0,
                entity_id="voice_xyz"
            )
        ]

        context = self.agent._prepare_m3_context(visuals, faces, voices)
        
        # Assertions
        self.assertIn("Visual Scene Summary:", context)
        self.assertIn("A sunny day in the park.", context)
        self.assertIn("Cast (Visual Entities Present):", context)
        self.assertIn("<face_abc>", context)
        self.assertIn("Audio Transcript:", context)
        self.assertIn("<voice_xyz>", context)
        self.assertIn('"Hello world"', context)

    @patch("conclave.agent.reasoning.openai.OpenAI")
    def test_generate_memory_structures(self, mock_openai_cls):
        # Mock the API response
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        
        # Re-init agent to pick up mock
        self.agent = ReasoningAgent(self.config)
        self.agent.client = mock_client
        
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
        {
            "episodic": ["<face_abc> led the way."],
            "semantic": ["Equivalence: <face_abc> is <voice_xyz>"]
        }
        """
        mock_client.chat.completions.create.return_value = mock_response

        # Execute
        memories = self.agent.generate_memory_structures(
            video_id="vid1", clip_id=1, visuals=[], faces=[], voices=[]
        )

        # Verify
        self.assertEqual(len(memories), 2)
        
        episodic = [m for m in memories if m.mem_type == MemoryType.EPISODIC][0]
        self.assertEqual(episodic.content, "<face_abc> led the way.")
        self.assertIn("face_abc", episodic.linked_entities)

        semantic = [m for m in memories if m.mem_type == MemoryType.SEMANTIC][0]
        self.assertEqual(semantic.content, "Equivalence: <face_abc> is <voice_xyz>")
        self.assertIn("face_abc", semantic.linked_entities)
        self.assertIn("voice_xyz", semantic.linked_entities)

if __name__ == "__main__":
    unittest.main()
