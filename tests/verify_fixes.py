import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies
qdrant_client_mock = MagicMock()
sys.modules["qdrant_client"] = qdrant_client_mock
sys.modules["qdrant_client.http"] = MagicMock()
sys.modules["qdrant_client.models"] = MagicMock()

# Ad-hoc sys path
sys.path.append(os.getcwd())

from conclave.core.schemas import MemoryNode, MemoryType
# We need to import ConclaveEngine, but we must mock its init to avoid loading real config/dbs
from conclave.core.engine import ConclaveEngine

class TestEngineFixes(unittest.TestCase):
    def setUp(self):
        # Patch the __init__ because it loads files and connects to DBs
        with patch.object(ConclaveEngine, '__init__', return_value=None):
            self.engine = ConclaveEngine("video_1")
            # Manually set up the mocks that __init__ would have set
            self.engine.video_id = "video_1"
            self.engine.vector_store = MagicMock()
            self.engine.graph_store = MagicMock()
            self.engine.collections = {"text": "text_memories"}
            self.engine.embedding_service = MagicMock()
            
            # Setup deterministic ID mock
            self.engine.embedding_service.generate_deterministic_id.side_effect = lambda content: f"hash_{content}"
            self.engine.embedding_service.get_embeddings_batched.return_value = [[0.1, 0.2]]
            self.engine.embedding_service.model = "mock_model"

    def test_add_memory_enforces_deterministic_id(self):
        """Verify add_memory overrides random UUID with deterministic ID."""
        mem = MemoryNode(video_id="v1", content="test_content", mem_type=MemoryType.EPISODIC)
        original_random_id = mem.mem_id
        
        self.engine.add_memory(mem)
        
        # Should be changed to hash_test_content
        self.assertEqual(mem.mem_id, "hash_test_content")
        self.assertNotEqual(mem.mem_id, original_random_id)
        print("\n[Passed] add_memory enforced deterministic ID.")

    def test_semantic_reinforcement_links_entities(self):
        """Verify _ingest_semantic_weighted links new entities to existing node."""
        mem = MemoryNode(
            video_id="v1", 
            content="semantic_content", 
            mem_type=MemoryType.SEMANTIC,
            linked_entities=["new_entity_1"]
        )
        
        # Simulate Vector Search Hit
        hit = MagicMock()
        hit.id = "existing_mem_id"
        hit.score = 0.90 # > 0.85 threshold
        self.engine.vector_store.search.return_value = [hit]
        
        # Run
        self.engine._ingest_semantic_weighted([mem])
        
        # Verify Reinforcement
        self.engine.graph_store.reinforce_node.assert_called_with("existing_mem_id", delta=1.0)
        
        # Verify Linking (THE FIX)
        self.engine.graph_store.link_memory_to_entity.assert_called_with("existing_mem_id", "new_entity_1", "MENTIONS")
        print("\n[Passed] _ingest_semantic_weighted linked new entities.")

if __name__ == '__main__':
    unittest.main()
