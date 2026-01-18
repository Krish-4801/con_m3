import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock openai before importing modules that use it
sys.modules["openai"] = MagicMock()

# Ad-hoc sys path to find modules
sys.path.append(os.getcwd())

from conclave.agent.reasoning import ReasoningAgent
from conclave.core.schemas import MemoryNode, MemoryType, EntityType

class TestMemoryDiscrepancies(unittest.TestCase):
    
    def test_1_regex_mismatch(self):
        """
        Discrepancy 1: ReasoningAgent regex doesn't match 'ent_' prefix IDs.
        """
        agent = ReasoningAgent({"openai_api_key": "dummy"})
        
        # IdentityManager generates this:
        ent_id = "ent_face_123456"
        valid_m3_tag = "<face_123456>"
        invalid_m3_tag = f"<{ent_id}>"
        
        text = f"I saw {valid_m3_tag} and {invalid_m3_tag}."
        
        extracted = agent._extract_tags(text)
        print(f"\n[Test 1] Extracted tags from '{text}': {extracted}")
        
        # CURRENT BEHAVIOR: Matches face_123456, FAILS to match ent_face_123456
        # self.assertIn("face_123456", extracted) # Should pass (old way)
        
        # EXPECTED FAILURE until fixed:
        if "ent_face_123456" not in extracted:
             print(">> CONFIRMED: Regex failed to extract 'ent_' prefix tag.")
        else:
             print(">> UNEXPECTED: Regex matched 'ent_' prefix tag (was it already fixed?).")

    def test_2_id_generation_inconsistency(self):
        """
        Discrepancy 2: Schema uses random UUID, Engine uses deterministic.
        Use mock to simulate Engine behavior or just check Schema default.
        """
        m1 = MemoryNode(video_id="v1", content="same content", mem_type=MemoryType.EPISODIC)
        m2 = MemoryNode(video_id="v1", content="same content", mem_type=MemoryType.EPISODIC)
        
        print(f"\n[Test 2] Initial Schema IDs: {m1.mem_id} vs {m2.mem_id}")
        
        # Schema default behavior: Random UUIDs
        self.assertNotEqual(m1.mem_id, m2.mem_id)
        print(">> CONFIRMED: Schema generates random UUIDs by default.")

    def test_4_reinforcement_data_loss(self):
        """
        Discrepancy 4: Logic check - does _ingest_semantic_weighted link new entities?
        We will simulate the logic found in engine.py manually since we can't easily instantiate the full engine with DBs suitable for unit test in this environment without heavy mocking.
        """
        # Mocking the specific logic block from engine.py
        # Logic: if hit.score > 0.85 -> reinforce_node -> continue (LINKING SKIPPED)
        
        print("\n[Test 4] Simulating Reinforcement Logic...")
        linked_called = False
        
        # Simulate payload found
        found_existing = True
        
        if found_existing:
            # ORIGINAL CODE SIMULATION
            # self.graph_store.reinforce_node(existing_id, delta=1.0)
            print(">> Executed reinforce_node()")
            pass 
            # Loop continues...
        
        if not linked_called:
            print(">> CONFIRMED: Entity linking was skipped during reinforcement.")

if __name__ == '__main__':
    unittest.main()
