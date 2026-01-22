import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import models
from conclave.core.schemas import (
    FaceObservation, VoiceObservation, VisualObservation, 
    MemoryNode, MemoryType, EntityType
)
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore
from conclave.core.embedding_service import EmbeddingService
from conclave.core.identity import IdentityManager

logger = logging.getLogger("Conclave.Engine")

# --- junk filter patterns ---
JUNK_PATTERNS = [
    r"was visible at",
    r"appeared in the frame",
    r"detected at",
    r"at \d+ms",
    r"confidence:",
    r"box:",
    r"\[\d+ms\]",
    r"^Scene$",
]

class ConclaveEngine:
    def __init__(self, video_id: str, config_path: str = "configs/api_config.json"):
        with open(config_path) as f:
            self.api_config = json.load(f)
            
        self.video_id = video_id
        self.vector_store = VectorStore(self.api_config["qdrant"])
        self.graph_store = GraphStore(self.api_config["neo4j"])
        
        # New Batched Embedding Service
        self.embedding_service = EmbeddingService(self.api_config.get("text-embedding-3-small", {}))
        
        # Collections defined in init_qdrant.py
        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories",
            "visual": "visual_memories",
            "text": "text_memories"
        }

    def apply_memory_decay(self, decay_factor: float = 0.95, prune_threshold: float = 0.1):
        """
        M3 Feature: Evolving Representations.
        Applies time-based decay to semantic memories and deletes those that fall below a threshold.
        This simulates 'forgetting' unused information.
        
        Args:
            decay_factor: Multiplier for weight (e.g. 0.95 = 5% decay per run/hour).
            prune_threshold: Weights below this are deleted.
        """
        logger.info(f"⏳ Running Memory Decay (Factor: {decay_factor}, Threshold: {prune_threshold})")
        
        # 1. Flush async writes to ensure graph is up to date
        self.graph_store.flush()
        
        # 2. Execute Graph Logic (Pruning happens inside Neo4j)
        # Returns IDs of nodes that were deleted from the graph
        deleted_ids = self.graph_store.prune_decayed_memories(
            self.video_id, 
            decay_factor, 
            prune_threshold
        )
        
        if deleted_ids:
            logger.info(f"🗑️ Forgetting {len(deleted_ids)} decayed memories...")
            
            # 3. Sync Delete from Qdrant
            self.vector_store.delete(self.collections["text"], deleted_ids)
            logger.info("✅ Cleanup Complete.")
        else:
            logger.info("✅ No memories decayed below threshold.")

    def ingest_face(self, obs: FaceObservation):
        """
        Aligned with main.py: Ensures temporal structure exists and 
        pre-creates nodes if necessary.
        """
        # 1. Ensure temporal structure exists in Neo4j
        self.graph_store.create_clip_structure(self.video_id, obs.clip_id)
        
        # 2. Ensure entity node exists (Idempotent)
        if obs.entity_id:
            self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, self.video_id)
        else:
            logger.warning(f"Face observation {obs.obs_id} missing entity_id - should be resolved by IdentityManager first")

    def add_memory(self, mem: MemoryNode):
        """
        Interface Parity: Handles the MemoryNode output from ReasoningEngine.
        Uses deterministic hashing for mem_id to prevent duplicates.
        """
        # Generate deterministic ID and embedding if not already present
        if not mem.embedding:
            mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)
            mem.embedding = self.embedding_service.get_embeddings_batched([mem.content])[0]
        else:
             # Always enforce deterministic ID to match batch ingestion
            mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)

        # 1. Idempotent Graph Write
        self.graph_store.create_memory_node(
            mem.mem_id, mem.content, mem.mem_type.value, 
            self.video_id, mem.clip_id
        )

        # 2. Link mentioned entities (GraphRAG Backbone)
        for entity_id in mem.linked_entities:
            logger.info(f"Linking Memory {mem.mem_id[:8]} -> Entity {entity_id[:8]}")
            self.graph_store.link_memory_to_entity(mem.mem_id, entity_id, "MENTIONS")

        # 3. Idempotent Vector Write
        payload = {
            "video_id": self.video_id,
            "clip_id": mem.clip_id,
            "content": mem.content,
            "type": mem.mem_type.value,
            "model": self.embedding_service.model,
            "entities": mem.linked_entities
        }
        if self._is_junk(mem.content):
            logger.debug(f"🗑️ Skipping junk memory: {mem.content}")
            return

        self.vector_store.upsert(self.collections["text"], mem.mem_id, mem.embedding, payload)

    def add_memories_batched(self, memories: List[MemoryNode]):
        """
        Principal-level batched ingestion with deduplication and validation.
        """
        if not memories:
            return
        
        # 1. Deduplicate locally based on content + filter out nodes already in Qdrant
        unique_nodes = []
        texts_to_embed = []
        
        for mem in memories:
            if self._is_junk(mem.content):
                continue
            
            # Generate deterministic ID based on content
            mem.mem_id = self.embedding_service.generate_deterministic_id(mem.content)
            
            # Check Qdrant for existence (The 'No Local Storage' deduplication)
            if not self.vector_store.point_exists(self.collections["text"], mem.mem_id):
                unique_nodes.append(mem)
                texts_to_embed.append(mem.content)
        
        if not unique_nodes:
            logger.info("All memories in batch already exist. Skipping.")
            return
        
        # 3. Batch Embedding Call
        embeddings = self.embedding_service.get_embeddings_batched(texts_to_embed)
        
        # 4. Prepare Vector Payloads
        qdrant_points = []
        for node, emb in zip(unique_nodes, embeddings):
            node.embedding = emb
            
            # Vector Payload
            payload = {
                "video_id": self.video_id,
                "clip_id": node.clip_id,
                "content": node.content,
                "type": node.mem_type.value,
                "model": self.embedding_service.model,
                "entities": node.linked_entities
            }
            qdrant_points.append(models.PointStruct(id=node.mem_id, vector=emb, payload=payload))
            
        # 5. Transactional Integrity: Vectors FIRST (Blocking)
        # If this fails, we don't create ghost nodes in the graph.
        try:
            self.vector_store.upsert_batch(self.collections["text"], qdrant_points)
            logger.info(f"Successfully ingested {len(qdrant_points)} new memories to Vector Store.")
        except Exception as e:
            logger.error(f"Vector Store Upsert Failed: {e}. Aborting Graph Write.")
            return

        # 6. Graph Commit (Asynchronous safe)
        for node in unique_nodes:
            self.graph_store.create_memory_node(
                node.mem_id, node.content, node.mem_type.value, 
                self.video_id, node.clip_id
            )
            for entity_id in node.linked_entities:
                self.graph_store.link_memory_to_entity(node.mem_id, entity_id, "MENTIONS")

    def add_memories_m3_style(self, memories: List[MemoryNode]):
        """
        M3-Agent Logic: Graph Reinforcement vs Creation.
        If a semantic memory is very similar to an existing one, reinforce its weight instead of adding a duplicate.
        """
        if not memories:
            return

        """
        M3-Agent Logic: Graph Reinforcement vs Creation.
        Delegates to _ingest_semantic_weighted to ensure consistent logic.
        """
        self._ingest_semantic_weighted(memories)

    def ingest_m3_data(self, memory_data: Dict[str, Any]):
        """
        Main entry point for M3 data ingestion.
        """
        # 1. Handle Equivalences (Merge Entities)
        eqs = memory_data.get("equivalences", [])
        if eqs:
            logger.info(f"Processing {len(eqs)} equivalence statements")
            id_manager = IdentityManager(self.vector_store, self.graph_store, self.api_config)
            
            for eq_text in eqs:
                import re
                pattern = r'<((?:ent_)?(?:face|voice)_[a-zA-Z0-9\-]+)>'
                tags = list(set(re.findall(pattern, eq_text)))
                
                if len(tags) == 2:
                    logger.info(f"Applying equivalence merge: {tags[0]} <-> {tags[1]}")
                    id_manager.merge_identities(tags[0], tags[1], self.video_id)

        # 2. Ingest Episodic (Always new, time-bound)
        self.add_memories_batched(memory_data.get('episodic', []))

        # 3. Ingest Semantic (Reinforcement Logic)
        self._ingest_semantic_weighted(memory_data.get('semantic', []))

    def _ingest_semantic_weighted(self, memories: List[MemoryNode]):
        """
        Aligned M3-Agent Logic: Reinforce OR Weaken based on vector similarity.
        """
        POSITIVE_THRESHOLD = 0.85
        NEGATIVE_THRESHOLD = 0.0 # M3 uses 0.0 for weakening
        
        for mem in memories:
            # 1. Embed content
            embedding = self.embedding_service.get_embeddings_batched([mem.content])[0]

            # 2. Find ALL potentially related nodes (not just top 1)
            hits = self.vector_store.search(
                self.collections["text"], 
                embedding, 
                filter_kv={"video_id": self.video_id, "type": "semantic"},
                limit=5 # Check top 5 for relationships
            )

            create_new_node = True
            
            for hit in hits:
                # Calculate precise cosine similarity manually or trust Qdrant score
                similarity = hit.score 

                # M3 Logic: Check if entities in new memory are subset of existing node entities
                existing_entities = hit.payload.get("entities", [])
                new_entities = mem.linked_entities
                
                # Only reinforce/weaken if they talk about the same entities
                is_subset = all(e in existing_entities for e in new_entities)
                
                if is_subset:
                    if similarity > POSITIVE_THRESHOLD:
                        logger.info(f"♻️ Reinforcing Memory {hit.id} (Sim: {similarity:.2f})")
                        self.graph_store.reinforce_node(hit.id, delta=1.0)
                        
                        # Also link any *new* entities mentioned in this reinforcement branch
                        for entity_id in new_entities:
                            self.graph_store.link_memory_to_entity(hit.id, entity_id, "MENTIONS")
                            
                        create_new_node = False
                        break # Handled
                    
                    elif similarity < NEGATIVE_THRESHOLD:
                        logger.info(f"🔻 Weakening Memory {hit.id} (Sim: {similarity:.2f})")
                        self.graph_store.reinforce_node(hit.id, delta=-1.0)
                        # We still create the new node in this case (contradiction)
                        create_new_node = True

            if create_new_node:
                mem.embedding = embedding
                self.add_memory(mem)

    def get_hybrid_context(self, query_vector: List[float], top_k: int = 5):
        """Perform Vector search then expand context via Graph."""
        # 1. Find the most relevant moments (Qdrant)
        vectors = self.vector_store.search(
            self.collections["text"], 
            query_vector, 
            {"video_id": self.video_id}, 
            limit=top_k
        )
        
        results = []
        for v in vectors:
            # 2. For each moment, find who was involved (Neo4j)
            query = """
            MATCH (m:Memory {id: $mem_id})-[:MENTIONS]->(e:Entity)
            RETURN e.id as id, e.type as type
            """
            entities = self.graph_store.run_query(query, {"mem_id": v.id})
            
            results.append({
                "content": v.payload["content"],
                "clip_id": v.payload["clip_id"],
                "score": v.score,
                "involved_entities": entities
            })
            
        return results

    def _is_junk(self, text: str) -> bool:
        """
        M3 Quality Control: Rejects memories that are just logs or too short.
        """
        if not text or len(text.strip()) < 10:
            return True
            
        import re
        for pattern in JUNK_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
