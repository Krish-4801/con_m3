import logging
import queue
import threading
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

logger = logging.getLogger("Conclave.GraphStore")

class GraphStore:
    def __init__(self, config: Dict[str, Any]):
        self.driver = GraphDatabase.driver(
            config["uri"], 
            auth=(config["user"], config["password"])
        )
        
        # 🚀 Async Writer
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._async_worker, daemon=True)
        self.worker_thread.start()
        
        # Initialize Schema Constraints
        self._init_constraints()

    def close(self):
        self.write_queue.join()
        self.driver.close()

    def flush(self):
        """
        Blocks the main thread until all pending async writes are completed.
        Call this before running complex read queries (like identity linking).
        """
        self.write_queue.join()

    def _init_constraints(self):
        queries = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE INDEX clip_lookup IF NOT EXISTS FOR (c:Clip) ON (c.id, c.video_id)",
            "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.type)"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def _async_worker(self):
        while True:
            task = self.write_queue.get()
            if task is None:
                self.write_queue.task_done()
                break
            query, params = task
            try:
                with self.driver.session() as session:
                    session.run(query, params)
            except Exception as e:
                logger.error(f"Async Graph Write Failed: {e}")
            finally:
                self.write_queue.task_done()

    # --- READ METHODS ---
    def run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return session.run(query, parameters).data()

    def get_connected_nodes(self, node_id: str, node_types: List[str] = None) -> List[str]:
        type_filter = ""
        if node_types:
            quoted_types = [f"'{t}'" for t in node_types]
            type_filter = f"WHERE t.type IN [{', '.join(quoted_types)}]"

        query = f"""
        MATCH (n {{id: $node_id}})-[r]-(t)
        {type_filter}
        RETURN t.id as id
        """
        results = self.run_query(query, {"node_id": node_id})
        return [r['id'] for r in results]

    # --- WRITE METHODS ---
    def execute_async(self, query: str, parameters: Dict[str, Any] = None):
        self.write_queue.put((query, parameters))

    def create_entity_node(self, entity_id: str, entity_type: str, video_id: str):
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET e.type = $type, e.video_id = $video_id
        """
        self.write_queue.put((query, {"entity_id": entity_id, "type": entity_type, "video_id": video_id}))

    def create_memory_node(self, mem_id: str, content: str, mem_type: str, video_id: str, clip_id: int):
        # UPDATED: Initialize last_accessed timestamp
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (m:Memory {id: $mem_id})
        ON CREATE SET 
            m.content = $content, 
            m.type = $mem_type, 
            m.video_id = $video_id, 
            m.weight = 1.0,
            m.last_accessed = timestamp()
        MERGE (c)-[:HAS_MEMORY]->(m)
        """
        self.write_queue.put((query, {
            "mem_id": mem_id, "content": content, "mem_type": mem_type,
            "video_id": video_id, "clip_id": clip_id
        }))

    def reinforce_node(self, mem_id: str, delta: float = 1.0):
        # UPDATED: Update last_accessed on reinforcement
        query = """
        MATCH (m:Memory {id: $mem_id})
        SET m.weight = coalesce(m.weight, 1.0) + $delta,
            m.last_accessed = timestamp()
        """
        self.write_queue.put((query, {"mem_id": mem_id, "delta": delta}))

    def prune_decayed_memories(self, video_id: str, decay_factor: float, threshold: float) -> List[str]:
        """
        Synchronous method to apply time-decay and delete weak memories.
        Returns the list of deleted memory IDs to sync with VectorStore.
        
        Logic:
        1. Calculate time passed since last access.
        2. Apply decay: weight = weight * decay_factor
        3. If weight < threshold, DELETE node and return ID.
        """
        # Note: We filter for 'semantic' usually, as episodic events shouldn't disappear,
        # but the requirement says "Evolving representations", so we might decay older semantic abstractions.
        query = """
        MATCH (m:Memory {video_id: $video_id, type: 'semantic'})
        WHERE m.last_accessed IS NOT NULL
        
        // Calculate hours since last access (timestamp is ms)
        WITH m, (timestamp() - m.last_accessed) / 3600000.0 AS hours_passed
        WHERE hours_passed > 1 // Only decay things older than an hour to avoid thrashing
        
        // Apply Decay
        SET m.weight = m.weight * ($decay_factor)
        
        // Check Threshold
        WITH m
        WHERE m.weight < $threshold
        
        // Prune
        DETACH DELETE m
        RETURN m.id as id
        """
        
        # We must run this synchronously to get the IDs back for Qdrant deletion
        with self.driver.session() as session:
            result = session.run(query, {
                "video_id": video_id, 
                "decay_factor": decay_factor, 
                "threshold": threshold
            }).data()
            
        return [r['id'] for r in result]

    def create_clip_structure(self, video_id: str, clip_id: int):
        query = """
        MERGE (v:Video {id: $video_id})
        MERGE (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (v)-[:HAS_CLIP]->(c)
        """
        self.write_queue.put((query, {"video_id": video_id, "clip_id": clip_id}))

    def link_memory_to_entity(self, mem_id: str, entity_id: str, rel_type: str = "MENTIONS"):
        query = f"""
        MATCH (m:Memory {{id: $mem_id}})
        MATCH (e:Entity {{id: $entity_id}})
        MERGE (m)-[:{rel_type}]->(e)
        """
        self.write_queue.put((query, {"mem_id": mem_id, "entity_id": entity_id}))

    def create_appearance_link(self, entity_id: str, clip_id: int, video_id: str, ts_ms: int, obs_id: str):
        query = """
        MERGE (v:Video {id: $video_id})
        MERGE (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (v)-[:HAS_CLIP]->(c)
        WITH c
        MATCH (e:Entity {id: $entity_id})
        MERGE (e)-[r:APPEARED_IN {obs_id: $obs_id}]->(c)
        ON CREATE SET r.ts_ms = $ts_ms
        """
        self.write_queue.put((query, {
            "entity_id": entity_id, "clip_id": clip_id, 
            "video_id": video_id, "ts_ms": ts_ms, "obs_id": obs_id
        }))
