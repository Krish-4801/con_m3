import logging
import queue
import threading
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

logger = logging.getLogger("Conclave.GraphStore")

class GraphStore:
    """
    Neo4j Backend for M3-Agent Logic.
    Replaces the in-memory 'VideoGraph' class from ByteDance's implementation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.driver = GraphDatabase.driver(
            config["uri"], 
            auth=(config["user"], config["password"])
        )
        
        # 🚀 Async Writer (Non-blocking)
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._async_worker, daemon=True)
        self.worker_thread.start()
        
        # Initialize Schema Constraints on startup
        self._init_constraints()

    def close(self):
        self.write_queue.join()
        self.driver.close()

    def _init_constraints(self):
        """M3-Specific Schema Constraints"""
        queries = [
            # Ensure unique IDs for all base types
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            
            # Index Clip IDs for fast temporal lookup
            "CREATE INDEX clip_lookup IF NOT EXISTS FOR (c:Clip) ON (c.id, c.video_id)",
            
            # 🔥 M3 Optimization: Index Memory Types for fast Equivalence search
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

    # -------------------------------------------------------------------------
    # READ METHODS (Synchronous - Needed for Logic)
    # -------------------------------------------------------------------------

    def run_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            return session.run(query, parameters).data()

    def get_connected_nodes(self, node_id: str, node_types: List[str] = None) -> List[str]:
        """
        🔥 M3 CORE LOGIC PORT: Equivalent to VideoGraph.get_connected_nodes
        Finds all nodes connected to a given node, optionally filtering by type.
        Used extensively for Retrieval and Context Generation.
        """
        # We look for relationships in both directions:
        # (Node)-[]-(Target)
        
        type_filter = ""
        if node_types:
            # Construct Cypher filter, e.g., "WHERE t.type IN ['episodic', 'semantic']"
            # Note: In our graph, Faces/Voices are Entities with property 'type', 
            # Memories are nodes with property 'type'.
            quoted_types = [f"'{t}'" for t in node_types]
            type_filter = f"WHERE t.type IN [{', '.join(quoted_types)}]"

        query = f"""
        MATCH (n {{id: $node_id}})-[r]-(t)
        {type_filter}
        RETURN t.id as id
        """
        results = self.run_query(query, {"node_id": node_id})
        return [r['id'] for r in results]

    # -------------------------------------------------------------------------
    # WRITE METHODS (Async / Fire-and-Forget)
    # -------------------------------------------------------------------------

    def execute_async(self, query: str, parameters: Dict[str, Any] = None):
        """Generic async write."""
        self.write_queue.put((query, parameters))

    def create_entity_node(self, entity_id: str, entity_type: str, video_id: str):
        """Creates Face, Voice, or Character nodes."""
        query = """
        MERGE (e:Entity {id: $entity_id})
        ON CREATE SET e.type = $type, e.video_id = $video_id
        """
        self.write_queue.put((query, {"entity_id": entity_id, "type": entity_type, "video_id": video_id}))

    def create_memory_node(self, mem_id: str, content: str, mem_type: str, video_id: str, clip_id: int):
        """Creates Episodic or Semantic memory nodes."""
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        MERGE (m:Memory {id: $mem_id})
        ON CREATE SET m.content = $content, m.type = $mem_type, m.video_id = $video_id
        MERGE (c)-[:HAS_MEMORY]->(m)
        """
        self.write_queue.put((query, {
            "mem_id": mem_id, "content": content, "mem_type": mem_type,
            "video_id": video_id, "clip_id": clip_id
        }))

    def link_memory_to_entity(self, mem_id: str, entity_id: str):
        """Links a Memory (text) to an Entity (Face/Voice/Character)."""
        query = """
        MATCH (m:Memory {id: $mem_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (m)-[:MENTIONS]->(e)
        """
        self.write_queue.put((query, {"mem_id": mem_id, "entity_id": entity_id}))

    def create_appearance_link(self, entity_id: str, clip_id: int, video_id: str, ts_ms: int, obs_id: str):
        """Links Entity to Clip (Temporal occurrence)."""
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
