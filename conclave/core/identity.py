import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Set
from conclave.core.schemas import FaceObservation, VoiceObservation, EntityType
from conclave.core.vector_store import VectorStore
from conclave.core.graph_store import GraphStore

logger = logging.getLogger("Conclave.Identity")

class IdentityManager:
    """
    🔥 UNIFIED IDENTITY RESOLUTION SYSTEM 🔥
    Merges logic from identity.py and identity_resolver.py into one canonical class.
    Handles:
    - Face/Voice resolution via vector similarity
    - Co-occurrence-based identity linking
    - Deep merge operations (Neo4j + Qdrant historical retagging)
    - Canonical entity index for fast lookups
    """
    
    def __init__(self, vector_store: VectorStore, graph_store: GraphStore, config: Dict[str, Any]):
        """
        SOTA Identity Resolution System with Deep Refactoring capabilities.
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        # Configuration Thresholds
        self.face_threshold = config.get("face_match_threshold", 0.75)
        self.voice_threshold = config.get("voice_match_threshold", 0.82)
        self.co_occurrence_threshold = config.get("min_co_occurrences_to_merge", 3)
        
        # 🔥 UNIFIED: Canonical Mapping (merged from IdentityResolver)
        # entity_id -> { 'type': ..., 'alias': ..., 'canonical_id': ... }
        self.entity_index: Dict[str, Dict[str, Any]] = {}
        
        # M3 Alignment: Character Mappings (Union-Find Cache)
        # character_id -> [face_id_1, voice_id_2, ...]
        self.character_mappings = {} 
        self.reverse_character_mappings = {} # face_id_1 -> character_id

        self.collections = {
            "face": "face_memories",
            "voice": "voice_memories"
        }

    def _generate_entity_id(self, prefix: str = "ent") -> str:
        """Generates a new deterministic entity ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def get_canonical_id(self, entity_id: str) -> str:
        """
        🔥 MERGED: From IdentityResolver
        Resolves aliased IDs to their canonical form.
        In production multi-agent systems, this prevents identity fragmentation.
        """
        if entity_id in self.entity_index:
            return self.entity_index[entity_id].get("canonical_id", entity_id)
        return entity_id

    def resolve_face(self, obs: FaceObservation) -> str:
        """
        Alignment: Logic to resolve face identity before ingestion.
        Uses vector similarity search to match against existing entities.
        """
        results = self.vector_store.search(
            collection=self.collections["face"],
            vector=obs.embedding,
            filter_kv={"video_id": obs.video_id},
            limit=1
        )
        
        if results and results[0].score >= self.face_threshold:
            matched_id = results[0].payload.get("entity_id")
            # Resolve to canonical ID if aliased
            obs.entity_id = self.get_canonical_id(matched_id)
            logger.info(f"Resolved Face {obs.obs_id[:8]} -> Existing Entity {obs.entity_id} (score: {results[0].score:.3f})")
        else:
            obs.entity_id = self._generate_entity_id("ent_face")
            logger.info(f"New Face Detected. Created Entity {obs.entity_id}")
            self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, obs.video_id)
            # Register in canonical index
            self.entity_index[obs.entity_id] = {
                "type": EntityType.PERSON.value,
                "canonical_id": obs.entity_id,
                "video_id": obs.video_id
            }
        
        return obs.entity_id

    def resolve_voice(self, obs: VoiceObservation) -> str:
        """
        Alignment: Logic to resolve voice identity via vectors or co-occurrence.
        Falls back to single-person heuristic if no vector match found.
        """
        results = self.vector_store.search(
            collection=self.collections["voice"],
            vector=obs.embedding,
            filter_kv={"video_id": obs.video_id},
            limit=1
        )
        
        if results and results[0].score >= self.voice_threshold:
            matched_id = results[0].payload.get("entity_id")
            # Resolve to canonical ID if aliased
            obs.entity_id = self.get_canonical_id(matched_id)
            logger.info(f"Resolved Voice {obs.obs_id[:8]} -> Existing Entity {obs.entity_id} (score: {results[0].score:.3f})")
        else:
            # Fallback to co-occurrence heuristic if single person visible
            query = """
            MATCH (e:Entity)-[:APPEARED_IN]->(c:Clip {id: $clip_id, video_id: $video_id})
            WHERE e.type = 'person'
            RETURN e.id as id
            """
            visible_people = self.graph_store.run_query(query, {
                "clip_id": obs.clip_id,
                "video_id": obs.video_id
            })
            
            if len(visible_people) == 1:
                matched_id = visible_people[0]['id']
                obs.entity_id = self.get_canonical_id(matched_id)
                logger.info(f"Inferred Voice Identity -> Entity {obs.entity_id} via Single-Person Co-occurrence")
            else:
                obs.entity_id = self._generate_entity_id("ent_voice")
                logger.info(f"New Voice Detected. Created Entity {obs.entity_id}")
                self.graph_store.create_entity_node(obs.entity_id, EntityType.PERSON.value, obs.video_id)
                # Register in canonical index
                self.entity_index[obs.entity_id] = {
                    "type": EntityType.PERSON.value,
                    "canonical_id": obs.entity_id,
                    "video_id": obs.video_id
                }
        
        return obs.entity_id

    def register_observation(self, obs: Union[FaceObservation, VoiceObservation]):
        """
        API Parity: Mandatory final step for observation ingestion.
        Syncs the resolved identity to both Vector and Graph databases.
        """
        # Ensure identity is resolved
        if not obs.entity_id:
            if isinstance(obs, FaceObservation):
                self.resolve_face(obs)
            else:
                self.resolve_voice(obs)

        col = self.collections["face"] if isinstance(obs, FaceObservation) else self.collections["voice"]
        
        # 1. Update Graph (Async Write via Queue)
        # We use the GraphStore's specialized method which puts this in the write_queue.
        # This ensures it runs AFTER the node creation tasks pushed by engine.ingest_face().
        self.graph_store.create_appearance_link(
            entity_id=obs.entity_id,
            clip_id=obs.clip_id,
            video_id=obs.video_id,
            ts_ms=obs.ts_ms,
            obs_id=obs.obs_id
        )

        # 2. Update Vector Store (Idempotent)
        payload = {
            "video_id": obs.video_id,
            "entity_id": obs.entity_id,
            "clip_id": obs.clip_id,
            "obs_id": obs.obs_id,
            "type": "resolved_identity"
        }
        self.vector_store.upsert(col, obs.obs_id, obs.embedding, payload)

    def merge_identities(self, source_id: str, target_id: str, video_id: str):
        """
        🔥 UNIFIED DEEP MERGE
        Step 3: Safe Merge (The "Deep Refactor").
        Re-tags all historical vectors and refactors the Graph.
        Uses Async Graph Writes to prevent blocking the perception loop.
        
        Args:
            source_id: The primary/canonical entity ID to keep
            target_id: The secondary entity ID to merge into source
            video_id: Video context for the merge operation
        """
        if source_id == target_id:
            return

        logger.warning(f"🔄 DEEP MERGE INITIATED: Unifying {target_id} into {source_id}")

        # 1. Update Graph Relationships (Neo4j) - ASYNC NOW
        merge_query = """
        MATCH (primary:Entity {id: $source_id})
        MATCH (secondary:Entity {id: $target_id})
        
        OPTIONAL MATCH (secondary)-[r:APPEARED_IN]->(c:Clip)
        FOREACH (x IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
            MERGE (primary)-[new_r:APPEARED_IN {obs_id: r.obs_id}]->(c)
            ON CREATE SET new_r.ts_ms = r.ts_ms
        )
        
        // Relink Memories
        WITH primary, secondary
        OPTIONAL MATCH (m:Memory)-[r2:MENTIONS]->(secondary)
        FOREACH (x IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
            MERGE (m)-[:MENTIONS]->(primary)
        )
        """
        # 🔥 FIX: Use execute_async
        self.graph_store.execute_async(merge_query, {
            "source_id": source_id,
            "target_id": target_id
        })


        # 2. Vector Payload Update (Qdrant) - Retag all historical vectors
        for col_name in self.collections.values():
            try:
                # Scroll through all points matching the old entity_id
                points_to_update, _ = self.vector_store.client.scroll(
                    collection_name=col_name,
                    scroll_filter={
                        "must": [
                            {"key": "entity_id", "match": {"value": target_id}},
                            {"key": "video_id", "match": {"value": video_id}}
                        ]
                    },
                    with_payload=False,
                    limit=1000  # Batch size
                )
                
                if points_to_update:
                    point_ids = [p.id for p in points_to_update]
                    # Update all points to reference the new canonical entity
                    self.vector_store.client.set_payload(
                        collection_name=col_name,
                        payload={"entity_id": source_id},
                        points=point_ids
                    )
                    logger.info(f"✓ Retagged {len(point_ids)} vectors in {col_name}")
                else:
                    logger.debug(f"No vectors found for {target_id} in {col_name}")
                    
            except Exception as e:
                logger.error(f"Error updating vectors in {col_name}: {e}")

        # 3. Update Canonical Index
        if target_id in self.entity_index:
            del self.entity_index[target_id]
        
        # Mark source as canonical
        if source_id not in self.entity_index:
            self.entity_index[source_id] = {
                "type": EntityType.PERSON.value,
                "canonical_id": source_id,
                "video_id": video_id
            }

        logger.info(f"✅ Deep merge complete: {target_id} fully unified into {source_id}")

    def link_modalities(self, video_id: str, clip_id: int = None):
        """
        Advanced Identity Fusion.
        Analyzes Neo4j to find consistent Face-Voice pairings and merges them.
        """
        # Check if we even have appearance data yet to avoid Neo4j Warning 01N51
        check_query = """
        MATCH ()-[r:APPEARED_IN]->() RETURN count(r) as cnt LIMIT 1
        """
        try:
            check = self.graph_store.run_query(check_query)
            if not check or check[0]['cnt'] == 0:
                logger.debug("Skipping link_modalities: No APPEARED_IN relationships found yet.")
                return
        except Exception:
            # If the relationship doesn't exist, the query might error or return 0
            return

        query = """
        MATCH (e1:Entity)-[:APPEARED_IN]->(c:Clip {video_id: $video_id})
        MATCH (e2:Entity)-[:APPEARED_IN]->(c)
        WHERE e1.id < e2.id AND e1.type = 'person' AND e2.type = 'person'
        WITH e1, e2, count(c) as co_occurrences
        WHERE co_occurrences >= $threshold
        RETURN e1.id as source, e2.id as target, co_occurrences
        ORDER BY co_occurrences DESC
        """
        potential_merges = self.graph_store.run_query(query, {
            "video_id": video_id,
            "threshold": self.co_occurrence_threshold
        })

        for merge in potential_merges:
            logger.info(f"🔗 Co-occurrence detected: {merge['source']} <-> {merge['target']} ({merge['co_occurrences']} clips)")
            self.merge_identities(merge['source'], merge['target'], video_id)

    def refresh_mappings(self, video_id: str):
        """
        Rebuilds the character mapping for the reasoning engine to use.
        Finds entities that consistently appear in clips with 'equivalence' memories.
        Triggers deep merges based on semantic understanding from the reasoning engine.
        """
        query = """
        MATCH (m:Memory {type: 'semantic'})-[:MENTIONS]->(e:Entity)
        WHERE (m.content CONTAINS "is the same as" OR m.content CONTAINS "also known as")
            AND e.video_id = $video_id
        MATCH (m)-[:MENTIONS]->(e2:Entity)
        WHERE e.id < e2.id AND e2.video_id = $video_id
        RETURN e.id as source, e2.id as target, m.content as evidence
        """
        merges = self.graph_store.run_query(query, {"video_id": video_id})
        
        for m in merges:
            logger.info(f"🧠 Semantic equivalence detected: {m['source']} = {m['target']}")
            logger.debug(f"Evidence: {m['evidence'][:100]}...")
            self.merge_identities(m['source'], m['target'], video_id)

    def refresh_equivalences(self, video_id: str):
        """
        M3-Agent Logic: Parses 'Equivalence: ...' text nodes from Neo4j
        and builds dynamic character mappings using Union-Find.
        """
        # 1. Fetch all semantic memories containing "Equivalence"
        query = """
        MATCH (m:Memory {type: 'semantic', video_id: $video_id})
        WHERE toLower(m.content) STARTS WITH 'equivalence'
        RETURN m.content as content
        """
        results = self.graph_store.run_query(query, {"video_id": video_id})
        
        # 2. Initialize Union-Find structure
        parent = {}
        
        def find(i):
            if i not in parent: parent[i] = i
            if parent[i] != i: parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j: parent[root_i] = root_j

        # 3. Process raw tags
        import re
        tag_pattern = re.compile(r'<((?:ent_)?(?:face|voice)_[a-zA-Z0-9\-]+)>')

        all_tags = set()
        
        # Build sets from Equivalence memories
        for res in results:
            content = res.get('content', '')
            tags = tag_pattern.findall(content)
            for tag in tags:
                all_tags.add(tag)
            
            # Link tags in the same equivalence statement
            if len(tags) >= 2:
                base = tags[0]
                for other in tags[1:]:
                    union(base, other)

        # 4. Group into Characters
        self.character_mappings = {}
        self.reverse_character_mappings = {}
        
        groups = {}
        for tag in all_tags:
            root = find(tag)
            if root not in groups: groups[root] = []
            groups[root].append(tag)
            
        # Crucial Step: M3 Naming Convention
        # Sort for deterministic IDs
        sorted_roots = sorted(list(groups.keys()))
        
        for idx, root in enumerate(sorted_roots):
            char_id = f"character_{idx}" # M3 naming convention
            tags = groups[root]
            
            self.character_mappings[char_id] = tags
            for tag in tags:
                self.reverse_character_mappings[tag] = char_id
                
        logger.info(f"M3 Equivalences: Mapped {len(self.character_mappings)} characters.")

    def translate_content(self, text: str) -> str:
        """
        M3 Logic: Converts internal IDs (<face_x>, <voice_y>) back to 
        readable 'Character N' labels for the LLM.
        """
        # Ensure mappings are fresh
        if not self.reverse_character_mappings:
            logger.warning("Mappings empty, returning raw text.")
            return text

        import re
        # Find all tags like <face_...> or <voice_...>
        pattern = re.compile(r'<((?:ent_)?(?:face|voice)_[a-zA-Z0-9\-]+)>')
        
        def replace_match(match):
            tag = match.group(1)
            # Return mapped character (e.g., "character_0") or original tag if not found
            return self.reverse_character_mappings.get(tag, tag)

        # Replace <tag> with character_id
        translated = pattern.sub(replace_match, text)
        return translated

    def get_entity_stats(self, video_id: str) -> Dict[str, Any]:
        """
        Returns statistics about entities in the current video.
        Useful for debugging and monitoring identity resolution quality.
        """
        query = """
        MATCH (e:Entity {video_id: $video_id})
        OPTIONAL MATCH (e)-[r:APPEARED_IN]->(c:Clip)
        WITH e, count(DISTINCT c) as clip_count
        RETURN e.id as entity_id, e.type as entity_type, clip_count
        ORDER BY clip_count DESC
        """
        results = self.graph_store.run_query(query, {"video_id": video_id})
        
        return {
            "total_entities": len(results),
            "entities": results,
            "canonical_index_size": len(self.entity_index)
        }