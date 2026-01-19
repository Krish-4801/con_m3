import os
import sys
import json
import logging
import argparse
import re
import ast
from typing import List, Dict, Any, Optional

import openai


# Import Conclave modules
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.m3_prompts import (
    prompt_generate_plan,
    prompt_generate_action_with_plan,
    prompt_generate_action_with_plan_multiple_queries,
    prompt_generate_action_with_plan_new_direction,
)

# Logging setup
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Conclave.QuerySystem")

class ConclaveQuerier:
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # Initialize Engine (Reuses connection logic and embedding service)
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # Reuse Engine's embedder to ensure consistency with ingestion
        self.embedder = self.engine.embedding_service
        
        # Identity Manager for character mappings (Union-Find)
        self.identity_manager = IdentityManager(self.engine.vector_store, self.engine.graph_store, self.engine.api_config)
        
        # LLM Client Setup
        api_conf = self.config.get("api", {})
        self.model = api_conf.get("model", "gpt-4o") # Default to high capability
        
        # Logic to handle Gemini vs OpenAI config
        if "gemini" in self.config and "gemini" in self.model.lower():
             g_conf = self.config["gemini"]
             self.client = openai.OpenAI(
                 api_key=g_conf.get("api_key"),
                 base_url=g_conf.get("base_url")
             )
        else:
             self.client = openai.OpenAI(
                 api_key=api_conf.get("openai_api_key") or api_conf.get("api_key")
             )

    def _fetch_clip_context(self, clip_id: int) -> Dict[str, Any]:
        """
        Directly retrieves what happened in a specific clip from Graph + Vector Store.
        Used when the agent specifically asks to "See CLIP_X".
        """
        query = """
        MATCH (c:Clip {id: $clip_id, video_id: $video_id})
        OPTIONAL MATCH (c)-[:HAS_MEMORY]->(m:Memory)
        OPTIONAL MATCH (p:Entity)-[r:APPEARED_IN]->(c)
        RETURN 
            collect(DISTINCT m.content) as memories,
            collect(DISTINCT {id: p.id, type: p.type}) as entities
        """
        result = self.engine.graph_store.run_query(query, {
            "clip_id": clip_id,
            "video_id": self.video_id
        })
        
        if not result or not result[0]:
            return {"error": f"Clip {clip_id} not found."}
            
        data = result[0]
        return {
            "source": f"Direct Lookup CLIP_{clip_id}",
            "memories": data['memories'],
            "entities_present": [e['id'] for e in data['entities']]
        }

    def back_translate(self, query: str) -> List[str]:
        """
        M3-Agent Logic: Expands "Character X" into all specific modality tags.
        """
        # Ensure mappings are fresh from Graph Analysis
        self.identity_manager.refresh_equivalences(self.video_id)
        
        # M3 stores mappings as: character_0 -> ['face_1', 'voice_2']
        mappings = self.identity_manager.character_mappings 
        
        expanded_queries = [query]
        
        # 1. Parse entities in the query
        # We look for "character_X"
        char_matches = list(set(re.findall(r'(character_\d+)', query)))
        
        to_be_translated = [query]
        
        for char_id in char_matches:
            if char_id in mappings:
                mapped_tags = mappings[char_id] # e.g. ['face_123', 'voice_456']
                
                new_variations = []
                for tag in mapped_tags:
                    # Create variants: "Who is character_0" -> "Who is <face_123>"
                    # M3 format uses angular brackets for specific IDs
                    for base_q in to_be_translated:
                        new_q = base_q.replace(char_id, f"<{tag}>")
                        new_variations.append(new_q)
                
                to_be_translated = new_variations
        
        # M3 logic specifically adds the translated queries to the list, 
        # it doesn't just replace them.
        return list(set(expanded_queries + to_be_translated))

    def retrieve_knowledge(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs GraphRAG: Vector Search -> Graph Context Expansion.
        Robustly handles both Episodic (Clip-bound) and Semantic (Global) memories.
        """
        # 1. Expand Query (M3 Logic)
        expanded_queries = self.back_translate(query_text)
        if len(expanded_queries) > 1:
            print(f"M3 Expansion: {len(expanded_queries)} variations generated.")

        all_context_results = []
        
        for q in expanded_queries:
            # 1. Embed Question
            query_vec = self.embedder.get_embeddings_batched([q])[0]
            
            # 2. Vector Search (Qdrant)
            hits = self.engine.vector_store.search(
                collection="text_memories",
                vector=query_vec,
                filter_kv={"video_id": self.video_id},
                limit=top_k
            )
            
            for hit in hits:
                mem_id = hit.id
                score = hit.score
                content = hit.payload.get("content")
                clip_id = hit.payload.get("clip_id") # Can be None for semantic memories
                
                # 3. Graph Expansion (Neo4j)
                params = {
                    "mem_id": mem_id, 
                    "video_id": self.video_id
                }
                
                # Base query: Get entities mentioned in the text
                cypher = """
                MATCH (m:Memory {id: $mem_id})
                OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
                WITH m, collect(e.id) as mentioned_ids
                """
                
                # Conditional query: If attached to a clip, get entities present in that clip
                if clip_id is not None:
                    cypher += """
                    OPTIONAL MATCH (c:Clip {id: $clip_id, video_id: $video_id})
                    OPTIONAL MATCH (p:Entity)-[:APPEARED_IN]->(c)
                    RETURN mentioned_ids, collect(DISTINCT p.id) as present_ids
                    """
                    params["clip_id"] = clip_id
                else:
                    cypher += """
                    RETURN mentioned_ids, [] as present_ids
                    """
    
                graph_data = self.engine.graph_store.run_query(cypher, params)
                
                # Robust unpacking
                g_res = graph_data[0] if graph_data else {"mentioned_ids": [], "present_ids": []}
                
                all_context_results.append({
                    "type": "retrieval",
                    "query": q,
                    "score": score,
                    "clip_id": clip_id,
                    "memory_text": content,
                    "entities": list(set(g_res['mentioned_ids'] + g_res['present_ids']))
                })
        
        # Deduplicate results based on memory_text
        seen_memories = set()
        unique_results = []
        for res in sorted(all_context_results, key=lambda x: x['score'], reverse=True):
            if res['memory_text'] not in seen_memories:
                unique_results.append(res)
                seen_memories.add(res['memory_text'])
            
        return unique_results[:top_k]

    def execute_m3_control_loop(self, question: str):
        """
        The M3-Agent Controller Logic.
        1. Generate Plan
        2. Iterative Loop (Search/Look <-> Reason)
        3. Final Answer
        """
        print(f"\n❓ Question: {question}")
        print("-" * 50)

        # --- STEP 1: Generate Retrieval Plan ---
        plan_messages = [
            {"role": "system", "content": prompt_generate_plan.format(question=question)}
        ]
        
        try:
            plan_resp = self.client.chat.completions.create(
                model=self.model, messages=plan_messages
            )
            retrieval_plan = plan_resp.choices[0].message.content
            print(f"📋 Plan:\n{retrieval_plan}\n")
        except Exception as e:
            print(f"❌ Error generating plan: {e}")
            return

        # --- STEP 2: The Control Loop ---
        knowledge_context = [] 
        max_steps = 5
        route_switch = False
        
        for step in range(max_steps):
            # Limit context size to avoid token overflow
            knowledge_str = json.dumps(knowledge_context[-15:], indent=2) 
            
            # M3 Logic: Switch prompt if we are stuck
            if route_switch:
                prompt_template = prompt_generate_action_with_plan_new_direction
            else:
                prompt_template = prompt_generate_action_with_plan_multiple_queries

            user_content = prompt_template.format(
                question=question,
                retrieval_plan=retrieval_plan,
                knowledge=knowledge_str
            )

            # Call LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": user_content}]
                )
                raw_output = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"❌ LLM Error: {e}")
                break

            print(f"🤖 Step {step+1} Output: {raw_output[:100]}...")

            # --- STEP 3: Parse Action ---
            if "[ANSWER]" in raw_output:
                final_answer = raw_output.split("[ANSWER]", 1)[1].strip()
                print("\n" + "="*60)
                print(f"✅ FINAL ANSWER")
                print("="*60)
                print(final_answer)
                return final_answer
            
            elif "[SEARCH]" in raw_output:
                # Robust extraction of queries from potential JSON or list text
                try:
                    search_part = raw_output.split("[SEARCH]", 1)[1].strip()
                    # Clean up markdown code blocks if present
                    search_part = search_part.replace("```json", "").replace("```", "").strip()
                    
                    if search_part.startswith("[") and search_part.endswith("]"):
                         queries = json.loads(search_part)
                    else:
                         # Fallback to simple split
                         queries = [line.strip("- \"'") for line in search_part.splitlines() if line.strip()]
                    
                    if isinstance(queries, str): queries = [queries]
                    queries = [str(q) for q in queries if q]

                except Exception as e:
                    print(f"⚠️ Failed to parse queries: {e}. Using raw text.")
                    queries = [raw_output.split("[SEARCH]", 1)[1].strip()]

                print(f"🔍 Searching: {queries}")

                # Execute Searches
                found_new_info = False
                for q in queries:
                    # 1. Handle "CLIP_x" direct lookups
                    clip_match = re.search(r"CLIP_(\d+)", q, re.IGNORECASE)
                    if clip_match:
                        clip_id = int(clip_match.group(1))
                        print(f"   -> Direct Lookup: Clip {clip_id}")
                        clip_data = self._fetch_clip_context(clip_id)
                        knowledge_context.append(clip_data)
                        found_new_info = True
                        continue

                    # 2. Handle Semantic/Vector Search
                    results = self.retrieve_knowledge(q, top_k=2)
                    if results:
                        found_new_info = True
                        for r in results:
                            knowledge_context.append({
                                "query": q,
                                "memory": r["memory_text"],
                                "related_entities": r["entities"],
                                "clip_id": r["clip_id"]
                            })

                if not found_new_info:
                    print("⚠️ No info found. Switching reasoning route.")
                    knowledge_context.append({"system_note": "Previous search yielded no results."})
                    route_switch = True
                else:
                    route_switch = False

            else:
                print("⚠️ Unrecognized action pattern. Retrying...")
                knowledge_context.append({"system_note": "Invalid output format. Please use [SEARCH] or [ANSWER]."})

        print("❌ Max steps reached without answer.")

    def generate_answer(self, user_query: str, context: List[Dict[str, Any]]):
        """
        Synthesizes the answer using LLM + Context (Standard RAG)
        """
        if not context:
            print("❌ No relevant information found.")
            return

        # Format context for the LLM
        context_str = ""
        for item in sorted(context, key=lambda x: x.get('clip_id') or -1):
            clip_info = f"[Clip ID: {item['clip_id']}]" if item.get('clip_id') is not None else "[Global Memory]"
            text = item.get('memory_text', '')
            entities = item.get('entities', [])
            
            context_str += f"""
            {clip_info}
            Description: {text}
            Related Entities: {entities}
            ------------------------------------------------
            """

        system_prompt = """
        You are Conclave, an advanced video intelligence AI.
        Answer the user's question based strictly on the provided Context.
        
        FORMATTING RULES:
        1. **Direct Answer**: Start with a clear, direct answer.
        2. **Reasoning**: Explain *why* you concluded this based on evidence.
        3. **Citations**: You MUST cite the [Clip ID] for every claim.
        4. **Timeline**: If describing a sequence, provide a chronological breakdown.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
                ]
            )
            
            answer = response.choices[0].message.content
            print("\n" + "="*60)
            print(f"🤖 CONCLAVE ANSWER")
            print("="*60)
            print(answer)
            print("="*60)
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Conclave Memory System")
    parser.add_argument("--query", type=str, required=True, help="Your question about the video")
    parser.add_argument("--video_id", type=str, required=True, help="The ID used during processing")
    parser.add_argument("--config", type=str, default="configs/api_config.json")
    parser.add_argument("--iterative", action="store_true", help="Enable M3 iterative reasoning loop")
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)

    querier = ConclaveQuerier(args.config, args.video_id)
    
    if args.iterative:
        querier.execute_m3_control_loop(args.query)
    else:
        knowledge = querier.retrieve_knowledge(args.query)
        querier.generate_answer(args.query, knowledge)
