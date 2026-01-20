
import os
import sys
import json
import logging
import argparse
import re
from typing import List, Dict, Any, Tuple, Optional
import openai

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Conclave.M3Control")

from conclave.prompts.query import SYSTEM_PROMPT, PLAN_GENERATION_PROMPT, FINAL_ANSWER_PROMPT

class M3Controller:
    def __init__(self, config_path: str, video_id: str, preloaded_engine: Optional[ConclaveEngine] = None):
        """
        Refactored to accept preloaded_engine.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # Reuse or Create Engine
        if preloaded_engine:
            self.engine = preloaded_engine
            self.engine.video_id = video_id # Switch context
        else:
            self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
            
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.engine.api_config
        )
        
        self.embedder = self.engine.embedding_service

        # Initialize LLM Client
        api_conf = self.config.get("api", {})
        self.model = api_conf.get("model", "gpt-4o")
        
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

        logger.info("🔄 Building Identity Mappings...")
        self.identity_manager.refresh_equivalences(self.video_id)

    # [Rest of retrieve_by_clip_aggregation, back_translate, run methods remain UNCHANGED]
    def back_translate(self, query: str) -> List[str]:
        # ... copy from previous ...
        expanded_queries = [query]
        char_matches = list(set(re.findall(r'(character_\d+)', query)))
        to_be_translated = [query]
        mappings = self.identity_manager.character_mappings
        for char_id in char_matches:
            if char_id in mappings:
                mapped_tags = mappings[char_id]
                new_variations = []
                for tag in mapped_tags:
                    for base_q in to_be_translated:
                        new_q = base_q.replace(char_id, f"<{tag}>")
                        new_variations.append(new_q)
                to_be_translated = new_variations
        return list(set(expanded_queries + to_be_translated))

    def retrieve_by_clip_aggregation(self, query: str, current_clips: List[int], top_k: int = 2):
        queries = self.back_translate(query)
        query_vecs = self.embedder.get_embeddings_batched(queries)
        clip_scores: Dict[int, List[float]] = {}
        for vec in query_vecs:
            hits = self.engine.vector_store.search("text_memories", vec, {"video_id": self.video_id}, 20)
            for hit in hits:
                c_id = hit.payload.get("clip_id")
                if c_id is not None:
                    if c_id not in clip_scores: clip_scores[c_id] = []
                    clip_scores[c_id].append(hit.score)
        final_clip_scores = {cid: max(scores) for cid, scores in clip_scores.items()}
        sorted_clips = sorted(final_clip_scores.items(), key=lambda x: x[1], reverse=True)
        new_top_clips = []
        for cid, score in sorted_clips:
            if cid not in current_clips:
                new_top_clips.append(cid)
            if len(new_top_clips) >= top_k:
                break
        new_memories = {}
        for cid in new_top_clips:
            cypher = "MATCH (c:Clip {id: $clip_id, video_id: $video_id})-[:HAS_MEMORY]->(m:Memory) RETURN m.content as content"
            results = self.engine.graph_store.run_query(cypher, {"clip_id": cid, "video_id": self.video_id})
            raw_texts = [r['content'] for r in results]
            translated_texts = [self.identity_manager.translate_content(t) for t in raw_texts]
            new_memories[cid] = translated_texts
            current_clips.append(cid)
        return new_memories, current_clips

    def run(self, question: str, max_rounds: int = 5):
        # 1. Generate Retrieval Plan
        logger.info(f"📋 Generating Retrieval Plan for: {question}")
        plan_messages = [
            {"role": "user", "content": PLAN_GENERATION_PROMPT.format(question=question)}
        ]
        try:
            plan_resp = self.client.chat.completions.create(model=self.model, messages=plan_messages)
            retrieval_plan = plan_resp.choices[0].message.content.strip()
            logger.info(f"📍 PLAN: {retrieval_plan}")
        except Exception as e:
            logger.error(f"Planning Error: {e}")
            retrieval_plan = "N/A"

        # 2. Main ReAct Loop
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                question=question, 
                retrieval_plan=retrieval_plan,
                knowledge="{}"
            )},
            {"role": "user", "content": "Begin search."}
        ]
        
        current_clips = [] 
        temp = 1.0 if "gemini" in self.model.lower() else 0.0

        for round_idx in range(max_rounds):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temp)
                output_text = response.choices[0].message.content.strip()
                logger.info(f"🔄 Round {round_idx+1}: {output_text[:100]}...")
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                break

            messages.append({"role": "assistant", "content": output_text})
            
            # Simple Action detection (Search vs Answer)
            if "[ANSWER]" in output_text:
                final_content = output_text.split("[ANSWER]")[-1].strip()
                return final_content
            
            if "[SEARCH]" in output_text:
                query_content = output_text.split("[SEARCH]")[-1].strip()
                # Run retrieval
                new_memories, current_clips = self.retrieve_by_clip_aggregation(query_content, current_clips)
                
                # Feedback to LLM
                search_result_text = f"Knowledge from {query_content}: " + json.dumps(new_memories, ensure_ascii=False)
                if not new_memories:
                    search_result_text = f"No new information found for query: {query_content}"
                
                messages.append({"role": "user", "content": search_result_text})
            else:
                if round_idx == max_rounds - 1:
                    return output_text
                messages.append({"role": "user", "content": "Please follow the format: [SEARCH] query OR [ANSWER] final answer."})

        return "Max rounds reached without definitive answer."
