# FILE: query_system.py
import os
import sys
import json
import logging
import argparse
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import openai

# Conclave Imports
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Conclave.M3Control")

# --- M3 PROMPTS (Embedded for exact replication) ---
SYSTEM_PROMPT = """You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.

Question: {question}"""

INSTRUCTION_PROMPT = """
Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

If the answer cannot be derived yet, the {content} should be a single search query that would help retrieve the missing information. The search {content} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{i}>" or "What is the character id of {name}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {content} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{i}>."""

class M3Controller:
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # Initialize Engine & Identity
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.engine.api_config
        )
        
        # Initialize Embedder (Reuse Engine's service)
        self.embedder = self.engine.embedding_service

        # Initialize LLM Client (RESTORED GEMINI LOGIC)
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

        # Build Character Mappings immediately (Critical for M3)
        logger.info("🔄 Building Identity Mappings...")
        self.identity_manager.refresh_equivalences(self.video_id)

    # ------------------------------------------------------------------
    # 1. M3 RETRIEVAL LOGIC (retrieve.py equivalent)
    # ------------------------------------------------------------------

    def back_translate(self, query: str) -> List[str]:
        """
        M3 Logic: Expands "character_X" into specific tags (<face_Y>, <voice_Z>).
        """
        expanded_queries = [query]
        
        # Regex for character_0, character_1...
        char_matches = list(set(re.findall(r'(character_\d+)', query)))
        
        to_be_translated = [query]
        mappings = self.identity_manager.character_mappings
        
        for char_id in char_matches:
            if char_id in mappings:
                mapped_tags = mappings[char_id] # e.g. ['face_1', 'voice_2']
                new_variations = []
                for tag in mapped_tags:
                    # M3 replaces character_id with <tag>
                    for base_q in to_be_translated:
                        new_q = base_q.replace(char_id, f"<{tag}>")
                        new_variations.append(new_q)
                to_be_translated = new_variations
        
        # Combine original + expanded
        return list(set(expanded_queries + to_be_translated))

    def retrieve_by_clip_aggregation(self, query: str, current_clips: List[int], top_k: int = 2) -> Tuple[Dict[int, List[str]], List[int]]:
        """
        M3 Logic: Search Vectors -> Score Clips -> Retrieve ALL context for top Clips.
        Unlike standard RAG, this preserves temporal coherence.
        """
        # 1. Back Translate Query
        queries = self.back_translate(query)
        
        # 2. Get Embeddings
        query_vecs = self.embedder.get_embeddings_batched(queries)
        
        # 3. Vector Search (Text Memories)
        # We need raw scores to aggregate by clip
        clip_scores: Dict[int, List[float]] = {}
        
        for vec in query_vecs:
            hits = self.engine.vector_store.search(
                collection="text_memories",
                vector=vec,
                filter_kv={"video_id": self.video_id},
                limit=20 # Fetch wider candidate pool
            )
            
            for hit in hits:
                c_id = hit.payload.get("clip_id")
                score = hit.score
                if c_id is not None:
                    if c_id not in clip_scores: clip_scores[c_id] = []
                    clip_scores[c_id].append(score)

        # 4. Aggregation Strategy (Max Score per Clip)
        final_clip_scores = {cid: max(scores) for cid, scores in clip_scores.items()}
        
        # Sort clips
        sorted_clips = sorted(final_clip_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out already seen clips
        new_top_clips = []
        for cid, score in sorted_clips:
            if cid not in current_clips:
                new_top_clips.append(cid)
            if len(new_top_clips) >= top_k:
                break
        
        # 5. Fetch Content for Top Clips
        new_memories = {}
        
        for cid in new_top_clips:
            # Query Graph/Vector for ALL text in this clip
            # Using Graph is cleaner here
            cypher = """
            MATCH (c:Clip {id: $clip_id, video_id: $video_id})-[:HAS_MEMORY]->(m:Memory)
            RETURN m.content as content
            """
            results = self.engine.graph_store.run_query(cypher, {
                "clip_id": cid, 
                "video_id": self.video_id
            })
            
            raw_texts = [r['content'] for r in results]
            
            # 6. Forward Translate (The "Reader" View)
            # Convert <face_x> -> character_x
            translated_texts = [self.identity_manager.translate_content(t) for t in raw_texts]
            
            new_memories[cid] = translated_texts
            current_clips.append(cid)
            
        return new_memories, current_clips

    # ------------------------------------------------------------------
    # 2. M3 CONTROL LOOP (control.py equivalent)
    # ------------------------------------------------------------------

    def run(self, question: str, max_rounds: int = 5):
        """
        Executes the exact ReAct loop from M3-Agent.
        """
        # Conversation History
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(question=question)},
            {"role": "user", "content": "Searched knowledge: {}"}
        ]
        
        current_clips = [] # State tracking for clips seen
        pattern = r"Action: \[(.*)\].*Content: (.*)"
        
        print(f"\n🚀 Starting M3 Control Loop for: {question}\n")

        # CHECK FOR TEMPERATURE COMPATIBILITY
        # Some Gemini proxy models only support temperature=1.0
        temp = 0.0
        if "gemini" in self.model.lower() or "gpt-5" in self.model.lower():
            temp = 1.0

        for round_idx in range(max_rounds):
            # 1. Prepare Prompt
            # Clone messages to avoid modifying history permanently with instruction
            round_messages = list(messages)
            
            # Inject instruction at the end of the last user message or as new system?
            # M3 appends instruction to the content of the last message usually.
            last_content = round_messages[-1]["content"]
            instruction_text = INSTRUCTION_PROMPT
            
            if round_idx == max_rounds - 1:
                instruction_text += "\n(The Action of this round must be [Answer].)"
            
            round_messages[-1]["content"] = last_content + instruction_text

            # 2. LLM Inference
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=round_messages,
                    temperature=temp
                )
                output_text = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                break

            # 3. Add Assistant Response to History
            messages.append({"role": "assistant", "content": output_text})
            print(f"🤖 Round {round_idx+1}: {output_text}")

            # 4. Parse Output (Consumer)
            match = re.search(pattern, output_text, re.DOTALL)
            
            action = "Search"
            content = None
            
            if match:
                action = match.group(1).strip()
                content = match.group(2).strip()
            else:
                # Fallback logic
                if "[Answer]" in output_text:
                    action = "Answer"
                    content = output_text.split("[Answer]")[-1].strip()
                elif "[Search]" in output_text:
                    action = "Search"
                    content = output_text.split("[Search]")[-1].strip()

            # 5. Execute Action
            if action == "Answer":
                print("\n✅ FINAL ANSWER:")
                print(content)
                return content
            
            else:
                # Search Logic
                search_result_text = ""
                new_memories = {}
                
                if content:
                    # Specialized logic for ID lookup vs General Search
                    if "character id" in content.lower() or "name of" in content.lower():
                        # Simple retrieval for mapping queries
                        # M3 usually treats these as vector searches too, or direct lookups
                        # We use the standard flow but top_k might differ
                        mems, current_clips = self.retrieve_by_clip_aggregation(content, current_clips)
                        new_memories.update(mems)
                    else:
                        mems, current_clips = self.retrieve_by_clip_aggregation(content, current_clips)
                        new_memories.update(mems)
                
                # Format Results
                if new_memories:
                    # JSON dump ensures the LLM can read the structure clearly
                    search_result_text = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False)
                else:
                    search_result_text = "Searched knowledge: \n(The search result is empty. Please try searching from another perspective.)"

                # Append result as User message for next round
                messages.append({"role": "user", "content": search_result_text})

        return "Max rounds reached without answer."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/api_config.json")
    parser.add_argument("--iterative", action="store_true", help="Included for backward compatibility")
    args = parser.parse_args()

    agent = M3Controller(args.config, args.video_id)
    agent.run(args.query)
