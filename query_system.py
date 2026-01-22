import os
import json
import logging
import argparse
import re
import ast
import time
from typing import List, Dict, Any, Tuple
import openai

# Conclave Imports
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager

# Configure Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger("Conclave.M3Control")

# ==============================================================================
# 1. M3 PROMPTS (Exact Replacements)
# ==============================================================================

PROMPT_GENERATE_PLAN = """You are given a clip from a specific video and a question about the video. There exists a memory bank that contains information about this video, but you will not be shown its contents.

The memory bank is structured as a temporally ordered sequence of entries. Each entry contains either:
	•	a fine-grained description of a specific moment in the video, or
	•	a high-level summary or abstraction of events.

Your task is to create a detailed and robust retrieval plan: a step-by-step outline describing what kinds of information should be retrieved from the memory bank to answer the question effectively.

Requirements:
	•	Do not answer the question.
	•	Instead, output a string list, where each item describes one retrieval step.
	•	Each step should specify a type of content, topic, or temporal segment to retrieve (e.g., "find entries describing character motivations" or "look for summaries of the climax").

Your plan must:
	1.	Ensure completeness:
		The plan must guide the retrieval process in such a way that all essential pieces of information required to answer the question will be retrieved — including context, reasoning chains, motivations, consequences, and temporal links, as relevant.
		Do not stop at partial evidence. Design the plan so that it systematically explores and gathers all necessary supporting elements.
	2.	Include contingency strategies:
		Anticipa what might go wrong or be missing during retrieval. For example:
		•	What if direct mentions of an event are not available?
		•	What if the memory bank contains conflicting interpretations?
		•	What if characters' intentions or relationships are implied but not explicitly stated?
		Your plan should include fallback options and indirect paths to cover these cases (e.g., using emotion cues, related scenes, earlier summaries, or surrounding context).
	3.	Follow a logical order:
		The steps should be ordered in a way that reflects effective reasoning — e.g., from specific to general, or from earlier scenes to later consequences.

Output format:
A list of strings. Example:

[
	"Step 1: Retrieve entries describing the initial context and setting of the video.",
	"Step 2: Look for interactions between the main characters relevant to the question.",
	"Step 3: Find summaries that explain the consequences of the key events."
]

Please response with only the string list of the plan (wrapped by "[]"), without any additional explanation or formatting.

Now start generating the plan.

Questions: {question}"""

# Standard Action Prompt
PROMPT_ACTION_WITH_PLAN = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan, which outlines the types of information that should be retrieved from a memory bank in order to answer the question. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [ANSWER] followed by the answer. If it is not sufficient, output [SEARCH] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank that contains detailed descriptions and high-level abstractions of the video, considering the question, the provided knowledge, and the retrieval plan.

Your response should contain two parts:
1.	Reasoning
	•	Analyze the question, the knowledge, and the retrieval plan.
	•	If the current information is sufficient, explain why and what conclusions you can draw.
	•	If not, clearly identify what is missing and why it is important.
2.	Answer or Search
	•	[ANSWER]: If the answer can be derived from the provided knowledge, output [ANSWER] followed by a short, clear, and direct answer.
		•	When referring to a character, always use their specific name if available.
		•	Do not use ID tags like <character_1> or <face_1>.
	•	[SEARCH]: If the answer cannot be derived yet, output [SEARCH] followed by a single search query that would help retrieve the missing information.

Instructions for [SEARCH] queries:
	•	Use the retrieval plan to inform what type of content should be searched for next. These contents should cover aspects that provide useful context or background to the question, such as character names, behaviors, relationships, personality traits, actions, and key events.
	•	Use keyword-based queries, not command sentences. Queries should be written as compact keyword phrases, not as full sentences or instructions. Avoid using directive language like "Retrieve", "Describe", or question forms such as "What", "When", "How".
	•	Keep each query short and focused on one point. Each query should target one specific type of information, without combining multiple ideas or aspects.
	•	Avoid over-complexity and unnecessary detail. Do not include too many qualifiers or conditions. Strip down to the most essential keywords needed to retrieve valuable content.
	•	The query should target information outside of the existing knowledge that might help answer the question.
	•	For time-sensitive or chronological information (e.g., events occurring in sequence, changes over time, or specific moments in a timeline), you can generate clip-based queries that reference specific clips or moments in time. These queries should include a reference to the clip number, indicating the index of the clip in the video (a number from 1 to N, where a smaller number indicates an earlier clip). Format these queries as "CLIP_x", where x should be an integer that indicates the clip index. Note only generate clip-based queries if the question is about a specific moment in time or a sequence of events.
	•	You can also generate queries that focus on specific characters or characters' attributes using the id shown in the knowledge.
	•	Make sure your generated query focus on some aspects that are not retrieved or asked yet. Do not repeatedly generate queries that have high semantic similarity with those generated before.

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

Output:"""

# Strategy Switch Prompt (New Direction)
PROMPT_ACTION_NEW_DIRECTION = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan, which outlines the types of information that should be retrieved from a memory bank in order to answer the question. Your task is to reason about whether the provided knowledge is sufficient to answer the question.

Important Context:
The previous retrieval attempt did not return any useful new information. Therefore, you must now shift your approach and think differently. Specifically, you must identify new angles or unexplored directions based on the retrieval plan that have not yet been considered. Your goal is to create search queries that are distinct from the ones used before, aiming to retrieve different types of content that could lead to an answer.

Your response must include two parts:
1. Reasoning:
	•	Analyze the question, the provided knowledge, and the retrieval plan.
	•	Evaluate why the previous queries may have failed and what new avenues should be explored now.
	•	Identify what specific types of information are still missing and why they matter.
	•	Suggest alternative directions that have not been fully explored yet, based on the retrieval plan.
2. Answer or Search:
	•	[ANSWER]: If the answer can now be derived from the current knowledge, output [ANSWER] followed by a short, clear, and direct answer.
		•	Use specific character names if available.
		•	Do not use generic tags like <character_1> or <face_1>.
	•	[SEARCH]: If more information is needed, output [SEARCH] followed by a new search query that are different from those used in the previous retrieval attempt.
		•	The new query must reflect a change in strategy, targeting unexplored or less obvious aspects.
		•	Use the retrieval plan to guide what different types of content should be searched for (e.g., overlooked characters, background events, personality traits, contextual clues).
		•	Include CLIP-based queries only if the question relates to specific moments or sequences in time, formatted as "CLIP_x" (noting that the clip ids are ordered chronologically).
		•	Avoid repeating previous query patterns or focusing on the same semantic areas.

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

Output:"""

# Multiple Queries Prompt
PROMPT_ACTION_MULTIPLE_QUERIES = """You are given a question and some relevant knowledge about a specific video. You are also provided with a retrieval plan...
(Abbreviated for brevity: Use strict logic to output [SEARCH] followed by a Python list of 5 queries wrapped in [])
...
Format the queries as a **Python-style string list wrapped by "[]"**: [SEARCH] ["Query 1", "Query 2", "Query 3", "Query 4", "Query 5"]

Now, generate your response for the following input:

Question: {question}

Retrieval Plan: {retrieval_plan}

Knowledge: {knowledge}

Output:"""

# Forced Final Answer Prompt
PROMPT_FINAL_ANSWER = """You are given a question about a specific video and a dictionary of some related information about the video. Each key in the dictionary is a clip ID (an integer), representing the index of a video clip. The corresponding value is a list of video descriptions from that clip.

Your task is to analyze the provided information, reason over it, and produce the most reasonable and well-supported answer to the question.

Output Requirements:
	•	Your response must begin with a brief reasoning process that explains how you arrive at the answer.
	•	Then, output [ANSWER] followed by your final answer.
	•	The format must be: Here is the reasoning... [ANSWER] Your final answer here.
	•	Your final answer must be definite and specific — even if the information is partial or ambiguous, you must infer and provide the most reasonable answer based on the given evidence.
	•	Do not refuse to answer or say that the answer is unknowable. Use reasoning to reach the best possible conclusion. Dont include Clip_x, clip names in the final output keep it short and simple

Input:
	•	Question: {question}
	•	Video Information: {information}

Output:"""


class M3Controller:
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # Engine & Identity
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.engine.api_config
        )
        self.embedder = self.engine.embedding_service

        # LLM Setup
        api_conf = self.config.get("api", {})
        self.model = api_conf.get("model", "gpt-4o")
        self.client = openai.OpenAI(
            api_key=api_conf.get("openai_api_key") or api_conf.get("api_key")
        )

        # Build Character Mappings
        logger.info("🔄 Building Identity Mappings...")
        self.identity_manager.refresh_equivalences(self.video_id)

    # ------------------------------------------------------------------
    # 1. RETRIEVAL LOGIC
    # ------------------------------------------------------------------

    def back_translate(self, queries: List[str]) -> List[str]:
        """
        M3 Logic: Expands specific IDs into all known aliases (Back Translation).
        Input: ["Who is <character_0>?"]
        Output: ["Who is <character_0>?", "Who is <face_101>?", "Who is <voice_55>?"]
        """
        translated_queries = []
        for query in queries:
            # Parse entities in query
            entities = re.findall(r'<([^<>]*_[^<>]*)>', query)
            to_be_translated = [query]
            
            for entity_str in set(entities):
                # Check mapping in IdentityManager
                # Assuming character_mappings maps 'character_0' -> ['face_1', 'voice_2']
                if entity_str in self.identity_manager.character_mappings:
                    mapped_tags = self.identity_manager.character_mappings[entity_str]
                    
                    new_variations = []
                    for tag in mapped_tags:
                        for base_q in to_be_translated:
                            # Replace <character_0> with <face_1>
                            new_q = base_q.replace(entity_str, f"{tag}")
                            new_variations.append(new_q)
                    to_be_translated.extend(new_variations)
            
            translated_queries.extend(to_be_translated)
        
        return list(set(translated_queries))

    def retrieve(self, queries: List[str], current_clips: List[int], top_k: int = 5) -> Tuple[Dict[str, List[str]], List[int]]:
        """
        M3 Search: Back Translate -> Vector Search -> Aggregate by Clip -> Forward Translate.
        Accepts a list of queries (for multiple_queries support).
        """
        # 1. Back Translate
        expanded_queries = self.back_translate(queries)
        
        # 2. Get Embeddings
        query_vecs = self.embedder.get_embeddings_batched(expanded_queries)
        
        # 3. Vector Search & Aggregate Scores
        clip_scores: Dict[int, List[float]] = {}
        
        for vec in query_vecs:
            # Search against text memories
            hits = self.engine.vector_store.search(
                collection="text_memories",
                vector=vec,
                filter_kv={"video_id": self.video_id},
                limit=20 
            )
            for hit in hits:
                c_id = hit.payload.get("clip_id")
                if c_id is not None:
                    if c_id not in clip_scores: clip_scores[c_id] = []
                    clip_scores[c_id].append(hit.score)

        # Max Score Aggregation (Standard M3)
        final_clip_scores = {cid: max(scores) for cid, scores in clip_scores.items()}
        
        # Filter existing clips
        sorted_clips = sorted(final_clip_scores.items(), key=lambda x: x[1], reverse=True)
        new_top_clips = []
        for cid, _ in sorted_clips:
            if cid not in current_clips:
                new_top_clips.append(cid)
            if len(new_top_clips) >= top_k:
                break
        
        # 4. Retrieve & Translate Content
        new_memories = {}
        for cid in new_top_clips:
            # Cypher to get all text for this clip
            cypher = """
            MATCH (c:Clip {id: $clip_id, video_id: $video_id})-[:HAS_MEMORY]->(m:Memory)
            RETURN m.content as content
            """
            results = self.engine.graph_store.run_query(cypher, {"clip_id": cid, "video_id": self.video_id})
            
            raw_texts = [r['content'] for r in results]
            
            # Forward Translate: <face_1> -> <character_0> (for LLM readability)
            translated_texts = [self.identity_manager.translate_content(t) for t in raw_texts]
            
            new_memories[f"CLIP_{cid}"] = translated_texts
            current_clips.append(cid)
            
        return new_memories, current_clips

    # ------------------------------------------------------------------
    # 2. LLM CALL HELPERS
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict]):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return ""

    def generate_plan(self, question: str) -> str:
        """Phase 1: Generate Retrieval Plan"""
        prompt = PROMPT_GENERATE_PLAN.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        logger.info("🧠 Generating Retrieval Plan...")
        return self._call_llm(messages)

    # ------------------------------------------------------------------
    # 3. CONTROL LOOP (M3 State Machine)
    # ------------------------------------------------------------------

    def run(self, question: str, max_steps: int = 5, multiple_queries: bool = False):
        """
        Executes the M3 Control Loop:
        Plan -> Reasoning -> Action -> Search -> Strategy Switch -> Final Answer.
        """
        # 1. Generate Plan
        retrieval_plan = self.generate_plan(question)
        logger.info(f"📝 Plan: {retrieval_plan}")

        context = []  # Stores reasoning/query history
        current_clips = []
        switch_strategy = False # Flag for empty results

        for step in range(max_steps):
            logger.info(f"\n--- Round {step+1}/{max_steps} (Switch: {switch_strategy}) ---")

            # 2. Select Prompt based on State
            if not switch_strategy:
                if multiple_queries:
                    prompt_template = PROMPT_ACTION_MULTIPLE_QUERIES
                else:
                    prompt_template = PROMPT_ACTION_WITH_PLAN
            else:
                # Use "New Direction" prompt if previous search failed
                # M3 uses prompt_generate_action_with_plan_new_direction
                prompt_template = PROMPT_ACTION_NEW_DIRECTION

            formatted_prompt = prompt_template.format(
                question=question,
                retrieval_plan=retrieval_plan,
                knowledge=json.dumps(context, indent=2, ensure_ascii=False)
            )

            messages = [{"role": "user", "content": formatted_prompt}]
            
            # 3. LLM Action Decision
            response_text = self._call_llm(messages)
            logger.info(f"🤖 Agent: {response_text[:200]}...")

            # 4. Parse Output
            reasoning = ""
            action_type = ""
            action_content = None

            if "[ANSWER]" in response_text:
                parts = response_text.split("[ANSWER]")
                reasoning = parts[0].strip()
                action_type = "answer"
                action_content = parts[1].strip()
            elif "[SEARCH]" in response_text:
                parts = response_text.split("[SEARCH]")
                reasoning = parts[0].strip()
                action_type = "search"
                raw_content = parts[1].strip()
                
                # Parse search content (List or String)
                try:
                    # Try parsing as python list if multiple queries
                    parsed_list = ast.literal_eval(raw_content)
                    if isinstance(parsed_list, list):
                        action_content = parsed_list
                    else:
                        action_content = [raw_content]
                except:
                    # Fallback if just a string
                    action_content = [raw_content]

            # 5. Execute Action
            if action_type == "answer":
                print(f"\n✅ FINAL ANSWER: {action_content}")
                return action_content
            
            elif action_type == "search":
                # Perform Search
                new_mems, updated_clips = self.retrieve(action_content, current_clips)
                current_clips = updated_clips
                
                # Check for Strategy Switch (Empty result)
                if not new_mems:
                    logger.warning("⚠️ No new memories found. Switching strategy for next round.")
                    switch_strategy = True
                else:
                    switch_strategy = False # Reset if we found something

                # Update Context
                context.append({
                    "reasoning": reasoning,
                    "query": action_content,
                    "retrieved_memories": new_mems
                })

        # 6. Forced Final Answer (if max steps reached)
        logger.info("🛑 Max steps reached. Forcing final answer.")
        final_prompt = PROMPT_FINAL_ANSWER.format(
            question=question,
            information=json.dumps(context, ensure_ascii=False)
        )
        final_response = self._call_llm([{"role": "user", "content": final_prompt}])
        
        final_ans = final_response.split("[ANSWER]")[-1].strip() if "[ANSWER]" in final_response else final_response
        print(f"\n✅ FORCED ANSWER: {final_ans}")
        return final_ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/api_config.json")
    parser.add_argument("--multi", action="store_true", help="Enable multiple queries per step")
    args = parser.parse_args()

    agent = M3Controller(args.config, args.video_id)
    agent.run(args.query, multiple_queries=args.multi)