import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
import openai
from conclave.agent.m3_prompts import (
    prompt_generate_plan,
    prompt_generate_action_with_plan,
    prompt_generate_action_with_plan_multiple_queries,
)

# Dynamic path handling
import os
import sys

# Add parent directory to path for package imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from conclave.core.engine import ConclaveEngine
from conclave.core.embedding_service import EmbeddingService
import ast

# Configure logging to be clean for CLI output
logging.basicConfig(level=logging.ERROR) 

class ConclaveQuerier:
    def __init__(self, config_path: str, video_id: str):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # We need the embedder to turn the question into a vector
        self.embedder = EmbeddingService(self.config["text-embedding-3-small"])
        
        # LLM Client
        api_conf = self.config.get("api", {})
        self.model = api_conf.get("model", "gemini-3-flash-preview")
        
        if "gemini" in self.model.lower():
            api_key = api_conf.get("gemini_api_key")
            base_url = api_conf.get("gemini_base_url")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            api_key = api_conf.get("openai_api_key") or api_conf.get("api_key")
            self.client = openai.OpenAI(api_key=api_key)

    def retrieve_knowledge(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs GraphRAG: Vector Search -> Graph Context Expansion
        """
        print(f"🔍 Searching Memory Bank for: '{query_text}'...")
        
        # 1. Embed Question
        query_vec = self.embedder.get_embeddings_batched([query_text])[0]
        
        # 2. Vector Search (Qdrant) - Find relevant episodic/semantic memories
        # We search 'text_memories' which contains the synthesis of Visual+Face+Voice
        hits = self.engine.vector_store.search(
            collection="text_memories",
            vector=query_vec,
            filter_kv={"video_id": self.video_id},
            limit=top_k
        )
        
        context_results = []
        
        for hit in hits:
            mem_id = hit.id
            score = hit.score
            content = hit.payload.get("content")
            clip_id = hit.payload.get("clip_id")
            
            # 3. Graph Expansion (Neo4j)
            # Find entities mentioned in this memory AND entities physically present in the clip
            query = """
            MATCH (m:Memory {id: $mem_id})
            // Get explicitly linked entities
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            WITH m, collect(e.id) as mentioned_entities
            
            // Get co-occurring entities in the same clip (Spatial Context)
            MATCH (c:Clip {id: $clip_id, video_id: $video_id})
            OPTIONAL MATCH (p:Entity)-[r:APPEARED_IN]->(c)
            
            RETURN mentioned_entities, collect({id: p.id, type: p.type, ts: r.ts_ms}) as present_entities
            """
            
            graph_data = self.engine.graph_store.run_query(query, {
                "mem_id": mem_id, 
                "clip_id": clip_id, 
                "video_id": self.video_id
            })
            
            context_results.append({
                "clip_id": clip_id,
                "score": score,
                "memory_text": content,
                "graph_context": graph_data[0] if graph_data else {}
            })
            
        return context_results

    def iterative_reasoning_loop(self, query: str, max_steps: int = 5):
        """
        M3-Agent Logic: Search -> Reason -> Answer Loop
        """
        current_knowledge = []
        history = []

        system_prompt = """
You are an intelligent agent answering questions about a long video.
You have a memory bank you can search.

Process:
1. Analyze the user question and current knowledge.
2. Decide if you have enough info to answer.
3. If NO: Output "Action: [Search] <query>"
4. If YES: Output "Action: [Answer] <final_answer>"

Constraint: Use specific entity IDs (e.g., face_1, voice_2) in searches if known.
"""

        for step in range(max_steps):
            context_str = "\n".join([f"KB Item: {k}" for k in current_knowledge])
            prompt = f"Question: {query}\n\nRetrieved Knowledge:\n{context_str}\n\nHistory:\n{history}\n\nWhat is your next action?"

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )

                content = response.choices[0].message.content
                history.append(content)
                print(f"Step {step+1}: {content}")

                if "[Answer]" in content:
                    final_ans = content.split("[Answer]", 1)[1].strip()
                    print(f"✅ Final Answer: {final_ans}")
                    return final_ans

                elif "[Search]" in content:
                    search_q = content.split("[Search]", 1)[1].strip()
                    vec = self.embedder.get_embeddings_batched([search_q])[0]
                    hits = self.engine.vector_store.search(self.engine.collections["text"], vec, {"video_id": self.video_id}, limit=3)
                    new_info = [h.payload.get('content') for h in hits if h.payload]
                    current_knowledge.extend(new_info)

                else:
                    print("⚠️ Unrecognized action, performing default search.")
                    vec = self.embedder.get_embeddings_batched([query])[0]
                    hits = self.engine.vector_store.search(self.engine.collections["text"], vec, {"video_id": self.video_id}, limit=3)
                    new_info = [h.payload.get('content') for h in hits if h.payload]
                    current_knowledge.extend(new_info)

            except Exception as e:
                print(f"LLM decision error: {e}")

        print("❌ Max steps reached.")
        return None

    def execute_m3_control_loop(self, question: str):
        """
        The M3-Agent Controller Logic.
        1. Generate Plan
        2. Iterative Loop (Search <-> Reason)
        3. Final Answer
        """
        print(f"❓ Question: {question}")

        # --- STEP 1: Generate Retrieval Plan ---
        plan_messages = [
            {"role": "system", "content": prompt_generate_plan.format(question=question)}
        ]
        plan_resp = self.client.chat.completions.create(
            model=self.model, messages=plan_messages
        )
        retrieval_plan = plan_resp.choices[0].message.content
        print(f"📋 Plan: {retrieval_plan}")

        # --- STEP 2: The Control Loop ---
        knowledge_context = [] # List of strings
        max_steps = 5
        
        for step in range(max_steps):
            # Prepare context for the "Thinker"
            # We construct the prompt dynamically based on collected knowledge
            knowledge_str = json.dumps(knowledge_context, indent=2)
            
            # Select prompt: If it's the first step, simple plan. 
            # If previous steps failed, M3 implies using _multiple_queries or _new_direction.
            # We'll use the robust _multiple_queries prompt for general cases.
            prompt_template = prompt_generate_action_with_plan_multiple_queries
            
            user_content = prompt_template.format(
                question=question,
                retrieval_plan=retrieval_plan,
                knowledge=knowledge_str
            )

            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_content}]
            )
            raw_output = response.choices[0].message.content

            # --- STEP 3: Parse Action ---
            if "[ANSWER]" in raw_output:
                # Extraction logic
                final_answer = raw_output.split("[ANSWER]")[1].strip()
                print(f"✅ Final Answer: {final_answer}")
                return final_answer
            
            elif "[SEARCH]" in raw_output:
                # Extraction logic for queries
                # M3 often outputs: [SEARCH] ["query1", "query2"]
                search_part = raw_output.split("[SEARCH]", 1)[1].strip()
                
                # Attempt robust parsing
                queries = []
                try:
                    # 1. Try to find a JSON-like list in the text
                    import re
                    json_match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', search_part, re.DOTALL)
                    if json_match:
                        queries = ast.literal_eval(json_match.group(0))
                    else:
                        # 2. Try literal_eval of the whole part
                        queries = ast.literal_eval(search_part)
                except Exception:
                    # 3. Fallback: Split by lines if it looks like a list
                    lines = [l.strip("-*•123456789. ") for l in search_part.splitlines() if l.strip()]
                    queries = [l for l in lines if l]
                
                # Final cleanup
                if isinstance(queries, str):
                    queries = [queries]
                elif not isinstance(queries, list):
                    queries = [str(queries)]
                
                # Filter out any non-string items or empty strings
                queries = [q for q in queries if isinstance(q, str) and q.strip()]
                
                if not queries:
                    # Last ditch fallback: use the first line of search_part
                    first_line = search_part.splitlines()[0].strip() if search_part else ""
                    if first_line:
                        queries = [first_line]

                print(f"🔍 Step {step+1} Searching: {queries}")


                # Execute Searches
                new_info = []
                for q in queries:
                    # Handle "CLIP_x" queries (M3 Logic)
                    if "CLIP_" in q:
                         # Extraction of clip ID logic would go here
                         # For now, treat as vector search
                         pass
                    
                    # Vector Search
                    q_vec = self.embedder.get_embeddings_batched([q])[0]
                    hits = self.engine.vector_store.search("text_memories", q_vec, {"video_id": self.video_id}, limit=2)
                    
                    for h in hits:
                        # M3 Format: {"query": q, "related_memories": ...}
                        new_info.append({
                            "query": q,
                            "found": h.payload['content']
                        })

                if not new_info:
                    print("⚠️ No new info found. Forcing next step to rethink.")
                    knowledge_context.append({"system": "Previous search returned no results."})
                else:
                    knowledge_context.extend(new_info)

            else:
                print("⚠️ LLM confusion. Retrying...")

        print("❌ Max steps reached without answer.")

    def generate_answer(self, user_query: str, context: List[Dict[str, Any]]):
        """
        Synthesizes the answer using LLM + Context
        """
        if not context:
            print("❌ No relevant information found in the video memory.")
            return

        # Format context for the LLM
        context_str = ""
        for item in sorted(context, key=lambda x: x['clip_id']):
            clip_idx = item['clip_id']
            text = item['memory_text']
            
            # Parse graph data
            g = item['graph_context']
            mentions = g.get('mentioned_entities', [])
            present = [p['id'] for p in g.get('present_entities', []) if p['id']]
            
            # Deduplicate entities
            all_entities = list(set(mentions + present))
            
            context_str += f"""
            [Clip ID: {clip_idx}]
            Description: {text}
            Related Entities (IDs): {all_entities}
            ------------------------------------------------
            """

        system_prompt = """
        You are Conclave, an advanced video intelligence AI.
        Answer the user's question based strictly on the provided Context.
        
        FORMATTING RULES:
        1. **Direct Answer**: Start with a clear, direct answer.
        2. **Reasoning**: Explain *why* you concluded this based on the visual/audio evidence in the context.
        3. **Citations**: You MUST cite the [Clip ID] for every claim.
        4. **Timeline**: If describing a sequence, provide a chronological breakdown.
        5. **Entities**: If referring to a specific person/object ID (e.g., ent_face_...), refer to them as "Person <ID>" unless a name is available in the text.
        
        If the information is missing, state clearly that it is not in the processed video memory.
        """

        print("🧠 Reasoning...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
                ],
                temperature=0.3 # Low temperature for factual accuracy
            )
            
            answer = response.choices[0].message.content
            
            print("\n" + "="*60)
            print(f"🤖 CONCLAVE ANSWER")
            print("="*60)
            print(answer)
            print("="*60)
            
        except Exception as e:
            print(f"Error generating answer: {e}")

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
        # Run iterative M3 control loop
        querier.execute_m3_control_loop(args.query)
    else:
        # Standard flow
        # 1. Retrieve
        knowledge = querier.retrieve_knowledge(args.query)
        
        # 2. Answer
        querier.generate_answer(args.query, knowledge)