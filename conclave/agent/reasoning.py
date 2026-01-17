import json
import logging
import re
from typing import List, Dict, Any, Tuple
import openai

from core.schemas import (
    FaceObservation,
    VoiceObservation,
    VisualObservation,
    MemoryNode,
    MemoryType,
)

logger = logging.getLogger("Conclave.ReasoningAgent")

class ReasoningAgent:
    """
    M3-Agent Logic Implementation:
    Generates 'Episodic' descriptions and 'Semantic' equivalences.
    Strictly enforces <face_uuid> and <voice_uuid> tagging to enable Graph resolution.
    """

    def __init__(self, config: Dict[str, Any]):
        api_key = config.get("openai_api_key") or config.get("api_key")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o") # M3 uses Qwen/GPT-4o
        self.max_retries = 3

    # ------------------------------------------------------------------
    # CONTEXT PREPARATION (M3 Style)
    # ------------------------------------------------------------------

    def _prepare_m3_context(
        self,
        visuals: List[VisualObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> str:
        """
        Formats raw perceptions into the specific layout M3-Agent expects.
        Explicitly maps IDs to descriptions to help the LLM link them.
        """
        context_blocks = []

        # 1. Visual Context (Scene Description)
        if visuals:
            # Use the most representative visual description (middle of clip)
            mid_v = visuals[len(visuals)//2]
            spatial = getattr(mid_v, "spatial_metadata", {})
            desc = spatial.get("dense_description", "A video clip.")
            ocr = ", ".join([t["text"] for t in mid_v.ocr_tokens]) if mid_v.ocr_tokens else "None"
            context_blocks.append(f"### Visual Scene:\n{desc}\nVisible Text: {ocr}")

        # 2. Face Features (List available tags)
        if faces:
            context_blocks.append("### Detected Faces (Visual Entities):")
            # Group by resolved Entity ID
            seen_entities = set()
            for f in faces:
                if f.entity_id and f.entity_id not in seen_entities:
                    seen_entities.add(f.entity_id)
                    # Note: In a real M3 implementation, we would describe the face's appearance
                    # (e.g., "Man with glasses"). Here we rely on the vector ID.
                    context_blocks.append(f"- Entity Tag: <{f.entity_id}> (Visible at {f.ts_ms}ms)")

        # 3. Voice Features (List available tags + Transcripts)
        if voices:
            context_blocks.append("### Detected Voices (Audio Entities):")
            for v in voices:
                if v.entity_id:
                    context_blocks.append(
                        f"- Entity Tag: <{v.entity_id}>\n"
                        f"  Transcript: '{v.asr_text}' (Time: {v.start_sec}-{v.end_sec}s)"
                    )

        return "\n\n".join(context_blocks)

    # ------------------------------------------------------------------
    # REASONING CORE
    # ------------------------------------------------------------------

    def generate_memory_structures(
        self,
        video_id: str,
        clip_id: int,
        visuals: List[VisualObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> List[MemoryNode]:
        """
        Generates BOTH Episodic (Event) and Semantic (Equivalence) memories in one pass.
        This mirrors M3's `generate_all_memories` function.
        """
        
        # 1. Build Context
        context_str = self._prepare_m3_context(visuals, faces, voices)
        
        # 2. M3 System Prompt
        # Key instruction: "Equivalence: <face_x> is <voice_y>"
        system_prompt = """
        You are the M3-Agent Multimodal Reasoning Engine.
        Your goal is to turn raw perception data into structured memory.
        
        INPUT DATA:
        - Visual Scene: Description of the environment.
        - Faces: List of <face_uuid> entities present.
        - Voices: List of <voice_uuid> entities and what they said.

        TASK:
        1. **Episodic Memory**: Write a chronological description of what happened. 
           - YOU MUST USE THE ENTITY TAGS (e.g., <face_...>, <voice_...>) when referring to people.
           - Example: "<face_a1> entered the room and argued with <face_b2>."
           - Example: "<voice_c3> said 'Hello' while <face_d4> waved."

        2. **Semantic Memory (Equivalences)**: 
           - Analyze the timing and context. Does a voice belong to a specific face?
           - If you are confident, output an equivalence statement.
           - Format: "Equivalence: <face_uuid> is <voice_uuid>"
           - Format: "Equivalence: <face_uuid> is the same as <face_uuid>" (if visual appearance matches context).

        OUTPUT FORMAT (JSON):
        {
            "episodic": ["string", "string"],
            "semantic": ["string", "string"]
        }
        """

        try:
            # 3. LLM Call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Perception Data:\n{context_str}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2 # Low temperature for consistent tagging
            )
            
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            
            memories = []

            # 4. Process Episodic
            for text in parsed.get("episodic", []):
                memories.append(MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id,
                    content=text,
                    mem_type=MemoryType.EPISODIC,
                    linked_entities=self._extract_tags(text)
                ))

            # 5. Process Semantic (The triggers for Union-Find)
            for text in parsed.get("semantic", []):
                memories.append(MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id, # Semantic can be tied to clip initially
                    content=text,
                    mem_type=MemoryType.SEMANTIC,
                    linked_entities=self._extract_tags(text)
                ))

            return memories

        except Exception as e:
            logger.error(f"Reasoning Error: {e}")
            return []

    def _extract_tags(self, text: str) -> List[str]:
        """Regex to pull <face_...> and <voice_...> tags for graph linking."""
        pattern = r'<((?:face|voice)_[a-zA-Z0-9\-]+)>'
        return list(set(re.findall(pattern, text)))

    # ------------------------------------------------------------------
    # (Optional) High-Level Distillation (Cross-Clip)
    # ------------------------------------------------------------------

    def distill_cross_clip_knowledge(self, video_id: str, recent_memories: List[str]) -> List[MemoryNode]:
        """
        M3 feature: Summarize multiple clips to find long-range equivalences 
        or plot points.
        """
        # Implementation would gather recent text nodes and ask LLM for summary.
        # Keeping it simple for now to focus on the Identity Logic.
        return []
