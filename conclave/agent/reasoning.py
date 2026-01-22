import json
import logging
import re
from typing import List, Dict, Any, Tuple
import openai

from conclave.core.schemas import (
    FaceObservation,
    VoiceObservation,
    VisualObservation,
    MemoryNode,
    MemoryType,
)

logger = logging.getLogger("Conclave.ReasoningAgent")

class ReasoningAgent:

    def __init__(self, config: Dict[str, Any]):
        self.model = config.get("model", "gemini-3-flash-preview")
        
        if "gemini" in self.model.lower():
            api_key = config.get("gemini_api_key")
            base_url = config.get("gemini_base_url")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            api_key = config.get("openai_api_key") or config.get("api_key")
            self.client = openai.OpenAI(api_key=api_key)
        self.max_retries = 3

    # ------------------------------------------------------------------
    # CONTEXT PREPARATION
    # ------------------------------------------------------------------

    def _prepare_m3_context(
        self,
        visuals: List[VisualObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> str:
        """
        Formats raw perceptions into a "Screenplay" format for the LLM.
        Aggregates raw timestamps into availability lists to prevent "log-style" outputs.
        """
        context_blocks = []

        # 1. Visual Context (The Stage)
        # Aggregate unique descriptions to avoid repetition
        descriptions = []
        ocr_texts = []
        for v in visuals:
            spatial = getattr(v, "spatial_metadata", {})
            desc = spatial.get("dense_description")
            if desc: descriptions.append(desc)
            if v.ocr_tokens:
                ocr_texts.extend([t["text"] for t in v.ocr_tokens])
        
        # Deduplicate descriptions loosely
        unique_desc = list(set(descriptions))
        scene_summary = " ".join(unique_desc[:3]) # Take top 3 unique descriptions
        unique_ocr = ", ".join(list(set(ocr_texts)))

        context_blocks.append(f"### Visual Scene Summary:\n{scene_summary}")
        if unique_ocr:
            context_blocks.append(f"### Visible Text/Slides:\n{unique_ocr}")

        # 2. Face Context (The Cast)
        context_blocks.append("Face features:")
        if faces:
            unique_faces = sorted(list(set([f.entity_id for f in faces if f.entity_id])))
            for f_id in unique_faces:
                 context_blocks.append(f"<{f_id}> detected.")
        else:
            context_blocks.append("No faces detected.")

        # 3. Voice Context (The Script)
        context_blocks.append("Voice features:")
        if voices:
            for v in voices:
                if v.entity_id and len(v.asr_text) > 2: # Filter empty noise
                    context_blocks.append(f"<{v.entity_id}>: {v.asr_text}")
        else:
            context_blocks.append("No voices detected.")

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
        """
        
        # 1. Build Context
        context_str = self._prepare_m3_context(visuals, faces, voices)
        
        # Updated to explicitly forbid "was visible" logging style.
        system_prompt = """
        You are a Multimodal Reasoning Engine.
        Your goal is to synthesize distinct perceptions (Visual Scene, Faces, Audio) into a cohesive narrative.

        INPUT DATA:
        - Visual Scene: Detailed description of the environment and actions.
        - Cast: List of <face_uuid> entities present in the video.
        - Transcript: List of <voice_uuid> entities and what they said.

        CRITICAL INSTRUCTIONS:
        1. **NO LOGGING**: Do NOT generate memories like "<face_x> was visible at 400ms" or "<face_y> appeared." These are useless.
        2. **SYNTHESIS**: Combine the Visual Scene description with the Entity Tags.
           - BAD: "<face_1> was seen. The scene shows a man at a podium."
           - GOOD: "<face_1> stood at the podium addressing the audience regarding the conference topics."
        3. **AUDIO INTEGRATION**: Attribute speech to faces based on context. 
           - If <voice_1> says "Welcome", and <face_1> is the only person on screen, write: "<face_1> welcomed the audience."
        
        OUTPUT TASKS:
        1. **Episodic Memory**: A rich, chronological story of the clip. Describe actions, emotions, and specific speech topics.
        2. **Semantic Memory**: High-level facts derived from the clip (e.g., "The event is the 2026 Esper Conference").
        3. **Equivalences**: If you are confident a voice belongs to a face, output an equivalence statement.
           - Format: "Equivalence: <face_uuid> is <voice_uuid>"

        OUTPUT FORMAT (JSON):
        {
            "episodic": ["string (narrative sentence)", "string"],
            "semantic": ["string (fact)", "string"],
            "equivalences": ["string"]
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
                temperature=1.0 # Gemini-3-flash-preview requires 1.0 or default
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
            
            # 6. Process Equivalences (Explicitly add as Semantic nodes for graph logic)
            for text in parsed.get("equivalences", []):
                memories.append(MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id,
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
        pattern = r'<((?:ent_)?(?:face|voice)_[a-zA-Z0-9\-]+)>'
        return list(set(re.findall(pattern, text)))

    # ------------------------------------------------------------------
    # (Optional) High-Level Distillation (Cross-Clip)
    # ------------------------------------------------------------------

    def distill_cross_clip_knowledge(self, video_id: str, recent_memories: List[str]) -> List[MemoryNode]:
        """
        M3 feature: Summarize multiple clips to find long-range equivalences 
        or plot points.
        """
        return []
