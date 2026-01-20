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
from conclave.prompts.reasoning import M3_SYSTEM_PROMPT

logger = logging.getLogger("Conclave.ReasoningAgent")

class ReasoningAgent:
    """
    M3-Agent Logic Implementation:
    Generates 'Episodic' descriptions and 'Semantic' equivalences.
    Strictly enforces <face_uuid> and <voice_uuid> tagging to enable Graph resolution.
    """

    def __init__(self, config: Dict[str, Any]):
        # Support various config structures (api section vs root)
        self.model = config.get("model", "gemini-1.5-flash") # Fallback to a valid default
        
        # Handle API Key extraction
        api_key = config.get("openai_api_key") or config.get("api_key") or config.get("gemini_api_key")
        base_url = config.get("base_url") or config.get("gemini_base_url")

        if "gemini" in self.model.lower():
            # Ensure we have the Gemini endpoint if using a Gemini model
            if not base_url and "googleapis" not in (base_url or ""):
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = openai.OpenAI(api_key=api_key)

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
        Formats raw perceptions into a "Screenplay" format for the LLM.
        """
        context_blocks = []

        # 1. Visual Context
        descriptions = []
        ocr_texts = []
        for v in visuals:
            spatial = getattr(v, "spatial_metadata", {})
            desc = spatial.get("dense_description")
            if desc: descriptions.append(desc)
            if v.ocr_tokens:
                ocr_texts.extend([t["text"] for t in v.ocr_tokens])
        
        unique_desc = list(set(descriptions))
        scene_summary = " ".join(unique_desc[:3]) 
        unique_ocr = ", ".join(list(set(ocr_texts)))

        context_blocks.append(f"### Visual Scene Summary:\n{scene_summary}")
        if unique_ocr:
            context_blocks.append(f"### Visible Text/Slides:\n{unique_ocr}")

        # 2. Face Context
        context_blocks.append("Face features:")
        if faces:
            unique_faces = sorted(list(set([f.entity_id for f in faces if f.entity_id])))
            for f_id in unique_faces:
                 context_blocks.append(f"<{f_id}> detected.")
        else:
            context_blocks.append("No faces detected.")

        # 3. Voice Context
        context_blocks.append("Voice features:")
        if voices:
            for v in voices:
                if v.entity_id and len(v.asr_text) > 2: 
                    context_blocks.append(f"<{v.entity_id}>: {v.asr_text}")
        else:
            context_blocks.append("No voices detected.")

        return "\n\n".join(context_blocks)

    # ------------------------------------------------------------------
    # REASONING CORE (The Missing Method!)
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
        
        # 2. M3 System Prompt
        system_prompt = M3_SYSTEM_PROMPT

        try:
            # 3. LLM Call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Perception Data:\n{context_str}"}
                ],
                response_format={"type": "json_object"},
                temperature=1.0 # High temp for Gemini creativity
            )
            
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            
            memories = []


            # 4. Process Episodic
            for text in parsed.get("episodic_memory", []):
                memories.append(MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id,
                    content=text,
                    mem_type=MemoryType.EPISODIC,
                    linked_entities=self._extract_tags(text)
                ))

            # 5. Process Semantic (includes Equivalences in this prompt)
            for text in parsed.get("semantic_memory", []):
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