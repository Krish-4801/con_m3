import json
import logging
import re
from typing import List, Dict, Any
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
    """
    State-of-the-art reasoning engine with:
    - Strict JSON enforcement
    - Automatic repair & retry loop
    - Knowledge Graph–safe outputs
    """

    def __init__(self, config: Dict[str, Any]):
        
        api_key = config.get("openai_api_key") or config.get("api_key")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-5-mini")
        self.max_retries = config.get("max_retries", 3)

        # Matches <ent_uuid> or raw UUIDs
        self.entity_pattern = re.compile(r"<(ent_[a-zA-Z0-9_]+|[a-f0-9\-]{36})>")

    # ------------------------------------------------------------------
    # JSON SAFETY
    # ------------------------------------------------------------------

    def _clean_json_string(self, raw: str) -> str:
        """Removes markdown wrappers and common LLM formatting mistakes."""
        cleaned = raw.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"```[a-zA-Z]*\n?", "", cleaned)
            cleaned = cleaned.rstrip("```")

        return cleaned.strip()

    def _safe_json_load(self, raw: str) -> Any:
        cleaned = self._clean_json_string(raw)
        return json.loads(cleaned)

    def _llm_json_call(
        self,
        system_prompt: str,
        user_prompt: str,
        required_key: str,
    ) -> Any:
        """
        Executes a strict JSON call with validation + repair loop.
        Forces object output and extracts required_key.
        """

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                            + "\nOUTPUT ONLY VALID JSON. NO MARKDOWN.",
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )

                raw = response.choices[0].message.content
                parsed = self._safe_json_load(raw)

                if required_key not in parsed:
                    raise KeyError(f"Missing key '{required_key}'")

                return parsed[required_key]

            except Exception as e:
                logger.warning(
                    f"JSON attempt {attempt}/{self.max_retries} failed: {e}"
                )

        logger.error("All JSON repair attempts failed")
        return []

    # ------------------------------------------------------------------
    # MULTIMODAL CONTEXT
    # ------------------------------------------------------------------

    def _prepare_multimodal_context(
        self,
        visuals: List[VisualObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> str:
        """
        Converts raw perception into LLM-readable structured context.
        """

        blocks: List[str] = []

        # Visuals
        for v in visuals:
            spatial = getattr(v, "spatial_metadata", {}) or {}
            desc = spatial.get("dense_description", "No description")
            ocr = ", ".join(t["text"] for t in v.ocr_tokens) if v.ocr_tokens else ""
            blocks.append(
                f"[{v.ts_ms}ms] Visual: {desc}. Text: [{ocr}]"
            )

        # Aggregate by entity
        entity_events: Dict[str, List[str]] = {}

        for f in faces:
            if f.entity_id:
                entity_events.setdefault(f.entity_id, []).append(
                    f"Face seen at {f.ts_ms}ms"
                )

        for v in voices:
            if v.entity_id:
                entity_events.setdefault(v.entity_id, []).append(
                    f"Spoke at {v.ts_ms}ms: '{v.asr_text}'"
                )

        for eid, events in entity_events.items():
            blocks.append(
                f"[Entity: <{eid}>] " + " | ".join(events)
            )

        return "\n".join(blocks)

    # ------------------------------------------------------------------
    # EPISODIC MEMORY
    # ------------------------------------------------------------------

    def generate_episodic_memory(
        self,
        video_id: str,
        clip_id: int,
        visuals: List[VisualObservation],
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> List[MemoryNode]:

        context = self._prepare_multimodal_context(
            visuals, faces, voices
        )

        system_prompt = """
You are the Conclave Multimodal Reasoning Engine.

RULES:
1. Output JSON only.
2. Return {"memories": [string, ...]}
3. Be factual and concise.
4. Use EXACT entity tags like <ent_uuid>.
5. Do not invent entities.
"""

        user_prompt = f"""
Video ID: {video_id}
Clip ID: {clip_id}

Perception Data:
{context}

Generate episodic memories.
"""

        memories = self._llm_json_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_key="memories",
        )

        nodes: List[MemoryNode] = []

        for text in memories:
            mentions = list(set(self.entity_pattern.findall(text)))

            nodes.append(
                MemoryNode(
                    video_id=video_id,
                    clip_id=clip_id,
                    content=text,
                    mem_type=MemoryType.EPISODIC,
                    linked_entities=mentions,
                )
            )

        return nodes

    def generate_episodic_memory_m3_style(
        self,
        video_id: str,
        clip_id: int,
        visual_obs: VisualObservation,
        faces: List[FaceObservation],
        voices: List[VoiceObservation],
    ) -> List[MemoryNode]:
        """
        M3-Agent Logic: visual-grounded prompt generation.
        We send the LLM the face crops so it associates <face_X> with visual traits.
        """

        # 1. Prepare Face Context (unique faces)
        unique_faces: Dict[str, str] = {}
        for f in faces:
            if getattr(f, "entity_id", None) and getattr(f, "base64_img", None):
                if f.entity_id not in unique_faces:
                    unique_faces[f.entity_id] = f.base64_img

        face_blocks: List[str] = []
        for fid, b64 in unique_faces.items():
            face_blocks.append(f"<{fid}>:")
            face_blocks.append(f"data:image/jpeg;base64,{b64}")

        # 2. Prepare Voice Context
        voice_log = []
        for v in voices:
            voice_log.append({
                "start": getattr(v, "start_sec", None),
                "end": getattr(v, "end_sec", None),
                "content": getattr(v, "asr_text", ""),
                "speaker": f"<{v.entity_id}>" if getattr(v, "entity_id", None) else "unknown",
            })

        voice_text = json.dumps(voice_log, indent=2)

        system_prompt = """
You are an AI agent building a memory graph for a video. 
Describe the video clip events in detail.

CRITICAL RULES:
1. When you see a person from the provided face images, refer to them EXACTLY as <face_id>.
2. When you hear a speaker from the logs, refer to them EXACTLY as <voice_id>.
3. Determine relationships: If <face_0> is moving their lips while <voice_1> speaks, output "Equivalence: <face_0> is <voice_1>".
4. Output JSON: {"episodic_memory": ["..."], "semantic_memory": ["..."], "equivalences": ["..."]}
"""

        user_prompt = (
            f"Video Clip ID: {clip_id}.\n\nReferential Faces:\n"
            + "\n".join(face_blocks)
            + f"\n\nAudio Transcript:\n{voice_text}\n\nDescribe the events." 
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            data = self._safe_json_load(raw)

            nodes: List[MemoryNode] = []
            for mem in data.get("episodic_memory", []):
                nodes.append(
                    MemoryNode(
                        video_id=video_id,
                        clip_id=clip_id,
                        content=mem,
                        mem_type=MemoryType.EPISODIC,
                    )
                )

            for mem in data.get("semantic_memory", []):
                nodes.append(
                    MemoryNode(
                        video_id=video_id,
                        clip_id=clip_id,
                        content=mem,
                        mem_type=MemoryType.SEMANTIC,
                    )
                )

            # Attach any equivalence statements as semantic nodes too
            for eq in data.get("equivalences", []):
                nodes.append(
                    MemoryNode(
                        video_id=video_id,
                        clip_id=clip_id,
                        content=eq,
                        mem_type=MemoryType.SEMANTIC,
                    )
                )

            return nodes

        except Exception as e:
            logger.error(f"M3 Memory Gen Failed: {e}")
            return []

    # ------------------------------------------------------------------
    # SEMANTIC DISTILLATION
    # ------------------------------------------------------------------

    def distillation_pass(
        self,
        video_id: str,
        episodic_nodes: List[MemoryNode],
        existing_semantic_context: str,
    ) -> List[MemoryNode]:

        episodes = "\n".join(n.content for n in episodic_nodes)

        system_prompt = """
You are a knowledge distillation engine.

RULES:
1. Output JSON only.
2. Return {"facts": [string, ...]}
3. Extract long-term, reusable knowledge.
4. Use entity tags <ent_uuid>.
"""

        user_prompt = f"""
Existing Knowledge:
{existing_semantic_context}

New Events:
{episodes}

Extract semantic facts.
"""

        facts = self._llm_json_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_key="facts",
        )

        semantic_nodes: List[MemoryNode] = []

        for fact in facts:
            mentions = list(set(self.entity_pattern.findall(fact)))
            semantic_nodes.append(
                MemoryNode(
                    video_id=video_id,
                    content=fact,
                    mem_type=MemoryType.SEMANTIC,
                    linked_entities=mentions,
                )
            )

        return semantic_nodes
