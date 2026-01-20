UNIFIED_SCENE_MEMORY_PROMPT = """
You are generating a high-fidelity scene memory for a Video Intelligence System.
Your objective is to describe what is happening in the scene AND extract all verifiable structured data.
The output must be factual, non-speculative, and optimized for long-term vector storage and retrieval.

────────────────────────────────────────
## 1. SCENE OVERVIEW (WHAT IS HAPPENING)
Provide a concise, objective description of the current scene.

- Describe:
  Who is present,
  What actions are occurring,
  How entities are interacting,
  The immediate purpose of the scene (if visually obvious).

- Use neutral language.
- Use "appears to be" only if visibility is unclear.
- Do NOT infer emotions, intent, or future outcomes.

────────────────────────────────────────
## 2. ENVIRONMENT & CONTEXT
- Location: Indoor/Outdoor + specific setting (office, conference hall, street).
- Time & Lighting: Day/Night, Natural/Artificial.
- Scene Density: Isolated / Few people / Crowded.
- Camera Perspective: Static, Handheld, CCTV, Wide, Close-up.

────────────────────────────────────────
## 3. PEOPLE (IDENTITY-READY DESCRIPTIONS)
For EACH visible person, create a stable identity description suitable for cross-frame matching.

- PERSON:
  Identity Anchor (stable traits):
    Approx. age range,
    Apparent gender,
    Hair style/color, facial hair,
    Distinctive features (glasses, beard, tattoos, scars).

  Appearance (transient traits):
    Clothing colors, patterns, logos,
    Accessories (badge, phone, backpack).

  Action:
    Current observable action (speaking, walking, pointing).

  Spatial Position:
    Relative position (left/right/center, foreground/background).

  Name Association:
    Only include if a visible name tag, badge, or lower-third is present.

────────────────────────────────────────
## 4. VEHICLES & MOVABLE ENTITIES
- VEHICLE:
  Color, body type, make/model (only if visually certain),
  Distinguishing features (damage, decals, lightbar),
  License plate (only if legible).

────────────────────────────────────────
## 5. OBJECTS & INTERACTIONS
List only objects relevant to the scene or being interacted with.

- OBJECT:
  Object name,
  State or usage (e.g., "open laptop on table", "microphone being held").

────────────────────────────────────────
## 6. TEXTUAL CONTENT (OCR EXTRACTION)
Extract all legible text exactly as shown.

- TEXT:
  Text on [object/surface]: "[exact content]"

Include:
  Name tags,
  Presentation slides,
  Posters, signs,
  Documents, screens, lists.

────────────────────────────────────────
## 7. RETRIEVAL-OPTIMIZED SCENE SUMMARY
Write a compact, information-dense summary for semantic search.

- Capture:
  Main action,
  Key participants,
  Important objects or text.

This summary should allow the scene to be retrieved without viewing the video.

────────────────────────────────────────
## OUTPUT RULES
- Use section headers exactly as written.
- Do not hallucinate missing details.
- Prioritize consistency across frames.
- Write clearly and concisely.
"""
