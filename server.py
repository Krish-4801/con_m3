import os
import sys
import json
import logging
import asyncio
import importlib
from typing import Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

# Ensure local modules are found
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules so they can be reloaded during requests
import conclave.perception.vision.scene
import conclave.perception.vision.face
import conclave.perception.audio.voice
import conclave.core.engine
import main
import query_system

from conclave.perception.vision.scene import SceneProcessor
from conclave.perception.vision.face import FaceProcessor
from conclave.perception.audio.voice import VoiceProcessor
from conclave.core.engine import ConclaveEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("Conclave.Server")

MODELS: Dict[str, object] = {}
CONFIG_PATH = "configs/api_config.json"


class CachedOrchestrator(main.ConclaveOrchestrator):
    """
    Subclass of ConclaveOrchestrator that uses preloaded model instances
    provided by the server lifespan to avoid reloading heavy models.
    """
    def __init__(self, config_path: str, video_id: str, preloaded_models: Dict[str, object]):
        self._preloaded = preloaded_models
        super().__init__(config_path, video_id)

    def _load_face_model(self):
        logger.info("⚡ Using Preloaded FaceProcessor")
        return self._preloaded["face"]

    def _load_voice_model(self):
        logger.info("⚡ Using Preloaded VoiceProcessor")
        return self._preloaded["voice"]

    def _load_scene_model(self):
        logger.info("⚡ Using Preloaded SceneProcessor")
        return self._preloaded["scene"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy models on startup and keep them resident.
    """
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    logger.info("🔥 WARMING UP SERVER - Loading Heavy Models into GPU...")

    base_process_config = config.get("processing", {})
    gemini_config = config.get("gemini", {})

    scene_config = base_process_config.copy()
    if gemini_config:
        scene_config["gemini"] = gemini_config

    voice_config = base_process_config.copy()

    # Load heavy models once and store
    MODELS["scene"] = SceneProcessor(scene_config)
    logger.info("✅ SceneProcessor Loaded")

    MODELS["face"] = FaceProcessor(base_process_config)
    logger.info("✅ FaceProcessor Loaded")

    MODELS["voice"] = VoiceProcessor(voice_config)
    logger.info("✅ VoiceProcessor Loaded")

    logger.info("🚀 SERVER READY. Waiting for requests...")
    yield

    MODELS.clear()
    logger.info("🛑 Server shutdown.")


app = FastAPI(lifespan=lifespan)


class IngestRequest(BaseModel):
    video_path: str
    video_id: str


class QueryRequest(BaseModel):
    query: str
    video_id: str


@app.post("/ingest")
async def ingest_video(req: IngestRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    def _run_pipeline_task(video_path: str, video_id: str):
        logger.info(f"▶️ Starting Background Pipeline for {video_id}")
        try:
            # Reload logic modules to pick up code changes without restarting server
            importlib.reload(conclave.agent.reasoning)
            importlib.reload(main)

            orch = CachedOrchestrator(config_path=CONFIG_PATH, video_id=video_id, preloaded_models=MODELS)
            orch.run_pipeline(video_path)
            logger.info(f"✅ Finished Pipeline for {video_id}")
        except Exception as e:
            logger.error(f"❌ Pipeline Failed: {e}", exc_info=True)

    background_tasks.add_task(_run_pipeline_task, req.video_path, req.video_id)
    return {"status": "processing_started", "video_id": req.video_id}


@app.post("/query")
def query_knowledge(req: QueryRequest):
    logger.info(f"❓ Querying {req.video_id}: {req.query}")
    try:
        importlib.reload(query_system)

        agent = query_system.M3Controller(config_path=CONFIG_PATH, video_id=req.video_id)
        answer = agent.run(req.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
