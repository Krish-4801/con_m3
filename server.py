import os
import sys
import json
import logging
import asyncio
import importlib
from typing import Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn

# Add local directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
# --- MODULE IMPORTS (We import modules, not classes, so we can reload them) ---
import conclave.perception.vision.scene
import conclave.perception.vision.face
import conclave.perception.audio.voice
import conclave.agent.reasoning
import conclave.core.engine
import main           # The Orchestrator module
import query_system   # The M3 Controller module

# Import classes for type hinting / initial loading
from conclave.perception.vision.scene import SceneProcessor
from conclave.perception.vision.face import FaceProcessor
from conclave.perception.audio.voice import VoiceProcessor
from conclave.core.engine import ConclaveEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("Conclave.Server")

# --- Global State ---
MODELS = {}
CONFIG_PATH = "configs/api_config.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load HEAVY models on startup. These stay in VRAM.
    We do NOT load lightweight logic classes (like ReasoningAgent) here anymore.
    """
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    logger.info("🔥 WARMING UP SERVER - Loading Heavy Models into GPU...")
    
    # Merge Configs
    base_process_config = config.get("processing", {})
    api_config = config.get("api", {})
    gemini_config = config.get("gemini", {})

    scene_config = base_process_config.copy()
    scene_config["gemini"] = gemini_config

    voice_config = base_process_config.copy()
    voice_config.update(api_config)

    # --- LOAD HEAVY MODELS ONLY ---
    MODELS["scene"] = SceneProcessor(scene_config)
    logger.info("✅ SceneProcessor Loaded")
    
    MODELS["face"] = FaceProcessor(base_process_config)
    logger.info("✅ FaceProcessor Loaded")
    
    MODELS["voice"] = VoiceProcessor(voice_config)
    logger.info("✅ VoiceProcessor Loaded")
    
    # We keep the Engine connection open, but we will reload the class definitions
    MODELS["engine"] = ConclaveEngine(video_id="INIT", config_path=CONFIG_PATH)
    logger.info("✅ Engine Connection Established")
    
    logger.info("🚀 SERVER READY (Hot-Reload Enabled). Waiting for requests...")
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

    def _run_pipeline_task(video_path, video_id):
        logger.info(f"▶️ Starting Background Pipeline for {video_id}")
        try:
            # --- HOT RELOAD LOGIC ---
            logger.info("🔄 Reloading Logic Modules...")
            importlib.reload(conclave.agent.reasoning) # Reload reasoning first (dependency)
            importlib.reload(main)                     # Reload Orchestrator
            
            # Re-read config in case you changed prompts/keys
            with open(CONFIG_PATH) as f:
                new_config = json.load(f)

            # Create FRESH Reasoning Agent (fast, no VRAM)
            # This ensures prompt changes in reasoning.py are picked up immediately
            new_reasoning_agent = conclave.agent.reasoning.ReasoningAgent(new_config.get("api", {}))
            
            # Inject the fresh agent into the model dict temporarily
            # We copy the global dict so we don't mess up other requests
            current_models = MODELS.copy()
            current_models["reasoning"] = new_reasoning_agent

            # Initialize FRESH Orchestrator class
            orch = main.ConclaveOrchestrator(
                config_path=CONFIG_PATH,
                video_id=video_id,
                preloaded_models=current_models
            )
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
        # --- HOT RELOAD LOGIC ---
        logger.info("🔄 Reloading Query System...")
        importlib.reload(conclave.core.identity) # Just in case identity logic changed
        importlib.reload(query_system)
        
        # Initialize FRESH Controller class
        # It will use the EXISTING Engine connection (fast) but NEW logic code
        agent = query_system.M3Controller(
            config_path=CONFIG_PATH,
            video_id=req.video_id,
            preloaded_engine=MODELS["engine"]
        )
        
        answer = agent.run(req.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)