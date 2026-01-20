
import os
import sys
import json
import logging
import argparse
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import numpy as np
import concurrent.futures
import queue
import time
import cv2
import ffmpeg
from tqdm import tqdm
from typing import Dict, Any, Optional

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from conclave.core.engine import ConclaveEngine
from conclave.core.identity import IdentityManager
from conclave.agent.reasoning import ReasoningAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Conclave.Orchestrator")

class ConclaveOrchestrator:
    def __init__(self, config_path: str, video_id: str, preloaded_models: Optional[Dict] = None):
        """
        Modified to accept preloaded_models for Server Mode.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # 1. Initialize Engine (Fast - Connection only)
        # If passed in preloaded, use it, otherwise create new
        if preloaded_models and "engine" in preloaded_models:
            self.engine = preloaded_models["engine"]
            # We must update the engine's video_id context manually
            self.engine.video_id = video_id 
        else:
            self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # 2. Initialize Identity Manager
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.config.get("processing", {})
        )
        
        # 3. MODEL LOADING (Conditional)
        if preloaded_models:
            logger.info("🚀 Using Pre-Loaded Models (Server Mode)")
            self.face_proc = preloaded_models["face"]
            self.voice_proc = preloaded_models["voice"]
            self.scene_proc = preloaded_models["scene"]
            self.reasoning_agent = preloaded_models.get("reasoning") or ReasoningAgent(self.config.get("api", {}))
        else:
            logger.info("🚀 Bootstrapping AI Models SEQUENTIALLY (Cold Start)...")
            t0 = time.time()
            self.face_proc = self._load_face_model()
            self.voice_proc = self._load_voice_model()
            self.scene_proc = self._load_scene_model()
            self.reasoning_agent = ReasoningAgent(self.config.get("api", {}))
            logger.info(f"✅ All Models Loaded in {time.time() - t0:.2f}s")

    def _load_face_model(self):
        from conclave.perception.vision.face import FaceProcessor
        logger.info("-> Loading FaceProcessor...")
        return FaceProcessor(self.config.get("processing", {}))

    def _load_voice_model(self):
        from conclave.perception.audio.voice import VoiceProcessor
        logger.info("-> Loading VoiceProcessor...")
        return VoiceProcessor(self.config.get("processing", {}))

    def _load_scene_model(self):
        from conclave.perception.vision.scene import SceneProcessor
        logger.info("-> Loading SceneProcessor (YOLO + SigLIP + Gemini)...")
        process_config = self.config.get("processing", {}).copy()
        if "gemini" in self.config:
            process_config["gemini"] = self.config["gemini"]
        return SceneProcessor(process_config)

    def run_pipeline(self, video_path: str, window_size: int = 30, overlap: int = 3):
        # ... [Rest of the run_pipeline code remains EXACTLY the same] ...
        # Copy the run_pipeline method from the previous main.py here
        # (It is omitted for brevity but assume the logic is identical)
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return

        logger.info(f"🎬 STARTING PIPELINE: {self.video_id}")
        
        try:
            probe = ffmpeg.probe(video_path)
            total_duration = float(probe['format']['duration'])
        except Exception as e:
            logger.error(f"FFProbe failed: {e}")
            return

        stride = window_size - overlap
        estimated_clips = int(np.ceil(total_duration / stride))
        pbar = tqdm(total=estimated_clips, desc="Processing Video Pipeline", unit="clip")

        curr = 0.0
        clip_id = 0
        
        while curr < total_duration:
            try:
                data = self._fast_extract(video_path, curr, window_size)
                current_start = curr
                pbar.set_description(f"Processing Clip {clip_id} [{current_start:.1f}s]")

                visuals = self.scene_proc.process_batch(
                    data["raw_frames"], self.video_id, clip_id, int(current_start * 1000), 1000
                )

                face_stride = 2
                selected_frames = data["raw_frames"][::face_stride]
                faces = []
                if selected_frames:
                    faces = self.face_proc.extract_from_frames(selected_frames, self.video_id, clip_id)

                voices = []
                if data.get("audio_bytes"):
                    voices = self.voice_proc.process_clip_audio(data["audio_bytes"], self.video_id, clip_id)

                for v in visuals:
                    self.engine.vector_store.upsert(
                        "visual_memories", v.obs_id, v.clip_embedding,
                        {"video_id": self.video_id, "clip_id": clip_id, "desc": v.detected_objects[0]}
                    )

                for f in self.face_proc.cluster_clip_faces(faces):
                    self.identity_manager.resolve_face(f)
                    self.engine.ingest_face(f)
                    self.identity_manager.register_observation(f)

                for v in voices:
                    self.identity_manager.resolve_voice(v)
                    self.identity_manager.register_observation(v)

                memories = self.reasoning_agent.generate_memory_structures(
                    self.video_id, clip_id, visuals, faces, voices
                )
                
                if memories:
                    from conclave.core.schemas import MemoryType
                    episodic = [m for m in memories if m.mem_type == MemoryType.EPISODIC]
                    semantic = [m for m in memories if m.mem_type == MemoryType.SEMANTIC]
                    
                    if episodic: self.engine.add_memories_batched(episodic)
                    if semantic: self.engine.add_memories_m3_style(semantic)

                if clip_id % 2 == 0: 
                    self.engine.graph_store.flush()
                    self.identity_manager.link_modalities(self.video_id)
                
                if clip_id % 50 == 0:
                    torch.cuda.empty_cache()
                
                pbar.update(1)
                curr += stride
                clip_id += 1

            except Exception as e:
                logger.error(f"Error at {curr}s: {e}")
                break

        pbar.close()
        logger.info("✅ Pipeline Execution Complete.")

    def _fast_extract(self, video_path: str, start: float, duration: float):
        # ... [Same as previous file] ...
        clip_data = {"audio_bytes": None, "raw_frames": []}
        try:
            out, _ = (
                ffmpeg.input(video_path, ss=start, t=duration)
                .output('pipe:', format='wav', acodec='pcm_s16le', ar='16000', ac='1', loglevel="quiet")
                .run(capture_stdout=True)
            )
            clip_data["audio_bytes"] = out
        except ffmpeg.Error:
            pass 

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        target_fps = 1
        step = max(1, int(fps / target_fps))
        frames_needed = int(duration * target_fps)
        frames_read = 0
        count = 0
        TARGET_H = 640

        while frames_read < frames_needed:
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                h, w = frame.shape[:2]
                if h > TARGET_H:
                    scale = TARGET_H / float(h)
                    new_w = int(w * scale)
                    frame = cv2.resize(frame, (new_w, TARGET_H), interpolation=cv2.INTER_AREA)
                clip_data["raw_frames"].append(frame)
                frames_read += 1
            count += 1
            if (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0) > (start + duration + 1):
                break
        cap.release()
        return clip_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    args = parser.parse_args()
    ConclaveOrchestrator("configs/api_config.json", args.video_id).run_pipeline(args.video)