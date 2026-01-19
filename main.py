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
from typing import Dict, Any


# Now the imports should work
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
    def __init__(self, config_path: str, video_id: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_id = video_id
        
        # 1. Initialize Engine (Fast)
        self.engine = ConclaveEngine(video_id=video_id, config_path=config_path)
        
        # 2. Initialize Identity Manager (Fast)
        self.identity_manager = IdentityManager(
            self.engine.vector_store, 
            self.engine.graph_store, 
            self.config.get("processing", {})
        )
        
        # 3. SEQUENTIAL MODEL LOADING
        logger.info("🚀 Bootstrapping AI Models SEQUENTIALLY...")
        t0 = time.time()
        
        self.face_proc = self._load_face_model()
        self.voice_proc = self._load_voice_model()
        self.scene_proc = self._load_scene_model()
            
        logger.info(f"✅ All Models Loaded in {time.time() - t0:.2f}s")
        
        # 4. Initialize Reasoning Agent
        self.reasoning_agent = ReasoningAgent(self.config.get("api", {}))

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
        
        # Start with processing config
        process_config = self.config.get("processing", {}).copy()
        
        # Inject Gemini config
        if "gemini" in self.config:
            process_config["gemini"] = self.config["gemini"]
        else:
            logger.warning("⚠️ No 'gemini' section found in config!")
        
        return SceneProcessor(process_config)

    def run_pipeline(self, video_path: str, window_size: int = 30, overlap: int = 3):
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return

        logger.info(f"🎬 STARTING PIPELINE: {self.video_id}")
        
        # Get total duration using ffprobe (fastest)
        try:
            probe = ffmpeg.probe(video_path)
            total_duration = float(probe['format']['duration'])
        except Exception as e:
            logger.error(f"FFProbe failed: {e}")
            return

        # ----------------------------------------------------------------
        # 1. Sequential Processing Loop
        # ----------------------------------------------------------------
        # Calculate estimated total clips for progress bar
        stride = window_size - overlap
        estimated_clips = int(np.ceil(total_duration / stride))
        
        # Initialize tqdm
        pbar = tqdm(total=estimated_clips, desc="Processing Video Pipeline", unit="clip")

        curr = 0.0
        clip_id = 0
        
        while curr < total_duration:
            try:
                # Synchronous Extract (No Disk I/O)
                data = self._fast_extract(video_path, curr, window_size)
                current_start = curr
                
                # Update description
                pbar.set_description(f"Processing Clip {clip_id} [{current_start:.1f}s]")

                # --- Sequential Tasks: Scene -> Face -> Voice ---
                logger.debug(f"⚡ Processing Clip {clip_id} - Scene...")
                visuals = self.scene_proc.process_batch(
                    data["raw_frames"], self.video_id, clip_id, int(current_start * 1000), 1000
                )

                logger.debug(f"⚡ Processing Clip {clip_id} - Face...")
                face_stride = 2
                selected_frames = data["raw_frames"][::face_stride]
                faces = []
                if selected_frames:
                    faces = self.face_proc.extract_from_frames(selected_frames, self.video_id, clip_id)

                logger.debug(f"⚡ Processing Clip {clip_id} - Voice...")
                voices = []
                if data.get("audio_bytes"):
                    voices = self.voice_proc.process_clip_audio(data["audio_bytes"], self.video_id, clip_id)

                # Commit visuals
                for v in visuals:
                    self.engine.vector_store.upsert(
                        "visual_memories", v.obs_id, v.clip_embedding,
                        {"video_id": self.video_id, "clip_id": clip_id, "desc": v.detected_objects[0]}
                    )

                # Faces: cluster & resolve
                for f in self.face_proc.cluster_clip_faces(faces):
                    self.identity_manager.resolve_face(f)
                    self.engine.ingest_face(f)
                    self.identity_manager.register_observation(f)

                # Voices
                for v in voices:
                    self.identity_manager.resolve_voice(v)
                    self.identity_manager.register_observation(v)

                # --- REASONING ---
                # UPDATED: Use M3 unified memory generation
                memories = self.reasoning_agent.generate_memory_structures(
                    self.video_id, clip_id, visuals, faces, voices
                )
                
                if memories:
                    # Split memories by type for correct handling in Engine
                    from conclave.core.schemas import MemoryType
                    
                    episodic = [m for m in memories if m.mem_type == MemoryType.EPISODIC]
                    semantic = [m for m in memories if m.mem_type == MemoryType.SEMANTIC]
                    
                    # 1. Episodic: Standard batched ingestion
                    if episodic:
                        self.engine.add_memories_batched(episodic)
                    
                    # 2. Semantic: Weighted reinforcement (M3 Logic)
                    if semantic:
                        self.engine.add_memories_m3_style(semantic)

                if clip_id % 2 == 0: 
                    # CRITICAL FIX: Ensure async writes are committed before reading
                    self.engine.graph_store.flush()
                    self.identity_manager.link_modalities(self.video_id)
                
                # Optimized Cleanup: Only clear cache if VRAM is actually tight
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
        """
        Extracts raw frames and audio bytes directly into RAM using FFmpeg pipes.
        Includes automatic resizing to 640p height for speed.
        """
        clip_data = {"audio_bytes": None, "raw_frames": []}
        
        # 1. Extract Audio
        try:
            out, _ = (
                ffmpeg
                .input(video_path, ss=start, t=duration)
                .output('pipe:', format='wav', acodec='pcm_s16le', ar='16000', ac='1', loglevel="quiet")
                .run(capture_stdout=True)
            )
            clip_data["audio_bytes"] = out
        except ffmpeg.Error:
            pass 

        # 2. Extract Frames
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25
        
        target_fps = 1 # Keep this at 1 FPS for speed
        step = max(1, int(fps / target_fps))
        
        frames_needed = int(duration * target_fps)
        frames_read = 0
        count = 0
        
        # OPTIMIZATION: Target Height
        TARGET_H = 640

        while frames_read < frames_needed:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                # --- GLOBAL RESIZE OPTIMIZATION ---
                h, w = frame.shape[:2]
                if h > TARGET_H:
                    scale = TARGET_H / float(h)
                    new_w = int(w * scale)
                    # Resize to (new_w, 640)
                    frame = cv2.resize(frame, (new_w, TARGET_H), interpolation=cv2.INTER_AREA)
                # ----------------------------------

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