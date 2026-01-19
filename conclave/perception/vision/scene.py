import cv2
import torch
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from PIL import Image
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    SiglipVisionModel, 
    SiglipProcessor
)
import easyocr
import base64
import concurrent.futures
import openai
from conclave.core.schemas import VisualObservation

logger = logging.getLogger("Conclave.Vision.Scene")

class SceneProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # 1. SETUP GEMINI API
        gemini_conf = config.get("gemini", {})
        api_key = gemini_conf.get("api_key")
        base_url = gemini_conf.get("base_url")
        model = gemini_conf.get("model", "gemini-1.5-flash")

        self.gemini_client = None
        self.vlm_model_name = model
        
        if api_key and base_url:
            logger.info(f"🌐 Connecting to Gemini API for Scene Captioning: {model}")
            self.gemini_client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=15.0
            )
        else:
            logger.warning("⚠️ Gemini API credentials missing. Falling back to local BLIP.")

        logger.info(f"Loading Scene Models (BLIP + SigLIP + YOLO) on {self.device}...")

        # 2. SigLIP Base (For Embeddings)
        self.siglip_model = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device, dtype=self.torch_dtype).eval()
        self.siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

        # 3. BLIP Base (Fallback For Detailed Captioning)
        self.vlm_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device, dtype=self.torch_dtype).eval()
        
        self.vlm_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        # 4. YOLO11n (For Object Detection)
        self.yolo_model = YOLO("yolo11n.pt") 

        # 5. EasyOCR (For Text)
        self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'), verbose=False)

    def _encode_image(self, image_np: np.ndarray) -> str:
        h, w = image_np.shape[:2]
        # Resize if too big (speed optimization for API upload)
        if max(h, w) > 720:
            scale = 720 / max(h, w)
            image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')

    def _get_caption_gemini(self, img_np: np.ndarray) -> str:
        if not self.gemini_client: return None
        try:
            b64 = self._encode_image(img_np)
            resp = self.gemini_client.chat.completions.create(
                model=self.vlm_model_name,
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Describe this video frame concisely in one sentence. Focus on key actions and people."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=200
            )
            if resp.choices:
                return resp.choices[0].message.content.strip()
            return None
        except Exception as e:
            logger.warning(f"Gemini API Error: {e}")
            return None

    @torch.no_grad()
    def _get_siglip_embedding_batch(self, pil_imgs: List[Image.Image]) -> List[List[float]]:
        # SigLIP expects RGB images
        inputs = self.siglip_processor(images=pil_imgs, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
        outputs = self.siglip_model(**inputs)
        
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        norms = torch.linalg.norm(embeddings, ord=2, dim=1, keepdim=True)
        return (embeddings / (norms + 1e-6)).tolist()

    @torch.no_grad()
    def _run_vlm_caption_sequential(self, pil_imgs: List[Image.Image], max_tokens: int = 128) -> List[str]:
        """
        Runs BLIP-Base on a list of images.
        """
        captions = []
        for img in tqdm(pil_imgs, desc="Scene Captioning", leave=False):
            inputs = self.vlm_processor(images=img, return_tensors="pt").to(self.device, dtype=self.torch_dtype)
            
            outputs = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_tokens
            )
            
            caption = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
            captions.append(caption)
            
        return captions

    def process_batch(self, frames_np: List[np.ndarray], video_id: str, clip_id: int, start_ts: int, interval_ms: int) -> List[VisualObservation]:
        if not frames_np: return []

        # Convert to PIL
        pil_imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_np]
        
        # 1. Embeddings (SigLIP - Batched)
        visual_vecs = self._get_siglip_embedding_batch(pil_imgs)
        
        # 2. Captions (Gemini with BLIP fallback)
        if self.gemini_client:
            # Send requests in parallel to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                captions = list(executor.map(self._get_caption_gemini, frames_np))
            
            # For any failed Gemini requests, fallback to BLIP
            failed_indices = [i for i, c in enumerate(captions) if c is None]
            if failed_indices:
                logger.info(f"Fallback: Running BLIP for {len(failed_indices)} frames.")
                fallback_imgs = [pil_imgs[i] for i in failed_indices]
                fallback_captions = self._run_vlm_caption_sequential(fallback_imgs, max_tokens=128)
                for i, idx in enumerate(failed_indices):
                    captions[idx] = fallback_captions[i]
        else:
            # Standard local BLIP flow
            captions = self._run_vlm_caption_sequential(pil_imgs, max_tokens=128)
        
        # 3. Object Detection (YOLO)
        yolo_results = self.yolo_model(frames_np, verbose=False, stream=False) 
        
        # 4. OCR (EasyOCR)
        ocr_results = []
        for f in frames_np:
            try:
                res = self.ocr_reader.readtext(f) 
                lines = []
                for (bbox, text, conf) in res:
                    if conf > 0.4:  # Higher confidence threshold for quality
                        lines.append({"text": text, "conf": float(conf)})
                ocr_results.append(lines)
            except:
                ocr_results.append([])

        # 5. Aggregate with Object Filtering
        observations = []
        for i, (vec, cap, y_res, ocr) in enumerate(zip(visual_vecs, captions, yolo_results, ocr_results)):
            
            # OBJECT FILTER: Ignore objects < 1% of frame area (removes tiny/noise detections)
            img_area = y_res.orig_shape[0] * y_res.orig_shape[1]
            objects = []
            for box in y_res.boxes:
                # Get width and height from xywh format
                w = box.xywh[0][2].item()
                h = box.xywh[0][3].item()
                obj_area = w * h
                
                # Only keep objects larger than 1% of image
                if (obj_area / img_area) > 0.01:
                    objects.append(y_res.names[int(box.cls)])
            
            ts = start_ts + (i * interval_ms)
            
            obs = VisualObservation(
                video_id=video_id,
                clip_id=clip_id,
                ts_ms=ts,
                clip_embedding=vec,
                ocr_tokens=ocr,
                detected_objects=[cap] + objects  # Caption first, then filtered objects
            )
            obs.__dict__['spatial_metadata'] = {"dense_description": cap}
            observations.append(obs)
            
        return observations