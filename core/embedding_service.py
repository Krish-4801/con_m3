import logging
import hashlib
import time
import uuid  # <-- Added
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAIError

logger = logging.getLogger("Conclave.Embedding")

class EmbeddingService:
    def __init__(self, config: Dict[str, Any]):
        self.model = config.get("model", "text-embedding-3-small")
        self.api_key = config.get("api_key")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 5)
        self.base_retry_delay = 2.0
        
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_deterministic_id(self, content: str) -> str:
        """
        Creates a valid UUIDv5 based on content.
        This fits Qdrant's strict ID requirements while remaining deterministic.
        """
        # Combine model and content to ensure uniqueness per embedding space
        unique_string = f"{self.model}:{content}"
        # Generate UUIDv5 using DNS namespace as a seed
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

    def _get_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [data.embedding for data in response.data]
            except OpenAIError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Embedding failed after {self.max_retries} attempts: {e}")
                    raise e
                
                delay = self.base_retry_delay * (2 ** attempt)
                logger.warning(f"Embedding API error: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        return []

    def get_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Filter out empty strings or non-strings which cause API errors
        valid_texts = []
        original_indices = []
        for i, t in enumerate(texts):
            if isinstance(t, str) and t.strip():
                valid_texts.append(t)
                original_indices.append(i)
            else:
                logger.warning(f"Skipping invalid embedding input at index {i}: {repr(t)}")
        
        if not valid_texts:
            # Default to 1536 (common for OpenAI) if we can't determine dimension
            dim = 1536
            if "large" in self.model: dim = 3072
            return [[0.0] * dim for _ in range(len(texts))]


        all_embeddings_map = {}
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i : i + self.batch_size]
            batch_indices = original_indices[i : i + self.batch_size]
            batch_embeddings = self._get_embeddings_with_backoff(batch)
            
            for idx, emb in zip(batch_indices, batch_embeddings):
                all_embeddings_map[idx] = emb
        
        # Reconstruct full list in original order, filling gaps with zero vectors
        first_emb = next(iter(all_embeddings_map.values()), None)
        dim = len(first_emb) if first_emb else 1536
        
        final_results = []
        for i in range(len(texts)):
            if i in all_embeddings_map:
                final_results.append(all_embeddings_map[i])
            else:
                final_results.append([0.0] * dim)
            
        return final_results
