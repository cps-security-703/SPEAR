from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import config

class DocumentEmbedder:


    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return [0.0] * self.embedding_dimension

    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:

        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            return [[0.0] * self.embedding_dimension] * len(texts)

    def get_embedding_dimension(self) -> int:

        return self.embedding_dimension

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
