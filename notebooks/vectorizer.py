import abc
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from dsp.modules.sentence_vectorizer import BaseSentenceVectorizer

class CustomSentenceVectorizer(BaseSentenceVectorizer):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        
    def __call__(self, inp_examples: List[str]) -> np.ndarray:
        texts = self._extract_text_from_examples(inp_examples)
        embeddings_list = self.model.encode(texts)
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings