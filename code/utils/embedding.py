from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


EMBEDDING_PROVIDER = "local_bge_m3"
EMBEDDING_MODEL_NAME_OR_PATH = "/path/to/bge-m3"
EMBEDDING_USE_FP16 = False
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_NORMALIZE = True
EMBEDDING_LOCAL_FILES_ONLY = True
EMBEDDING_DEVICE: str | None = None


@dataclass
class EmbeddingConfig:
    provider: str = EMBEDDING_PROVIDER
    model_name_or_path: str = EMBEDDING_MODEL_NAME_OR_PATH
    use_fp16: bool = EMBEDDING_USE_FP16
    batch_size: int = EMBEDDING_BATCH_SIZE
    normalize_embeddings: bool = EMBEDDING_NORMALIZE
    local_files_only: bool = EMBEDDING_LOCAL_FILES_ONLY
    device: str | None = EMBEDDING_DEVICE


class EmbeddingProvider(Protocol):
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        ...


class BGEM3EmbeddingProvider:
    @staticmethod
    def _validate_local_model_dir(model_dir: Path) -> None:
        required_any = [
            ["pytorch_model.bin"],
            ["model.safetensors"],
        ]
        required_all = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        missing = [name for name in required_all if not (model_dir / name).exists()]
        has_weight = any(all((model_dir / name).exists() for name in group) for group in required_any)

        if not has_weight:
            missing.append("pytorch_model.bin or model.safetensors")

        if missing:
            missing_text = ", ".join(missing)
            raise FileNotFoundError(
                f"Local BGE-M3 directory looks incomplete: {model_dir}. Missing required files: {missing_text}. "
                "Set EMBEDDING_MODEL_NAME_OR_PATH in utils/embedding.py to a complete local model directory or snapshot directory."
            )

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "BGE-M3 embedding requires FlagEmbedding. Install it with `pip install FlagEmbedding torch`."
            ) from exc

        model_path = Path(config.model_name_or_path).expanduser()
        if model_path.exists():
            self._validate_local_model_dir(model_path)
            if config.local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"

        self.model = BGEM3FlagModel(
            config.model_name_or_path,
            use_fp16=config.use_fp16,
            normalize_embeddings=config.normalize_embeddings,
            devices=config.device,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [str(text).strip() for text in texts if str(text).strip()]
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)

        result = self.model.encode(
            text_list,
            batch_size=self.config.batch_size,
        )
        embeddings = np.asarray(result["dense_vecs"], dtype=np.float32)

        if self.config.normalize_embeddings and embeddings.size > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        return embeddings


def build_embedding_provider(config: EmbeddingConfig | None = None) -> EmbeddingProvider:
    resolved_config = config or EmbeddingConfig()
    provider = resolved_config.provider.strip().lower()
    if provider == "local_bge_m3":
        return BGEM3EmbeddingProvider(resolved_config)
    raise ValueError(f"Unsupported embedding provider: {resolved_config.provider}")





