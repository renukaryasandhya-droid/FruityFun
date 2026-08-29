from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_llm_model: str = "gpt-5.6-terra"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_image_model: str = "gpt-image-2"

    pinecone_api_key: str = ""
    pinecone_dense_index: str = "fruity-fun-dense"
    pinecone_namespace: str = "fruit-corpus-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_text_field: str = "text"

    top_k_dense: int = Field(default=12, ge=1, le=100)
    top_k_sparse: int = Field(default=12, ge=1, le=100)
    top_k_final: int = Field(default=8, ge=1, le=30)
    rrf_k: int = Field(default=60, ge=1)
    confidence_threshold: float = Field(default=0.65, ge=0, le=1)

    pdf_corpus_dir: Path = Path("./data/pdfs")
    processed_corpus_path: Path = Path("./data/processed/fruit_chunks.json")
    image_output_dir: Path = Path("./outputs/generated")

    @property
    def missing_runtime_secrets(self) -> list[str]:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
