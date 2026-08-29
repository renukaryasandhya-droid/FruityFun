from __future__ import annotations

from itertools import islice

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from .config import Settings
from .corpus import extract_pdfs, save_chunks


def _batches(values: list, size: int = 100):
    iterator = iter(values)
    while batch := list(islice(iterator, size)):
        yield batch


def ingest(settings: Settings, create_index: bool = True) -> int:
    if not settings.openai_api_key or not settings.pinecone_api_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY are required for ingestion.")
    chunks = extract_pdfs(settings.pdf_corpus_dir)
    if not chunks:
        raise RuntimeError(f"No PDF files found in {settings.pdf_corpus_dir}")
    save_chunks(chunks, settings.processed_corpus_path)

    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
    )
    client = Pinecone(api_key=settings.pinecone_api_key)
    existing = {index.name for index in client.list_indexes()}
    if settings.pinecone_dense_index not in existing:
        if not create_index:
            raise RuntimeError(f"Pinecone index {settings.pinecone_dense_index!r} does not exist")
        dimension = len(embeddings.embed_query("fruit"))
        client.create_index(
            name=settings.pinecone_dense_index,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    index = client.Index(settings.pinecone_dense_index)
    for batch in _batches(chunks):
        vectors = embeddings.embed_documents([chunk.text for chunk in batch])
        index.upsert(
            namespace=settings.pinecone_namespace,
            vectors=[
                {
                    "id": chunk.id,
                    "values": vector,
                    "metadata": {
                        settings.pinecone_text_field: chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        **chunk.metadata,
                    },
                }
                for chunk, vector in zip(batch, vectors, strict=True)
            ],
        )
    return len(chunks)
