from fruity_fun.config import get_settings
from fruity_fun.ingest import ingest

if __name__ == "__main__":
    count = ingest(get_settings())
    print(f"Ingested {count} chunks into Pinecone and the local BM25 corpus.")
