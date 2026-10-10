"""chunks.jsonl -> BGE-M3 (dense + sparse) -> Qdrant collection.

Run on a GPU (Kaggle):  python -m src.index
"""
import json
import uuid

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.config import CHUNKS_PATH, COLLECTION_NAME, EMBED_MODEL, QDRANT_PATH

BATCH_SIZE = 32
MAX_LENGTH = 1024      # tokens; our chunks are <= 2500 chars, ~600-900 tokens
DENSE_DIM = 1024       # size of a BGE-M3 dense vector


def load_chunks() -> list[dict]:
    """Read chunks.jsonl: one JSON object per line."""
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be an int or a UUID. uuid5 is deterministic: same chunk_id -> same UUID,
    so re-indexing overwrites points instead of duplicating them."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def create_collection(client: QdrantClient) -> None:
    """(Re)create the collection: start clean on every run (idempotent script)."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def to_sparse(weights: dict) -> models.SparseVector:
    """BGE-M3 sparse output {"12345": 0.31, ...} -> Qdrant SparseVector(indices, values)."""
    return models.SparseVector(
        indices=[int(k) for k in weights],
        values=[float(v) for v in weights.values()],
    )


def main():
    from FlagEmbedding import BGEM3FlagModel   # heavy import, only when we actually index

    chunks = load_chunks()
    print(f"{len(chunks)} chunks loaded from {CHUNKS_PATH}")
    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=True)
    client = QdrantClient(path=QDRANT_PATH)
    create_collection(client)

    for start in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[start:start + BATCH_SIZE]
        out = model.encode(
            [c["text"] for c in batch],
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
        )
        points = [
            models.PointStruct(
                id=point_id(c["chunk_id"]),
                vector={"dense": dense.tolist(), "sparse": to_sparse(sparse)},
                payload=c,
            )
            for c, dense, sparse in zip(batch, out["dense_vecs"], out["lexical_weights"])
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    n = client.count(COLLECTION_NAME).count
    print(f"collection '{COLLECTION_NAME}': {n} points (expected {len(chunks)})")
    assert n == len(chunks), "some chunks were not indexed (duplicate chunk_id?)"
    client.close()


if __name__ == "__main__":
    main()
