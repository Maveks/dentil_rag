"""
Pipeline: JSON transcripts → LlamaIndex chunks → OpenAI embeddings → ChromaDB
Usage: python ingest.py --transcripts-dir ./transcripts --course implantology_101
"""
import argparse
import json
import pathlib

import chromadb
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI

import config


def load_transcripts(transcripts_dir: pathlib.Path) -> list[Document]:
    documents = []
    for json_file in sorted(transcripts_dir.glob("lesson_*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        course = data["course"]
        lesson = data["lesson"]
        lesson_file = data["lesson_file"]

        for seg in data["segments"]:
            doc = Document(
                text=seg["text"],
                metadata={
                    "course": course,
                    "lesson": lesson,
                    "lesson_file": lesson_file,
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "timestamp": seg["timestamp"],
                },
            )
            documents.append(doc)

    print(f"Завантажено {len(documents)} сегментів з {transcripts_dir}")
    return documents


def chunk_documents(documents: list[Document]) -> list:
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Чанкінг: {len(documents)} сегментів → {len(nodes)} чанків")
    return nodes


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
            dimensions=config.EMBEDDING_DIMENSIONS,
        )
        all_embeddings.extend([r.embedding for r in response.data])
        print(f"  Ембедінги: {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


def upsert_to_chroma(
    nodes: list,
    embeddings: list[list[float]],
    collection_name: str,
) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    try:
        chroma_client.delete_collection(collection_name)
        print(f"Стару колекцію '{collection_name}' видалено")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(nodes))]
    texts = [node.get_content() for node in nodes]
    metadatas = [node.metadata for node in nodes]

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        print(f"  Upsert: {min(i + batch_size, len(ids))}/{len(ids)} чанків")

    print(f"Колекція '{collection_name}': {collection.count()} чанків збережено")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest transcripts into ChromaDB")
    parser.add_argument("--transcripts-dir", default=str(config.TRANSCRIPTS_DIR))
    parser.add_argument("--course", default=config.COURSE_NAME)
    args = parser.parse_args()

    transcripts_dir = pathlib.Path(args.transcripts_dir)
    if not transcripts_dir.exists():
        print(f"Директорія не знайдена: {transcripts_dir}")
        return

    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    documents = load_transcripts(transcripts_dir)
    nodes = chunk_documents(documents)

    nodes = [n for n in nodes if n.get_content().strip()]
    texts = [node.get_content() for node in nodes]
    print(f"Генерація ембедінгів для {len(texts)} чанків...")
    embeddings = embed_texts(openai_client, texts)

    upsert_to_chroma(nodes, embeddings, collection_name=args.course)
    print(f"\nГотово! База даних: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
