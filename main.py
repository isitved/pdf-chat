
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

PDF_PATH = "story.pdf"

# Word counts. Overlap keeps context that would otherwise sit on a chunk boundary.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3


def extract_pages(pdf_path):
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            # Skip blank pages so they never become empty chunks.
            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


def create_chunks(pages):
    chunks = []
    # Slide forward by less than a full chunk so neighbors share words.
    step = CHUNK_SIZE - CHUNK_OVERLAP

    if step <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than CHUNK_OVERLAP"
        )

    for page in pages:
        words = page["text"].split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_SIZE]

            chunks.append({
                "page": page["page"],
                "text": " ".join(chunk_words)
            })

    return chunks


def create_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    if not texts:
        return np.empty((0, 384))  # MiniLM-L6-v2 vector size

    # Normalized vectors so dot product equals cosine similarity.
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def search(query, chunks, embeddings, model, top_k=TOP_K):
    # 1. Convert the query into an embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # 2. Calculate similarity with every chunk
    similarities = embeddings @ query_embedding

    # 3. Get indices of the highest-scoring chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 4. Return the matching chunks and their scores
    results = []

    for index in top_indices:
        results.append({
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(similarities[index])
        })

    return results


def main():
    print("Extracting PDF...")
    pages = extract_pages(PDF_PATH)
    print(f"Pages extracted: {len(pages)}")

    print("\nCreating chunks...")
    chunks = create_chunks(pages)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print("No text found in the PDF.")
        return

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("\nCreating embeddings...")
    embeddings = create_embeddings(chunks, model)
    print(f"Embeddings created: {len(embeddings)}")

    # Keep asking questions until the user exits
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(query, chunks, embeddings, model)

        print("\nSemantic search results:")

        for rank, result in enumerate(results, start=1):
            print("\n" + "=" * 60)
            print(f"Rank: {rank}")
            print(f"Page: {result['page']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Text:\n{result['text']}")


if __name__ == "__main__":
    main()
