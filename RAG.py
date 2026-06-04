
import ollama 

EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

EXAMPLE_FACTS = [
    "My cats name is Pepper",
    "Violet is my favorite color",
    "My age is a prime number"
]


def add_chunk_to_db(db, chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    db.append((chunk, embedding)) 


def initialize_db():
    db = []
    #
    # Example memory setup
    #
    for i, chunk in enumerate(EXAMPLE_FACTS):
        add_chunk_to_db(db, chunk)
        print(f"Commited fact #{i}: [{chunk}] to memory.")

    return db


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


def db_retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    print(f"Query: {query} Embedding:{len(query_embedding)}")
    similarities = []

    for chunk, embedding in VECTOR_DB:
        similarity_score = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity_score))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in similarities[:top_n]]    


VECTOR_DB = initialize_db()
