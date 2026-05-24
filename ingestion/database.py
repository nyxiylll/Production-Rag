import json
import hashlib
from pathlib import Path
from langchain_classic.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


PERSIST_DIR = Path("chroma_db")
DATA_DIR = Path("data")
MANIFEST_FILE = PERSIST_DIR / "manifest.json"


def get_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_data = file.read()
    return hashlib.sha256(file_data).hexdigest()


def get_current_state():

    state = {}
    for file in DATA_DIR.rglob("*"):
        if file.is_file():
            state[str(file)] = get_file_hash(file)
    return state

def sync_vectorstore(loader, cleaner, chunker):

    current_state = get_current_state()

    old_state = {}

    if MANIFEST_FILE.exists():

        with open(MANIFEST_FILE, "r") as file:
            old_state = json.load(file)

    if current_state == old_state:

        print("No changes found. Loading existing database...")

        db = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings
        )

        return db

    print("Changes detected. Creating new database...")

    # Load documents
    docs = loader()

    docs = cleaner(docs)

    docs = chunker(docs)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR)
    )

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_FILE, "w") as file:
        json.dump(current_state, file)

    print("Database updated successfully!")
    return db