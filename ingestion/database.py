from langchain_classic.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_PATH / "chroma")

class VectorStore:
    def __init__(self,presist_dir,embedding_model):
        self.presist_dir = presist_dir
        self.embedding_model = embedding_model
        self.embeddings = _get_embeddings()

    def _get_embeddings(self):
