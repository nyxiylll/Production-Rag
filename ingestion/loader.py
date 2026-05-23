import os
from pathlib import Path
import re
import hashlib
from typing import List
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    JSONLoader,
    UnstructuredPowerPointLoader,
    UnstructuredEmailLoader,
    UnstructuredImageLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_core.documents import Document
LOADER_MAP = {
    "**/*.pdf":   (PyPDFLoader,                   {}),
    "**/*.docx":  (Docx2txtLoader,                {}),
    "**/*.doc":   (Docx2txtLoader,                {}),
    "**/*.txt":   (TextLoader,                    {"encoding": "utf-8"}),
    "**/*.md":    (TextLoader,                    {"encoding": "utf-8"}),
    "**/*.html":  (UnstructuredHTMLLoader,         {}),
    "**/*.htm":   (UnstructuredHTMLLoader,         {}),
    "**/*.csv":   (CSVLoader,                     {}),
    "**/*.xlsx":  (UnstructuredExcelLoader,        {}),
    "**/*.xls":   (UnstructuredExcelLoader,        {}),
    "**/*.pptx":  (UnstructuredPowerPointLoader,   {}),
    "**/*.eml":   (UnstructuredEmailLoader,        {}),
    "**/*.png":   (UnstructuredImageLoader,        {}),
    "**/*.jpg":   (UnstructuredImageLoader,        {}),
    "**/*.jpeg":  (UnstructuredImageLoader,        {}),
    "**/*.json":  (JSONLoader,                    {"jq_schema": ".", "text_content": False}),
}
# ... (LOADER_MAP remains the same as your snippet) ...

# 1. Use / operator for cleaner path joining
PROJECT_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_DIR / "data"

def get_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def list_files(data_dir=DATA_DIR):
    return [f for f in data_dir.rglob("*") if f.is_file()]

def loader(loader_map=LOADER_MAP):

    files = list_files()
    all_docs = []

    for obj_path in files:
        ext = obj_path.suffix.lower()

        match_key = next((key for key in loader_map if key.endswith(ext)), None)

        if match_key:
            try:
                loader_cls, loader_args = loader_map[match_key]
                
                instance = loader_cls(
                    file_path=str(obj_path),
                    **loader_args
                )
                
                docs = instance.load()

                file_id = get_file_hash(obj_path)
                
                for doc in docs:
                    doc.metadata["file_id"] = file_id
                    doc.metadata["file_name"] = obj_path.name
                
                all_docs.extend(docs)
                print(f" Successfully loaded: {obj_path.name}")

            except Exception as e:
                print(f"Error loading {obj_path.name}: {e}")
        else:
            print(f"No loader found for: {obj_path.name}")
            
    return all_docs


def cleaner(docs):
    cleaned :List[Document] = []
    for doc in docs:
        text = doc.page_content or ""
        text = text.replace("\x00","").replace("\r\n","\n").strip()
        text = re.sub(r'\n{3,}','\n\n',text)
        text = re.sub(r'[\t]{2,}',' ',text)

        if len(text) < 30:
            continue
        doc.page_content = text
        cleaned.append(doc)
    return cleaned 


from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

def chunker(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function=len,
        add_start_index = True
    )
    return splitter.split_documents(docs)



# Execute
if __name__ == "__main__":
    docs = loader()
    print(f"\nTotal documents loaded: {len(docs)}")