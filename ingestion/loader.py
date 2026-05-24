import os
import re
import hashlib
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
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

load_dotenv()

model = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

LOADER_MAP = {
    "**/*.pdf":   (PyPDFLoader,                  {}),
    "**/*.docx":  (Docx2txtLoader,               {}),
    "**/*.doc":   (Docx2txtLoader,               {}),
    "**/*.txt":   (TextLoader,                   {"encoding": "utf-8"}),
    "**/*.md":    (TextLoader,                   {"encoding": "utf-8"}),
    "**/*.html":  (UnstructuredHTMLLoader,        {}),
    "**/*.htm":   (UnstructuredHTMLLoader,        {}),
    "**/*.csv":   (CSVLoader,                    {}),
    "**/*.xlsx":  (UnstructuredExcelLoader,       {}),
    "**/*.xls":   (UnstructuredExcelLoader,       {}),
    "**/*.pptx":  (UnstructuredPowerPointLoader,  {}),
    "**/*.eml":   (UnstructuredEmailLoader,       {}),
    "**/*.png":   (UnstructuredImageLoader,       {}),
    "**/*.jpg":   (UnstructuredImageLoader,       {}),
    "**/*.jpeg":  (UnstructuredImageLoader,       {}),
    "**/*.json":  (JSONLoader,                   {"jq_schema": ".", "text_content": False}),
}

PROJECT_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_DIR / "data"


class Topic(BaseModel):
    topic: str = Field(description="Just one word for the given topic")


#def get_file_hash(file_path: Path) -> str:
#    sha256_hash = hashlib.sha256()
#    with open(file_path, "rb") as f:
#        for byte_block in iter(lambda: f.read(4096), b""):
#            sha256_hash.update(byte_block)
#    return sha256_hash.hexdigest()

def list_files(data_dir=DATA_DIR):
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return []
    return [f for f in data_dir.rglob("*") if f.is_file()]

def get_topic(text: str):
    if not text or len(text.strip()) < 10:
        return "Unknown"
    
    template = """You are a text summarizer. 
    Analyze the following text and summarize its main subject in exactly ONE word.
    
    Text: {text}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    structured_llm = model.with_structured_output(Topic)
    chain = prompt | structured_llm
    
    try:
        response = chain.invoke({"text": text[:3000]})
        return str(response)
    except Exception as e:
        print(f"LLM Error extracting topic: {e}")
        return "General"


def loader(loader_map=LOADER_MAP):
    files = list_files()
    all_docs = []

    for obj_path in files:
        ext = obj_path.suffix.lower()
        match_key = next((key for key in loader_map if key.endswith(ext)), None)

        if match_key:
            try:
                loader_cls, loader_args = loader_map[match_key]
                instance = loader_cls(file_path=str(obj_path), **loader_args)
                docs = instance.load()

                #file_id = get_file_hash(obj_path)
                
                for doc in docs:
                    #doc.metadata["file_id"] = file_id
                    doc.metadata["file_name"] = obj_path.name
                    doc.metadata["file_type"] = ext
                    doc.metadata["topic"] = get_topic(doc.page_content)
                
                all_docs.extend(docs)
                print(f"Successfully loaded: {obj_path.name}")

            except Exception as e:
                print(f"Error loading {obj_path.name}: {e}")
        else:
            print(f"No loader found for: {obj_path.name}")
            
    return all_docs

def cleaner(docs: List[Document]):
    cleaned: List[Document] = []
    for doc in docs:
        text = doc.page_content or ""
        text = text.replace("\x00", "").replace("\r\n", "\n").strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[\t]{2,}', ' ', text)

        if len(text) < 30:
            continue
            
        doc.page_content = text
        cleaned.append(doc)
    return cleaned 

def chunker(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    return splitter.split_documents(docs)

