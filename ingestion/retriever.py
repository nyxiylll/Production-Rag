from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

def load_retriever(db, docs):
    vector_retriever = db.as_retriever(
        search_kwargs={"k": 10}
    )

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    return ensemble_retriever


def setup_rerank_pipeline(base_retriever, top_n=3):
    compressor = FlashrankRerank.client(top_n=top_n)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever