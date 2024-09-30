import os
import pickle
import faiss
import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import torch

# 设置日志
logging.basicConfig(level=logging.INFO)

# 设置OpenAI API密钥

# 强制使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)

class LocalEmbeddings(Embeddings):
    def __init__(self, model_path="/Users/shiwenbin/Desktop/bce-emb"):
        self.model = SentenceTransformer(model_path, device='cpu')  # 强制使用CPU

    def embed_documents(self, texts):
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return np.ascontiguousarray(embeddings.astype('float32'))
        except Exception as e:
            logging.error(f"Error in embed_documents: {e}")
            return None

    def embed_query(self, text):
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return np.ascontiguousarray(embedding.astype('float32'))
        except Exception as e:
            logging.error(f"Error in embed_query: {e}")
            return None

def save_metadata(metadata, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")

def load_metadata(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        return None

def main(query: str):
    """Call to get the retriever."""
    faiss_index_path = './vectorstore.index'
    metadata_path = './metadata.pkl'

    # 检查是否存在已有的知识库
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        try:
            print("Loading existing index and metadata...")
            index = faiss.read_index(faiss_index_path)
            metadata = load_metadata(metadata_path)
            if metadata is None:
                raise ValueError("Failed to load metadata")
        except Exception as e:
            logging.error(f"Error loading index or metadata: {e}")
            return
    else:
        # 如果不存在知识库，则创建新的知识库
        print("Creating new index and metadata...")
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]
        # Load documents
        docs = []
        for url in urls:
            try:
                docs.extend(WebBaseLoader(url).load())
            except Exception as e:
                logging.error(f"Error loading {url}: {e}")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)

        local_embeddings = LocalEmbeddings()

        # Prepare embeddings
        embeddings = local_embeddings.embed_documents([doc.page_content for doc in doc_splits])
        if embeddings is None:
            logging.error("Failed to create embeddings")
            return

        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # 使用内积索引
        index.add(embeddings)
        try:
            faiss.write_index(index, faiss_index_path)
            metadata = {'doc_splits': doc_splits}
            save_metadata(metadata, metadata_path)
        except Exception as e:
            logging.error(f"Error saving index or metadata: {e}")
            return

    # 处理查询
    local_embeddings = LocalEmbeddings()
    query_embedding = local_embeddings.embed_query(query)
    if query_embedding is None:
        logging.error("Failed to create query embedding")
        return

    k = 5
    try:
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        results = []
        for idx in indices[0]:
            results.append(metadata['doc_splits'][idx].page_content)

        return results
    except Exception as e:
        logging.error(f"Error during index search: {e}")
        return []

retriever_tool = main
if __name__ == "__main__":
    result = main("What does Lilian Weng say about the types of agent memory?")
    print(result)
