import pickle
import numpy as np
from PyPDF2 import PdfReader
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
import torch
import os
import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import list_collections


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
        """
        Embeds a list of texts into a list of float vectors using the model.

        Args:
            texts (list of str): The texts to be embedded.

        Returns:
            np.ndarray: A NumPy array of float vectors representing the embeddings,
                        or None if an error occurs.
        """
        try:
            # Ensure texts is a non-empty list
            if not isinstance(texts, list) or len(texts) == 0:
                logging.error("Input texts must be a non-empty list.")
                return None

            # Encode the texts using the model
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            # Convert embeddings to a contiguous array of type float32
            embeddings = np.ascontiguousarray(embeddings.astype('float32'))

            # Check the shape of the embeddings
            if embeddings.ndim != 2:
                logging.error("Embeddings should be a 2D array.")
                return None

            logging.info(f"Successfully generated embeddings of shape {embeddings.shape}.")
            return embeddings

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



def retriever_tool(query: str):
    """调用以获取检索器。"""
    # 连接到 Milvus
    connections.connect("default", host="localhost", port="19530")

    collection_name = "document_embeddings"

    # 定义集合的模式
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="page_number", dtype=DataType.INT64)
    ]
    schema = CollectionSchema(fields, description="文档嵌入")

    # 检查集合是否存在，如果不存在则创建
    collections = list_collections()
    if collection_name not in collections:
        print("创建新集合和元数据...")

        # 设置文档路径
        doc_path = '/Users/shiwenbin/Downloads/红楼梦脂批本完美版pdf版.pdf'

        # 检查文档是否存在
        if not os.path.exists(doc_path):
            print(f"文件 {doc_path} 不存在。请确保文件路径正确。")
            return

        # 使用 PyPDF2 读取文档以提取页码信息
        reader = PdfReader(doc_path)
        page_texts = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text().encode('utf-8').decode('utf-8')

            if text:
                page_texts.append((text, page_number))

        token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        print("拆分文档...")

        doc_splits = []
        for text, page_number in page_texts:
            splits = token_splitter.split_text(text)
            for split in splits:
                doc_splits.append((split, page_number))

        local_embeddings = LocalEmbeddings()

        # 准备嵌入和其他信息
        contents = [doc[0] for doc in doc_splits]
        page_numbers = [doc[1] for doc in doc_splits]
        embeddings = local_embeddings.embed_documents(contents)

        if embeddings is None:
            logging.error("创建嵌入失败")
            return

        # 检查数据长度匹配
        if len(embeddings) != len(contents) or len(contents) != len(page_numbers):
            logging.error("数据长度不匹配，请检查数据。")
            return

        # 打印调试信息
        if len(embeddings) > 0:
            print(f"Embeddings count: {len(embeddings)}, dimension: {len(embeddings[0])}")
        else:
            print("No embeddings found.")
        print(f"Contents count: {len(contents)}")
        print(f"Page numbers count: {len(page_numbers)}")

        # 创建集合
        collection = Collection(name=collection_name, schema=schema)

        insert_data = [
            {"embedding": embedding, "content": content, "page_number": page_number}
            for embedding, content, page_number in zip(embeddings, contents, page_numbers)
        ]
        collection.insert(insert_data)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "IP"
        }

        collection.create_index(field_name="embedding", index_params=index_params)

        # 本地保存元数据
        metadata = {'doc_splits': doc_splits}
        save_metadata(metadata, './metadata.pkl')

    else:
        print("加载现有集合和元数据...")
        collection = Collection(name=collection_name)
        metadata = load_metadata('./metadata.pkl')
        if metadata is None:
            logging.error("加载元数据失败")
            return

    # 加载集合
    collection.load()


    # 处理查询
    local_embeddings = LocalEmbeddings()
    query_embedding = local_embeddings.embed_query(query)
    if query_embedding is None:
        logging.error("创建查询嵌入失败")
        return

    if query_embedding is not None:
        print(f"Query embedding dimension: {len(query_embedding)}")

    k = 5
    try:
        # 查询集合
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=None
        )

        if not results:
            print("搜索没有返回任何结果")
            return []

        retrieved_docs = []
        for hit in results[0]:  # results[0] 因为我们只搜索了一个查询向量
            # 使用 hit.id 查询文档内容
            doc_info = collection.query(
                expr=f"id == {hit.id}",
                output_fields=["content", "page_number"]
            )
            if doc_info:
                content = doc_info[0]['content']
                page_number = doc_info[0]['page_number']
                retrieved_docs.append((content, page_number, hit.distance))
            else:
                print(f"未找到ID为{hit.id}的文档")

        return retrieved_docs

    except Exception as e:
        logging.error(f"集合搜索过程中出错: {str(e)}")
        return []


if __name__ == "__main__":
    result = retriever_tool("宁国府除夕祭宗祠")
    print(result)
