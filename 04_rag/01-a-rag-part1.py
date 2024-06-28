import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
 

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = "database/docs/GameOfThronesBundle.pdf"
db_directory = "database/vectorstores"
persistent_directory = os.path.join(current_directory, "..", db_directory, "got-chroma-store")

# define function to load the documents
def document_loader_init(file_path):
    """
    Initializes a document loader with the given file path and returns the loader object.
    """
    # alter file path to go back on step
    file_path = os.path.join(current_directory, "..", file_path)
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

# define function to split the documents into chunks
def text_splitter_init(documents):
    """
    Initializes and returns a RecursiveCharacterTextSplitter object with the specified chunk size and chunk overlap.

    Returns:
        RecursiveCharacterTextSplitter: The initialized text splitter object.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents=documents)
    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")
    
    return docs

# define function to embed the documents
def embedding_init():
    from sentence_transformers import SentenceTransformer
    from langchain.embeddings import SentenceTransformerEmbeddings
    print("\n--- Creating embeddings model ---")

    # model_name = "all-mpnet-base-v2"
    # model = SentenceTransformer(model_name_or_path=model_name)
    # print("\n--- Embedding documents ---")
    # embeddings = model.encode(
    #     sentences=[doc.page_content for doc in docs],
    #     show_progress_bar=True,
    #     batch_size=40,
    #     device="mps",
    #     )
    # print(f"Embeddings shape (dimensions): {embeddings.shape}")  # (number of documents, number of dimensions of embeddings.shape)
    # print("\n--- Embedding finished!!!!!! ---")
    
    
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
        )

    
    return embedding_model


# define function to initialize the vector store
def pinecone_vector_store_init(docs, embedding):
    """
    Initializes a Chroma vector store with the given documents and embeddings and returns the vector store object.
    """
    print("\n--- Creating vector store ---")
    
    from pinecone import Pinecone,ServerlessSpec
    
    
    client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "got-pinecone-index"
    
    if index_name not in client.list_indexes().names():
        
        dimension = len(embedding.embed_query(
            "This is a sample document string to get the dimension of the embedding"
            ))
        
        client.create_index(
        name="got-pinecone-index",
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = client.Index(index_name)

    print("\n--- Pinecone Vector store created successfully!! ---")
    print("Index status: ", index.describe_index_stats())
    
    vector_count = index.describe_index_stats()["total_vector_count"]
    docs_count = len(docs)
    print(f"Total vectors in the index: {vector_count}")
    print(f"Number of document chunks: {len(docs)}")
    
    if vector_count == docs_count:
        print("\n--- Vector store is up to date ---\n")
        return
        
    # Add documents to the vector store
    from langchain_pinecone import Pinecone

    print("\n--- Adding data to Vector store ---")
    
    Pinecone.from_documents(
        documents=docs,
        embedding=embedding,
        index_name="got-pinecone-index",
    )
    
    print("\n--- data added to Vector store  successfully!! ---")
    
    # db = Chroma(
    #     persist_directory=persistent_directory,
    #     embedding_function=embedding
    # )
    # # Prepare texts and metadatas from documents
    # texts = [doc.page_content for doc in docs]
    # metadatas = [doc.meta for doc in docs]
    
    # # Batch size limit
    # batch_size_limit = 1000
    
    # # Split the data into batches
    # for i in range(0, len(texts), batch_size_limit):
    #     batch_texts = texts[i:i + batch_size_limit]
    #     batch_metadatas = metadatas[i:i + batch_size_limit]
        
    #     print("processing batch........", i)
        
    #     # Add texts and metadatas in batches
    #     db.add_texts(
    #         texts=batch_texts,
    #     )
    
    # db.persist()
    

if __name__ == "__main__":
    documents = document_loader_init(file_path)
    docs = text_splitter_init(documents)
    embedding = embedding_init()
    db = pinecone_vector_store_init(docs, embedding)
    
    print("Docs chucked and added to Vector store  successfully!!")