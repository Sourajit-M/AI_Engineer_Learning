from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

text = """Artificial intelligence is rapidly changing the way people learn, work, and solve problems. 
One of the most useful applications is retrieval-augmented generation, commonly known as RAG, which allows a language model to answer questions using information retrieved from an external knowledge base. In a typical RAG system, documents are first collected and divided into smaller chunks so that relevant pieces of information can be efficiently retrieved later. 
Each chunk is converted into a numerical representation called an embedding, which captures its semantic meaning. These embeddings are stored in a vector database, where they can be searched using the embedding of a user's query. When a user asks a question, the system retrieves the most relevant chunks and provides them as context to the language model. The model then generates an answer based on the retrieved information rather than relying only on its internal knowledge. 
The quality of chunking is particularly important because chunks that are too small may lose important context, while chunks that are too large may contain unnecessary information and reduce retrieval accuracy. For this reason, developers often experiment with chunk size, overlap, and semantic boundaries to create a retrieval system that balances context and precision.
"""

#fixed size chunking
print("Fixed Size chunking")
fixed_chunk = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0
)

for i, chunk in enumerate(fixed_chunk.split_text(text), 1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("*"*50)


#paragraph chunking
paragaraph_chunk = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=0
)

print("Paragraph Splitter")
for i, chunk in enumerate(fixed_chunk.split_text(text), 1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("*"*50)


recursive_chunk = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

print("Recursive Character Chunking")
for i, chunk in enumerate(fixed_chunk.split_text(text), 1):
    print(f"\nChunk {i}:")
    print(chunk)
    print("*"*50)