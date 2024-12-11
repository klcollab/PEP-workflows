import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def main():
    loader = DirectoryLoader(config.exit_interview_dir, glob="**/*.txt", show_progress=True)
    df = pd.DataFrame({"interview": loader.load()})

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': False, 'device': 'cuda'})

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\nQ:", "\n\nA:", "\n\n", "\n", " ", ""],
        chunk_size=2048,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    documents = text_splitter.split_documents(df["interview"])

    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local("out/chunk_vectors/", "chunks")


if __name__ == '__main__':
    main()
