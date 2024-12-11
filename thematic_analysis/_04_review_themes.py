import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from tqdm.auto import tqdm

import config
import prompts

tqdm.pandas()


def main():
    model = Ollama(model=config.model, temperature=config.LLM_TEMPERATURE)

    themes = pd.read_csv(f"out/thematic_analysis_themes_{config.model}.csv")

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': False, 'device': 'cpu'})

    vector = FAISS.load_local(
        "out/chunk_vectors/",
        embeddings, index_name="chunks",
        allow_dangerous_deserialization=True)
    retriever = vector.as_retriever(search_kwargs={"k": 10})

    prompt = PromptTemplate.from_template(prompts.develop_theme)

    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    themes["developed_theme"] = themes["theme"].progress_apply(
        lambda x: retrieval_chain.invoke({"input": x})["answer"].strip())

    themes.sort_values("theme").to_csv(f"out/thematic_analysis_themes_{config.model}.csv", index=False)
    print(f"themes written to out/thematic_analysis_themes_{config.model}.csv")


if __name__ == '__main__':
    main()
