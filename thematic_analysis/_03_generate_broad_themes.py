import json
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from tqdm.auto import tqdm

import config
import prompts

tqdm.pandas()


def main():
    model = Ollama(model=config.model, temperature=config.LLM_TEMPERATURE)

    loader = DirectoryLoader(config.exit_interview_dir, glob="**/*.txt", show_progress=True)
    df = pd.DataFrame({"interview": loader.load()})
    df["filename"] = df["interview"].apply(lambda x: Path(x.metadata["source"]).name)
    df = df.merge(pd.read_csv(f"out/thematic_analysis_codes_{config.model}.csv"), on="filename")
    df["codes"] = df["codes"].apply(json.loads)
    codes = df["codes"].sum()
    codes = [x.lower() for x in codes]
    codes = [x.replace("_", " ").strip() for x in codes]
    codes = sorted(list(set(codes)))

    prompt = PromptTemplate.from_template(prompts.generate_broad_themes)

    chain_model = prompt | model | StrOutputParser()

    print("Generating broad themes")
    themes = json.loads(chain_model.invoke({"input": str(codes)}))
    pd.DataFrame({"theme": themes}).to_csv(f"out/thematic_analysis_themes_{config.model}.csv", index=False)
    print(f"themes written to out/thematic_analysis_themes_{config.model}.csv")


if __name__ == '__main__':
    main()
