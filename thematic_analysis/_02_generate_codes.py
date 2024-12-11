import json
import sys
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


def invoke_and_test(model, input_s) -> list:
    attempts_remaining = 5
    while attempts_remaining > 0:
        output = model.invoke({"input": input_s}).strip()
        try:
            json.loads(output)
            return output
        except json.JSONDecodeError:
            tqdm.write("Bad output, trying again!", file=sys.stderr)
            attempts_remaining -= 1
    raise Exception("Could not get valid list output from LLM. Check the prompt")


def main():
    loader = DirectoryLoader(config.exit_interview_dir, glob="**/*.txt", show_progress=True)
    df = pd.DataFrame({"interview": loader.load()})
    df["filename"] = df["interview"].apply(lambda x: Path(x.metadata["source"]).name)

    model = Ollama(model=config.model, temperature=config.LLM_TEMPERATURE)

    prompt = PromptTemplate.from_template(prompts.generate_codes)

    chain_model = prompt | model | StrOutputParser()

    print("Processing interviews")
    df["codes"] = df["interview"].progress_apply(lambda x: invoke_and_test(chain_model, x))
    df.sort_values("filename")[["filename", "codes"]].to_csv(
        f"out/thematic_analysis_codes_{config.model}.csv", index=False)
    print(f"codes written to out/thematic_analysis_codes_{config.model}.csv")


if __name__ == '__main__':
    main()
