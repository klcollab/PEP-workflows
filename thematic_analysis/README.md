This directory contains scripts for utilizing LLMs to perform Thematic Analysis as described by Virginia Braun and
Victoria Clarke. It reads interview transcripts and outputs generated codes, initial themes, and reviewed themes
(Phases 2, 3, and 4/5).

## Getting Started

### Running Ollama

The first step is to get an ollama server up and running. This can be accomplished by running
```bash
docker pull ollama/ollama
```
and
```bash
docker run -d --gpus=all -v /path/to/ollama_models/:/root/.ollama/models -p 11434:11434 --name ollama ollama/ollama
```

If the container is already setup:
```bash
docker container start ollama
```

To start a chat session with an LLM:
```bash
docker exec -it ollama ollama run Meta-Llama-3-70B-Instruct-Q5_K_S
```

### Building container for thematic_analysis

Run the following command in the current directory to build an image called `thematic_analysis`:
```bash
docker build -f Dockerfile -t thematic_analysis .
```

### Running thematic_analysis container

```bash
docker run --gpus all -v /path/to/exit_interviews/:/app/exit_interviews -v /path/to/out/:/app/out/ \
  --network container:ollama thematic_analysis sh /app/run.sh
```

- `--network` bridges the ollama server with the running container so it can be accessed.
- `--gpus all` provides access to GPU hardware.
- `-v` binds the input and output directories. Update the code block above as appropriate.
- `sh /app/run.sh` runs a script that will run each step of the process. Replace with something like
`python /app/_01_create_vector_store.py` to run an individual step.

There are 4 scripts/steps:
```
_01_create_vector_store.py: Chunks interview transcripts and stores them into a FAISS index store.
_02_generate_codes.py: Generates codes from each of the interviews and outputs them into a CSV file.
_03_generate_broad_themes.py: Uses the generated codes and outputs ~10 broad themes to a CSV file.
_04_review_themes.py: Uses broad themes and interview chunks to develop into more detailed themes with definitions and quotes.
```