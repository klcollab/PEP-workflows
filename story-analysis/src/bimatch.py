"""
bimatch.py - Bidirectional Theme-Story Matching Using Semantic Embeddings

This script performs bidirectional semantic similarity matching between themes and stories
using vector embeddings and FAISS similarity search. It creates two complementary indexes:
1) Theme-to-Story matching: finds stories most relevant to each theme
2) Story-to-Theme matching: finds themes most relevant to each story

Overview:
  Uses HuggingFace embeddings (Qwen3-Embedding-8B) and FAISS vector database to compute
  semantic similarity between thematic concepts and narrative text. The script generates
  two CSV outputs showing bidirectional matches above a configurable similarity threshold.

Usage:
  python bimatch.py -tf THEMES_CSV [options] STORY_FILE1 STORY_FILE2 ...

Arguments:
  stories                      One or more text files containing story content (positional)

Required Options:
  -tf, --themes PATH          Path to CSV file containing themes with columns for theme,
                              refined_theme, and definition

Optional Arguments:
  -st, --sim_thresh FLOAT     Similarity threshold for matching (0.0-1.0, default: 0.95)
                              Higher values include lower semantic similarity

Input Format:
  Themes CSV: Expected to contain columns describing thematic elements. Common columns:
    - theme: Short theme name
    - refined_theme: Expanded theme description
    - definition: Detailed definition of the theme

  Story files: Plain text files, one story per file

Output:
  Two CSV files are generated based on the input themes filename:

  1. Theme-indexed file (*_theme_indexed.csv):
     Columns: match num, theme, matches
     - Lists all stories matching each theme above similarity threshold

  2. Story-indexed file (*_story_indexed.csv):
     Columns: match num, file, theme
     - Lists all themes matching each story above similarity threshold

Processing Steps:
  1. Load all story files and themes from CSV
  2. Initialize HuggingFace embeddings model (Qwen3-Embedding-8B on CUDA)
  3. Create FAISS vector database of story embeddings
  4. For each theme, search for similar stories above threshold
  5. Create FAISS vector database of theme embeddings
  6. For each story, search for similar themes above threshold
  7. Output statistics and write results to CSV files

Features:
  - Bidirectional semantic matching (themes→stories and stories→themes)
  - GPU-accelerated embeddings (CUDA support)
  - Configurable similarity threshold
  - L2-normalized vector search for consistent similarity scores
  - Detailed matching statistics and reporting
  - Handles multi-line theme definitions with structured parsing

Example:
  python bimatch.py -tf themes_kept.csv -st 0.90 story1.txt story2.txt story3.txt

  Output files:
    - themes_theme_indexed.csv (themes with matching stories)
    - themes_story_indexed.csv (stories with matching themes)

Dependencies:
  - pandas: Data manipulation and CSV I/O
  - langchain_community: FAISS vector store integration
  - langchain_huggingface: HuggingFace embeddings wrapper
  - langchain_core: Document handling
  - FAISS: Fast similarity search library
  - transformers/torch: HuggingFace model backend (with CUDA support)

Performance Notes:
  - Uses GPU (CUDA) for embedding generation
  - Embedding model: Qwen/Qwen3-Embedding-8B (trust_remote_code enabled)
  - Alternative model commented out: Alibaba-NLP/gte-base-en-v1.5
  - Processing time scales with number of stories × themes

Output Statistics:
  Theme-to-Story: Reports count of themes matching ≥1 story vs. no matches
  Story-to-Theme: Reports count of stories matching ≥1 theme vs. no matches
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def print_stats(successes: int, fails: int, entity_type: str, other_entity: str) -> None:
  """Print matching statistics."""
  print(f"{successes} {entity_type} match at least 1 {other_entity}. {fails} {entity_type} match no {other_entity}.")

def read_file(fn):
  with open(fn, "r") as f: return f.read()

def theme_join(row):
  if not isinstance(row, pd.Series): return "No theme"

  parts = [str(x) for x in row if pd.notna(x)]
  if not parts: return "No theme"
  # Dynamically build the output for each non-null column
  lines = []
  for col_name, value in row.items():
    if pd.notna(value):
      lines.append(f"{col_name}: {value}")
  return "\n".join(lines)

def parse_theme_string(theme_str):
  """
  Parse a theme string into separate components
  Args:
    theme_str: String in format "theme: <name>\nrefined_theme: <phrase>\ndefinition: <text>"
  Returns:
    Dictionary with 'theme', 'refined_theme', and 'definition' keys
  """
  result = {"theme": "", "refined_theme": "", "definition": ""}

  if not isinstance(theme_str, str):
    return result

  # Split by newlines and process each line
  lines = theme_str.split("\n")
  current_key = None
  current_value = []

  for line in lines:
    # Check if line starts with a known key
    if line.startswith("theme: "):
      # Save previous key-value if exists
      if current_key and current_value:
        result[current_key] = "\n".join(current_value).strip()
      current_key = "theme"
      current_value = [line[7:]]  # Remove "theme: " prefix
    elif line.startswith("refined_theme: "):
      # Save previous key-value if exists
      if current_key and current_value:
        result[current_key] = "\n".join(current_value).strip()
      current_key = "refined_theme"
      current_value = [line[15:]]  # Remove "refined_theme: " prefix
    elif line.startswith("definition: "):
      # Save previous key-value if exists
      if current_key and current_value:
        result[current_key] = "\n".join(current_value).strip()
      current_key = "definition"
      current_value = [line[12:]]  # Remove "definition: " prefix
    else:
      # Continuation of previous value
      if current_key:
        current_value.append(line)

  # Save the last key-value pair
  if current_key and current_value:
    result[current_key] = "\n".join(current_value).strip()

  return result


##
# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
  "-st",
  "--sim_thresh",
  type=float,
  default=0.95,
  required=False,
  help="Similarity Threshold.",
)
parser.add_argument(
  "-tf",
  "--themes",
  type=str,
  required=True,
  help="CSV of qualities that may be present in a story. E.g. a CSV of themes derived from the TA workflow.",
)
parser.add_argument("stories", nargs="+", help="Text files.")
args = parser.parse_args()

##
# start main script
##

storynames = []
stories = []
for storyfile in args.stories:
  storynames.append(Path(storyfile).stem)
  story = read_file(storyfile)
  stories.append(story)
storydf = pd.DataFrame(dtype="string")
storydf["Files"] = storynames
storydf["Stories"] = stories

# modelname = 'Alibaba-NLP/gte-base-en-v1.5'
modelname = "Qwen/Qwen3-Embedding-8B"
# embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cpu'})
embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={"trust_remote_code": True, "device": "cuda"})

##
#  Theme-to-Story Search: Find stories most relevant to each theme
story_vdb = FAISS.from_documents([Document(x) for x in stories], embeddings, normalize_L2=True)
ti_name = args.themes
ti_out = ti_name.replace("kept", "theme_indexed")
if ti_out == ti_name: ti_out = Path(ti_name).stem + '_theme_indexed' + '.csv'
tidf = pd.DataFrame()
themedf = pd.read_csv(args.themes)
fails = 0
successes = 0
themes = []  # for the theme_index output
for ndx, row in themedf.iterrows():
  theme = theme_join(row)
  themes.append(theme)
  simstories = story_vdb.similarity_search(theme, k=story_vdb.index.ntotal, score_threshold=args.sim_thresh)
  matches = []
  for simstory in simstories:
    txt = simstory.page_content
    match = storydf[storydf["Stories"] == txt]
    matches.append(list(match["Files"])[0])
  if not matches:
    fails = fails + 1
  else:
    successes = successes + 1
  tirow = pd.DataFrame({"match num": str(len(matches)), "theme": row.iloc[0], "matches": [matches]})
  tidf = pd.concat([tidf, tirow])
tidf.to_csv(ti_out, index=False)
print_stats(successes, fails, "themes", "stories")

##
# Story-to-Theme Search: Find themes most relevant to each story
theme_vdb = FAISS.from_documents([Document(x) for x in themes], embeddings, normalize_L2=True)
si_name = args.themes
si_out = si_name.replace("kept", "story_indexed")
if si_out == si_name: si_out = Path(si_name).stem + '_story_indexed' + '.csv'
sidf = pd.DataFrame()
fails = 0
successes = 0
for ndx, row in storydf.iterrows():
  story = row["Stories"]
  simthemes = theme_vdb.similarity_search(story, k=theme_vdb.index.ntotal, score_threshold=args.sim_thresh)
  matches = []
  for simtheme in simthemes:
    match = parse_theme_string(simtheme.page_content)["theme"]
    matches.append(match)
  if not matches:
    fails = fails + 1
  else:
    successes = successes + 1
  sirow = pd.DataFrame({"match num": str(len(matches)), "file": row["Files"], "theme": [matches]})
  sidf = pd.concat([sidf, sirow])
sidf.to_csv(si_out, index=False)
print_stats(successes, fails, "stories", "themes")
