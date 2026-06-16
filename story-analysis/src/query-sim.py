"""
Embed themes and retrieve all the stories that are similar to that theme.
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-st', '--sim_thresh', type=float, default=0.95, required=False, help='Similarity Threshold.')
parser.add_argument('-tf','--themes', type=str, required=True, help='CSV of qualities that may be present in a story. E.g. a CSV of themes derived from the TA workflow.')
parser.add_argument('stories', nargs='+', help='Text files.')
args = parser.parse_args()

import os
import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def read_file(fn):
  with open(fn,'r') as f: return(f.read())

def theme_join(row):
    if not isinstance(row, pd.Series):
        return "No theme"

    parts = [str(x) for x in row if pd.notna(x)]
    if not parts:
        return "No theme"

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
    result = {
        'theme': '',
        'refined_theme': '',
        'definition': ''
    }

    if not isinstance(theme_str, str):
        return result

    # Split by newlines and process each line
    lines = theme_str.split('\n')
    current_key = None
    current_value = []

    for line in lines:
        # Check if line starts with a known key
        if line.startswith('theme: '):
            # Save previous key-value if exists
            if current_key and current_value:
                result[current_key] = '\n'.join(current_value).strip()
            current_key = 'theme'
            current_value = [line[7:]]  # Remove "theme: " prefix
        elif line.startswith('refined_theme: '):
            # Save previous key-value if exists
            if current_key and current_value:
                result[current_key] = '\n'.join(current_value).strip()
            current_key = 'refined_theme'
            current_value = [line[15:]]  # Remove "refined_theme: " prefix
        elif line.startswith('definition: '):
            # Save previous key-value if exists
            if current_key and current_value:
                result[current_key] = '\n'.join(current_value).strip()
            current_key = 'definition'
            current_value = [line[12:]]  # Remove "definition: " prefix
        else:
            # Continuation of previous value
            if current_key:
                current_value.append(line)

    # Save the last key-value pair
    if current_key and current_value:
        result[current_key] = '\n'.join(current_value).strip()

    return result

storynames = []
stories = []
for storyfile in args.stories:
  storynames.append(Path(storyfile).stem)
  story = read_file(storyfile)
  stories.append(story)
storydf = pd.DataFrame(dtype='string')
storydf['Files'] = storynames
storydf['Stories'] = stories

#modelname = 'Alibaba-NLP/gte-base-en-v1.5'
modelname = 'Qwen/Qwen3-Embedding-8B'
#embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cpu'})
embeddings = HuggingFaceEmbeddings(model_name=modelname, model_kwargs={'trust_remote_code': True, 'device': 'cuda'})

story_vdb = FAISS.from_documents([Document(x) for x in stories], embeddings, normalize_L2=True)

ti_name = args.themes
ti_name = ti_name.replace('kept', 'theme_indexed')
tidf = pd.DataFrame()
themedf = pd.read_csv(args.themes)
fails = 0
successes = 0
themes = [] # for the theme_index output
for ndx,row in themedf.iterrows():
  theme = theme_join(row)
  themes.append(theme)
  simstories = story_vdb.similarity_search(theme,k=story_vdb.index.ntotal,score_threshold=args.sim_thresh)
  matches = []
  for simstory in simstories:
    txt = simstory.page_content
    match = storydf[storydf['Stories'] == txt]
    matches.append(list(match['Files'])[0])
  if not matches: fails = fails+1
  else: successes = successes+1
  tirow = pd.DataFrame({'match num': str(len(matches)), 'theme': row.iloc[0], 'matches': [matches]})
  tidf = pd.concat([tidf,tirow])
tidf.to_csv(ti_name, index=False)
print(str(successes)+' themes match at least 1 story. '+str(fails)+' themes match no stories.')

theme_vdb = FAISS.from_documents([Document(x) for x in themes], embeddings, normalize_L2=True)
si_name = args.themes
si_name = si_name.replace('kept','story_indexed')
sidf = pd.DataFrame()
fails = 0
successes = 0
for ndx,row in storydf.iterrows():
  story = row['Stories']
  simthemes = theme_vdb.similarity_search(story,k=theme_vdb.index.ntotal,score_threshold=args.sim_thresh)
  matches = []
  for simtheme in simthemes:
    match = parse_theme_string(simtheme.page_content)['theme']
    matches.append(match)
  if not matches: fails = fails + 1
  else: successes = successes + 1
  sirow = pd.DataFrame({'match num': str(len(matches)), 'file': row['Files'], 'theme': [matches]})
  sidf = pd.concat([sidf,sirow])
sidf.to_csv(si_name, index=False)
print(str(successes)+' stories match at least 1 theme. '+str(fails)+' stories match no themes.')
