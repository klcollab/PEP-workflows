generate_codes = """You are a world-class qualitative researcher. The following input is an interview transcript between a clinician (Q) and a participant (A) with chronic low back-pain.
<input>{input}</input>
Perform the following tasks:
1. Using the steps as described in Thematic Analysis, identify initial codes or labels for each unit of meaning (e.g., sentence, phrase) in the data.
2. Provide your response as a Python list of strings. Format strings using double-quotes (\"). Do not provide any additional information except the Python list.
Here are the list of codes:
"""

generate_broad_themes = """The following input is a collection of initial codes from a Thematic Analysis extracted from several participants.
<input>{input}</input>
Perform the following tasks:
1. Combine similar codes and generate approximately 10 broad themes.
2. Provide your response as a Python list of strings. Format strings using double-quotes (\"). Do not provide any additional information except the Python list.
Here are the list of themes:"
"""

develop_theme = """The following context contains snippets of interview dialogues between a clinician (Q) and multiple participants (A) with chronic low back-pain.
<context>{context}</context>
Perform the following tasks:
1. Using the Thematic Analysis methodology, rephrase the term '{input}' into a more descriptive theme.
2. Define the theme as it relates to the group of participants.
3. Provide quotes from the participants that help establish the theme.
Here is the refined Theme and definition with supporting participant quotes:
"""
