# Story Analysis

1. Choose your variables from the Data Dictionary and list them in a file. E.g.
   ```
   C_SEHHINCOME
   C_SEEDLEVEL
   C_SEEMPSTAT
   C_SEOCCUPCLS
   C_SERELSTAT
   C_LBLENGTH
   C_DMAGE
   C_DMSEX
   C_DMGENDER
   ```
2. Suffix the variable descriptions and write to new file:

   `python var-desc.py -dd <data_dictionary>.csv <variable_list>.csv`
3. Query the participant's CSV data for their values:

   `python values.py <vars_with_descriptions>.csv <participants' CSV files>`
4. Query the LLMs for their analyses:

   `python cur.py -p prompt.txt -d <data>.csv -s <story>.txt`
5. Get embedding vectors for the analyses:

   `python embed_docs.py <report_files>`
