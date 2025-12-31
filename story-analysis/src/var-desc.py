"""
Column insert variable descriptions between variable name and longitudinal measures
Inputs:
  CSV formatted data dictionary. E.g.
    Domain,"Variable Description","Key Analysis Variable","Flags for Analysis","ANALYSIS DATASET(S) INCLUSION","CB STANDARIZED variable
  List of variables (e.g. C_ANBMI, C_ANBMISRCL, C_ANBMICAT), one per line
Output - CSV with variable descriptions as the 2nd column
"""
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dict', type=str, required=True, help='Data dictionary for the data set to be queried.')
parser.add_argument('-v', '--variables', type=str, required=True, help='List of variables to lookup in the dictionary.')
args = parser.parse_args()

dd = pd.read_csv(args.data_dict)
vd = dd.iloc[:,[5,1]]
dv = pd.read_csv(args.variables)
dvwd = vd.loc[vd.iloc[:,0].isin(dv.iloc[:,0])]
dvwd.to_csv('vars-with-desc.csv',index=False)
