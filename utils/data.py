import json
import pandas as pd

def src_data():
    datafile = 'data/df_CA.json'

    with open(datafile) as json_file:     
        data = json_file.readlines()
        data = list(map(json.loads, data)) 

    df = pd.DataFrame(data)
    res_list = df["name"].unique().tolist()
    
    return df, res_list
