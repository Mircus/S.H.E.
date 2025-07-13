import xgi
import pandas as pd

def build_simplicial_set_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    H = xgi.Hypergraph()

    for _, row in df.iterrows():
        members = row['members'].split(';')
        attributes = {k: row[k] for k in df.columns if k != 'members'}
        H.add_edge(members, **attributes)

    return H