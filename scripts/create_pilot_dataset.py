import pandas as pd
import gzip

def create_pilot_subset(source_file, dest_file, n_rows=1000):
    df = pd.read_csv(source_file)
    df.head(n_rows).to_csv(dest_file, index=False, compression='gzip')

if __name__ == "__main__":
    create_pilot_subset("cdvae/data/mp_20/train.csv", "data/mp_20_pilot/train.csv.gz")
    create_pilot_subset("cdvae/data/mp_20/val.csv", "data/mp_20_pilot/val.csv.gz")
    create_pilot_subset("cdvae/data/mp_20/test.csv", "data/mp_20_pilot/test.csv.gz")
