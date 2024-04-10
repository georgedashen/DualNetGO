import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input sequence csv file')
    parser.add_argument('--output', type=str, default='out.fasta')
    args = parser.parse_args()

    outfile = f'../data/cafa3/{args.output}'

    df = pd.read_csv(args.input)

    with open(outfile, 'w') as f:
        for i in range(len(df)):
            if df['protein'][i] == 0:
                f.write(f'>seq_{i}\n')
            else:
                f.write(f">{df['protein'][i]}\n")

            f.write(f"{df['sequences'][i]}\n")

