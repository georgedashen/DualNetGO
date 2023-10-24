from Bio import SeqIO
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/9606.protein.sequences.v11.5.fa', help='input sequence fasta file')
    parser.add_argument('--org', type=str, default='human')
    args = parser.parse_args()

    prot_id = np.genfromtxt(f'../data/{args.org}_protein_id.txt', dtype=str, skip_header=1)
    with open(f'./{args.org}.fa', 'w') as fout:
        for record in SeqIO.parse(args.input,'fasta'):
            if record.id in prot_id:
                SeqIO.write(record, fout, 'fasta')
