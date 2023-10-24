import numpy as np
import argparse
from Bio import SeqIO

def CT(sequence):
    classMap = {'G':'1','A':'1','V':'1','L':'2','I':'2','F':'2','P':'2',
            'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
            'R':'5','K':'5','D':'6','E':'6','C':'7'}

    seq = ''.join([classMap[x] for x in sequence if x in classMap.keys()])
    length = len(seq)
    coding = np.zeros(343,dtype=int)
    for i in range(length-2):
        index = int(seq[i]) + (int(seq[i+1])-1)*7 + (int(seq[i+2])-1)*49 - 1
        coding[index] = coding[index] + 1

    return coding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../data/9606.protein.sequences.v11.5.fa', help='input sequence fasta file')
    parser.add_argument('--org', type=str, default='human')
    parser.add_argument('--output', type=str, default='human_CT_343.npy', help='output CT encoding numpy file, eg. human_CT_343.npy')
    args = parser.parse_args()

    ##load seq file and preprocess
    prot_id = np.genfromtxt(f'../data/{args.org}_protein_id.txt', dtype=str, skip_header=1)
    record_id = []
    seqs = []
    for record in SeqIO.parse(args.input,'fasta'):
        record_id.append(record.id)
        seqs.append(str(record.seq))
    
    idx = []
    for i in prot_id:
        if i in record_id:
            idx.append(record_id.index(i))
        else:
            print(-1)

    ##CT
    CT_encoder = np.zeros((len(prot_id),343))
    for i,k in enumerate(idx):
        CT_encoder[i,:] = CT(seqs[k])

    ##save
    np.save(args.output, CT_encoder)
