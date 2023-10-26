# BLAST

BLAST method transfers GO terms of a target protein in training set to the query protein in test set via BLAST algorithm, and the identity score of alignment is used as a coefficient for all assigned terms.

Some data is generated from the main DualNetGO pipeline, so please make sure to run the codes in the **Data preprocessing** section for DualNetGO first.

## Generating Fasta files

Sequence data can be downloaded from STRING database (version 11.5), or you can use the link [here](https://version-11-5.string-db.org/cgi/download?sessionId=bomegv4bG3lg&species_text=Homo+sapiens) for the human data.

```
python split_fasta.py --org human --aspect P
python split_fasta.py --org human --aspect F
python split_fasta.py --org human --aspect C
python split_fasta.py --org mouse --aspect P
python split_fasta.py --org mouse --aspect F
python split_fasta.py --org mouse --aspect C
```

Because different GO aspect contains different proteins for one species, GO apsect has to be specified through argument `--aspect`.

## Sequence alignment

Next use blast to align sequences in the test set to those in the training set to find out the most similiar protein for each of the query protein in the test set, and also get the identity scores from alignment results.

run the bash script to generate protein databases for training set and perform sequence alignment for all aspects in human and mouse.

```
sh blast.sh
```

## Assigning functions and evaluation

Assign a protein in the test set with the functions of a protein that is the most similar to it in the training set, with the identity score as a coefficient for all GO terms.

```
python BLAST.py --org human --aspect P
```

Change the arguments for other aspects in human or mouse.

One can also use the `output.py` script to output AUPR values for each GO term.
