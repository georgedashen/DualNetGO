# Naive

Naive method simply assigns the relative frequency of a term over all proteins in training set as the score for this term for all proteins in test set.

## Calculating frequencies and evaluation

Scores representing the frequency of terms across all proteins in the training set are assigned to all proteins in the test set.

```
python Naive.py 
```

Change the arguments for other aspects in human or mouse.

One can also use the `output.py` script to output AUPR values for each GO term.
