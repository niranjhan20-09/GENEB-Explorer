"""The deployed app's existing 64-feature, L1-normalized 3-mer representation."""
from itertools import product

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

MAX_BASES = 100_000
VOCABULARY = ["".join(chars) for chars in product("ACGT", repeat=3)]


def clean_sequence(text):
    """Accept one plain DNA sequence; normalize whitespace and letter case."""
    sequence = "".join(text.split()).upper()
    if len(sequence) < 3:
        raise ValueError("Enter a DNA sequence at least 3 bases long.")
    if len(sequence) > MAX_BASES:
        raise ValueError(f"Use at most {MAX_BASES:,} bases for this interactive demo.")
    if not set(sequence).issubset(set("ACGT")):
        raise ValueError("Only A, C, G and T are supported. Remove FASTA headers and ambiguous bases such as N.")
    return sequence


def dna_to_kmers(sequence):
    return [sequence[i:i + 3] for i in range(len(sequence) - 2)
            if set(sequence[i:i + 3]) <= set("ACGT")]


def sequence_to_features(sequence):
    sequence = clean_sequence(sequence)
    vectorizer = CountVectorizer(
        analyzer=dna_to_kmers, vocabulary=VOCABULARY, lowercase=False
    )
    return normalize(vectorizer.transform([sequence]), norm="l1").toarray()
