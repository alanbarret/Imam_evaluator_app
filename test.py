from operator import xor
from typing import List

# These imports should be in your Python module path
# after installing the `pyacoustid` package from PyPI.
import acoustid
import chromaprint


def get_fingerprint(filename: str) -> List[int]:
    """
    Reads an audio file from the filesystem and returns a
    fingerprint.

    Args:
        filename: The filename of an audio file on the local
            filesystem to read.

    Returns:
        Returns a list of 32-bit integers. Two fingerprints can
        be roughly compared by counting the number of
        corresponding bits that are different from each other.
    """
    _, encoded = acoustid.fingerprint_file(filename)
    fingerprint, _ = chromaprint.decode_fingerprint(
        encoded
    )
    return fingerprint


def fingerprint_distance(
    f1: List[int],
    f2: List[int],
    fingerprint_len: int,
) -> float:
    """
    Returns a normalized distance between two fingerprints.

    Args:
        f1: The first fingerprint.

        f2: The second fingerprint.

        fingerprint_len: Only compare the first `fingerprint_len`
            integers in each fingerprint. This is useful
            when comparing audio samples of a different length.

    Returns:
        Returns a number between 0.0 and 1.0 representing
        the distance between two fingerprints. This value
        represents distance as like a percentage.
    """
    max_hamming_weight = 32 * fingerprint_len
    hamming_weight = sum(
        sum(
            c == "1"
            for c in bin(xor(f1[i], f2[i]))
        )
        for i in range(fingerprint_len)
    )
    return hamming_weight / max_hamming_weight
