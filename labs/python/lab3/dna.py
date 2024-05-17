"""
Computer Systems Architecture - Lab 3

Threadpool assignment:
    Use a pool of threads to search for a DNA sequence in a list of DNA samples
"""
from concurrent.futures import ThreadPoolExecutor as ThreadPool

random.seed(0)

# generate many DNA samples, need to be large, otherwise there's no point of a concurrent program
dna_samples = []  # TODO 1.

SEARCH_SEQUENCE = []  # TODO 2.


def search_dna_sequence(sequence, sample):
    """
    TODO 4: Search a DNA sample in a DNA sequence
    :return: True if the sequence was found, False otherwise
    """
    raise NotImplementedError


def thread_job(sample_index):
    """
    Each thread searches the sequence in a given sample
    """
    if search_dna_sequence(SEARCH_SEQUENCE, dna_samples[sample_index]):
        return "DNA sequence found in sample {}".format(sample_index)
    return "DNA sequence not found in sample {}".format(sample_index)


if __name__ == "__main__":

    thread_pool = # TODO 3.

    futures = []

    with thread_pool:
        # TODO5: execute thread job using submit or map

        # TODO6: print results
