import os
from queue import Empty
import re
from collections import OrderedDict
from functools import reduce
from gc import collect
from traceback import print_exc

from charset_normalizer import from_bytes
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from nltk_light import download, word_tokenize
from nltk_light.stem import RSLPStemmer
from tqdm import tqdm

from .partial_index import partial_index_cb, merge_counts, merge_indexes
from .util import count, get_visible, ignored_words, warc_loader, partitioned_loader

stemmer = RSLPStemmer()


def count_worker(document: bytes, idx: int) -> None:
    """Writes to the file idx a mapping of the tokens in document to their counts.

    Args:
        document (bytes): Document to be processed.
        idx (int): The document's index.
    """
    vis = get_visible(str(from_bytes(document).best())) # O(len(document))
    tokens = word_tokenize(vis, "portuguese") # O(len(document))
    ntokens = len(tokens)
    tokens = filter(lambda word: not re.search(r"[^\w]|[\d]|\_", word), tokens) # O(len(document))
    tokens = filter(lambda word: word not in ignored_words, tokens) # O(len(document))
    tokens = map(stemmer.stem, tokens) # O(len(document))
    tokens = sorted(tokens) # O(len(tokens)log len(tokens)), in the worst case len(tokens) = len(document) <- dominating
    tokens = reduce(count, tokens, OrderedDict()) # O(len(document))

    with open(f"cache/pre_ind/{idx}", "w", encoding="UTF-8") as f:
        f.write(f"{ntokens}\n")
        for token, c in tokens.items():
            f.write(f"{token}: {c}\n")


def count_worker_plain(document: bytes, idx: int) -> None:
    """Writes to the file idx a mapping of the tokens in document to their counts. Plaintext version.

    Args:
        document (bytes): Document to be processed.
        idx (int): The document's index.
    """
    tokens = word_tokenize(str(from_bytes(document).best()), "portuguese")
    ntokens = len(tokens)
    tokens = filter(lambda word: not re.search(r"[^\w]|[\d]|\_", word), tokens)
    tokens = map(stemmer.stem, tokens)
    tokens = filter(lambda word: word not in ignored_words, tokens)
    tokens = sorted(tokens)
    tokens = reduce(count, tokens, OrderedDict())

    with open(f"cache/pre_ind/{idx}", "w", encoding="UTF-8") as f:
        f.write(f"{ntokens}\n")
        for token, c in tokens.items():
            f.write(f"{token}: {c}\n")


def create_count(document: bytes, idx: int) -> None:
    """Calls index_worker(/2) and collects garbage after its execution. O(1)"""
    try:
        count_worker(document, idx)
    except Exception as e:
        print(e)
        print_exc()
    # collect()


def index_manager(corpus_path: str, max_memory: int, ndocs=None, plaintext=False) -> None:
    """Manages the index creation process. First it creates the token -> count files for
    each document in the corpus (located in the documents_path). Then it creates partial
    indexes by merging said counts in sets of a 1000. Finally it merges the partial indexes.
    It does that without surpassing the memory limit provided by max_memory.

    It adapts to how much memory is available by assuming that a process running the create_count
    function won't exceed 250MB and a process merging the counts won't exceed 100MB. Therefore we can create
    $max_memory//count_mem$ and $max_memory//partial_mem$ processes for each step. Since it wouldn't make
    sense to create too many more processes than twice the ammount of cpu cores, we limit the process count
    to that ammount.

    Since the final merging step opens as many file pointers as there are partial indexes,
    it could exceed max_memory or the max open files limit, this could be fixed by merging
    the partial indexes until there were less than a set amount of partial indexes. Even
    then it could still exceed the memory limit if the lines of the partial indexes were too big.
    This won't happen for less than 10**6 documents.

    Args:
        documents_path (str): Path to the document corpus.
        max_memory (int): Max memory in MB that the indexer can use at any given moment.
        ndocs (int|optional): Number of documents in the corpus.
        plaintext(bool|optional): If the corpus contains only plaintext files. Set to False by default.
    """
    download("rslp")

    cpu_count = os.cpu_count() or 4
    count_mem = 150 if not plaintext else 120
    count_jobs = min(((max_memory // count_mem) - 1, cpu_count))
    partial_mem = 100
    partial_jobs = min(((max_memory // partial_mem) - 1, cpu_count))
    loader = warc_loader(corpus_path, total=ndocs)
    countf = create_count if not plaintext else count_worker_plain

    print("COUNTING TERMS:")

    """
    Python refuses to completely free allocated memory (https://rushter.com/blog/python-garbage-collector/)
    even when calling gc.collect(/0), it does however reuse it. For the 10**6 documents provided this does
    not pose a problem. However for larger values, or different collections of documents it may. The following
    code kills the processes after 10000 documents (1 WARC file), and restarts them to process the next 10000 documents.
    The memory usage of the main process still increases nevertheless, because python just does that, but this
    could be mitigated by doing something similar with index_manager(/4), where an alternate version of this
    function runs in a separate process that restarts after some time.

    \"If you create a large object and delete it again, Python has probably released the memory, but the
    memory allocators involved don’t necessarily return the memory to the operating system\"
    http://effbot.org/pyfaq/why-doesnt-python-release-the-memory-when-i-delete-a-large-object.htm

    \"The only really reliable way to ensure that a large but temporary use of memory DOES return
    all resources to the system when it's done, is to have that use happen in a subprocess,
    which does the memory-hungry work then terminates.\"
    https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python/1316799#1316799

    If done without restarting the processes count_meme has to be set to 250 if not plaintext else 150, and create_count
    should call collect(/0).
    """
    while True:
        try:
            Parallel(n_jobs=count_jobs)(
                delayed(countf)(doc, idx) for doc, idx in partitioned_loader(loader, 10000) # O((n log n)*|Corpus|)
            )
        except (RuntimeError, StopIteration, Empty):
            # partitioned loader throws StopIteration, but this exception is caught by Parallel and it throws RuntimeError.
            print("Finished Count")
            break


        get_reusable_executor().shutdown(wait=True)
        collect()
    get_reusable_executor().shutdown(wait=True)
    collect()

    with Parallel(n_jobs=partial_jobs) as parallel:
        num_counts = len(os.listdir("cache/pre_ind"))
        step = 1000
        print("CREATING PARTIAL INDEXES:")
        parallel(
            delayed(partial_index_cb)("cache/pre_ind", start, start + step) # O(n*nfiles) n = docsize
            for start in tqdm(range(0, num_counts, step))
        )

    get_reusable_executor().shutdown(wait=True)
    collect()
    print("MERGING PARTIAL INDEXES:")
    merge_counts()
    collect()
    merge_indexes("cache/partial_indexes")# O(nterms*nfiles)
    collect()
