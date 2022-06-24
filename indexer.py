import argparse
import os
import resource
import shutil
import sys
from zipfile import ZipFile

from index.index_manager import index_manager

MEGABYTE = 1024 * 1024


def memory_limit(value):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def mkdir_safe(dir: str):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def main(mem: int):
    mkdir_safe("final")
    mkdir_safe("cache")
    mkdir_safe("cache/partial_counts")
    mkdir_safe("cache/partial_indexes")
    mkdir_safe("cache/pre_ind")
    index_manager("archive.zip", mem, ndocs=950493, plaintext=False)
    shutil.rmtree("cache")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "-m", dest="memory_limit", action="store", required=True, type=int, help="memory available"
    )
    args = parser.parse_args()
    memory_limit(args.memory_limit)
    try:
        main(args.memory_limit)
    except MemoryError:
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)


# You CAN (and MUST) FREELY EDIT this file (add libraries, arguments, functions and calls) to implement your indexer
# However, you should respect the memory limitation mechanism and guarantee
# it works correctly with your implementation
