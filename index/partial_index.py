import os
import shutil
from gc import collect

from tqdm import tqdm

from .file_buffer import FileBuffer


def create_partial_index(count_path: str, start_f: int, end_f: int) -> None:
    """Create partial indexes from the count mappings in count_path from start_f to end_f.
    Also create the partial total term counts for the processed documents.

    Args:
        count_path (str): Path where the count mappings are located.
        start_f (int): File to start from.
        end_f (int): File to end on.
    """
    if start_f == end_f:
        return
    f_buf = {FileBuffer(count_path, f"{fileidx}") for fileidx in range(start_f, end_f)}
    f_buf = {f for f in f_buf if f.token is not None}
    collect()

    with open(f"cache/partial_counts/{start_f}_{end_f}", "w", encoding="UTF-8") as f:
        for buf in f_buf:
            if not hasattr(buf, "total"):
                print()
            f.write(f"{buf.id}: {buf.total}\n")

    last = ""
    with open(f"cache/partial_indexes/{start_f}_{end_f}", "w", encoding="UTF-8") as out:
        while f_buf:
            # Of all FileBuffers, get the one with the lexicographically smallest token and docid.
            m = min(f_buf)
            if m.token != last:
                if last:
                    out.write("]\n")
                last = m.token
                out.write(f"{m.token}: [")

            out.write(f"({m.id}, {m.value()}),")
            m.next()
            if m.token is None:
                m.close()
                f_buf.remove(m)
        out.write("]")


def partial_index_cb(count_path: str, start_f: int, end_f: int):
    """Callback to collect garbage after the partial index is created. Probably not necessary."""
    create_partial_index(count_path, start_f, end_f)
    collect()


def merge_indexes(partial_path):
    """Merge partial indexes in partial_path, also create a word mapping
    the word to the line it occurs on the final index.

    Args:
        partial_path (str): Path containing the partial index.
    """
    f_buf = {FileBuffer(partial_path, filename) for filename in os.listdir(partial_path)}
    f_buf = {f for f in f_buf if f.token is not None}
    last = ""
    cur_word_id = 1
    collect_interval = 10**6
    with tqdm() as pbar:
        with open("final/index", "w", encoding="UTF-8") as out:
            while f_buf:
                # Of all FileBuffers, get the one with the lexicographically smallest token and docid.
                m = min(f_buf)
                if m.token != last:
                    pbar.update(1)
                    cur_word_id += 1
                    if not cur_word_id % collect_interval:
                        collect()
                    if last:
                        out.write("]\n")
                    last = m.token
                    out.write(f"{m.token}: [")

                out.write(",".join([f"({id},{count})" for id, count in m.value()]))
                out.write(",")
                m.next()
                if m.token is None:
                    f_buf.remove(m)
                    collect()

            out.write("]")


def merge_counts():
    """Merge the partial term counts."""
    with open("final/count", "w", encoding="UTF-8") as out:
        for filename in os.listdir("cache/partial_counts"):
            with open(os.path.join("cache/partial_counts", filename), "r", encoding="UTF-8") as f:
                shutil.copyfileobj(f, out)
