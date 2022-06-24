import ast
from typing import List, TextIO

from .structs import Tup


class Index:
    def __init__(self, index_path: str) -> None:
        print("Creating index")
        self.idfp = open(index_path, "r", encoding="UTF-8")
        self.set_offsets()
        self.cur_line = 0

    def close(self):
        self.idfp.close()

    def set_offsets(self):
        """Creates a dictionary that maps tokens to their offset in bytes in the file. O(filesize)
        """
        print("Setting offsets")
        self.line_offset = {}
        offset = 0
        for line in self.idfp:
            split = line.index(":")

            self.line_offset[line[:split]] = offset
            offset += len(line.encode("utf-8"))
        self.idfp.seek(0)

    def get_value(self, i: int):
        self.idfp.seek(i)
        line = self.idfp.readline()
        split = line.index(":")
        return ast.literal_eval(line[split + 1 :])

    def next_line(self):
        self.idfp.seek(self.line_offset[self.cur_line + 1])
        self.cur_line += 1

    def __getitem__(self, key: str):
        i = self.line_offset.get(key, "")
        return self.get_value(i) if i else []

class PartialIndex:
    def __init__(self, index_path: str, terms: List[str]) -> None:
        """ Constructs a PartialIndex, which is an index containing the terms provided as a parameter.
         """
        with open(index_path, "r", encoding="UTF-8") as idfp:
            self.set_index(idfp, terms)


    def set_index(self, file: TextIO, terms: List[str]):
        """Creates a dictionary that maps terms to a dictionary that maps docids to counts. O(filesize)"""
        self.index = {}
        for line in file:
            split = line.index(":")
            term = line[:split]
            if term in terms:
                self.index[term] = dict(ast.literal_eval(line[split + 1 :]))

    def __getitem__(self, key: str):
        # O(1)
        return self.index.get(key, [])
