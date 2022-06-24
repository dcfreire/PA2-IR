import ast
import os


class FileBuffer:
    """File buffer for count files and partial indexes. Hashable by docid, and comparable by
    token then (if the token is equal) docid."""

    def __init__(self, fdir: str, filename: str):
        self.id = int(filename.split(sep="_")[0])
        try:
            self.fp = open(os.path.join(fdir, filename), "r", encoding="UTF-8")
        except FileNotFoundError:
            self.token = None
            return
        self.token = None
        self.next()

    def close(self):
        self.fp.close()

    def value(self):
        line = self.fp.readline()

        if not line:
            return None

        val = ast.literal_eval(line)
        return val

    def next(self):
        """Move the buffer to the next (token, value) pair"""
        token = ""

        char = self.fp.read(1)

        if not char:
            self.token = None
            return

        token = []
        while char != ":":
            token.append(char)
            char = self.fp.read(1)
            if char == "\n":
                self.total = int("".join(token))
                self.next()
                return

        self.token = "".join(token)

    def __gt__(self, other) -> bool:
        if self.token is None:
            return False
        if other.token is None:
            return True
        return self.token > other.token if self.token != other.token else self.id > other.id

    def __lt__(self, other) -> bool:
        if self.token is None:
            return False
        if other.token is None:
            return True
        return self.token < other.token if self.token != other.token else self.id < other.id

    def __hash__(self) -> int:
        return hash(self.id)
