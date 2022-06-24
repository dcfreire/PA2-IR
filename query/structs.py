import heapq


class PriorityQueue:
    """Simple priority queue that iterates in descending order."""

    def __init__(self, maxsize: int) -> None:
        self.h = []
        self.maxsize = maxsize

    def put(self, elem):
        if len(self.h) < self.maxsize:
            heapq.heappush(self.h, elem)
        else:
            heapq.heappushpop(self.h, elem)

    def get(self):
        return heapq.heappop(self.h)

    def empty(self):
        return not self.h

    def __iter__(self):
        h = sorted(self.h, reverse=True)
        for elem in h:
            yield elem

    def __len__(self):
        return len(self.h)


class Tup:
    """Hashable data structure that holds two values, docid and count, and can be compared with integers."""

    def __init__(self, docid, count) -> None:
        self.docid = docid
        self.count = count

    def __eq__(self, __o: object) -> bool:
        return self.docid == __o

    def __ne__(self, __o: object) -> bool:
        return self.docid != __o

    def __hash__(self) -> int:
        return hash(self.docid)

    def __getitem__(self, key):
        return self.docid if not key else self.count
