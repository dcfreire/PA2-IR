import math
import os
import re
from statistics import mean
from time import time
from typing import Dict, List, Set

from joblib import Parallel, delayed
from nltk_light import download, word_tokenize
from nltk_light.stem import RSLPStemmer

from index.util import ignored_words

from .index import PartialIndex
from .logger import Logger
from .structs import PriorityQueue


class QueryProcessor:
    def __init__(self, ipath: str, qpath: str, rfunc: str):
        """Class that processes the queries in qpath, using the ranking function rfunc,
        on the index in ipath that contains the urls in urls_path.

        Args:
            ipath (str): Path to the index file.
            qpath (str): Path to the queries file.
            rfunc (str): Ranking function to use.
            urls_path (str): Path to the urls mapping.
        """
        download("rslp")
        self.qpath = qpath
        self.ipath = ipath
        self.rfunc = self.bm25_query if rfunc == "BM25" else self.tf_idf_query
        self.load_urls()
        self.load_count()
        self.mean_len = mean(self.count.values())
        self.stemmer = RSLPStemmer()
        self.logger = Logger()

    def load_count(self):
        """Load the term count file. O(countsize)"""
        print("Loading counts...")

        self.count: Dict[int, int] = {}
        cpath = os.path.join(''.join(os.path.split(self.ipath)[:-1]), "count")
        with open(cpath, "r", encoding="UTF-8") as ufile:
            for line in ufile:
                split = line.index(":")
                self.count[int(line[:split])] = int(line[split + 1 :])

    def load_urls(self):
        """Load the urls mapping file, that maps documentids to their respective urls. O(urlssize)"""
        print("Loading urls...")
        self.urls: Dict[int, str] = {}
        urls_path = os.path.join(''.join(os.path.split(self.ipath)[:-1]), "url_index")
        with open(urls_path, "r", encoding="UTF-8") as ufile:
            for line in ufile:
                split = line.index(":")
                self.urls[int(line[:split])] = line[split + 1 :]

    def tf(self, term: str, document: int) -> float:
        """Compute the term frequency for the specified term and document.

        Args:
            term (str): Term to compute the term frequency of.
            document (int): Document id of the document to compute the term frequency on.

        Returns:
            float: Term frequency.
        """
        return self.index[term][document] / self.count[document]

    def idf(self, term: str) -> float:
        """Compute the inverse document frequency of the term.

        Args:
            term (str): Term to compute the IDF of.

        Returns:
            float: inverse document frequency.
        """
        n = len(self.urls)
        div = len(self.index[term]) or math.inf
        return math.log(n / div)

    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    def tf_idf(self, term: str, document: int) -> float:
        """Compute the TF-IDF of the term over the document.

        Args:
            term (str): Term
            document (int): document id

        Returns:
            float: TF-IDF
        """
        return self.tf(term, document) * self.idf(term)

    def has_term(self, term: str, document: int) -> bool:
        """Checks if the document "document" has the term "term".

        Args:
            term (str): Term to check.
            document (int): Document to check.

        Returns:
            bool: If the document has the term.
        """

        return document in self.index[term]

    def tf_idf_query(self, query: List[str]) -> PriorityQueue:
        """Compute the total TF-IDF over all terms in the query.

        Args:
            query (List[str]): List of terms

        Returns:
            PriorityQueue: Priority queue containing the top 10 documents.
        """
        relevants: Set[int] = self.get_relevants(query)
        res = PriorityQueue(maxsize=10)

        for document in relevants:
            total = 0
            for token in query:
                total += self.tf_idf(token, document)
            res.put((total, document))

        return res

    def get_relevants(self, query: List[str]) -> Set[int]:
        """Get the relevant documents for a query by performing a conjunctive
        document-at-a-time matching.

        Args:
            query (List[str]): Search query.

        Returns:
            Set[int]: Set of relevant document ids.
        """
        relevants: Set[int] = set()
        for document in self.urls:
            for token in sorted(query, key=lambda x: len(self.index[x])):
                if not self.has_term(token, document):
                    break
            else:
                relevants.add(document)

        return relevants

    def bm_idf(self, term: str) -> float:
        """IDF for the BM25 ranking function.

        Args:
            term (str): Term to compute the idf for.

        Returns:
            float: IDF of the term.
        """
        N = len(self.urls)
        n = len(self.index[term])
        return math.log(((N - n + 0.5) / (n + 0.5)) + 1)

    # https://en.wikipedia.org/wiki/Okapi_BM25
    def bm25(self, document: int, query: List[str]) -> float:
        """Compute the BM25 score of the query on the document.

        Args:
            document (int): Document to compute the score of.
            query (List[str]): Search query.

        Returns:
            float: BM25 of the document.
        """
        score = 0
        k1 = 1.5
        b = 0.75
        for token in query:
            score += self.bm_idf(token) * (
                self.tf(token, document)
                * (k1 + 1)
                / (self.tf(token, document) + k1 * (1 - b + (b * (self.count[document] / self.mean_len))))
            )
        return score

    def bm25_query(self, query: List[str]) -> PriorityQueue:
        """Compute the BM25 score of all relevant documents.


        Args:
            query (List[str]): Search query.

        Returns:
            PriorityQueue: Top 10 documents.
        """
        relevants: Set[int] = self.get_relevants(query)
        res = PriorityQueue(maxsize=10)
        for document in relevants:
            res.put((self.bm25(document, query), document))

        return res

    def process_query(self, query: str) -> PriorityQueue:
        """Tokenize, stem and remove stopwords of query.

        Args:
            query (str): Search query.

        Returns:
            PriorityQueue: Top 10 documents.
        """
        preprocessed_query = self.preprocess_query(query)
        self.index = PartialIndex(self.ipath, preprocessed_query)
        return self.rfunc(preprocessed_query)

    def preprocess_query(self, query: str) -> List[str]:
        """Tokenizes, removes ponctuation, stems and remove stopwords from query.

        Args:
            query (str): Query to preprocess

        Returns:
            List[str]: Processed tokens
        """
        tokens = word_tokenize(query, "portuguese")
        tokens = filter(lambda word: not re.search(r"[^\w]|[\d]|\_", word), tokens)
        tokens = filter(lambda word: word not in ignored_words, tokens)
        tokens = map(self.stemmer.stem, tokens)
        return list(tokens)

    def query_worker(self, query: str):
        """Process the query and send a formated json string to the logger.

        Args:
            query (str): Search query.
        """
        s = time()
        res = self.process_query(query)
        e = time()
        out = {}
        out["Query"] = query.strip()
        out["Results"] = [{"URL": self.urls[document][2:-3], "Score": score} for score, document in res]
        self.logger.add_message(f"{e-s},")

    def process_queries(self):
        """Process all queries in self.qpath"""
        print("Processing queries...")
        with open(self.qpath, "r", encoding="UTF-8") as qfile:
            Parallel(n_jobs=8)(delayed(self.query_worker)(query) for query in qfile)

        self.logger.shutdown()
