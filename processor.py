import argparse

from query import QueryProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the index file")
    parser.add_argument(
        "-i", dest="index_path", action="store", required=True, type=str, help="Path to the index file"
    )
    parser.add_argument(
        "-q",
        dest="query_path",
        action="store",
        required=True,
        type=str,
        help="Path to the file containing a list of queries",
    )
    parser.add_argument(
        "-r",
        dest="ranking_function",
        action="store",
        required=True,
        type=str,
        help='Ranking function to be used. Valid arguments are: "TFIDF" and "BM25"',
    )
    args = parser.parse_args()
    if args.ranking_function not in ["TFIDF", "BM25"]:
        raise ValueError(f'{args.ranking_function} is not a valid argument for -r. Valid arguments are: "TFIDF" and "BM25"')

    QueryProcessor(args.index_path, args.query_path, args.ranking_function).process_queries()
