import os
from collections import OrderedDict
from gc import collect
from typing import Iterable, Tuple
from contextlib import closing

from bs4 import BeautifulSoup, SoupStrainer
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
import zipp

# Stopwords obtained from nltk.stopwords. Those are hardcoded here because nltk's function returns a list, and a set is more apropiate for string lookups
ignored_words = {
    "de",
    "a",
    "o",
    "que",
    "e",
    "é",
    "do",
    "da",
    "em",
    "um",
    "para",
    "com",
    "não",
    "uma",
    "os",
    "no",
    "se",
    "na",
    "por",
    "mais",
    "as",
    "dos",
    "como",
    "mas",
    "ao",
    "ele",
    "das",
    "à",
    "seu",
    "sua",
    "ou",
    "quando",
    "muito",
    "nos",
    "já",
    "eu",
    "também",
    "só",
    "pelo",
    "pela",
    "até",
    "isso",
    "ela",
    "entre",
    "depois",
    "sem",
    "mesmo",
    "aos",
    "seus",
    "quem",
    "nas",
    "me",
    "esse",
    "eles",
    "você",
    "essa",
    "num",
    "nem",
    "suas",
    "meu",
    "às",
    "minha",
    "numa",
    "pelos",
    "elas",
    "qual",
    "nós",
    "lhe",
    "deles",
    "essas",
    "esses",
    "pelas",
    "este",
    "dele",
    "tu",
    "te",
    "vocês",
    "vos",
    "lhes",
    "meus",
    "minhas",
    "teu",
    "tua",
    "teus",
    "tuas",
    "nosso",
    "nossa",
    "nossos",
    "nossas",
    "dela",
    "delas",
    "esta",
    "estes",
    "estas",
    "aquele",
    "aquela",
    "aqueles",
    "aquelas",
    "isto",
    "aquilo",
    "estou",
    "está",
    "estamos",
    "estão",
    "estive",
    "esteve",
    "estivemos",
    "estiveram",
    "estava",
    "estávamos",
    "estavam",
    "estivera",
    "estivéramos",
    "esteja",
    "estejamos",
    "estejam",
    "estivesse",
    "estivéssemos",
    "estivessem",
    "estiver",
    "estivermos",
    "estiverem",
    "hei",
    "há",
    "havemos",
    "hão",
    "houve",
    "houvemos",
    "houveram",
    "houvera",
    "houvéramos",
    "haja",
    "hajamos",
    "hajam",
    "houvesse",
    "houvéssemos",
    "houvessem",
    "houver",
    "houvermos",
    "houverem",
    "houverei",
    "houverá",
    "houveremos",
    "houverão",
    "houveria",
    "houveríamos",
    "houveriam",
    "sou",
    "somos",
    "são",
    "era",
    "éramos",
    "eram",
    "fui",
    "foi",
    "fomos",
    "foram",
    "fora",
    "fôramos",
    "seja",
    "sejamos",
    "sejam",
    "fosse",
    "fôssemos",
    "fossem",
    "for",
    "formos",
    "forem",
    "serei",
    "será",
    "seremos",
    "serão",
    "seria",
    "seríamos",
    "seriam",
    "tenho",
    "tem",
    "temos",
    "tém",
    "tinha",
    "tínhamos",
    "tinham",
    "tive",
    "teve",
    "tivemos",
    "tiveram",
    "tivera",
    "tivéramos",
    "tenha",
    "tenhamos",
    "tenham",
    "tivesse",
    "tivéssemos",
    "tivessem",
    "tiver",
    "tivermos",
    "tiverem",
    "terei",
    "terá",
    "teremos",
    "terão",
    "teria",
    "teríamos",
    "teriam",
}


def count(dic: OrderedDict, word: str) -> OrderedDict:
    """Count up doc[word] by one. If it is not set yet, set to 0."""
    dic[word] = dic.setdefault(word, 0) + 1
    return dic


def tag_visible(*tag):
    """Visible tags filter."""
    if tag[0] in [
        "html",
        "style",
        "script",
        "head",
        "meta",
        "[document]",
    ]:
        return False
    return True


def get_visible(html: str) -> str:
    """Get the visible text from html. Probably O(len(html))

    Args:
        html (str): An html document (hopefully).

    Returns:
        str: The visible text from the html.
    """
    strainer = SoupStrainer(tag_visible)
    soup = BeautifulSoup(html, "html.parser", parse_only=strainer)
    visible_text = soup.get_text(separator=" ")
    soup.decompose()
    return visible_text


# Formats that were found in the corpus that should not be processed.
# Ideally this filtering would've been done in the corpus building stage,
# since indexing these types of doduments goes beyond the scope of this assignment
# and checking the filetype just by the url is not very reliable, and loading each file
# to check for the type uses too much memory.
excluded_formats = {
    ".mp4",
    ".png",
    ".fdm",
    ".pdf",
    ".doc",
    ".dll",
    ".exe",
    ".jpg",
    ".sh",
    ".yml",
    ".xsl",
    ".xml",
    ".mpq",
}


def warc_loader(documents_path: str, total) -> Iterable[Tuple[bytes, int]]:
    """Generator that yields the documents in each warc file in the zip file specified by documents_path, at the same
    time it writes a bijective mapping of integers to the urls of the documents.

    Args:
        documents_path (str): Path to a zip file containing the warc files.

    Yields:
        Tuple[bytes, int]: The document and its index
    """
    with tqdm(total=total) as pbar:
        with open("final/url_index", "w", encoding="UTF-8") as urlidx:
            idx = 0
            root = zipp.Path(documents_path)
            for file in root.iterdir():
                if file.suffix != ".kaggle":
                    continue
                with file.open(mode="rb") as stream:
                    with closing(ArchiveIterator(stream)) as ai:
                        for record in ai:
                            url: str = record.rec_headers.get_header("WARC-Target-URI")

                            if os.path.splitext(url)[1].lower() in excluded_formats:
                                pbar.update(1)
                                continue

                            doc = record.content_stream().read()
                            urlidx.write(f'{idx}: "{url}",\n')
                            pbar.update(1)
                            yield doc, idx
                            idx += 1

                collect()


def partitioned_loader(loader, n):
    for _ in range(n):
        yield next(loader)
