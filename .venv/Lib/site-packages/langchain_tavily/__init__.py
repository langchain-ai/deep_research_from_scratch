from importlib import metadata

from langchain_tavily.tavily_crawl import TavilyCrawl
from langchain_tavily.tavily_extract import TavilyExtract
from langchain_tavily.tavily_map import TavilyMap
from langchain_tavily.tavily_search import TavilySearch

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "TavilySearch",
    "TavilyExtract",
    "TavilyCrawl",
    "TavilyMap",
    "__version__",
]
