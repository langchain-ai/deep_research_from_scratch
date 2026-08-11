"""Tavily tools."""

from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_tavily._utilities import TavilySearchAPIWrapper


class TavilySearchInput(BaseModel):
    """Input for [TavilySearch]"""

    query: str = Field(description=("Search query to look up"))
    include_domains: Optional[List[str]] = Field(
        default=[],
        description="""A list of domains to restrict search results to.

        Use this parameter when:
        1. The user explicitly requests information from specific websites (e.g., "Find climate data from nasa.gov")
        2. The user mentions an organization or company without specifying the domain (e.g., "Find information about iPhones from Apple")

        In both cases, you should determine the appropriate domains (e.g., ["nasa.gov"] or ["apple.com"]) and set this parameter.

        Results will ONLY come from the specified domains - no other sources will be included.
        Default is None (no domain restriction).
        """,  # noqa: E501
    )
    exclude_domains: Optional[List[str]] = Field(
        default=[],
        description="""A list of domains to exclude from search results.

        Use this parameter when:
        1. The user explicitly requests to avoid certain websites (e.g., "Find information about climate change but not from twitter.com")
        2. The user mentions not wanting results from specific organizations without naming the domain (e.g., "Find phone reviews but nothing from Apple")

        In both cases, you should determine the appropriate domains to exclude (e.g., ["twitter.com"] or ["apple.com"]) and set this parameter.

        Results will filter out all content from the specified domains.
        Default is None (no domain exclusion).
        """,  # noqa: E501
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls search thoroughness and result comprehensiveness.
    
        Use "basic" for simple queries requiring quick, straightforward answers.
        
        Use "advanced" (default) for complex queries, specialized topics, 
        rare information, or when in-depth analysis is needed.
        """,  # noqa: E501
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="""Determines if the search returns relevant images along with text results.
   
        Set to True when the user explicitly requests visuals or when images would 
        significantly enhance understanding (e.g., "Show me what black holes look like," 
        "Find pictures of Renaissance art").
        
        Leave as False (default) for most informational queries where text is sufficient.
        """,  # noqa: E501
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="""Limits results to content published within a specific timeframe.
        
        ONLY set this when the user explicitly mentions a time period 
        (e.g., "latest AI news," "articles from last week").
        
        For less popular or niche topics, use broader time ranges 
        ("month" or "year") to ensure sufficient relevant results.
   
        Options: "day" (24h), "week" (7d), "month" (30d), "year" (365d).
        
        Default is None.
        """,  # noqa: E501
    )
    topic: Optional[Literal["general", "news", "finance"]] = Field(
        default="general",
        description="""Specifies search category for optimized results.
   
        Use "general" (default) for most queries, INCLUDING those with terms like 
        "latest," "newest," or "recent" when referring to general information.

        Use "finance" for markets, investments, economic data, or financial news.

        Use "news" ONLY for politics, sports, or major current events covered by 
        mainstream media - NOT simply because a query asks for "new" information.
        """,  # noqa: E501
    )
    include_favicon: Optional[bool] = Field(
        default=False,
        description="""Determines whether to include favicon URLs for each search result.
        
        When enabled, each search result will include the website's favicon URL,
        which can be useful for:
        - Building rich UI interfaces with visual website indicators
        - Providing visual cues about the source's credibility or brand
        - Creating bookmark-like displays with recognizable site icons
        
        Set to True when creating user interfaces that benefit from visual branding
        or when favicon information enhances the user experience.
        
        Default is False to minimize response size and API usage.
        """,  # noqa: E501
    )
    start_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or after this date.
        
        Use this parameter when you need to:
        - Find recent developments or updates on a topic
        - Exclude outdated information from search results
        - Focus on content within a specific timeframe
        - Combine with end_date to create a custom date range
        
        Format must be YYYY-MM-DD (e.g., "2024-01-15" for January 15, 2024).
        
        Examples:
        - "2024-01-01" - Results from January 1, 2024 onwards
        - "2023-12-25" - Results from December 25, 2023 onwards
        
        When combined with end_date, creates a precise date range filter.
        
        Default is None (no start date restriction).
        """,  # noqa: E501
    )
    end_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or before this date.
        
        Use this parameter when you need to:
        - Exclude content published after a certain date
        - Study historical information or past events
        - Research how topics were covered during specific time periods
        - Combine with start_date to create a custom date range
        
        Format must be YYYY-MM-DD (e.g., "2024-03-31" for March 31, 2024).
        
        Examples:
        - "2024-03-31" - Results up to and including March 31, 2024
        - "2023-12-31" - Results up to and including December 31, 2023
        
        When combined with start_date, creates a precise date range filter.
        For example: start_date="2024-01-01", end_date="2024-03-31" 
        returns results from Q1 2024 only.
        
        Default is None (no end date restriction).
        """,  # noqa: E501
    )


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []

    search_depth = params.get("search_depth")
    exclude_domains = params.get("exclude_domains")
    include_domains = params.get("include_domains")
    time_range = params.get("time_range")
    topic = params.get("topic")

    if time_range:
        suggestions.append("Remove time_range argument")
    if include_domains:
        suggestions.append("Remove include_domains argument")
    if exclude_domains:
        suggestions.append("Remove exclude_domains argument")
    if search_depth == "basic":
        suggestions.append("Try a more detailed search using 'advanced' search_depth")
    if topic != "general":
        suggestions.append("Try a general search using 'general' topic")

    return suggestions


class TavilySearch(BaseTool):  # type: ignore[override]
    """Tool that queries the Tavily Search API and gets back json.

    Setup:
        Install ``langchain-tavily`` and set environment variable ``TAVILY_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-tavily
            export TAVILY_API_KEY="your-api-key"

    Instantiate:

        .. code-block:: python
            from langchain_tavily import TavilySearch

            tool = TavilySearch(
                max_results=1,
                topic="general",
                # include_answer=False,
                # include_raw_content=False,
                # include_images=False,
                # include_image_descriptions=False,
                # search_depth="basic",
                # time_range="day",
                # include_domains=None,
                # exclude_domains=None,
                # country=None
                # include_favicon=False
            )

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({"query": "What happened at the last wimbledon"})

        .. code-block:: json

            {
                'query': 'What happened at the last wimbledon',
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [{'title': "Andy Murray pulls out of the men's singles draw at his last Wimbledon",
                            'url': 'https://www.nbcnews.com/news/sports/andy-murray-wimbledon-tennis-singles-draw-rcna159912',
                            'content': "NBC News Now LONDON — Andy Murray, one of the last decade's most successful ..."
                            'score': 0.6755297,
                            'raw_content': None
                            }],
                'response_time': 1.31
            }

    """  # noqa: E501

    name: str = "tavily_search"
    description: str = (
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Useful for when you need to answer questions about current events. "
        "It not only retrieves URLs and snippets, but offers advanced search depths, "
        "domain management, time range filters, and image search, this tool delivers "
        "real-time, accurate, and citation-backed results."
        "Input should be a search query."
    )

    args_schema: Type[BaseModel] = TavilySearchInput
    handle_tool_error: bool = True

    auto_parameters: Optional[bool] = None
    """
    When `auto_parameters` is enabled, Tavily automatically configures search parameters
    based on your query's content and intent. You can still set other parameters 
    manually, and your explicit values will override the automatic ones. The parameters 
    `include_answer`, `include_raw_content`, and `max_results` must always be set 
    manually, as they directly affect response size. Note: `search_depth` may be 
    automatically set to advanced when it's likely to improve results. This uses 2 API
    credits per request. To avoid the extra cost, you can explicitly set `search_depth` 
    to `basic`. 

    Default is `False`.
    """

    include_domains: Optional[List[str]] = None
    """A list of domains to specifically include in the search results

    default is None
    """
    exclude_domains: Optional[List[str]] = None
    """A list of domains to specifically exclude from the search results

    default is None
    """
    search_depth: Optional[Literal["basic", "advanced"]] = None
    """The depth of the search. It can be 'basic' or 'advanced'
    
    default is "basic"
    """
    include_images: Optional[bool] = None
    """Include a list of query related images in the response
    
    default is False
    """
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    """The time range back from the current date to filter results
    
    default is None
    """
    max_results: Optional[int] = None
    """Max search results to return, 
    
    default is 5
    """
    topic: Optional[Literal["general", "news", "finance"]] = None
    """The category of the search. Can be "general", "news", or "finance".
    
    Default is "general".
    """
    include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = None
    """Include a short answer to original query in the search results. 
    
    Default is False.
    """
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = None
    """ Include the cleaned and parsed HTML content of each search result. 
    "markdown" returns search result content in markdown format. "text" 
    returns the plain text from the results and may increase latency.
    
    Default is "markdown".
    """
    include_image_descriptions: Optional[bool] = None
    """Include a descriptive text for each image in the search results.
    
    Default is False.
    """
    country: Optional[str] = None
    """Boost search results from a specific country. This will prioritize content from 
    the selected country in the search results. Available only if topic is general.
    
    To see the countries supported visit our docs https://docs.tavily.com/documentation/api-reference/endpoint/search
    Default is None.
    """
    include_favicon: Optional[bool] = None
    """Whether to include the favicon URL for each result.
    
    Default is False.
    """
    api_wrapper: TavilySearchAPIWrapper = Field(default_factory=TavilySearchAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key and api_base_url if provided
        if "tavily_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "tavily_api_key" in kwargs:
                wrapper_kwargs["tavily_api_key"] = kwargs["tavily_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = TavilySearchAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a search query using the Tavily Search API.

        Returns:
            Dict[str, Any]: Search results containing:
                - query: Original search query
                - results: List of search results, each with:
                    - title: Title of the page
                    - url: URL of the page
                    - content: Relevant content snippet
                    - score: Relevance score
                - images: List of relevant images (if include_images=True)
                - response_time: Time taken for the search
        """
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                query=query,
                include_domains=self.include_domains
                if self.include_domains
                else include_domains,
                exclude_domains=self.exclude_domains
                if self.exclude_domains
                else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon
                if self.include_favicon
                else include_favicon,
                country=self.country,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                auto_parameters=self.auto_parameters,
                start_date=start_date,
                end_date=end_date,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                    "topic": topic,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = "basic",
        include_images: Optional[bool] = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = "general",
        include_favicon: Optional[bool] = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                include_domains=self.include_domains
                if self.include_domains
                else include_domains,
                exclude_domains=self.exclude_domains
                if self.exclude_domains
                else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon
                if self.include_favicon
                else include_favicon,
                country=self.country,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                auto_parameters=self.auto_parameters,
                start_date=start_date,
                end_date=end_date,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                    "topic": topic,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your search parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}
