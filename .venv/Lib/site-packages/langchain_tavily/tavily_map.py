"""Tool for the Tavily Map API."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_tavily._utilities import TavilyMapAPIWrapper


class TavilyMapInput(BaseModel):
    """Input for [TavilyMap]"""

    url: str = Field(description=("The root URL to begin the mapping."))
    max_depth: Optional[int] = Field(
        default=1,
        description="""Max depth of the mapping. Defines how far from the base URL the crawler can explore.

        Increase this parameter when:
        1. To map large websites and get a comprehensive overview of its structure.
        2. To map a website that has a lot of links to other pages.

        Set this parameter to 1 when:
        1. To stay local to the base_url
        2. To map a single page

        max_depth must be greater than 0
        """,  # noqa: E501
    )
    max_breadth: Optional[int] = Field(
        default=20,
        description="""Max number of links to follow per level of the tree (i.e., per page).

        tavily-map uses a BFS Depth: referring to the number of link hops from the root URL. 
        A page directly linked from the root is at BFS depth 1, regardless of its URL structure.

        Increase this parameter when:
        1. You want many links from each page to be mapped.

        max_breadth must be greater than 0
        """,  # noqa: E501
    )
    limit: Optional[int] = Field(
        default=50,
        description="""Total number of links the mapper will process before stopping.
        
        limit must be greater than 0
        """,  # noqa: E501
    )
    instructions: Optional[str] = Field(
        default=None,
        description="""Natural language instructions for the mapper.

        The instructions parameter allows the mapper to intelligently navigate through a website using natural language.
        Take the users request to set the instructions parameter to guide the mapper in the direction of the users request.
        
        ex. "I want to find all the Javascript SDK documentation from Tavily" ---> instructions = "Javascript SDK documentation"
        """,  # noqa: E501
    )
    select_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs with specific path patterns.

        Use when the user explicitly asks for a specific path from a website.
        
        ex. "Only map the /api/v1 path" ---> ["/api/v1.*"] 
        ex. "Only map the /documentation path" ---> ["/documentation/.*"]
        """,  # noqa: E501
    )
    select_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to select only URLs from specific domains or subdomains.
   
        Use when the user explicitly asks for a specific domain or subdomain from a website.

        ex. "Map only the docs.tavily.com subdomain" ---> ["^docs\\.tavily\\.com$"]
        """,  # noqa: E501
    )
    exclude_paths: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from the map with specific path patterns.

        Use when the user explicitly asks to exclude a specific path from a website.

        ex. "Map example.com but exclude the /api/v1 path form the map" ---> ["/api/v1.*"] 
        ex. "Map example.com but exclude the /documentation path from the map" ---> ["/documentation/.*"]
        """,  # noqa: E501
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="""Regex patterns to exclude URLs from specific domains or subdomains.

        Use when the user explicitly asks to exclude a specific domain or subdomain from a website.

        ex. "Map tavily.com but exclude the docs.tavily.com subdomain from the map" ---> ["^docs\\.tavily\\.com$"]
        """,  # noqa: E501
    )
    allow_external: Optional[bool] = Field(
        default=False,
        description="""Allow the crawler to follow external links.

        Use when the user explicitly asks to allow or deny external links.
        """,  # noqa: E501
    )
    categories: Optional[
        List[
            Literal[
                "Careers",
                "Blogs",
                "Documentation",
                "About",
                "Pricing",
                "Community",
                "Developers",
                "Contact",
                "Media",
            ]
        ]
    ] = Field(
        default=None,
        description="""Direct the crawler to crawl specific categories of a website.

        Set this field to the category that best matches the user's request. Use the following guide to choose the appropriate category:

            Careers: Crawl pages related to job listings, open positions, and company career information.
            Blogs: Crawl blog posts, news articles, and editorial content.
            Documentation: Crawl technical documentation, user guides, API references, and manuals.
            About: Crawl 'About Us' pages, company background, mission statements, and team information.
            Pricing: Crawl pages that detail product or service pricing, plans, and cost comparisons.
            Community: Crawl forums, discussion boards, user groups, and community-driven content.
            Developers: Crawl developer portals, SDKs, API documentation, and resources for software developers.
            Contact: Crawl contact information pages, support forms, and customer service details.
            Media: Crawl press releases, media kits, newsrooms, and multimedia content.


        ex. "Crawl apple.com for career opportunities" ---> categories="Careers"
        ex. "Crawl tavily.com for API documentation" ---> categories="Documentation"
    """,  # noqa: E501
    )


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed crawl parameters."""
    suggestions = []

    instructions = params.get("instructions")
    select_paths = params.get("select_paths")
    select_domains = params.get("select_domains")
    exclude_paths = params.get("exclude_paths")
    exclude_domains = params.get("exclude_domains")
    categories = params.get("categories")

    if instructions:
        suggestions.append("Try a more consice instructions")
    if select_paths:
        suggestions.append("Remove select_paths argument")
    if select_domains:
        suggestions.append("Remove select_domains argument")
    if exclude_paths:
        suggestions.append("Remove exclude_paths argument")
    if exclude_domains:
        suggestions.append("Remove exclude_domains argument")
    if categories:
        suggestions.append("Remove categories argument")

    return suggestions


class TavilyMap(BaseTool):  # type: ignore[override]
    """Tool that sends requests to the Tavily Map API with dynamically settable parameters."""  # noqa: E501

    name: str = "tavily_map"
    description: str = """"A powerful web mapping tool that creates a structured map of website URLs, allowing 
        you to discover and analyze site structure, content organization, and navigation paths. 
        Perfect for site audits, content discovery, and understanding website architecture.
        """  # noqa: E501

    args_schema: Type[BaseModel] = TavilyMapInput
    handle_tool_error: bool = True

    max_depth: Optional[int] = None
    """Max depth of the crawl. Defines how far from the base URL the crawler can explore.

    max_depth must be greater than 0

    default is 1
    """  # noqa: E501
    max_breadth: Optional[int] = None
    """The maximum number of links to follow per level of the tree (i.e., per page).

    max_breadth must be greater than 0

    default is 20
    """
    limit: Optional[int] = None
    """Total number of links the crawler will process before stopping.

    limit must be greater than 0

    default is 50
    """
    instructions: Optional[str] = None
    """Natural language instructions for the crawler.

    ex. "Python SDK"
    """
    select_paths: Optional[List[str]] = None
    """Regex patterns to select only URLs with specific path patterns.

    ex. ["/api/v1.*"]
    """
    select_domains: Optional[List[str]] = None
    """Regex patterns to select only URLs from specific domains or subdomains.

    ex. ["^docs\\.example\\.com$"]
    """
    exclude_paths: Optional[List[str]] = None
    """
    Regex patterns to exclude URLs with specific path patterns 
    ex.  [/private/.*, /admin/.*]
    """
    exclude_domains: Optional[List[str]] = None
    """
    Regex patterns to exclude specific domains or subdomains from mapping 
    ex. [^private\\.example\\.com$]
    """
    allow_external: Optional[bool] = None
    """Whether to allow following links that go to external domains.

    default is False
    """
    categories: Optional[
        List[
            Literal[
                "Careers",
                "Blogs",
                "Documentation",
                "About",
                "Pricing",
                "Community",
                "Developers",
                "Contact",
                "Media",
            ]
        ]
    ] = None
    """Filter URLs using predefined categories like 'Documentation', 'Blogs', etc.
    """
    api_wrapper: TavilyMapAPIWrapper = Field(default_factory=TavilyMapAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create api_wrapper with tavily_api_key and api_base_url if provided
        if "tavily_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "tavily_api_key" in kwargs:
                wrapper_kwargs["tavily_api_key"] = kwargs["tavily_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["api_wrapper"] = TavilyMapAPIWrapper(**wrapper_kwargs)

        super().__init__(**kwargs)

    def _run(
        self,
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        categories: Optional[
            List[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ]
        ] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Execute a mapping using the Tavily Map API.

        Returns:
            - base_url (str): The base URL that was mapped
                Example: "https://tavily.com/"

            - results (List[str]): A list of mapped URLs
                - url (str): The URL that was mapped
                    Example: "https://tavily.com/#features"

            - response_time (float): Time in seconds it took to complete the request

        """
        try:
            # Execute search with parameters directly
            raw_results = self.api_wrapper.raw_results(
                url=url,
                max_depth=self.max_depth if self.max_depth else max_depth,
                max_breadth=self.max_breadth if self.max_breadth else max_breadth,
                limit=self.limit if self.limit else limit,
                instructions=self.instructions if self.instructions else instructions,
                select_paths=self.select_paths if self.select_paths else select_paths,
                select_domains=self.select_domains
                if self.select_domains
                else select_domains,
                exclude_paths=self.exclude_paths
                if self.exclude_paths
                else exclude_paths,
                exclude_domains=self.exclude_domains
                if self.exclude_domains
                else exclude_domains,
                allow_external=self.allow_external
                if self.allow_external
                else allow_external,
                categories=self.categories if self.categories else categories,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "exclude_paths": exclude_paths,
                    "exclude_domains": exclude_domains,
                    "categories": categories,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."  # noqa: E501
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
        url: str,
        max_depth: Optional[int] = None,
        max_breadth: Optional[int] = None,
        limit: Optional[int] = None,
        instructions: Optional[str] = None,
        select_paths: Optional[List[str]] = None,
        select_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        allow_external: Optional[bool] = None,
        categories: Optional[
            List[
                Literal[
                    "Careers",
                    "Blogs",
                    "Documentation",
                    "About",
                    "Pricing",
                    "Community",
                    "Developers",
                    "Contact",
                    "Media",
                ]
            ]
        ] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                url=url,
                max_depth=self.max_depth if self.max_depth else max_depth,
                max_breadth=self.max_breadth if self.max_breadth else max_breadth,
                limit=self.limit if self.limit else limit,
                instructions=self.instructions if self.instructions else instructions,
                select_paths=self.select_paths if self.select_paths else select_paths,
                select_domains=self.select_domains
                if self.select_domains
                else select_domains,
                exclude_paths=self.exclude_paths
                if self.exclude_paths
                else exclude_paths,
                exclude_domains=self.exclude_domains
                if self.exclude_domains
                else exclude_domains,
                allow_external=self.allow_external
                if self.allow_external
                else allow_external,
                categories=self.categories if self.categories else categories,
            )

            # Check if results are empty and raise a specific exception
            if not raw_results.get("results", []):
                search_params = {
                    "instructions": instructions,
                    "select_paths": select_paths,
                    "select_domains": select_domains,
                    "categories": categories,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No crawl results found for '{url}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your crawl parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}
