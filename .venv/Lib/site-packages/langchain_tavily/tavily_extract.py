"""Tool for the Tavily Extract API."""

from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_tavily._utilities import TavilyExtractAPIWrapper


class TavilyExtractInput(BaseModel):
    """
    Input for [TavilyExtract]
    Extract web page content from one or more specified URLs using Tavily Extract.
    """

    urls: List[str] = Field(description="list of urls to extract")
    extract_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls the thoroughness of web content extraction.
        
        Use "basic" for faster extraction of main text content.
        
        Use "advanced" (default) to retrieve comprehensive content including 
        tables and embedded elements. Always use "advanced" for LinkedIn 
        and YouTube URLs for optimal results.
        
        Better for complex websites but may increase response time.
        """,  # noqa: E501
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="""Determines whether to extract and include images from the source URLs.
   
        Set to True when visualizations are needed for better context or understanding.
    
        Default is False (extracts text content only).
        """,  # noqa: E501
    )
    include_favicon: Optional[bool] = Field(
        default=False,
        description="Whether to include the favicon URL for each result.",
    )


def _generate_suggestions(params: dict) -> list:
    """Generate helpful suggestions based on the failed search parameters."""
    suggestions = []

    if params.get("extract_depth") and params["extract_depth"] == "basic":
        suggestions.append(
            "Try a more detailed extraction using 'advanced' extract_depth"
        )

    return suggestions


class TavilyExtract(BaseTool):  # type: ignore[override, override]
    """Tool that queries the Tavily Extract API with dynamically settable parameters."""

    name: str = "tavily_extract"
    description: str = (
        "Extracts comprehensive content from web pages based on provided URLs. "
        "This tool retrieves raw text of a web page, with an option to include images. "
        "It supports two extraction depths: 'basic' for standard text extraction and "
        "'advanced' for a more comprehensive extraction with higher success rate. "
        "Ideal for use cases such as content curation, data ingestion for NLP models, "
        "and automated information retrieval, this endpoint seamlessly integrates into "
        "your content processing pipeline. Input should be a list of one or more URLs."
    )

    args_schema: Type[BaseModel] = TavilyExtractInput
    handle_tool_error: bool = True

    # Default parameters
    extract_depth: Optional[Literal["basic", "advanced"]] = None
    """The depth of the extraction process. 
    'advanced' extraction retrieves more data than 'basic',
    with higher success but may increase latency.
    
    Default is 'basic'
    """
    include_images: Optional[bool] = None
    """Include a list of images extracted from the URLs in the response.
    
    Default is False
    """
    format: Optional[str] = None
    """
    The format of the extracted web page content. markdown returns content in markdown 
    format. text returns plain text and may increase latency.
    
    Default is 'markdown'
    """
    include_favicon: Optional[bool] = None
    """Whether to include the favicon URL for each result.
    
    Default is False.
    """
    apiwrapper: TavilyExtractAPIWrapper = Field(default_factory=TavilyExtractAPIWrapper)  # type: ignore[arg-type]

    def __init__(self, **kwargs: Any) -> None:
        # Create apiwrapper with tavily_api_key and api_base_url if provided
        if "tavily_api_key" in kwargs or "api_base_url" in kwargs:
            wrapper_kwargs = {}
            if "tavily_api_key" in kwargs:
                wrapper_kwargs["tavily_api_key"] = kwargs["tavily_api_key"]
            if "api_base_url" in kwargs:
                wrapper_kwargs["api_base_url"] = kwargs["api_base_url"]
            kwargs["apiwrapper"] = TavilyExtractAPIWrapper(**wrapper_kwargs)
        super().__init__(**kwargs)

    def _run(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        include_favicon: Optional[bool] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool."""
        try:
            # Execute search with parameters directly
            raw_results = self.apiwrapper.raw_results(
                urls=urls,
                extract_depth=self.extract_depth
                if self.extract_depth
                else extract_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                include_favicon=self.include_favicon
                if self.include_favicon
                else include_favicon,
                format=self.format,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth,
                    "include_images": include_images,
                }
                suggestions = _generate_suggestions(search_params)

                # Construct a detailed message for the agent
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters with one of these approaches."  # noqa: E501
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
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        include_favicon: Optional[bool] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Use the tool asynchronously."""
        try:
            raw_results = await self.apiwrapper.raw_results_async(
                urls=urls,
                extract_depth=self.extract_depth
                if self.extract_depth
                else extract_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                include_favicon=self.include_favicon
                if self.include_favicon
                else include_favicon,
                format=self.format,
            )

            # Check if results are empty and raise a specific exception
            results = raw_results.get("results", [])
            failed_results = raw_results.get("failed_results", [])
            if not results or len(failed_results) == len(urls):
                search_params = {
                    "extract_depth": extract_depth,
                    "include_images": include_images,
                }
                suggestions = _generate_suggestions(search_params)
                error_message = (
                    f"No extracted results found for '{urls}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                    f"Try modifying your extract parameters with one of these approaches."  # noqa: E501
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            # Re-raise tool exceptions
            raise
        except Exception as e:
            return {"error": e}
