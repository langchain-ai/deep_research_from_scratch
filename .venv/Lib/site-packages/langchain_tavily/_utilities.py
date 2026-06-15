"""Util that calls Tavily Search + Extract API.

In order to set this up, follow instructions at:
https://docs.tavily.com/docs/tavily-api/introduction
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union

import aiohttp
import requests
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, SecretStr, model_validator

TAVILY_API_URL = "https://api.tavily.com"


class TavilySearchAPIWrapper(BaseModel):
    """Wrapper for Tavily Search API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        query: str,
        max_results: Optional[int],
        search_depth: Optional[Literal["basic", "advanced"]],
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
        include_images: Optional[bool],
        include_image_descriptions: Optional[bool],
        include_favicon: Optional[bool],
        topic: Optional[Literal["general", "news", "finance"]],
        time_range: Optional[Literal["day", "week", "month", "year"]],
        country: Optional[str],
        auto_parameters: Optional[bool],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Dict:
        params = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
            "include_image_descriptions": include_image_descriptions,
            "include_favicon": include_favicon,
            "topic": topic,
            "time_range": time_range,
            "country": country,
            "auto_parameters": auto_parameters,
            "start_date": start_date,
            "end_date": end_date,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/search",
            json=params,
            headers=headers,
        )
        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
        self,
        query: str,
        max_results: Optional[int],
        search_depth: Optional[Literal["basic", "advanced"]],
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
        include_images: Optional[bool],
        include_image_descriptions: Optional[bool],
        include_favicon: Optional[bool],
        topic: Optional[Literal["general", "news", "finance"]],
        time_range: Optional[Literal["day", "week", "month", "year"]],
        country: Optional[str],
        auto_parameters: Optional[bool],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Dict:
        """Get results from the Tavily Search API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "include_favicon": include_favicon,
                "topic": topic,
                "time_range": time_range,
                "country": country,
                "auto_parameters": auto_parameters,
                "start_date": start_date,
                "end_date": end_date,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }
            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/search", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)


class TavilyExtractAPIWrapper(BaseModel):
    """Wrapper for Tavily Extract API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        urls: List[str],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_images: Optional[bool],
        include_favicon: Optional[bool],
        format: Optional[str],
    ) -> Dict:
        params = {
            "urls": urls,
            "include_images": include_images,
            "include_favicon": include_favicon,
            "extract_depth": extract_depth,
            "format": format,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/extract",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
        self,
        urls: List[str],
        include_images: Optional[bool],
        include_favicon: Optional[bool],
        extract_depth: Optional[Literal["basic", "advanced"]],
        format: Optional[str],
    ) -> Dict:
        """Get results from the Tavily Extract API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "urls": urls,
                "include_images": include_images,
                "include_favicon": include_favicon,
                "extract_depth": extract_depth,
                "format": format,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }

            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/extract", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)


class TavilyCrawlAPIWrapper(BaseModel):
    """Wrapper for Tavily Crawl API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        include_images: Optional[bool],
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
        ],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_favicon: Optional[bool],
        format: Optional[str],
    ) -> Dict:
        params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "include_images": include_images,
            "categories": categories,
            "extract_depth": extract_depth,
            "include_favicon": include_favicon,
            "format": format,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/crawl",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
        include_images: Optional[bool],
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
        ],
        extract_depth: Optional[Literal["basic", "advanced"]],
        include_favicon: Optional[bool],
        format: Optional[str],
    ) -> Dict:
        """Get results from the Tavily Crawl API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "instructions": instructions,
                "select_paths": select_paths,
                "select_domains": select_domains,
                "exclude_paths": exclude_paths,
                "exclude_domains": exclude_domains,
                "allow_external": allow_external,
                "include_images": include_images,
                "categories": categories,
                "extract_depth": extract_depth,
                "include_favicon": include_favicon,
                "format": format,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }

            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/crawl", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)


class TavilyMapAPIWrapper(BaseModel):
    """Wrapper for Tavily Map API."""

    tavily_api_key: SecretStr
    api_base_url: Optional[str] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        tavily_api_key = get_from_dict_or_env(
            values, "tavily_api_key", "TAVILY_API_KEY"
        )
        values["tavily_api_key"] = tavily_api_key

        return values

    def raw_results(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
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
        ],
    ) -> Dict:
        params = {
            "url": url,
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "limit": limit,
            "instructions": instructions,
            "select_paths": select_paths,
            "select_domains": select_domains,
            "exclude_paths": exclude_paths,
            "exclude_domains": exclude_domains,
            "allow_external": allow_external,
            "categories": categories,
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "X-Client-Source": "langchain-tavily",
        }

        base_url = self.api_base_url or TAVILY_API_URL
        response = requests.post(
            # type: ignore
            f"{base_url}/map",
            json=params,
            headers=headers,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", {})
            error_message = (
                detail.get("error") if isinstance(detail, dict) else "Unknown error"
            )
            raise ValueError(f"Error {response.status_code}: {error_message}")
        return response.json()

    async def raw_results_async(
        self,
        url: str,
        max_depth: Optional[int],
        max_breadth: Optional[int],
        limit: Optional[int],
        instructions: Optional[str],
        select_paths: Optional[List[str]],
        select_domains: Optional[List[str]],
        exclude_paths: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        allow_external: Optional[bool],
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
        ],
    ) -> Dict:
        """Get results from the Tavily Map API asynchronously."""

        # Function to perform the API call
        async def fetch() -> str:
            params = {
                "url": url,
                "max_depth": max_depth,
                "max_breadth": max_breadth,
                "limit": limit,
                "instructions": instructions,
                "select_paths": select_paths,
                "select_domains": select_domains,
                "exclude_paths": exclude_paths,
                "exclude_domains": exclude_domains,
                "allow_external": allow_external,
                "categories": categories,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            headers = {
                "Authorization": f"Bearer {self.tavily_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                "X-Client-Source": "langchain-tavily",
            }
            base_url = self.api_base_url or TAVILY_API_URL
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/map", json=params, headers=headers
                ) as res:
                    if res.status == 200:
                        data = await res.text()
                        return data
                    else:
                        raise Exception(f"Error {res.status}: {res.reason}")

        results_json_str = await fetch()

        return json.loads(results_json_str)
