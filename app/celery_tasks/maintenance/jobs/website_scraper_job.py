import logging

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, ContentTypeFilter, DomainFilter
from pymilvus import MilvusClient, DataType
from uuid import uuid4
import pandas as pd
from bs4 import BeautifulSoup

from app.services.inference.model import AiModel
from app.config import settings

# Github issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/128
# TODO general code refactoring would be nice


# TODO Create MongoDB collection pertaining to the scraped websites and APIs
# analogous to AssetCollection


async def populate_collections_wrapper() -> None:
    # TODO we should use MilvusEmbeedingStore instead
    client = MilvusClient(uri=settings.MILVUS.HOST, token=settings.MILVUS.MILVUS_TOKEN)
    model = AiModel(device=AiModel.get_device())

    website_df = pd.concat([await scraper(website) for website in settings.CRAWLER.WEBSITES])
    api_df = pd.concat([await scraper(api) for api in settings.CRAWLER.API_WEBSITES])

    populate_collection(model, client, settings.CHATBOT.WEBSITE_COLLECTION_NAME, website_df)
    populate_collection(model, client, settings.CHATBOT.API_COLLECTION_NAME, api_df)


def populate_collection(
    model: AiModel, client: MilvusClient, collection_name: str, crawled_content: pd.DataFrame
) -> None:
    if client.has_collection(collection_name):
        update_content_collection(model, client, collection_name, crawled_content)
    else:
        create_content_collection(client, collection_name)
        website_data = prepare_data(model, crawled_content)
        client.insert(collection_name=collection_name, data=website_data)


def last_modified_time(html_content: str) -> str | None:
    metadata_time = extract_meta_content(html_content, "article:modified_time")
    if metadata_time:
        return metadata_time
    else:
        span_time = extract_span_content(
            html_content,
            "git-revision-date-localized-plugin git-revision-date-localized-plugin-datetime",
        )
        if span_time:
            return span_time
        else:
            return ""


def extract_meta_content(html_content: str, property_name: str) -> str | None:
    """
    Extracts the content of a meta tag with a specific 'property' attribute.

    Args:
        html_content (str): The HTML content as a string.
        property_name (str): The value of the 'property' attribute to search for
                             (e.g., "article:modified_time").

    Returns:
        str | None: The value of the 'content' attribute if found, otherwise None.
    """
    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the meta tag with the specified property
    meta_tag = soup.find("meta", attrs={"property": property_name})

    # If the meta tag is found, return its 'content' attribute
    if meta_tag and "content" in meta_tag.attrs:
        return meta_tag["content"]
    else:
        return ""


def extract_span_content(html_content: str, class_name: str) -> str | None:
    soup = BeautifulSoup(html_content, "html.parser")

    # Find the meta tag with the specified clas
    span_tag = soup.find("span", attrs={"class": class_name})
    # If the meta tag is found, return its 'content' attribute
    if span_tag is not None and span_tag.text:
        return span_tag.text
    else:
        return ""


async def scraper(anchor_url: str) -> pd.DataFrame:
    filter_chain = FilterChain(
        [
            # Only follow URLs with specific patterns
            # URLPatternFilter(patterns=["*guide*", "*tutorial*"]),
            # Only crawl specific domains
            DomainFilter(blocked_domains=settings.CRAWLER.BLOCKED_WEBSITES),
            # Only include specific content types
            ContentTypeFilter(allowed_types=["text/html"]),
        ]
    )
    # ignore_images =True
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=0 if settings.AIOD.TESTING else 3,  # configure crawl level as needed
            include_external=False,
            filter_chain=filter_chain,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False,
        scan_full_page=True,
        scroll_delay=0.5,
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(anchor_url, config=config)

        content_list = []
        url_list = []
        id_list = []
        modified_time_list = []

        api_content_list = []
        api_url_list = []
        api_id_list = []
        api_modified_time_list = []
        for result in results:  # Show first 3 results
            if result.url not in url_list:  # make sure no duplicates are stored
                if str(extract_meta_content(result.html, "x-dont-crawl")).lower() != "true":
                    # only index pages that should be crawled
                    if "api" in result.url:
                        last_modified = str(last_modified_time(result.html))
                        api_modified_time_list.append(last_modified)
                        api_content_list.append(result.markdown)
                        api_url_list.append(result.url)
                        api_id_list.append(str(uuid4()))
                    else:
                        last_modified = str(last_modified_time(result.html))
                        modified_time_list.append(last_modified)
                        content_list.append(result.markdown)
                        url_list.append(result.url)
                        id_list.append(str(uuid4()))

        data = pd.DataFrame(
            {
                "content": content_list,
                "url": url_list,
                "last_modified": modified_time_list,
                "id": id_list,
            }
        )
        api_data = pd.DataFrame(
            {
                "content": api_content_list,
                "url": api_url_list,
                "last_modified": api_modified_time_list,
                "id": api_id_list,
            }
        )

        logging.info(f"Crawled {len(results)} pages in total from {anchor_url}")
        # logging.info(f"\t{len(url_list)} pages are from AIoD websites")
        # logging.info(f"\t{len(api_url_list)} pages are from AIoD APIs")

        # TODO since the crawler doesn't work properly now, we will simply merge the two dataframes
        # and return them as one
        return pd.concat([data, api_data])


def create_content_collection(client: MilvusClient, collection_name: str) -> None:
    schema = client.create_schema(
        enable_dynamic_field=True,
    )

    # content, url, hash, embeddings
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(
        field_name="content", datatype=DataType.VARCHAR, max_length=65535
    )  # max_length = between 1 and 65535 bytes
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="last_modified", datatype=DataType.VARCHAR, max_length=256)

    schema.verify()

    vector_index_kwargs = {
        "index_type": "FLAT" if settings.MILVUS.USE_LITE else "HNSW_SQ",
        "metric_type": "COSINE",
        "params": {} if settings.MILVUS.USE_LITE else {"sq_type": "SQ8"},
    }
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", **vector_index_kwargs)
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )


def update_content_collection(
    model: AiModel, client: MilvusClient, collection_name: str, content: pd.DataFrame
) -> None:
    # not useful as long as there is no consistent way to figure out if something on the webpage has changed
    formatted_urls = ", ".join(f"'{url}'" for url in content["url"].tolist())
    where_clause = f"url in [{formatted_urls}]"

    entries_to_examine = client.query(
        collection_name=collection_name,
        filter=where_clause,
        output_fields=["id", "url", "last_modified"],
    )
    entries_to_delete = []
    entries_to_add = []

    for entry in entries_to_examine:
        subset = content[content["url"] == entry["url"]]  # the url exists in the db already
        if not subset.empty:
            if (
                entry["last_modified"] not in subset["last_modified"].tolist()
            ):  # the content has changed
                entries_to_delete.append(entry["id"])
                entries_to_add += subset["id"].tolist()

    # find pages that have no representation in the milvus db and add them to the add list
    urls_to_examine = [x["url"] for x in entries_to_examine]
    content_url_list = content["url"].tolist()
    for new_entry in content_url_list:
        if new_entry not in urls_to_examine:
            entries_to_add += content[content["url"] == new_entry]["id"].tolist()

    # find pages that are not reachable on the web anymore and add their ids to the delete list
    for index, element in enumerate(urls_to_examine):
        if element not in content_url_list:
            entries_to_delete.append(entries_to_examine[index]["id"])

    # delete entries
    logging.info(f"Deleting {len(entries_to_delete)} entries from {collection_name}")
    if entries_to_delete:
        client.delete(collection_name=collection_name, ids=entries_to_delete)

    # insert entries
    logging.info(f"Adding {len(entries_to_add)} entries to {collection_name}")
    if entries_to_add:
        content_to_add = content[content["id"].isin(entries_to_add)]
        new_data = prepare_data(model, content_to_add)
        client.insert(collection_name=collection_name, data=new_data)


def prepare_data(
    model: AiModel,
    website_content: pd.DataFrame,
) -> list[dict]:
    result_df = pd.DataFrame()
    result_vectors = []
    result_content: list[str] = []
    result_url = []
    result_last_modified = []

    for _, row in website_content.iterrows():
        content = row["content"] or ""
        url = row["url"] or ""
        last_modified = row["last_modified"] or ""

        if not content:
            continue
        if model.text_splitter is None:
            # TODO this is somewhat brittle, we need to change this later on...
            raise ValueError(
                "Text splitter is not set. You need to change the model to use chunking -> STORE_CHUNKS"
            )

        chunks = model.text_splitter(content)
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 0]
        if len(chunks) == 0:
            continue

        embedd_chunks = [model.compute_query_embeddings(chunk)[0] for chunk in chunks]
        result_vectors += embedd_chunks
        result_content += chunks
        result_url += [url for x in chunks]
        result_last_modified += [str(last_modified) for _ in chunks]

    result_df["vector"] = result_vectors
    result_df["content"] = result_content
    result_df["url"] = result_url
    result_df["last_modified"] = result_last_modified

    return result_df.to_dict(orient="records")
