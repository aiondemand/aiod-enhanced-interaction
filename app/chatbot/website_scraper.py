import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, ContentTypeFilter, DomainFilter  # URLPatternFilter
from pymilvus import MilvusClient, DataType, utility
from uuid import uuid4
import pandas as pd
import torch
from dotenv import load_dotenv
import os
from app.services.inference.architecture import Basic_EmbeddingModel, SentenceTransformerToHF
from bs4 import BeautifulSoup

load_dotenv("../../.env.app")

milvus_uri = os.getenv("MILVUS__URI")
milvus_token = os.getenv('MILVUS__USER')+":"+os.getenv('MILVUS__PASS')
embedding_llm = os.getenv('MODEL_LOADPATH')
use_gpu = os.getenv('USE_GPU')
window_size = int(os.getenv("AIOD__WINDOW_SIZE"))
window_overlap = float(os.getenv("AIOD__WINDOW_OVERLAP"))

load_dotenv(".env.chatbot")
web_collection_name = os.getenv("WEBSITE_COLLECTION")
api_collection_name = os.getenv("API_COLLECTION")

c = MilvusClient(
        uri=milvus_uri,
        token=milvus_token
    )

relevant_pages = ["https://www.aiodp.ai/", "aiod.eu"]
filter_chain= FilterChain([
    # Only follow URLs with specific patterns
    # URLPatternFilter(patterns=["*guide*", "*tutorial*"]),

    # Only crawl specific domains
    DomainFilter(
        #allowed_domains=["docs.example.com"],
        blocked_domains=["https://auth.aiod.eu"]
    ),

    # Only include specific content types
    ContentTypeFilter(allowed_types=["text/html"])
])


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
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find the meta tag with the specified property
    meta_tag = soup.find('meta', attrs={'property': property_name})

    # If the meta tag is found, return its 'content' attribute
    if meta_tag and 'content' in meta_tag.attrs:
        return meta_tag['content']
    else:
        return ""


async def scraper(anchor_url):
    #ignore_images =True

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,  # configure crawl level as needed
            include_external=False,
            filter_chain=filter_chain
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False,
        scan_full_page=True,
        scroll_delay=0.5,
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(anchor_url, config=config)

        print(f"Crawled {len(results)} pages in total")

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
                if "api.aiod.eu" in result.url:
                    last_modified = str(extract_meta_content(result.html, "article:modified_time"))
                    api_modified_time_list.append(last_modified)
                    api_content_list.append(result.markdown)
                    api_url_list.append(result.url)
                    api_id_list.append(str(uuid4()))
                else:
                    last_modified = str(extract_meta_content(result.html, "article:modified_time"))
                    modified_time_list.append(last_modified)
                    content_list.append(result.markdown)
                    url_list.append(result.url)
                    id_list.append(str(uuid4()))
        print("url_list", len(url_list), url_list)
        print("api_url_list", len(api_url_list), api_url_list)
        data = {
                "content": content_list,
                'url': url_list,
                'last_modified': modified_time_list,
                'id': id_list
                }
        df = pd.DataFrame(data)
        df.to_csv("test_8.csv", sep=",", index=False)
        # df.to_json()

        api_data = {
            "content": api_content_list,
            'url': api_url_list,
            'last_modified': api_modified_time_list,
            'id': api_id_list
        }
        api_df = pd.DataFrame(api_data)
        api_df.to_csv("api_test_8.csv", sep=",", index=False)
        # api_df.to_json()
        return df, api_df


def create_content_collection(collection_name: str, client: MilvusClient):
    schema = client.create_schema(
        enable_dynamic_field=True,
    )

    # content, url, hash, embeddings
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # max_length = between 1 and 65535 bytes
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name='last_modified', datatype=DataType.VARCHAR, max_length=256)
    # schema.add_field(field_name="hash", datatype=DataType.INT64)

    schema.verify()

    index_params = client.prepare_index_params()

    vector_index_kwargs = {
        "index_type": "HNSW_SQ",
        "metric_type": "COSINE",
        "params": {"sq_type": "SQ8"},
    }

    index_params.add_index(field_name="vector", **vector_index_kwargs)

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    return True


def update_content_collection(collection_name: str, client: MilvusClient, content: pd.DataFrame):
    # not useful as long as there is no consistent way to figure out if something on the webpage has changed
    formatted_urls = ", ".join(f"'{url}'" for url in content['url'].tolist())
    where_clause = f"url in [{formatted_urls}]"

    entries_to_examine = client.query(
        collection_name=collection_name,
        filter=where_clause,
        output_fields=["id", "url", "last_modified"]
    )
    print(entries_to_examine[0])
    entries_to_delete = []
    entries_to_add = []

    for entry in entries_to_examine:
        subset = content[content['url'] == entry['url']]  # the url exists in the db already
        if not subset.empty:
            print('last_modified', entry['last_modified'], subset['last_modified'].tolist())
            if entry['last_modified'] not in subset['last_modified'].tolist():  # the content has changed
                entries_to_delete.append(entry['id'])
                entries_to_add += subset['id'].tolist()

    urls_to_examine = [x['url'] for x in entries_to_examine]
    for new_entry in content['url'].tolist():
        if new_entry not in urls_to_examine:
            entries_to_add += content[content['url'] == new_entry]['id'].tolist()

    # delete entries
    print("del", len(entries_to_delete), entries_to_delete)
    if entries_to_delete:
        res = client.delete(collection_name=collection_name, ids=entries_to_delete)
        print(res)
        client.flush(collection_name)

    # insert entries
    print("add", len(entries_to_add), entries_to_add)
    if entries_to_add:
        content_to_add = content[content['id'].isin(entries_to_add)]
        new_data = prepare_data(content_to_add)
        res = client.insert(collection_name=collection_name, data=new_data)
        print(res)
        client.flush(collection_name)
    return


@torch.no_grad
def embed_content(content_list: list[str]):
    # compute_asset_embeddings -> 61:model.py
    # compute_query_embeddings -> 69:model.py
    transformer = SentenceTransformerToHF(embedding_llm, trust_remote_code=True)
    if torch.cuda.is_available() and use_gpu:
        # print("use cuda")
        transformer.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        # print("use cpu")
    model = Basic_EmbeddingModel(
        transformer,
        transformer.tokenizer,
        pooling="none",
        document_max_length=4096,
        dev=device,
    )
    embedded_content = []
    for content in content_list:
        embedded_content.append(model.forward(content)[0].cpu())  # 31:architecture.py -> hopefully does the embeddings

    return embedded_content


class SlidingWindowChunking:
    def __init__(self, window_size=window_size, step=window_size*window_overlap):
        self.window_size = window_size
        self.step = int(step)

    def chunk(self, text):
        words = text.split()
        chunks = []
        if len(words) < self.window_size:
            return [text]
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(' '.join(words[i:i + self.window_size]))
        return chunks


def chunking(content: str):
    chunker = SlidingWindowChunking()
    chunks = chunker.chunk(content)
    # print(len(chunks))
    return chunks


def prepare_data(website_content: pd.DataFrame):
    print("prepare_data")
    result_df = pd.DataFrame()
    result_vectors = []
    result_content = []
    result_url = []
    result_last_modified = []
    for index, row in website_content.iterrows():
        content = website_content["content"].iloc[index]
        url = website_content['url'].iloc[index]
        print("prep", url)
        last_modified = website_content['last_modified'].iloc[index]
        chunked = chunking(content)
        embedd_chunks = embed_content(chunked)
        result_vectors += embedd_chunks
        result_content += chunked
        result_url += [url for x in chunked]
        result_last_modified += [str(last_modified) for x in chunked]

    result_df["vector"] = result_vectors
    result_df['content'] = result_content
    result_df['url'] = result_url
    result_df['last_modified'] = result_last_modified

    return result_df.to_dict(orient='records')


def populate_webcontent_collection(collection_name: str, client: MilvusClient):
    # website_content = scraper()
    website_content = pd.read_csv("test_8.csv")
    client.drop_collection(collection_name)
    # print(website_content.head())
    # client.drop_collection(collection_name=collection_name)
    if client.has_collection(collection_name):
        return update_content_collection(collection_name, client, website_content)

    else:
        print("create new collection")
        # create a new collection to store the web content
        create_content_collection(collection_name, client)
        print("collection created")
        # embed the markdown of the crawled content
        website_data = prepare_data(website_content)
        print("data prepared")
        # insert the data into the newly created collection
        res = client.insert(collection_name=collection_name, data=website_data)
        # write the collection into persistent storage
        client.flush(collection_name=collection_name)
        return res


def populate_api_collection(api_collection_name: str, client: MilvusClient):
    # website_content = scraper()
    website_content = pd.read_csv("api_test_8.csv")
    client.drop_collection(api_collection_name)
    # print(website_content.head())
    # client.drop_collection(collection_name=collection_name)
    if client.has_collection(api_collection_name):
        return update_content_collection(api_collection_name, client, website_content)

    else:
        print("create new collection")
        # create a new collection to store the web content
        create_content_collection(api_collection_name, client)
        print("collection created")
        # embed the markdown of the crawled content
        website_data = prepare_data(website_content)
        print("data prepared")
        # insert the data into the newly created collection
        res = client.insert(collection_name=api_collection_name, data=website_data)
        # write the collection into persistent storage
        client.flush(collection_name=api_collection_name)
        return res


def populate_collections():
    print("populate website:")
    populate_webcontent_collection(web_collection_name, c)
    print("populate api:")
    populate_api_collection(api_collection_name, c)


# asyncio.run(scraper("https://aiod.eu"))
populate_collections()

