import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, ContentTypeFilter  # URLPatternFilter, DomainFilter
from pymilvus import MilvusClient, DataType
from uuid import uuid4
import pandas as pd
import torch
from dotenv import load_dotenv
import os
from app.services.inference.architecture import Basic_EmbeddingModel, SentenceTransformerToHF


load_dotenv("../../.env.app")

milvus_uri = os.getenv("MILVUS__URI")
milvus_token = os.getenv('MILVUS__USER')+":"+os.getenv('MILVUS__PASS')
embedding_llm = os.getenv('MODEL_LOADPATH')
use_gpu = os.getenv('USE_GPU')

c = MilvusClient(
        uri=milvus_uri,
        token=milvus_token
    )


filter_chain = FilterChain([
    # Only follow URLs with specific patterns
    # URLPatternFilter(patterns=["*guide*", "*tutorial*"]),

    # Only crawl specific domains
    # DomainFilter(
        # allowed_domains=["docs.example.com"],
        # blocked_domains=["old.docs.example.com"]
    # ),

    # Only include specific content types
    ContentTypeFilter(allowed_types=["text/html"])
])


async def scraper():
    # TODO make sure each url is only scraped once
    #  not all pages are scraped correctly
    #  change hashes so that they are consistent
    #ignore_images =True

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=0,  # configure crawl level as needed
            include_external=False,
            # filter_chain=filter_chain
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun("https://aiod.eu", config=config)

        print(f"Crawled {len(results)} pages in total")

        content_list = []
        url_list = []
        hash_list = []
        id_list = []
        for result in results:  # Show first 3 results
            # print(f"URL: {result.url}")
            # print(f"Depth: {result.metadata.get('depth', 0)}")
            # relevant_info_list.append((result.markdown, result.url, hash(result.markdown)))
            print(hash(result.html))
            print(result.html)
            print(result.metadata)
            content_list.append(result.markdown)
            url_list.append(result.url)
            hash_list.append(hash(result.html))  # TODO hashing doesn't work -420105403813510547, -2706637908361790186
            id_list.append(str(uuid4()))
        print(url_list)
        data = {
            "content": content_list,
            'url': url_list,
            'hash': hash_list,
            'id': id_list
        }
        df = pd.DataFrame(data)
        df.to_csv("test_3.csv", sep=",", index=False)
        df.to_json()
        return df


def create_webcontent_collection(collection_name: str, client: MilvusClient):

    schema = client.create_schema(
        enable_dynamic_field=True,
    )

    # content, url, hash, embeddings
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=40000)  # TODO implement chunking? max_length = between 1 and 65535 bytes
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="hash", datatype=DataType.INT64)

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


def update_webcontent_collection(collection_name: str, client: MilvusClient, content: pd.DataFrame):
    """for index, row in content.iterrows():
        url_from_content = row['url']
        res = client.query(
            collection_name=collection_name,
            filter='url=={}'.format(url_from_content),
            output_fields=["id", "url", "hash"]
        )
        print(url_from_content, print(res))"""

    formatted_urls = ", ".join(f"'{url}'" for url in content['url'].tolist())
    where_clause = f"url in [{formatted_urls}]"

    entries_to_examine = client.query(
        collection_name=collection_name,
        filter=where_clause,
        output_fields=["id", "url", "hash"]
    )
    print(entries_to_examine[0])
    entries_to_delete = []
    entries_to_add = []

    for entry in entries_to_examine:
        subset = content[content['url'] == entry['url']]  # the url exists in the db already
        if not subset.empty:
            print('hash', entry['hash'], subset['hash'].tolist())  # TODO hashes of the the same page are not the same
            if entry['hash'] not in subset['hash'].tolist():  # the content has changed
                entries_to_delete.append(entry['id'])
                entries_to_add += subset['id'].tolist()

    urls_to_examine = [x['url'] for x in entries_to_examine]
    for new_entry in content['url'].tolist():
        if new_entry not in urls_to_examine:
            entries_to_add += content[content['url'] == new_entry]['id'].tolist()

    # todo delete entries
    print("del", len(entries_to_delete), entries_to_delete)
    if entries_to_delete:
        res = client.delete(collection_name=collection_name, ids=entries_to_delete)
        print(res)
        client.flush(collection_name)
    # new_entries = content[content['url'] == entry['url']]
    # print(res)
    # todo insert entries
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


def prepare_data(website_content: pd.DataFrame):
    embedded = embed_content(website_content["content"].tolist())
    website_content['vector'] = embedded
    # reduce the data to be loaded to the relevant information from the dataframe
    website_content_no_id = website_content[['vector', 'content', 'url', 'hash']]
    website_data = website_content_no_id.to_dict(orient='records')
    return website_data


def populate_webcontent_collection(collection_name: str, client: MilvusClient):
    # website_content = scraper()
    website_content = pd.read_csv("test_2.csv")
    # print(website_content.head())
    # client.drop_collection(collection_name=collection_name)
    if client.has_collection(collection_name):
        # print(client.list_collections())
        print("here")
        return update_webcontent_collection(collection_name, client, website_content)
    else:
        print("there")
        # create a new collection to store the web content
        create_webcontent_collection(collection_name, client)
        # embed the markdown of the crawled content
        website_data = prepare_data(website_content)
        # insert the data into the newly created collection
        res = client.insert(collection_name=collection_name, data=website_data)
        # write the collection into persistent storage
        client.flush(collection_name=collection_name)
        return res






asyncio.run(scraper())


# populate_webcontent_collection("test", c)


#collection, results = Collection.construct_from_dataframe  # -> there is an error in the pymilvus
            # implementation of this function
            # name=collection_name,
            # primary_field="id",
            # dataframe=website_content )

