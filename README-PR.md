*This is a belated Christmas present by yours truly :)*

# Main changes

This is a rather large PR containing various modifications to the Semantic Search service, namely:
- Incorporation of <ins>Metadata filtering</ins> logic
    - Extraction of metadata from assets (for now we <ins>only support extracting data from HuggingFace datasets</ins>)
    - User query parsing and retrieving filters that could be used to narrow down a list of retrieved assets
        - <ins>Automatic user query parsing utilizing an LLM</ins> (LLM inference is run via Ollama service)
        - <ins>Manually user-defined filters</ins> (filters are defined in the POST request, so an LLM inference is skipped)
- Small changes to the API endpoints
    - Expanded the endpoints with <ins>pagination</ins> support and an option to <ins>retrieve entire AIoD assets</ins> directly (not only asset IDs)
    - Added blocking endpoints that wait till the results to a user query are retrieved
    - Endpoints also return `num_hits` argument representing the number of assets in the entire Milvus DB that match a specific set of user criteria
- Simplified deployment process
    - Instead of having a ton of docker-compose files that differ from each other in a way they combine services we utilize a <ins>Jinja2 templates to build a specific docker-compose</ins> with services we wish to include on the fly
- Augmented garbage collector logic
    - Recurring (on a monthly basis) Milvus DB garbage collector logic -><ins>delete embeddings associated with old, outdated assets</ins> (*this functionality has already been implemented*)
    - Recurring (on a daily basis) TinyDB garbage collector logic -> <ins>delete expired queries</ins>
        - <ins>User queries have an expiration date</ins> -> 1 hour after being resolved

## Metadata filtering

Metadata filtering functionality consists of two main processes that need to be taken care of:
a) Extracting and storing of metadata associated with individual assets
b) Extracting of filters/conditions found in the user query /user query parsing/

### Extracting of metadata from assets

Currently this process applicability is restricted to only a specific subset of assets we can <ins>manually extract metadata</ins> from, from <ins>Huggingface datasets</ins> to be exact. Due to this extraction process being very time demanding, we have opted to perform the metadata extraction manually without any use of LLMs for now. 

Since the datasets as an asset type is the most prevalent asset in the AIoD platform, we have decided to apply metadata filtering on said asset, whilst choosing solely Huggingface datasets that share a common metadata structure as it can be used to retrieve a handful of metadata fields to be stored in Milvus database.

### Extracting of filters from user queries

Since we wish to minimize the costs and increase computational/time efficiency pertaining to serving of an LLM, we have opted to go for a rather small LLM (Llama 3.1 8B). Due to its size, performing more advanced tasks or multiple tasks at once can often lead incorrect results. To mitigate this to a certain degree, we have divided the <ins>user query parsing process into 2-3 LLM steps</ins> that further dissect and scrutinize a user query on different levels of granularity. The LLM steps are:
- STEP 1: Extraction of natural language conditions from query (extraction of spans representing a condtions/filters each associated with a specific metadata field)
- STEP 2: Analysis and transformation of each natural language condition (a span from user query) to a structure representing the condition performed separately
- [Optional STEP 3]: Further validation of transformed value against a list of permitted values for a particular metadata field

Unlike the former process, the extraction of metadata from assets, that can be performed manually without the use of an LLM to only a limited degree, <ins>the manual approach of defining filters explicitly</ins> is a full-fledged alternative to an automated user query parsing by an LLM. To this end, a user can define the filters himself, which can eliminate possibility for an LLM to misinterpret or omit some conditions user wanted to apply in the first place.

# Brief description of noteworthy files 

Here I briefly describe the contents of some files whose purpose may not be clear or they're simply too big to be easily understood
- `api/models/filter.py`: Represents structured filters either extracted using an LLM or defined in the body of an HTTP request
    - `validate_filter_or_raise` function: Checks the type/value of values associated with each expression tied to a particular metadata field based on its defined restrictions the value should adhere to. This function in particular important to check the validity of manually user-defined filters
- `api/schemas/asset_metadata/base.py`: Contains various annotation and schema operations that can be used for validation or creation of dynamic types on runtime.
- `api/schemas/asset_metadata/dataset_metadata.py`: Contains Pydantic model representing fields we wish to extract. This Pydantic model is then passed to an LLM functioning as an output schema an LLM is supposed to conform to. Each field may also have associated field validators with it that can further restrict the values permitted by said field.
- `api/services/inference/text_operations.py`: This file has been extended with additional functions that perform the manual (no LLM) extraction of metadata from Huggingface datasets
- `api/services/inference/llm_query_parsing.py`: This file contains all the logic regarding user query parsing using an LLM. For now I’d suggest you not to delve too much into this particular file as it is a quite a mess now, but functional nevertheless. In any case, this file contains:
    - Pydantic classes used as output schemas for individual LLM steps
    - `Llama_ManualFunctionCalling`: Class representing use of function calling performed through prompt engineering only rather than relying on Ollama/Langchain tool calls
    - `UserQueryParsingStages`: Class containing functions for performing individual LLM steps. Each LLM step is a variation, an instance of `Llama_ManualFunctionCalling` class
    - `UserQueryParsing`: Class encapsulating all the LLM steps to be performed for user query parsing purposes. This wrapper class is then used in other parts of our application to perform the user query parsing functionality with an LLM.

# Potential problems

## Asset pagination and the total number of assets fulfilling a set of criteria

Implementation of these two features for Milvus embeddings is rather trivial, but there are two main obstacles that can potentially be surmounted but at the great cost. For now we have chosen to stick to implementing of only an <ins>approximate pagination and retrieval of the total number of assets tied to a particular query</ins>.

*Example 1: I wish to retrieve a page (offset=5000, limit=100) associated with a particular user query*

**Obstacle 1:** The underlying AIoD assets of embeddings are constantly changing
- Since AIoD assets are constantly changing, we cannot guarantee that the first 5000 assets (*example 1*) are valid and up-to-date. Not to mention that there could be an <ins>overlap of assets in between pages</ins> if we were to retrieve the page with offset of 5000 before moving to the page with offset of 4900 (if there were any outdated assets that is...)
- This also applies to retrieving a total number of assets that comply with user filters -> We don't know the exact number of assets that are valid in the time of the user request.
- **Solution**: Always check all the preceding assets up to the page we're interested in (or even check all the assets in the Milvus DB in the case of determining the total number of assets compliant with user queries with no filters). In our This is obviously a ludicrously expensive and stupid...

**Obstacle 2:** One AIoD asset may be divided into N separate chunks each of them represented by its own Milvus embedding
- So far, we have assumed that each AIoD asset corresponds to one and only embedding. However this is not the case
- There's actually a hidden layer of abstraction that we have never wished to delve into: <ins>The Milvus operations are performed on embeddings rather than on assets, but user defines pagination parameters in the number of assets</ins>. We conceal this fact by prompting for more embeddings than the number of assets required and then we retrieve only embeddings associated with distinct assets
- This further exacerbates the precision of the page offset as the Milvus offset itself is applied on the embeddings rather than on assets
- This leads to <ins>overlaps of asset between pages</ins>
- **Solution**: Yet again the solution is to check all the assets preceding our page. For instance, to truly get an asset offset of 5000 (*example 1*), we would need to retrieve the top 5100 embeddings, actually it would be more than that to account for the assets that are tied to multiple embeddings. Then we would retrieve 5000 actual distinct and still existing assets from the embeddings and return only a specific window user requested. This approach is straight forward yet very expensive.

## Preserving AIoD assets locally (for a limited time)

In order for us to be able to return the entire assets instead of their IDs only, we need to temporarily store them to our database (*this would not necessarily be the case with blocking endpoints but that is a discussion for another day*). The problem arises with the fact that 1) not only do <ins>we store asset related data that is not immediately deleted once it gets removed on original platform as well (Huggingface, ...)</ins>, but we may also risk <ins>potentially serving outdated AIoD assets to our users</ins>.

*Query expiration date: We have introduced a concept of expiration date of queries that states up to what time a specific query is accessible to the user*
- *Still, even if a query is not expired yet, we DO NOT GUARANTEE results validity as the changes to AIoD assets can be done whenever... I suppose we could potentially check the results validity each time a user requests the results, but still that would not make impervious to being invalidated in the meantime...*
- *For now we have set the expiration date to be one hour after the user query is resolved*

**Problems:**
- To alleviate the 1) first problem, we have introduced an <ins>additional daily job that gets rid of all the expired queries</ins>. This job ensures the TinyDB size not to get out of hand whilst also <ins>removing asset metadata that might pertain to old or deleted AIoD assets</ins>
- The 2) second problem cannot be easily addressed I suppose. By further <ins>reducing the expiration duration</ins> we could avoid some situations when an AIoD asset is no longer up-to-date, but <ins>this problem cannot be nullified completely</ins>. Actually, even if we were to provide the asset IDs only, said IDs would still be susceptible to being out-of-date. The silver lining that makes the serving of invalid asset IDs somewhat acceptable compared to serving of invalid AIoD assets is the fact that the user needs to additionally prompt the AIoD platform to retrieve corresponding assets and thus he's subsequently informed of the invalidity of our results, which would not be the case with our response that returns entire assets.

# Additional planned features
- <ins>Extend the list of assets we can apply metadata filtering on</ins>. This entails: 
    - creating a separate class, a Pydantic model, defining metadata fields to extract from a specific asset
    - <ins>utilizing an LLM to extract said fields automatically</ins> instead of relying on common fragile metadata structure that could be potentially changed in the future
- Make DB updating job more time efficient -> the DB updating job should contain multiple tasks run in parallel to speed up the whole process. The job should contain the following tasks associated with: 
    - 1) fetching AIoD assets 
    - 2) computing embeddings
    - 3) LLM metadata extraction
- Distinguish between manual and automatized filter extraction in service config/settings => so that in the case we don't have an access to an LLM, we could still potentially perform metadata filtering manually (if there's a support for manually extracted asset metadata that is)
    - *Currently we either support all the forms of metadata filtering (an LLM is necessary) or none*
- Add DEBUG logs associated with time spent performing various processes (fetching of AIoD assets, computing/storing of embeddings, invoking an LLM, ...) so that we can determine what the weak link is once the processes become unbearably slow
- Perform changes to the non-blocking endpoints to appease AR (issue #3)

# TODO for deploying on AIoD

- [ ] We should repopulate Milvus and TinyDB databases from scratch, otherwise some unexpected behavior may be encountered as I have not implemented any counter measures for dealing with old, not up-to-date schema in databases, etc, ...
    - [x] I have precomputed embeddings and metadata on our cluster
- [x] I have yet to test `api/scripts/populate_milvus.py` script