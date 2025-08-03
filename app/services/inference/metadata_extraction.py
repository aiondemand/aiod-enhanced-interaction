import sys

sys.path.append(".")


from app.services.inference.text_operations import ConvertJsonToString


import logfire
import os
import json

from pydantic import ValidationError

from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.schemas.asset_metadata.dataset_metadata import (
    HuggingFaceDatasetMetadataTemplate,
)

# TODO make this class work with any asset type, in turn any output schema


class SimpleMetadataExtractor:
    SYSTEM_PROMPT = (
        "You are an expert metadata extractor. Given a description of any machine learning asset "
        "(a model card, dataset card, README, paper abstract, blog post, etc.), return a JSON "
        "object that adheres to the provided metadata schema. Only extract values that are "
        "explicitly present in the text - do not fabricate values that are not directly stated. "
        "It is normal and expected that many fields will be empty (null) since most assets will "
        "lack many of the fields. Never make up field values just to fill out the schema."
    )

    def __init__(self, enable_reasoning: bool = True) -> None:
        # Ollama model
        self.model = OpenAIModel(
            model_name="qwen3:4b", provider=OpenAIProvider(base_url="http://localhost:11434/v1")
        )
        self.enable_reasoning = enable_reasoning

        self.model_settings = ModelSettings(
            max_tokens=1_000,
        )
        self.agent = Agent(
            model=self.model,
            system_prompt=self.SYSTEM_PROMPT,
            output_type=HuggingFaceDatasetMetadataTemplate,
            output_retries=5,
            model_settings=self.model_settings,
        )

    def extract_metadata(self, document: str) -> HuggingFaceDatasetMetadataTemplate:
        if self.enable_reasoning is False:
            document = f"{document}\n\n/no_think"

        try:
            run_output = self.agent.run_sync(document, model_settings=self.model_settings)
            return run_output.output
        except ValidationError as e:
            # TODO: handle this better
            raise e


def main():
    os.environ["LOGFIRE_TOKEN"] = ""
    logfire.configure()
    logfire.instrument_pydantic_ai()

    # Sample document for testing
    with open("dataset.json", "r") as f:
        json_document = json.load(f)
    sample_document = ConvertJsonToString.extract_relevant_info(
        json_document, asset_type="datasets", stringify=False
    )

    for _ in range(5):
        try:
            simple_extractor = SimpleMetadataExtractor(enable_reasoning=False)
            result = simple_extractor.extract_metadata(sample_document)
            print(f"Extracted metadata: {result}")
        except Exception as e:
            print(f"Error with simple extractor: {e}")


if __name__ == "__main__":
    main()
