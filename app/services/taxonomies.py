import logging
from collections.abc import Sequence
from typing import cast

from app.config import AIOD_TAXONOMIES
from app.services.aiod import get_aiod_taxonomy


class TaxonomyService:
    def __init__(self, taxonomy_names: Sequence[str]):
        self.taxonomy_names = list(taxonomy_names)
        self.taxonomy_terms: dict[str, list[str]] = {}

    def refresh(self) -> None:
        for taxonomy in self.taxonomy_names:
            try:
                self.taxonomy_terms[taxonomy] = cast(list[str], get_aiod_taxonomy(taxonomy))
            except Exception as exc:
                logging.warning("Failed to load AIoD taxonomy '%s': %s", taxonomy, exc)

    def get_terms(self, taxonomy: str) -> list[str]:
        return self.taxonomy_terms.get(taxonomy, [])

    def all_terms(self) -> dict[str, list[str]]:
        return self.taxonomy_terms


taxonomy_service = TaxonomyService(AIOD_TAXONOMIES)


def initialize_taxonomies() -> None:
    logging.info("Loading AIoD taxonomies from remote service")
    taxonomy_service.refresh()

    import json

    with open("all_tax.json", "w") as f:
        json.dump(taxonomy_service.all_terms(), f)

    print()
