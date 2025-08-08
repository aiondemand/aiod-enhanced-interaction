from typing import Annotated
from pydantic import Field


AssetId = Annotated[
    str,
    Field(
        description="AIoD Asset ID. Must be a 3 or 4 letter code followed by a 24 character random case-sensitive alphanumeric sequence",
        pattern=r"^[a-z]{3,4}_[a-zA-Z0-9]{24}$",
    ),
]
