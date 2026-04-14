from dataclasses import dataclass
from typing import Any

@dataclass
class AdapterNote:
    message: str = (
        "Placeholder adapter layer. In a full integration, this module should "
        "convert SHEHyperstructure objects into the weighted simplicial format "
        "used by she_geofield."
    )

def from_she_hyperstructure(obj: Any):
    raise NotImplementedError(
        "Wire this to SHEHyperstructure once working against the live public repo."
    )
