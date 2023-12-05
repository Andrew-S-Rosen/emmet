from typing import Dict, Optional

from maggma.builders.dag_map_builder import MapBuilder
from maggma.core import Store
from pymatgen.core.structure import Structure

from emmet.core.robocrys import RobocrystallogapherDoc
from emmet.core.utils import jsanitize


class RobocrystallographerBuilder(MapBuilder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query: Optional[Dict] = None,
        chunk_size: int = 300,
        allow_bson=True,
        **kwargs
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.oxidation_states = source_keys["oxidation_states"]
        self.robocrys = target_keys["robocrys"]
        self.query = query or {}
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.kwargs = kwargs

        self.robocrys.key = "material_id"
        self.oxidation_states.key = "material_id"

        super().__init__(
            source=self.oxidation_states,
            target=self.robocrys,
            query=self.query,
            chunk_size=self.chunk_size,
            projection=["material_id", "structure", "deprecated"],
            **kwargs
        )

    def unary_function(self, item):
        structure = Structure.from_dict(item["structure"])
        mpid = item["material_id"]
        deprecated = item["deprecated"]

        doc = RobocrystallogapherDoc.from_structure(
            structure=structure,
            material_id=mpid,
            deprecated=deprecated,
            fields=[],
        )

        return jsanitize(doc.model_dump(), allow_bson=self.allow_bson)
