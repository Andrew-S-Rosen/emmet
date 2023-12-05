from typing import Dict

from maggma.builders.dag_map_builder import MapBuilder
from maggma.core import Store
from pymatgen.core.structure import Structure

from emmet.core.chemenv import ChemEnvDoc
from emmet.core.utils import jsanitize


class ChemEnvBuilder(MapBuilder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query=None,
        chunk_size: int = 300,
        allow_bson=True,
        **kwargs
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.oxidation_states = source_keys["oxidation_states"]
        self.chemenv = target_keys["chemenv"]
        self.query = query or {}
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.kwargs = kwargs

        self.chemenv.key = "material_id"
        self.oxidation_states.key = "material_id"

        super().__init__(
            source=self.oxidation_states,
            target=self.chemenv,
            chunk_size=self.chunk_size,
            query=self.query,
            projection=["material_id", "structure", "deprecated"],
            **kwargs
        )

    def unary_function(self, item):
        structure = Structure.from_dict(item["structure"])
        mpid = item["material_id"]
        deprecated = item["deprecated"]

        doc = ChemEnvDoc.from_structure(
            structure=structure,
            material_id=mpid,
            deprecated=deprecated,
        )

        return jsanitize(doc.model_dump(), allow_bson=self.allow_bson)
