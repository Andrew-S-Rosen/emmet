from typing import Dict

from maggma.builders.dag_map_builder import MapBuilder
from maggma.core import Store
from pymatgen.core import Structure

from emmet.core.oxidation_states import OxidationStateDoc
from emmet.core.utils import jsanitize


class OxidationStatesBuilder(MapBuilder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query=None,
        chunk_size: int = 300,
        allow_bson=True,
        **kwargs,
    ):
        """
        Creates Oxidation State documents from materials

        Args:
            materials: Store of materials docs
            oxidation_states: Store to update with oxidation state document
            query : query on materials to limit search
        """
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.materials = source_keys["materials"]
        self.oxidation_states = target_keys["oxidation_states"]
        self.query = query or {}
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.kwargs = kwargs

        # Enforce that we key on material_id
        self.materials.key = "material_id"
        self.oxidation_states.key = "material_id"
        super().__init__(
            source=self.materials,
            target=self.oxidation_states,
            chunk_size=chunk_size,
            projection=["structure", "deprecated"],
            query=query,
            **kwargs,
        )

    def unary_function(self, item):
        structure = Structure.from_dict(item["structure"])
        mpid = item["material_id"]
        deprecated = item["deprecated"]

        oxi_doc = OxidationStateDoc.from_structure(
            structure=structure, material_id=mpid, deprecated=deprecated
        )
        doc = jsanitize(oxi_doc.model_dump(), allow_bson=self.allow_bson)

        return doc
