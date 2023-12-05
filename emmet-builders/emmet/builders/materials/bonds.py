from typing import Dict

from maggma.builders.dag_map_builder import MapBuilder
from maggma.core import Store
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from emmet.core.bonds import BondingDoc
from emmet.core.utils import jsanitize


class BondingBuilder(MapBuilder):
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
        Creates Bonding documents from structures, ideally with
        oxidation states already annotated but will also work from any
        collection with structure and mp-id.

        Args:
            oxidation_states: Store of oxidation states
            bonding: Store to update with bonding documents
            query : query on materials to limit search
        """
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.oxidation_states = source_keys["oxidation_states"]
        self.bonding = target_keys["bonding"]
        self.query = query or {}
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.kwargs = kwargs

        # Enforce that we key on material_id
        self.oxidation_states.key = "material_id"
        self.bonding.key = "material_id"
        super().__init__(
            source=self.oxidation_states,
            target=self.bonding,
            chunk_size=self.chunk_size,
            projection=["structure", "deprecated"],
            **kwargs,
        )

    def unary_function(self, item):
        structure = Structure.from_dict(item["structure"])
        mpid = item["material_id"]
        deprecated = item["deprecated"]

        # temporarily convert to conventional structure inside this builder,
        # in future do structure setting operations in a separate builder
        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

        bonding_doc = BondingDoc.from_structure(
            structure=structure, material_id=mpid, deprecated=deprecated
        )
        doc = jsanitize(bonding_doc.model_dump(), allow_bson=self.allow_bson)

        return doc
