from collections import defaultdict
from datetime import datetime
from math import ceil
from typing import Dict, Iterable, List, Optional, Tuple

from maggma.core import Builder, Store
from maggma.utils import grouper
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.core.structure import Structure

from emmet.builders.settings import EmmetBuildSettings
from emmet.core.provenance import ProvenanceDoc, SNLDict
from emmet.core.utils import get_sg, jsanitize


class ProvenanceBuilder(Builder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        settings: Optional[EmmetBuildSettings] = None,
        chunk_size: int = 200,
        allow_bson=True,
        query: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Creates provenance from source SNLs and materials

        Args:
            materials: Store of materials docs to tag with SNLs
            provenance: Store to update with provenance data
            source_snls: List of locations to grab SNLs
            query : query on materials to limit search
        """
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.materials = source_keys["materials"]
        self.source_snls = [
            val for key, val in source_keys.items() if key != "materials"
        ]
        self.provenance = target_keys["provenance"]
        self.settings = EmmetBuildSettings.autoload(settings)
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.query = query or {}
        self.kwargs = kwargs

        self.materials.key = "material_id"
        self.provenance.key = "material_id"
        for s in self.source_snls:
            s.key = "snl_id"

        super().__init__(
            sources=[self.materials, *self.source_snls],
            targets=[self.provenance],
            chunk_size=self.chunk_size,
            **kwargs,
        )

    def ensure_indicies(self):
        self.materials.ensure_index("material_id", unique=True)
        self.materials.ensure_index("formula_pretty")

        self.provenance.ensure_index("material_id", unique=True)
        self.provenance.ensure_index("formula_pretty")

        for s in self.source_snls:
            s.ensure_index("snl_id")
            s.ensure_index("formula_pretty")

    def prechunk(self, number_splits: int) -> Iterable[Dict]:  # pragma: no cover
        self.ensure_indicies()

        # Find all formulas for materials that have been updated since this
        # builder was last ran
        q = self.query
        updated_materials = self.provenance.newer_in(
            self.materials, criteria=q, exhaustive=True
        )
        forms_to_update = set(
            self.materials.distinct(
                "formula_pretty", {"material_id": {"$in": updated_materials}}
            )
        )

        # Find all new SNL formulas since the builder was last run
        for source in self.source_snls:
            new_snls = self.provenance.newer_in(source)
            forms_to_update |= set(
                source.distinct("formula_pretty", {source.key: {"$in": new_snls}})
            )

        # Now reduce to the set of formulas we actually have
        forms_avail = set(self.materials.distinct("formula_pretty", self.query))
        forms_to_update = forms_to_update & forms_avail

        mat_ids = set(
            self.materials.distinct(
                "material_id", {"formula_pretty": {"$in": list(forms_to_update)}}
            )
        ) & set(updated_materials)

        N = ceil(len(mat_ids) / number_splits)

        self.logger.info(
            f"Found {len(mat_ids)} new/updated systems to distribute to workers "
            f"in {N} chunks."
        )

        for chunk in grouper(mat_ids, N):
            yield {"query": {"material_id": {"$in": chunk}}}

    def get_items(self) -> Tuple[List[Dict], List[Dict]]:  # type: ignore
        """
        Gets all materials to assocaite with SNLs
        Returns:
            generator of materials and SNLs that could match
        """
        self.logger.info("Provenance Builder Started")

        self.logger.info("Setting indexes")
        self.ensure_indicies()

        # Find all formulas for materials that have been updated since this
        # builder was last ran
        q = self.query
        updated_materials = self.provenance.newer_in(
            self.materials, criteria=q, exhaustive=True
        )
        forms_to_update = set(
            self.materials.distinct(
                "formula_pretty", {"material_id": {"$in": updated_materials}}
            )
        )

        # Find all new SNL formulas since the builder was last run
        for source in self.source_snls:
            new_snls = self.provenance.newer_in(source)
            forms_to_update |= set(
                source.distinct("formula_pretty", {source.key: {"$in": new_snls}})
            )

        # Now reduce to the set of formulas we actually have
        forms_avail = set(self.materials.distinct("formula_pretty", self.query))
        forms_to_update = forms_to_update & forms_avail

        mat_ids = set(
            self.materials.distinct(
                "material_id", {"formula_pretty": {"$in": list(forms_to_update)}}
            )
        ) & set(updated_materials)

        self.total = len(mat_ids)

        self.logger.info(f"Found {self.total} new/updated systems to process")

        return [
            list(mat_ids)[i : i + self.chunk_size]
            for i in range(0, len(mat_ids), self.chunk_size)
        ]

    def get_processed_docs(self, mats):
        self.materials.connect()

        for store in self.source_snls:
            store.connect()

        all_docs = []

        for mat_id in mats:
            mat = self.materials.query_one(
                properties=[
                    "material_id",
                    "last_updated",
                    "structure",
                    "initial_structures",
                    "formula_pretty",
                    "deprecated",
                ],
                criteria={"material_id": mat_id},
            )

            snls = []  # type: list
            for source in self.source_snls:
                snls.extend(
                    source.query(criteria={"formula_pretty": mat["formula_pretty"]})
                )

            snl_groups = defaultdict(list)
            for snl in snls:
                struc = Structure.from_dict(snl)
                snl_sg = get_sg(struc)
                struc.snl = SNLDict(**snl)
                snl_groups[snl_sg].append(struc)

            mat_sg = get_sg(Structure.from_dict(mat["structure"]))

            snl_structs = snl_groups[mat_sg]

            self.logger.debug(f"Found {len(snl_structs)} potential snls for {mat_id}")

            all_docs.append((mat, snl_structs))

        self.materials.close()

        for store in self.source_snls:
            store.close()

        return all_docs

    def process_item(self, items) -> Dict:
        """
        Matches SNLS and Materials
        Args:
            items (list[tuple]): a list of tuples of materials and snls
        Returns:
            list(dict): a list of collected snls with material ids
        """
        docs = []
        for item in items:
            if not item:
                continue

            mat, snl_structs = item
            formula_pretty = mat["formula_pretty"]
            snl_doc = None
            self.logger.debug(f"Finding Provenance {formula_pretty}")

            # Match up SNLS with materials

            matched_snls = self.match(snl_structs, mat)

            if len(matched_snls) > 0:
                doc = ProvenanceDoc.from_SNLs(
                    material_id=mat["material_id"],
                    structure=Structure.from_dict(mat["structure"]),
                    snls=matched_snls,
                    deprecated=mat["deprecated"],
                )
            else:
                doc = ProvenanceDoc(
                    material_id=mat["material_id"],
                    structure=Structure.from_dict(mat["structure"]),
                    deprecated=mat["deprecated"],
                    created_at=datetime.utcnow(),
                )

            doc.authors.append(self.settings.DEFAULT_AUTHOR)
            doc.history.append(self.settings.DEFAULT_HISTORY)
            doc.references.append(self.settings.DEFAULT_REFERENCE)

            snl_doc = jsanitize(
                doc.dict(exclude_none=False), allow_bson=self.allow_bson
            )

            docs.append(snl_doc)

        return docs

    def match(self, snl_structs, mat):
        """
        Finds a material doc that matches with the given snl
        Args:
            snl_structs ([dict]): the snls struct list
            mat (dict): a materials doc
        Returns:
            generator of materials doc keys
        """

        m_strucs = [Structure.from_dict(mat["structure"])] + [
            Structure.from_dict(init_struc) for init_struc in mat["initial_structures"]
        ]

        sm = StructureMatcher(
            ltol=self.settings.LTOL,
            stol=self.settings.STOL,
            angle_tol=self.settings.ANGLE_TOL,
            primitive_cell=True,
            scale=True,
            attempt_supercell=False,
            allow_subset=False,
            comparator=ElementComparator(),
        )

        snls = []

        for s in m_strucs:
            for snl_struc in snl_structs:
                if sm.fit(s, snl_struc):
                    if snl_struc.snl not in snls:
                        snls.append(snl_struc.snl)

        self.logger.debug(f"Found {len(snls)} SNLs for {mat['material_id']}")
        return snls

    def update_targets(self, items):
        """
        Inserts the new SNL docs into the SNL collection
        """
        if not items:
            return

        self.provenance.connect()

        snls = list(filter(None, items))

        if len(snls) > 0:
            self.logger.info(f"Found {len(snls)} SNLs to update")
            self.provenance.update(snls)
        else:
            self.logger.info("No items to update")

        self.provenance.close()
