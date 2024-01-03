from math import ceil
from typing import Dict, Iterator, Optional

from maggma.builders import Builder
from maggma.stores import Store
from maggma.utils import grouper
from pymatgen.core.structure import Structure

from emmet.core.magnetism import MagnetismDoc
from emmet.core.utils import jsanitize

__author__ = "Shyam Dwaraknath <shyamd@lbl.gov>, Matthew Horton <mkhorton@lbl.gov>"


class MagneticBuilder(Builder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query: Optional[Dict] = None,
        chunk_size: int = 300,
        allow_bson=True,
        **kwargs,
    ):
        """
        Creates a magnetism collection for materials

        Args:
            materials (Store): Store of materials documents to match to
            magnetism (Store): Store of magnetism properties

        """
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.materials = source_keys["materials"]
        self.tasks = source_keys["tasks"]
        self.magnetism = target_keys["magnetism"]
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.query = query or {}
        self.kwargs = kwargs

        self.materials.key = "material_id"
        self.tasks.key = "task_id"
        self.magnetism.key = "material_id"

        super().__init__(
            sources=[self.materials, self.tasks],
            targets=[self.magnetism],
            chunk_size=self.chunk_size,
            **kwargs,
        )

    def prechunk(self, number_splits: int) -> Iterator[Dict]:  # pragma: no cover
        """
        Prechunk method to perform chunking by the key field
        """
        q = dict(self.query)

        q.update({"deprecated": False})

        keys = self.magnetism.newer_in(self.materials, criteria=q, exhaustive=True)

        N = ceil(len(keys) / number_splits)
        for split in grouper(keys, N):
            yield {"query": {self.materials.key: {"$in": list(split)}}}

    def get_items(self):
        """
        Gets all items to process

        Returns:
            Generator or list relevant tasks and materials to process
        """

        self.logger.info("Magnetism Builder Started")

        q = dict(self.query)

        q.update({"deprecated": False})

        mat_ids = self.materials.distinct(self.materials.key, criteria=q)
        mag_ids = self.magnetism.distinct(self.magnetism.key)

        mats_set = set(
            self.magnetism.newer_in(target=self.materials, criteria=q, exhaustive=True)
        ) | (set(mat_ids) - set(mag_ids))

        mats = [mat for mat in mats_set]

        self.logger.info(f"Processing {len(mats)} materials for magnetism data")

        return [
            mats[i : i + self.chunk_size] for i in range(0, len(mats), self.chunk_size)
        ]

    def get_processed_docs(self, mats):
        self.materials.connect()
        self.tasks.connect()

        all_docs = []

        for mat in mats:
            mat_doc = self.materials.query_one(
                {self.materials.key: mat},
                [
                    self.materials.key,
                    "origins",
                    "last_updated",
                    "structure",
                    "deprecated",
                ],
            )

            for origin in mat_doc["origins"]:
                if origin["name"] == "structure":
                    task_id = origin["task_id"]

            task_query = self.tasks.query_one(
                properties=["last_updated", "calcs_reversed"],
                criteria={self.tasks.key: task_id},
            )

            task_updated = task_query["last_updated"]
            total_magnetization = task_query["calcs_reversed"][-1]["output"]["outcar"][
                "total_magnetization"
            ]

            mat_doc.update(
                {
                    "task_id": task_id,
                    "total_magnetization": total_magnetization,
                    "task_updated": task_updated,
                    self.materials.key: mat_doc[self.materials.key],
                }
            )

            all_docs.append(mat_doc)

        self.materials.close()
        self.tasks.close()

        return all_docs

    def process_item(self, items):
        docs = []
        for item in items:
            if not item:
                continue

            structure = Structure.from_dict(item["structure"])
            mpid = item["material_id"]
            origin_entry = {
                "name": "magnetism",
                "task_id": item["task_id"],
                "last_updated": item["task_updated"],
            }

            doc = MagnetismDoc.from_structure(
                structure=structure,
                material_id=mpid,
                total_magnetization=item["total_magnetization"],
                origins=[origin_entry],
                deprecated=item["deprecated"],
                last_updated=item["last_updated"],
            )

            docs.append(jsanitize(doc.model_dump(), allow_bson=self.allow_bson))

        return docs

    def update_targets(self, items):
        """
        Inserts the new magnetism docs into the magnetism collection
        """
        if not items:
            return

        self.magnetism.connect()

        docs = list(filter(None, items))

        if len(docs) > 0:
            self.logger.info(f"Found {len(docs)} magnetism docs to update")
            self.magnetism.update(docs)
        else:
            self.logger.info("No items to update")

        self.magnetism.close()
