from math import ceil
from typing import Dict, Optional

import numpy as np
from maggma.builders import Builder
from maggma.core import Store
from maggma.utils import grouper
from pymatgen.core.structure import Structure

from emmet.core.polar import PiezoelectricDoc
from emmet.core.utils import jsanitize


class PiezoelectricBuilder(Builder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query: Optional[Dict] = None,
        chunk_size: int = 300,
        allow_bson=True,
        **kwargs,
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.materials = source_keys["materials"]
        self.tasks = source_keys["tasks"]
        self.piezoelectric = target_keys["piezoelectric"]
        self.chunk_size = chunk_size
        self.allow_bson = allow_bson
        self.query = query or {}
        self.kwargs = kwargs

        self.materials.key = "material_id"
        self.tasks.key = "task_id"
        self.piezoelectric.key = "material_id"

        super().__init__(
            sources=[self.materials, self.tasks],
            targets=[self.piezoelectric],
            chunk_size=self.chunk_size,
            **kwargs,
        )

    def prechunk(self, number_splits: int):  # pragma: no cover
        """
        Prechunk method to perform chunking by the key field
        """
        q = dict(self.query)

        # Ensure no centrosymmetry
        q.update(
            {
                "symmetry.point_group": {
                    "$nin": [
                        "-1",
                        "2/m",
                        "mmm",
                        "4/m",
                        "4/mmm",
                        "-3",
                        "-3m",
                        "6/m",
                        "6/mmm",
                        "m-3",
                        "m-3m",
                    ]
                }
            }
        )

        keys = self.piezoelectric.newer_in(self.materials, criteria=q, exhaustive=True)

        N = ceil(len(keys) / number_splits)
        for split in grouper(keys, N):
            yield {"query": {self.materials.key: {"$in": list(split)}}}

    def get_items(self):
        """
        Gets all items to process

        Returns:
            generator or list relevant tasks and materials to process
        """

        self.logger.info("Piezoelectric Builder Started")

        q = dict(self.query)

        # Ensure no centrosymmetry
        q.update(
            {
                "symmetry.point_group": {
                    "$nin": [
                        "-1",
                        "2/m",
                        "mmm",
                        "4/m",
                        "4/mmm",
                        "-3",
                        "-3m",
                        "6/m",
                        "6/mmm",
                        "m-3",
                        "m-3m",
                    ]
                }
            }
        )

        mat_ids = self.materials.distinct(self.materials.key, criteria=q)
        piezo_ids = self.piezoelectric.distinct(self.piezoelectric.key)

        mats_set = set(
            self.piezoelectric.newer_in(
                target=self.materials, criteria=q, exhaustive=True
            )
        ) | (set(mat_ids) - set(piezo_ids))

        mats = [mat for mat in mats_set]

        self.logger.info(f"Processing {len(mats)} materials for piezoelectric data")

        return [
            mat_ids[i : i + self.chunk_size]
            for i in range(0, len(mat_ids), self.chunk_size)
        ]

    def get_processed_docs(self, mats):
        self.materials.connect()
        self.tasks.connect()

        all_docs = []

        for mat in mats:
            docs = []

            mat_doc = self.materials.query_one(
                {self.materials.key: mat},
                [
                    self.materials.key,
                    "structure",
                    "task_types",
                    "run_types",
                    "deprecated_tasks",
                    "last_updated",
                ],
            )

            task_types = mat_doc["task_types"].items()

            potential_task_ids = []

            for task_id, task_type in task_types:
                if task_type == "DFPT Dielectric":
                    if task_id not in mat_doc["deprecated_tasks"]:
                        potential_task_ids.append(task_id)

            for task_id in potential_task_ids:
                task_query = self.tasks.query_one(
                    properties=[
                        "last_updated",
                        "input.is_hubbard",
                        "orig_inputs.kpoints",
                        "orig_inputs.poscar.structure",
                        "input.parameters",
                        "input.structure",
                        "output.piezo_tensor",
                        "output.piezo_ionic_tensor",
                        "output.bandgap",
                    ],
                    criteria={self.tasks.key: str(task_id)},
                )
                if task_query["output"]["bandgap"] > 0:
                    try:
                        structure = task_query["orig_inputs"]["poscar"]["structure"]
                    except KeyError:
                        structure = task_query["input"]["structure"]

                    is_hubbard = task_query["input"]["is_hubbard"]

                    if (
                        task_query["orig_inputs"]["kpoints"]["generation_style"]
                        == "Monkhorst"
                        or task_query["orig_inputs"]["kpoints"]["generation_style"]
                        == "Gamma"
                    ):
                        nkpoints = np.prod(
                            task_query["orig_inputs"]["kpoints"]["kpoints"][0], axis=0
                        )

                    else:
                        nkpoints = task_query["orig_inputs"]["kpoints"]["nkpoints"]

                    lu_dt = mat_doc["last_updated"]
                    task_updated = task_query["last_updated"]

                    docs.append(
                        {
                            "task_id": task_id,
                            "is_hubbard": int(is_hubbard),
                            "nkpoints": int(nkpoints),
                            "piezo_static": task_query["output"]["piezo_tensor"],
                            "piezo_ionic": task_query["output"]["piezo_ionic_tensor"],
                            "structure": structure,
                            "updated_on": lu_dt,
                            "task_updated": task_updated,
                            self.materials.key: mat_doc[self.materials.key],
                        }
                    )

            if len(docs) > 0:
                sorted_docs = sorted(
                    docs,
                    key=lambda entry: (
                        entry["is_hubbard"],
                        entry["nkpoints"],
                        entry["updated_on"],
                    ),
                    reverse=True,
                )
                all_docs.append(sorted_docs[0])

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
                "name": "piezoelectric",
                "task_id": item["task_id"],
                "last_updated": item["task_updated"],
            }

            doc = PiezoelectricDoc.from_ionic_and_electronic(
                structure=structure,
                material_id=mpid,
                origins=[origin_entry],
                deprecated=False,
                ionic=item["piezo_ionic"],
                electronic=item["piezo_static"],
                last_updated=item["updated_on"],
            )

            docs.append(jsanitize(doc.model_dump(), allow_bson=self.allow_bson))

        return docs

    def update_targets(self, items):
        """
        Inserts the new dielectric docs into the dielectric collection
        """
        if not items:
            return

        self.piezoelectric.connect()

        docs = list(filter(None, items))

        if len(docs) > 0:
            self.logger.info(f"Found {len(docs)} piezoelectric docs to update")
            self.piezoelectric.update(docs)
        else:
            self.logger.info("No items to update")

        self.piezoelectric.close()
