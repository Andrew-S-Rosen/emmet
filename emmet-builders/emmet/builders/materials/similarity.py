from typing import Dict, Optional

import numpy as np
from maggma.builders import Builder
from maggma.core import Store

__author__ = "Nils E. R. Zimmermann <nerz@lbl.gov>"


# TODO:
# 1) ADD DOCUMENT MODEL


class StructureSimilarityBuilder(Builder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query: Optional[Dict] = None,
        chunk_size: int = 300,
        fp_type="csf",
        **kwargs,
    ):
        """
        Calculates similarity metrics between structures on the basis
        of site descriptors.

        Args:
            site_descriptors (Store): storage of site-descriptors data
                                      such as tetrahedral order parameter
                                      or percentage of 8-fold coordination.
            structure_similarity (Store): storage of structure similarity
                                          metrics.
            fp_type (str): target site fingerprint type to be
                           used for similarity computation
                           ("csf" (based on matminer's
                           CrystalSiteFingerprint class)
                           or "opsf" (based on matminer's
                           OPSiteFingerprint class)).
        """

        self.source_keys = source_keys
        self.target_keys = target_keys

        self.site_descriptors = source_keys["site_descriptors"]
        self.structure_similarity = target_keys["similarity"]
        self.fp_type = fp_type

        self.chunk_size = chunk_size
        self.query = query

        super().__init__(
            sources=[self.site_descriptors],
            targets=[self.structure_similarity],
            chunk_size=self.chunk_size,
            **kwargs,
        )

    def get_items(self):
        """
        Gets all materials that need new site descriptors.

        Returns:
            generator of materials to calculate site descriptors.
        """

        self.logger.info("Structure Similarity Builder Started")

        self.logger.info("Setting indexes")

        # TODO: re-introduce last-updated filtering.
        task_ids = list(self.site_descriptors.distinct(self.site_descriptors.key))
        n_task_ids = len(task_ids)

        return [
            task_ids[i : i + self.chunk_size]
            for i in range(0, n_task_ids, self.chunk_size)
        ]

    def get_processed_docs(self, tasks):
        self.site_descriptors.connect()

        all_docs = []

        for i in range(len(tasks) - 1):
            d1 = self.site_descriptors.query_one(
                properties=[self.site_descriptors.key, "statistics"],
                criteria={self.site_descriptors.key: tasks[i]},
            )
            for j in range(i + 1, len(tasks)):
                d2 = self.site_descriptors.query_one(
                    properties=[self.site_descriptors.key, "statistics"],
                    criteria={self.site_descriptors.key: tasks[j]},
                )
                all_docs.append(list([d1, d2]))

        self.site_descriptors.close()

        return all_docs

    def process_item(self, items):
        """
        Calculates site descriptors for the structures

        Args:
            item (list): a list (length 2) with each one document that
                         carries a task ID in "task_id" and a statistics
                         vector from OP site-fingerprints in
                         "statistics".

        Returns:
            dict: similarity measures.
        """

        docs = []

        for item in items:
            if not item:
                continue

            self.logger.debug(
                "Similarities for {} and {}".format(
                    item[0][self.site_descriptors.key],
                    item[1][self.site_descriptors.key],
                )
            )

            sim_doc = {}
            sim_doc = self.get_similarities(item[0], item[1])
            sim_doc[self.structure_similarity.key] = tuple(
                sorted(
                    [
                        item[0][self.site_descriptors.key],
                        item[1][self.site_descriptors.key],
                    ]
                )
            )

            docs.append(sim_doc)

        return docs

    def update_targets(self, items):
        """
        Inserts the new task_types into the task_types collection.

        Args:
            items ([[dict]]): a list of list of site-descriptors dictionaries to update.
        """
        if not items:
            return

        self.structure_similarity.connect()

        if len(items) > 0:
            self.logger.info("Updating {} structure-similarity docs".format(len(items)))
            self.structure_similarity.update(docs=items)
        else:
            self.logger.info("No items to update")

        self.structure_similarity.close()

    def get_similarities(self, d1, d2):
        doc = {}

        # Compute similarty metrics.
        try:
            dout = {}
            l = {}
            v = {}
            for i, li in enumerate(
                [d1["statistics"][self.fp_type], d2["statistics"][self.fp_type]]
            ):
                v[i] = []
                l[i] = []
                # for optype, stats in d.items():
                for opdict in li:
                    for stattype, val in opdict.items():
                        if stattype != "name":
                            v[i].append(val)
                            l[i].append("{} {}".format(opdict["name"], stattype))
            if len(l[0]) != len(l[1]):
                raise RuntimeError(
                    "Site-fingerprint statistics dictionaries"
                    " have different sizes ({}, {})".format(len(l[0]), len(l[1]))
                )
            for k in l[0]:
                if k not in l[1]:
                    raise RuntimeError(
                        'Label "{}" not found in second site-'
                        "fingerprint statistics "
                        "dictionary".format(k)
                    )
            v1 = np.array([v[0][k] for k in range(len(l[0]))])
            v2 = np.array([v[1][l[1].index(k)] for k in l[0]])
            dout["cos"] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            dout["dist"] = np.linalg.norm(v1 - v2)
            doc = dout

        except Exception as e:
            self.logger.error(
                "Failed calculating structure similarity" "metrics: {}".format(e)
            )

        return doc
