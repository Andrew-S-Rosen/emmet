from datetime import datetime
from itertools import chain
from math import ceil
from typing import Any, Dict, Iterable, Iterator, List, Optional

from maggma.builders import Builder
from maggma.core import Store
from maggma.utils import grouper

from emmet.builders.settings import EmmetBuildSettings
from emmet.core.molecules.summary import MoleculeSummaryDoc
from emmet.core.utils import jsanitize

# from monty.serialization import loadfn, dumpfn


__author__ = "Evan Spotte-Smith"

SETTINGS = EmmetBuildSettings()


class SummaryBuilder(Builder):
    """
    The SummaryBuilder collects all property documents and gathers their properties
    into a single MoleculeSummaryDoc

    The process is as follows:
        1. Gather MoleculeDocs by formula
        2. For each doc, grab the relevant property docs
        3. Convert property docs to MoleculeSummaryDoc
    """

    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        chunk_size: int = 100,
        allow_bson=True,
        query: Optional[Dict] = None,
        settings: Optional[EmmetBuildSettings] = None,
        **kwargs,
    ):
        self.bonds = source_keys["molecules_bonds"]
        self.charges = source_keys["molecules_charges"]
        self.metal_binding = source_keys["molecules_metal_binding"]
        self.molecules = source_keys["molecules"]
        self.orbitals = source_keys["molecules_orbitals"]
        self.redox = source_keys["molecules_redox"]
        self.spins = source_keys["molecules_spins"]
        self.thermo = source_keys["molecules_thermo"]
        self.vibes = source_keys["molecules_vibrations"]
        self.summary = target_keys["molecules_summary"]
        self.query = query if query else dict()
        self.settings = EmmetBuildSettings.autoload(settings)
        self.kwargs = kwargs

        super().__init__(
            sources=[
                self.molecules,
                self.charges,
                self.spins,
                self.bonds,
                self.metal_binding,
                self.orbitals,
                self.redox,
                self.thermo,
                self.vibes,
            ],
            targets=[self.summary],
            chunk_size=self.chunk_size,
            **kwargs,
        )
        # Uncomment in case of issue with mrun not connecting automatically to collections
        # for i in [
        #     self.molecules,
        #     self.charges,
        #     self.spins,
        #     self.bonds,
        #     self.metal_binding,
        #     self.orbitals,
        #     self.redox,
        #     self.thermo,
        #     self.vibes,
        #     self.summary
        # ]:
        #     try:
        #         i.connect()
        #     except Exception as e:
        #         print("Could not connect,", e)

    def ensure_indexes(self):
        """
        Ensures indices on the collections needed for building
        """

        # Search index for molecules
        self.molecules.ensure_index("molecule_id")
        self.molecules.ensure_index("last_updated")
        self.molecules.ensure_index("task_ids")
        self.molecules.ensure_index("formula_alphabetical")

        # Search index for charges
        self.charges.ensure_index("molecule_id")
        self.charges.ensure_index("method")
        self.charges.ensure_index("task_id")
        self.charges.ensure_index("solvent")
        self.charges.ensure_index("lot_solvent")
        self.charges.ensure_index("property_id")
        self.charges.ensure_index("last_updated")
        self.charges.ensure_index("formula_alphabetical")

        # Search index for charges
        self.spins.ensure_index("molecule_id")
        self.spins.ensure_index("method")
        self.spins.ensure_index("task_id")
        self.spins.ensure_index("solvent")
        self.spins.ensure_index("lot_solvent")
        self.spins.ensure_index("property_id")
        self.spins.ensure_index("last_updated")
        self.spins.ensure_index("formula_alphabetical")

        # Search index for charges
        self.bonds.ensure_index("molecule_id")
        self.bonds.ensure_index("method")
        self.bonds.ensure_index("task_id")
        self.bonds.ensure_index("solvent")
        self.bonds.ensure_index("lot_solvent")
        self.bonds.ensure_index("property_id")
        self.bonds.ensure_index("last_updated")
        self.bonds.ensure_index("formula_alphabetical")

        # Search index for metal_binding
        self.metal_binding.ensure_index("molecule_id")
        self.metal_binding.ensure_index("solvent")
        self.metal_binding.ensure_index("lot_solvent")
        self.metal_binding.ensure_index("property_id")
        self.metal_binding.ensure_index("last_updated")
        self.metal_binding.ensure_index("formula_alphabetical")
        self.metal_binding.ensure_index("method")

        # Search index for orbitals
        self.orbitals.ensure_index("molecule_id")
        self.orbitals.ensure_index("task_id")
        self.orbitals.ensure_index("solvent")
        self.orbitals.ensure_index("lot_solvent")
        self.orbitals.ensure_index("property_id")
        self.orbitals.ensure_index("last_updated")
        self.orbitals.ensure_index("formula_alphabetical")

        # Search index for orbitals
        self.redox.ensure_index("molecule_id")
        self.redox.ensure_index("task_id")
        self.redox.ensure_index("solvent")
        self.redox.ensure_index("lot_solvent")
        self.redox.ensure_index("property_id")
        self.redox.ensure_index("last_updated")
        self.redox.ensure_index("formula_alphabetical")

        # Search index for thermo
        self.thermo.ensure_index("molecule_id")
        self.thermo.ensure_index("task_id")
        self.thermo.ensure_index("solvent")
        self.thermo.ensure_index("lot_solvent")
        self.thermo.ensure_index("property_id")
        self.thermo.ensure_index("last_updated")
        self.thermo.ensure_index("formula_alphabetical")

        # Search index for vibrational properties
        self.vibes.ensure_index("molecule_id")
        self.vibes.ensure_index("task_id")
        self.vibes.ensure_index("solvent")
        self.vibes.ensure_index("lot_solvent")
        self.vibes.ensure_index("property_id")
        self.vibes.ensure_index("last_updated")
        self.vibes.ensure_index("formula_alphabetical")

        # Search index for molecules
        self.summary.ensure_index("molecule_id")
        self.summary.ensure_index("last_updated")
        self.summary.ensure_index("formula_alphabetical")

    def prechunk(self, number_splits: int) -> Iterable[Dict]:  # pragma: no cover
        """Prechunk the builder for distributed computation"""

        temp_query = dict(self.query)
        temp_query["deprecated"] = False

        self.logger.info("Finding documents to process")
        all_mols = list(
            self.molecules.query(
                temp_query, [self.molecules.key, "formula_alphabetical"]
            )
        )

        processed_docs = set([e for e in self.summary.distinct("molecule_id")])
        to_process_docs = {d[self.molecules.key] for d in all_mols} - processed_docs
        to_process_forms = {
            d["formula_alphabetical"]
            for d in all_mols
            if d[self.molecules.key] in to_process_docs
        }

        N = ceil(len(to_process_forms) / number_splits)

        for formula_chunk in grouper(to_process_forms, N):
            yield {"query": {"formula_alphabetical": {"$in": list(formula_chunk)}}}

    def get_items(self) -> Iterator[List[Dict]]:
        """
        Gets all items to process into summary documents.
        This does no datetime checking; relying on on whether
        task_ids are included in the summary Store

        Returns:
            generator or list relevant tasks and molecules to process into documents
        """

        self.logger.info("Summary builder started")
        self.logger.info("Setting indexes")
        self.ensure_indexes()

        # Save timestamp to mark buildtime
        self.timestamp = datetime.utcnow()

        # Get all processed molecules
        temp_query = dict(self.query)
        temp_query["deprecated"] = False

        self.logger.info("Finding documents to process")
        all_mols = list(
            self.molecules.query(
                temp_query, [self.molecules.key, "formula_alphabetical"]
            )
        )

        processed_docs = set([e for e in self.summary.distinct("molecule_id")])
        to_process_docs = {d[self.molecules.key] for d in all_mols} - processed_docs
        to_process_forms = {
            d["formula_alphabetical"]
            for d in all_mols
            if d[self.molecules.key] in to_process_docs
        }

        self.logger.info(f"Found {len(to_process_docs)} unprocessed documents")
        self.logger.info(f"Found {len(to_process_forms)} unprocessed formulas")

        # Set total for builder bars to have a total
        self.total = len(to_process_forms)

        return [
            to_process_forms[i : i + self.chunk_size]
            for i in range(0, self.total, self.chunk_size)
        ]

    def get_processed_docs(self, molecules):
        self.molecules.connect()

        all_docs = []

        temp_query = dict(self.query)
        temp_query["deprecated"] = False
        for formula in molecules:
            mol_query = dict(temp_query)
            mol_query["formula_alphabetical"] = formula
            molecules = list(self.molecules.query(criteria=mol_query))

            all_docs += molecules

        self.molecules.close()

        return all_docs

    def process_item(self, items: List[Dict]) -> List[Dict]:
        """
        Process the tasks into a MoleculeSummaryDoc

        Args:
            tasks List[Dict] : a list of MoleculeDocs in dict form

        Returns:
            [dict] : a list of new orbital docs
        """

        def _group_docs(docs: List[Dict[str, Any]], by_method: bool = False):
            """Helper function to group docs by solvent"""
            grouped: Dict[str, Any] = dict()

            for doc in docs:
                solvent = doc.get("solvent")
                method = doc.get("method")
                if not solvent:
                    # Need to group by solvent
                    continue
                if by_method and method is None:
                    # Trying to group by method, but no method present
                    continue

                if not by_method:
                    grouped[solvent] = doc
                else:
                    if solvent not in grouped:
                        grouped[solvent] = {method: doc}
                    else:
                        grouped[solvent][method] = doc

            return (grouped, by_method)

        if not items:
            return

        self.bonds.connect()
        self.charges.connect()
        self.metal_binding.connect()
        self.orbitals.connect()
        self.redox.connect()
        self.spins.connect()
        self.thermo.connect()
        self.vibes.connect()

        mols = items
        formula = mols[0]["formula_alphabetical"]
        mol_ids = [m["molecule_id"] for m in mols]
        self.logger.debug(f"Processing {formula} : {mol_ids}")

        summary_docs = list()

        for mol in mols:
            mol_id = mol["molecule_id"]

            d = {
                "molecules": mol,
                "partial_charges": _group_docs(
                    list(self.charges.query({"molecule_id": mol_id})), True
                ),
                "partial_spins": _group_docs(
                    list(self.spins.query({"molecule_id": mol_id})), True
                ),
                "bonding": _group_docs(
                    list(self.bonds.query({"molecule_id": mol_id})), True
                ),
                "metal_binding": _group_docs(
                    list(self.metal_binding.query({"molecule_id": mol_id})), True
                ),
                "orbitals": _group_docs(
                    list(self.orbitals.query({"molecule_id": mol_id})), False
                ),
                "redox": _group_docs(
                    list(self.redox.query({"molecule_id": mol_id})), False
                ),
                "thermo": _group_docs(
                    list(self.thermo.query({"molecule_id": mol_id})), False
                ),
                "vibration": _group_docs(
                    list(self.vibes.query({"molecule_id": mol_id})), False
                ),
            }

            to_delete = list()

            for k, v in d.items():
                if isinstance(v, dict) and len(v) == 0:
                    to_delete.append(k)

            for td in to_delete:
                del d[td]

            # # For debugging; keep because it might be needed again
            # dumpfn(d, f"{mol_id}.json.gz")
            # break

            summary_doc = MoleculeSummaryDoc.from_docs(molecule_id=mol_id, docs=d)
            summary_docs.append(summary_doc)

        self.logger.debug(f"Produced {len(summary_docs)} summary docs for {formula}")

        self.bonds.close()
        self.charges.close()
        self.metal_binding.close()
        self.orbitals.close()
        self.redox.close()
        self.spins.close()
        self.thermo.close()
        self.vibes.close()

        return jsanitize([doc.model_dump() for doc in summary_docs], allow_bson=True)

    def update_targets(self, items: List[List[Dict]]):
        """
        Inserts the new documents into the summary collection

        Args:
            items [[dict]]: A list of documents to update
        """

        if not items:
            return

        self.summary.connect()

        docs = list(chain.from_iterable(items))  # type: ignore

        # Add timestamp
        for item in docs:
            item.update(
                {
                    "_bt": self.timestamp,
                }
            )

        molecule_ids = list({item["molecule_id"] for item in docs})

        if len(items) > 0:
            self.logger.info(f"Updating {len(docs)} summary documents")
            self.summary.remove_docs({self.summary.key: {"$in": molecule_ids}})
            self.summary.update(
                docs=docs,
                key=["molecule_id"],
            )
        else:
            self.logger.info("No items to update")

        self.summary.close()
