from itertools import chain, combinations
from typing import Dict, List, Optional, Tuple, Union

from maggma.builders import Builder
from maggma.core import Store
from matminer.datasets import load_dataset
from pymatgen.analysis.alloys.core import (
    KNOWN_ANON_FORMULAS,
    AlloyMember,
    AlloyPair,
    AlloySystem,
    InvalidAlloy,
)
from pymatgen.core.structure import Structure
from tqdm import tqdm

from emmet.core.thermo import ThermoType

# rough sort of ANON_FORMULAS by "complexity"
ANON_FORMULAS = sorted(KNOWN_ANON_FORMULAS, key=lambda af: len(af))

# Combinatorially, cannot StructureMatch every single possible pair of materials
# Use a loose spacegroup for a pre-screen (in addition to standard spacegroup)
LOOSE_SPACEGROUP_SYMPREC = 0.5

# A source of effective masses, should be replaced with MP-provided effective masses.
BOLTZTRAP_DF = load_dataset("boltztrap_mp")


class AlloyPairBuilder(Builder):
    """
    This builder iterates over anonymous_formula and builds AlloyPair.
    It does not look for members of an AlloyPair.
    """

    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        query: Optional[Dict] = None,
        num_phase_diagram_eles: Optional[int] = None,
        chunk_size: int = 8,
        thermo_type: Union[ThermoType, str] = ThermoType.GGA_GGA_U_R2SCAN,
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.materials = source_keys["materials"]
        self.thermo = source_keys["thermo"]
        self.electronic_structure = source_keys["electronic_structure"]
        self.provenance = source_keys["provenance"]
        self.oxi_states = source_keys["oxi_states"]
        self.alloy_pairs = target_keys["alloy_pairs"]

        self.num_phase_diagram_eles = num_phase_diagram_eles
        self.query = query or {}
        self.chunk_size = chunk_size

        t_type = thermo_type if isinstance(thermo_type, str) else thermo_type.value
        valid_types = {*map(str, ThermoType.__members__.values())}
        if invalid_types := {t_type} - valid_types:
            raise ValueError(
                f"Invalid thermo type(s) passed: {invalid_types}, valid types are: {valid_types}"
            )

        self.thermo_type = t_type

        super().__init__(
            sources=[
                self.materials,
                self.thermo,
                self.electronic_structure,
                self.provenance,
                self.oxi_states,
            ],
            targets=[self.alloy_pairs],
            chunk_size=self.chunk_size,
        )

    def ensure_indexes(self):
        self.alloy_pairs.ensure_index("pair_id")
        self.alloy_pairs.ensure_index("_search.id")
        self.alloy_pairs.ensure_index("_search.formula")
        self.alloy_pairs.ensure_index("_search.member_ids")
        self.alloy_pairs.ensure_index("alloy_pair.chemsys")

    def get_items(self):
        self.ensure_indexes()

        return [
            ANON_FORMULAS[i : i + self.chunk_size]
            for i in range(0, len(ANON_FORMULAS), self.chunk_size)
        ]

    def get_processed_docs(self, formulas):
        self.materials.connect()
        self.thermo.connect()
        self.provenance.connect()
        self.electronic_structure.connect()
        self.oxi_states.connect()

        all_docs = []

        for af in formulas:
            # if af != "AB":
            #     continue

            thermo_docs = self.thermo.query(
                criteria={
                    "formula_anonymous": af,
                    "deprecated": False,
                    "thermo_type": self.thermo_type,
                },
                properties=[
                    "material_id",
                    "energy_above_hull",
                    "formation_energy_per_atom",
                ],
            )

            thermo_docs = {d["material_id"]: d for d in thermo_docs}

            mpids = list(thermo_docs.keys())

            docs = self.materials.query(
                criteria={
                    "material_id": {"$in": mpids},
                    "deprecated": False,
                },  # , "material_id": {"$in": ["mp-804", "mp-661"]}},
                properties=["structure", "material_id", "symmetry.number"],
            )
            docs = {d["material_id"]: d for d in docs}

            electronic_structure_docs = self.electronic_structure.query(
                {"material_id": {"$in": mpids}},
                properties=["material_id", "band_gap", "is_gap_direct"],
            )
            electronic_structure_docs = {
                d["material_id"]: d for d in electronic_structure_docs
            }

            provenance_docs = self.provenance.query(
                {"material_id": {"$in": mpids}},
                properties=["material_id", "theoretical", "database_IDs"],
            )
            provenance_docs = {d["material_id"]: d for d in provenance_docs}

            oxi_states_docs = self.oxi_states.query(
                {"material_id": {"$in": mpids}, "state": "successful"},
                properties=["material_id", "structure"],
            )
            oxi_states_docs = {d["material_id"]: d for d in oxi_states_docs}

            for material_id in mpids:
                d = docs[material_id]

                d["structure"] = Structure.from_dict(d["structure"])

                if material_id in oxi_states_docs:
                    d["structure_oxi"] = Structure.from_dict(
                        oxi_states_docs[material_id]["structure"]
                    )
                else:
                    d["structure_oxi"] = d["structure"]

                # calculate loose space group
                d["spacegroup_loose"] = d["structure"].get_space_group_info(
                    LOOSE_SPACEGROUP_SYMPREC
                )[1]

                d["properties"] = {}
                # patch in BoltzTraP data if present
                row = BOLTZTRAP_DF.loc[BOLTZTRAP_DF["mpid"] == material_id]
                if len(row) == 1:
                    d["properties"]["m_n"] = float(row.m_n)
                    d["properties"]["m_p"] = float(row.m_p)

                if material_id in electronic_structure_docs:
                    for key in ("band_gap", "is_gap_direct"):
                        d["properties"][key] = electronic_structure_docs[material_id][
                            key
                        ]

                for key in ("energy_above_hull", "formation_energy_per_atom"):
                    d["properties"][key] = thermo_docs[material_id][key]

                if material_id in provenance_docs:
                    for key in ("theoretical",):
                        d["properties"][key] = provenance_docs[material_id][key]

            # print(
            #     f"Starting {af} with {len(docs)} materials, anonymous formula {idx} of {len(ANON_FORMULAS)}"
            # )

            all_docs.append(docs)

        self.materials.close()
        self.thermo.close()
        self.provenance.close()
        self.electronic_structure.close()
        self.oxi_states.close()

        return all_docs

    def process_item(self, items):
        docs = []
        for item in items:
            if not item:
                continue

            pairs = []
            for mpids in tqdm(list(combinations(item.keys(), 2))):
                if (
                    item[mpids[0]]["symmetry"]["number"]
                    == item[mpids[1]]["symmetry"]["number"]
                ) or (
                    item[mpids[0]]["spacegroup_loose"]
                    == item[mpids[1]]["spacegroup_loose"]
                ):
                    # optionally, could restrict based on band gap too (e.g. at least one end-point semiconducting)
                    # if (item[mpids[0]]["band_gap"] > 0) or (item[mpids[1]]["band_gap"] > 0):
                    try:
                        pair = AlloyPair.from_structures(
                            structures=[
                                item[mpids[0]]["structure"],
                                item[mpids[1]]["structure"],
                            ],
                            structures_with_oxidation_states=[
                                item[mpids[0]]["structure_oxi"],
                                item[mpids[1]]["structure_oxi"],
                            ],
                            ids=[mpids[0], mpids[1]],
                            properties=[
                                item[mpids[0]]["properties"],
                                item[mpids[1]]["properties"],
                            ],
                        )
                        pairs.append(
                            {
                                "alloy_pair": pair.as_dict(),
                                "_search": pair.search_dict(),
                                "pair_id": pair.pair_id,
                            }
                        )
                    except InvalidAlloy:
                        pass
                    except Exception as exc:
                        print(exc)

            if pairs:
                print(f"Found {len(pairs)} alloy(s)")

            docs.append(pairs)

        return docs

    def update_targets(self, items):
        if not items:
            return

        self.alloy_pairs.connect()

        docs = list(chain.from_iterable(items))

        if docs:
            self.alloy_pairs.update(docs)

        self.alloy_pairs.close()


class AlloyPairMemberBuilder(Builder):
    """
    This builder iterates over available AlloyPairs by chemical system
    and searches for possible members of those AlloyPairs.
    """

    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        chunk_size: int = 200,
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.alloy_pairs = source_keys["alloy_pairs"]
        self.materials = source_keys["materials"]
        self.snls = source_keys["snls_icsd"]
        self.alloy_pair_members = target_keys["alloy_pair_members"]

        self.chunk_size = chunk_size

        super().__init__(
            sources=[self.alloy_pairs, self.materials, self.snls],
            targets=[self.alloy_pair_members],
            chunk_size=self.chunk_size,
        )

    def ensure_indexes(self):
        self.alloy_pairs.ensure_index("pair_id")
        self.alloy_pairs.ensure_index("_search.id")
        self.alloy_pairs.ensure_index("_search.formula")
        self.alloy_pairs.ensure_index("_search.member_ids")
        self.alloy_pairs.ensure_index("alloy_pair.chemsys")
        self.alloy_pairs.ensure_index("alloy_pair.anonymous_formula")

    def get_items(self):
        all_alloy_chemsys = set(self.alloy_pairs.distinct("alloy_pair.chemsys"))
        all_known_chemsys = set(self.materials.distinct("chemsys")) | set(
            self.snls.distinct("chemsys")
        )
        possible_chemsys = all_known_chemsys.intersection(all_alloy_chemsys)

        print(
            f"There are {len(all_alloy_chemsys)} alloy chemical systems of which "
            f"{len(possible_chemsys)} may have members."
        )

        return [
            list(possible_chemsys)[i : i + self.chunk_size]
            for i in range(0, len(possible_chemsys), self.chunk_size)
        ]

    def get_processed_docs(self, mats):
        self.alloy_pairs.connect()
        self.materials.connect()
        self.snls.connect()

        all_docs = []

        for possible_chemsys in mats:
            for idx, chemsys in enumerate(possible_chemsys):
                pairs = self.alloy_pairs.query(criteria={"alloy_pair.chemsys": chemsys})
                pairs = [AlloyPair.from_dict(d["alloy_pair"]) for d in pairs]

                mp_docs = self.materials.query(
                    criteria={"chemsys": chemsys, "deprecated": False},
                    properties=["structure", "material_id"],
                )
                mp_structures = {
                    d["material_id"]: Structure.from_dict(d["structure"])
                    for d in mp_docs
                }

                snl_docs = self.snls.query({"chemsys": chemsys})
                snl_structures = {d["snl_id"]: Structure.from_dict(d) for d in snl_docs}

                structures = mp_structures
                structures.update(snl_structures)

                if structures:
                    all_docs.append((pairs, structures))

        self.alloy_pairs.close()
        self.materials.close()
        self.snls.close()

        return all_docs

    def process_item(self, items: List[Tuple[List[AlloyPair], Dict[str, Structure]]]):
        docs = []

        for item in items:
            if not item:
                continue

            pairs, structures = item

            all_pair_members = []
            for pair in pairs:
                pair_members = {"pair_id": pair.pair_id, "members": []}
                for db_id, structure in structures.items():
                    try:
                        if pair.is_member(structure):
                            db, _ = db_id.split("-")
                            member = AlloyMember(
                                id_=db_id,
                                db=db,
                                composition=structure.composition,
                                is_ordered=structure.is_ordered,
                                x=pair.get_x(structure.composition),
                            )
                            pair_members["members"].append(member.as_dict())
                    except Exception as exc:
                        print(f"Exception for {db_id}: {exc}")
                if pair_members["members"]:
                    all_pair_members.append(pair_members)

            docs.append(all_pair_members)

        return docs

    def update_targets(self, items):
        if not items:
            return

        self.alloy_pair_members.connect()

        for item in items:
            docs = list(chain.from_iterable(items))
            if docs:
                self.alloy_pair_members.update(docs)

        self.alloy_pair_members.close()


class AlloySystemBuilder(Builder):
    """
    This builder stitches together the results of
    AlloyPairBuilder and AlloyPairMemberBuilder. The output
    of this collection is the one served by the AlloyPair API.
    It also builds AlloySystem.
    """

    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        chunk_size: int = 8,
        **kwargs,
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.alloy_pairs = source_keys["alloy_pairs"]
        self.alloy_pair_members = source_keys["alloy_pair_members"]
        self.alloy_pairs_merged = target_keys["alloy_pairs_merged"]
        self.alloy_systems = target_keys["alloy_systems"]

        self.chunk_size = chunk_size
        self.kwargs = kwargs

        super().__init__(
            sources=[self.alloy_pairs, self.alloy_pair_members],
            targets=[self.alloy_pairs_merged, self.alloy_systems],
            chunk_size=self.chunk_size,
            **kwargs,
        )

    def get_items(self):
        return [
            ANON_FORMULAS[i : i + self.chunk_size]
            for i in range(0, len(ANON_FORMULAS), self.chunk_size)
        ]

    def get_processed_docs(self, formulas):
        self.alloy_pairs.connect()
        self.alloy_pair_members.connect()

        all_docs = []

        for af in formulas:
            # comment out to only calculate a single anonymous formula for debugging
            # if af != "AB":
            #     continue

            docs = list(self.alloy_pairs.query({"alloy_pair.anonymous_formula": af}))
            pair_ids = [d["pair_id"] for d in docs]
            members = {
                d["pair_id"]: d
                for d in self.alloy_pair_members.query({"pair_id": {"$in": pair_ids}})
            }

            if docs:
                all_docs.append((docs, members))

        self.alloy_pairs.close()
        self.alloy_pair_members.close()

        return all_docs

    def process_item(self, items):
        docs = []
        for item in items:
            if not item:
                continue

            pair_docs, members = item

            for doc in pair_docs:
                if doc["pair_id"] in members:
                    doc["alloy_pair"]["members"] = members[doc["pair_id"]]["members"]
                    doc["_search"]["member_ids"] = [
                        m["id_"] for m in members[doc["pair_id"]]["members"]
                    ]
                else:
                    doc["alloy_pair"]["members"] = []
                    doc["_search"]["member_ids"] = []

            pairs = [AlloyPair.from_dict(d["alloy_pair"]) for d in pair_docs]
            systems = AlloySystem.systems_from_pairs(pairs)

            system_docs = [
                {
                    "alloy_system": system.as_dict(),
                    "alloy_id": system.alloy_id,
                    "_search": {"member_ids": [m.id_ for m in system.members]},
                }
                for system in systems
            ]

            for system_doc in system_docs:
                # Too big to store, will need to reconstruct separately from pair_ids
                system_doc["alloy_system"]["alloy_pairs"] = None

            docs.append((pair_docs, system_docs))

        return docs

    def update_targets(self, items):
        if not items:
            return

        self.alloy_pairs_merged.connect()
        self.alloy_systems.connect()

        pair_docs, system_docs = [p for p, s in items], [s for p, s in items]

        pair_docs = list(chain.from_iterable(pair_docs))
        if pair_docs:
            self.alloy_pairs_merged._collection.insert_many(pair_docs)

        system_docs = list(chain.from_iterable(system_docs))
        if system_docs:
            self.alloy_systems._collection.insert_many(system_docs)

        self.alloy_pairs_merged.close()
        self.alloy_systems.close()
