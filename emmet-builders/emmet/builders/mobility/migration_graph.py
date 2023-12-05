from typing import Dict, Tuple

from maggma.builders.dag_map_builder import MapBuilder
from maggma.core import Store
from pymatgen.analysis.diffusion.neb.full_path_mapper import MigrationGraph
from pymatgen.apps.battery.insertion_battery import InsertionElectrode

from emmet.builders.utils import get_hop_cutoff
from emmet.core.mobility.migrationgraph import MigrationGraphDoc
from emmet.core.utils import jsanitize


class MigrationGraphBuilder(MapBuilder):
    def __init__(
        self,
        source_keys: Dict[str, Store],
        target_keys: Dict[str, Store],
        chunk_size: int = 300,
        algorithm: str = "hops_based",
        min_hop_distance: float = 1,
        max_hop_distance: float = 7,
        populate_sc_fields: bool = True,
        min_length_sc: float = 8,
        minmax_num_atoms: Tuple[int, int] = (80, 120),
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5,
        **kwargs,
    ):
        self.source_keys = source_keys
        self.target_keys = target_keys

        self.insertion_electrode = source_keys["insertion_electrodes"]
        self.migration_graph = target_keys["migration_graph"]
        self.chunk_size = chunk_size

        self.algorithm = algorithm
        self.min_hop_distance = min_hop_distance
        self.max_hop_distance = max_hop_distance
        self.populate_sc_fields = populate_sc_fields
        self.min_length_sc = min_length_sc
        self.minmax_num_atoms = minmax_num_atoms
        self.ltol = ltol
        self.stol = stol
        self.angle_tol = angle_tol

        super().__init__(
            source=self.insertion_electrode,
            target=self.migration_graph,
            chunk_size=self.chunk_size,
            **kwargs,
        )

    def unary_function(self, item):
        warnings = []

        # get entries and info from insertion electrode
        ie = InsertionElectrode.from_dict(item["electrode_object"])
        entries = ie.get_all_entries()
        wi_entry = ie.working_ion_entry

        # get migration graph structure
        structs = MigrationGraph.get_structure_from_entries(entries, wi_entry)
        if len(structs) == 0:
            warnings.append("cannot generate migration graph from entries")
            d = None
        else:
            if len(structs) > 1:
                warnings.append(
                    f"migration graph ambiguous: {len(structs)} possible options"
                )
            # get hop cutoff distance
            d = get_hop_cutoff(
                migration_graph_struct=structs[0],
                mobile_specie=wi_entry.composition.chemical_system,
                algorithm=self.algorithm,
                min_hop_distance=self.min_hop_distance,
                max_hop_distance=self.max_hop_distance,
            )

        # get migration graph doc
        try:
            mg_doc = MigrationGraphDoc.from_entries_and_distance(
                battery_id=item["battery_id"],
                grouped_entries=entries,
                working_ion_entry=wi_entry,
                hop_cutoff=d,
                populate_sc_fields=self.populate_sc_fields,
                min_length_sc=self.min_length_sc,
                minmax_num_atoms=self.minmax_num_atoms,
                ltol=self.ltol,
                stol=self.stol,
                angle_tol=self.angle_tol,
                warnings=warnings,
            )
        except Exception as e:
            mg_doc = MigrationGraphDoc(
                battery_id=item["battery_id"],
                entries_for_generation=entries,
                working_ion_entry=wi_entry,
                hop_cutoff=d,
                migration_graph=None,
                populate_sc_fields=self.populate_sc_fields,
                min_length_sc=self.min_length_sc,
                minmax_num_atoms=self.minmax_num_atoms,
                ltol=self.ltol,
                stol=self.stol,
                angle_tol=self.angle_tol,
                warnings=warnings,
                deprecated=True,
            )
            self.logger.error(f"error getting MigrationGraphDoc: {e}")
            return jsanitize(mg_doc)

        return jsanitize(mg_doc.model_dump())
