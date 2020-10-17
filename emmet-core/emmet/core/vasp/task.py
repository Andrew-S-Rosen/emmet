""" Core definition of a VASP Task Document """
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial
from typing import ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, Field, create_model
from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer, Ordering
from pymatgen.analysis.structure_analyzer import oxide_type

from emmet.core.structure import StructureMetadata
from emmet.core.vasp.calc_types import (
    CalcType,
    RunType,
    TaskType,
    calc_type,
    run_type,
    task_type,
)
from emmet.stubs import ComputedEntry, Matrix3D, Vector3D, Structure


class Status(Enum):
    """
    VASP Calculation State
    """

    SUCESS = "successful"
    FAILED = "failed"


class InputSummary(BaseModel):
    """
    Summary of inputs for a VASP calculation
    """

    structure: Structure = Field(None, description="The input structure object")
    parameters: Dict = Field(
        None,
        description="Input parameters from VASPRUN for the last calculation in the series",
    )
    pseudo_potentials: Dict = Field(
        None, description="Summary of the pseudopotentials used in this calculation"
    )

    potcar_spec: List[Dict] = Field(
        None, description="Potcar specification as a title and hash"
    )
    parameters: Dict = Field(None, description="parameters for this VASP calculation")


class OutputSummary(BaseModel):
    """
    Summary of the outputs for a VASP calculation
    """

    structure: Structure = Field(None, description="The output structure object")
    energy: float = Field(
        None, description="The final total DFT energy for the last calculation"
    )
    energy_per_atom: float = Field(
        None, description="The final DFT energy per atom for the last calculation"
    )
    bandgap: float = Field(None, description="The DFT bandgap for the last calculation")
    forces: List[Vector3D] = Field(
        None, description="Forces on atoms from the last calculation"
    )
    stress: Matrix3D = Field(
        None, description="Stress on the unitcell from the last calculation"
    )
    


class RunStatistics(BaseModel):
    """
    Summary of the Run statistics for a VASP calculation
    """

    average_memory: float = Field(None, description="The average memory used in kb")
    max_memory: float = Field(None, description="The maximum memory used in kb")
    elapsed_time: float = Field(None, description="The real time elapsed in seconds")
    system_time: float = Field(None, description="The system CPU time in seconds")
    user_time: float = Field(
        None, description="The user CPU time spent by VASP in seconds"
    )
    total_time: float = Field(
        None, description="The total CPU time for this calculation"
    )
    cores: int = Field(None, description="The number of cores used by VASP")


class TaskDocument(BaseModel):
    """
    Definition of VASP Task Document
    """

    dir_name: str = Field(None, description="The directory for this VASP task")
    run_stats: Dict[str, RunStatistics] = Field(
        None,
        description="Summary of runtime statisitics for each calcualtion in this task",
    )
    completed_at: datetime = Field(
        None, description="Timestamp for when this task was completed"
    )
    last_updated: datetime = Field(
        None, description="Timestamp for this task document was last updateed"
    )

    is_valid: bool = Field(
        True, description="Whether this task document passed validation or not"
    )

    input: InputSummary
    output: OutputSummary

    state: Status

    orig_inputs: Dict[str, Dict] = Field(
        None, description="Summary of the original VASP inputs"
    )
    task_id: str = Field(None, description="the Task ID For this document")
    tags: List[str] = Field(None, description="Metadata tags for this task document")

    sandboxes: List[str] = Field(
        ["core"], description="List of sandboxes this task document is allowed in"
    )

    @property
    def run_type(self) -> RunType:
        return run_type(self.input.parameters)

    @property
    def task_type(self):
        return task_type(self.orig_inputs)

    @property
    def calc_type(self):
        return calc_type(self.orig_inputs, self.input.parameters)

    @property
    def entry(self):
        """ Turns a Task Doc into a ComputedEntry"""
        entry_dict = {
            "correction": 0.0,
            "entry_id": self.task_id,
            "composition": self.output.structure.composition,
            "energy": self.output.energy,
            "parameters": {
                "potcar_spec": self.input.potcar_spec,
                "run_type": self.run_type,
            },
            "data": {
                "oxide_type": oxide_type(self.output.structure),
                "last_updated": self.last_updated,
            },
        }

        return ComputedEntry.from_dict(entry_dict)
