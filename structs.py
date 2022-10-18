from typing import *
from dataclasses import dataclass, field

from torch import Tensor 


@dataclass
class ProgramNode:
    step: int
    function: str
    inputs: Sequence[int]
    concept_input: Optional[str] = field(default=None)
    value_input: Optional[str] = field(default=None)
    _outputs: Optional[Sequence[int]]  = field(default=None)

    def __repr__(self):
        return f"({str(self.step)}): {self.function}{'{'+self.concept_input+'}' if self.concept_input is not None else ''}{'['+self.value_input+']' if self.value_input is not None else ''}({','.join(list(map(str, self.inputs))) if self.inputs else ''})"


Program = Sequence[ProgramNode]