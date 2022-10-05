from typing import *
from dataclasses import dataclass, field


@dataclass
class ProgramNode:
    step: int
    function: str
    value_input: Optional[str]
    inputs: Sequence[int]
    _outputs: Optional[Sequence[int]] = field(default=None)


Program = Sequence[ProgramNode]

