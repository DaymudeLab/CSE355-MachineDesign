from __future__ import annotations
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing import Dict, TypedDict, KeysView, ValuesView

    from .. import automata
    from ..automata import base as automata_base

    class _Registry(TypedDict):
        dfa: Dict[str, automata.DFA]
        nfa: Dict[str, automata.NFA]


_registry: _Registry = {"dfa": {}, "nfa": {}}


def add_to_registry(
    automata_type: str, question_number: int, submission: automata_base._Automata
) -> None:
    _registry[automata_type][f"Question_{question_number}"] = submission


def get_dfa(question_number: int) -> Union[automata.DFA, None]:
    return _registry["dfa"].get(f"Question_{question_number}")


def get_nfa(question_number: int) -> Union[automata.NFA, None]:
    return _registry["nfa"].get(f"Question_{question_number}")
