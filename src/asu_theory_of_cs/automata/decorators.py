from typing import Callable, Dict, Set, Tuple

from .dfa import _DFA
from ..registry import dfa_registry

DeclareDFAArgType = Callable[
    [], Tuple[Set[str], Set[str], Dict[Tuple, str], str, Set[str]]
]


def _DeclareDFA(problem: str, render=False) -> Callable[[DeclareDFAArgType], None]:
    def inner(defintion: DeclareDFAArgType) -> None:
        dfa = _DFA(*defintion())
        if render:
            dfa.render()
        dfa_registry[problem] = dfa

    return inner
