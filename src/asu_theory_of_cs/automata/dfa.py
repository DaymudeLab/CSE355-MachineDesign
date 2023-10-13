from typing import Callable, Dict, Set, Tuple
from asu_theory_of_cs.automata.base import _Automata
from itertools import product
from .. import registry


class _DFA(_Automata):
    transition_table: Dict[Tuple[str, str], str]
    final_states: Set[str]

    def __init__(self, Q, Sigma, delta, q0, F) -> None:
        self.states = Q
        self.input_symbols = Sigma
        self.transition_table = delta
        self.start_state = q0
        self.final_states = F
        super().__init__()

    def verify(self) -> None:
        transitions_valid = True

        for k in self.transition_table:
            transitions_valid &= k[0] in self.states and k[1] in self.input_symbols

        if not transitions_valid:
            raise RuntimeError(
                "Delta contains states and/or symbols not found in Q and/or Sigma"
            )

        all_entries = set(product(self.states, self.input_symbols))

        # Compute symmetric difference and check if empty
        diff = all_entries ^ set(self.transition_table.keys())
        transitions_complete = len(diff) == 0

        if not transitions_complete:
            raise RuntimeError(
                "The following state symbol pairs are not found in both the transition table and state/symbol declarations\n{}".format(
                    diff
                )
            )

        start_valid = self.start_state in self.states
        if not start_valid:
            raise RuntimeError("q_0 not found in Q")

        final_valid = self.final_states.issubset(self.states)
        if not final_valid:
            raise RuntimeError("F not a subset of in Q")

    def display(self) -> str:
        res = 'strict digraph {rankdir="LR";'
        for state in self.states:
            if state == self.start_state:
                continue
            res += '"{}" [shape={}];'.format(
                state, "doublecircle" if state in self.final_states else "circle"
            )
        for (from_state, input_symbol), to_state in self.transition_table.items():
            res += '"{}" -> "{}" [label="{}"];'.format(
                from_state, to_state, input_symbol
            )
        res += 'subgraph cluster {rankdir="LR";peripheries=0;'
        res += 'n0 [label= "", shape=none,height=.0,width=.0];'
        res += '"{}" [shape={}];'.format(
            self.start_state,
            "doublecircle" if self.start_state in self.final_states else "circle",
        )
        res += 'n0 -> "{}"'.format(self.start_state)
        res += "}}"
        return res


DeclareDFAArgType = Callable[
    [], Tuple[Set[str], Set[str], Dict[Tuple, str], str, Set[str]]
]


def _DeclareDFA(problem: str, render=False) -> Callable[[DeclareDFAArgType], None]:
    def inner(defintion: DeclareDFAArgType) -> None:
        dfa = _DFA(*defintion())
        if render:
            dfa.render()
        registry.automata_registry[problem] = dfa

    return inner
