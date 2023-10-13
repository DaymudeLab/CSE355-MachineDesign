from collections import defaultdict
from typing import Dict, Set, Tuple
from asu_theory_of_cs.automata.base import _Automata
from itertools import product


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
        res = 'digraph {rankdir="LR";ranksep=0.2;edge[minlen=3];'
        res += 'subgraph cluster {edge[minlen=default];rankdir="LR";peripheries=0;'
        res += 'n0 [label= "", shape=none,height=.0,width=.0];'
        res += '"{}" [shape={}];'.format(
            self.start_state,
            "doublecircle" if self.start_state in self.final_states else "circle",
        )
        res += 'n0 -> "{}"'.format(self.start_state)
        res += "};"
        for state in self.states:
            if state == self.start_state:
                continue
            res += '"{}" [shape={}];'.format(
                state, "doublecircle" if state in self.final_states else "circle"
            )
        tt_restructured: defaultdict[Tuple[str, str], list[str]] = defaultdict(list)

        for (from_state, input_symbol), to_state in self.transition_table.items():
            tt_restructured[(from_state, to_state)].append(input_symbol)

        for (from_state, to_state), transition_list in tt_restructured.items():
            label = "{}".format(transition_list[0])
            for i, input_symbol in enumerate(transition_list):
                if i == 0:
                    continue
                label += ","
                if i % 3 == 2:
                    label += "\n"
                label += "{}".format(transition_list[i])
            res += '"{}" -> "{}" [label="{}"];'.format(from_state, to_state, label)

        res += "}"
        return res
