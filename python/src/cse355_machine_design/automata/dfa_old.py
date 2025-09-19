from collections import defaultdict
from typing import Dict, Set, Tuple, Union
from cse355_machine_design.automata.base import _Automata
from itertools import product
from cse355_machine_design.errors import DetailedError
from cse355_machine_design import registry


class _DFA(_Automata):
    _transition_table: Dict[Tuple[str, str], str]
    _final_states: Set[str]

    def __init__(
        self,
        Q: Set[str],
        Sigma: Set[str],
        delta: Dict[Tuple, str],
        q0: str,
        F: Set[str],
    ) -> None:
        _delta: Dict[Tuple[str, str], str] = delta  # type: ignore
        self._states = Q
        self._input_symbols = Sigma
        self._transition_table = _delta
        self._start_state = q0
        self._final_states = F
        super().__init__()

    def verify_legality(self) -> None:
        transitions_valid = True
        seen_transition: set[Tuple[str, str]] = set()
        err_str = ""
        for k, v in self._transition_table.items():
            transition_current_is_state = k[0] in self._states
            transition_final_is_state = v in self._states
            transition_input_is_alphabet = k[1] in self._input_symbols
            if k in seen_transition:
                err_str += f'delta has an attempt to redefine the transition from "{k[0]}" while reading input "{k[1]}"\n'
            else:
                seen_transition.add(k)
            if not transition_current_is_state:
                err_str += f'delta has a transition from "{k[0]}" on input "{k[1]}" to "{v}", but "{k[0]}" is not in Q\n'
            if not transition_input_is_alphabet:
                err_str += f'delta has a transition from "{k[0]}" on input "{k[1]}" to "{v}", but "{k[1]}" is not in Sigma\n'
            if not transition_final_is_state:
                err_str += f'delta has a transition from "{k[0]}" on input "{k[1]}" to "{v}", but "{v}" is not in Q\n'

            transitions_valid &= (
                transition_current_is_state
                and transition_input_is_alphabet
                and transition_final_is_state
            )

        if not transitions_valid:
            raise DetailedError(
                "(Q, Sigma and/or delta are incorrectly defined)", err_str
            )

        err_str = ""

        all_entries = set(product(self._states, self._input_symbols))

        # check all entries to make sure transition table is complete
        transitions_complete = True
        for necessary_transition in all_entries:
            transition_found = necessary_transition in self._transition_table.keys()
            if not transition_found:
                err_str += f'Based on Q,Sigma, we need a transition from "{necessary_transition[0]}" to some other state on input symbol "{necessary_transition[1]}", but this transition was not found\n'
            transitions_complete &= transition_found

        if not transitions_complete:
            raise DetailedError(
                "(Based on the definitions of Q and Sigma, it is the case that delta is missing transitions)",
                err_str,
            )

        start_valid = self._start_state in self._states
        if not start_valid:
            raise DetailedError(
                "(Start state q_0 not found in Q)",
                f'"{self._start_state}" was not found in Q',
            )

        err_str = ""
        final_valid = True
        for potential_state in self._final_states:
            potential_state_valid = potential_state in self._states
            if not potential_state_valid:
                err_str += f'"{potential_state}" was declared as a final state, but it is not found in Q\n'
            final_valid &= potential_state_valid
        final_valid = self._final_states.issubset(self._states)
        if not final_valid:
            raise DetailedError("(F is not a subset of Q)", err_str)

    def evaluate_one_step(
        self, input_char: str, current_state: str, enable_trace=False
    ) -> str:
        new_state = self._transition_table[(current_state, input_char)]
        if enable_trace:
            print(
                f'Reading input "{input_char}". This causes us to transition from "{current_state}" to "{new_state}"'
            )
        return new_state

    def evaluate(self, input_str: str, enable_trace=False) -> Union[bool, None]:
        if enable_trace:
            print(f'Starting at state "{self._start_state}"')
        current_state = self._start_state
        for input_char in input_str:
            current_state = self.evaluate_one_step(
                input_char, current_state, enable_trace
            )
        is_final = current_state in self._final_states
        accept_str = ("", "accept") if is_final else (" not", "reject")
        if enable_trace:
            print(
                f'Finished reading input. We are now in state "{current_state}". This is{accept_str[0]} a final state, so we {accept_str[1]}'
            )
        return current_state in self._final_states

    def _generate_dot_string(self) -> str:
        res = 'digraph {rankdir="LR";ranksep=0.2;edge[minlen=3];'
        res += 'subgraph cluster {rank=same;edge[minlen=default];rankdir="LR";peripheries=0;'
        res += 'n0 [label="", shape=none, width=0, height=0];'
        res += '"{}" [shape={}];'.format(
            self._start_state,
            "doublecircle" if self._start_state in self._final_states else "circle",
        )
        res += 'n0 -> "{}"'.format(self._start_state)
        res += "};"
        for state in self._states:
            if state == self._start_state:
                continue
            res += '"{}" [shape={}];'.format(
                state, "doublecircle" if state in self._final_states else "circle"
            )
        tt_restructured: defaultdict[Tuple[str, str], list[str]] = defaultdict(list)

        for (from_state, input_symbol), to_state in self._transition_table.items():
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

    def submit_as_answer(self, question_number: int):
        registry.add_to_registry("dfa", question_number, self)
