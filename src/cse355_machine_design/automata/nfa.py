from collections import defaultdict
from typing import Any, DefaultDict, Dict, Set, Tuple, Union
from cse355_machine_design.automata.base import _Automata
from itertools import product
from cse355_machine_design.errors import DetailedError
from cse355_machine_design import registry
from cse355_machine_design.util import quote_str, set_str


class _NFA(_Automata):
    _transition_table: Dict[Tuple[str, str], Set[str]]
    _final_states: Set[str]
    _epsilon: str

    def __init__(
        self,
        Q: Set[str],
        Sigma: Set[str],
        delta: Dict[Tuple, Set[str]],
        q0: str,
        F: Set[str],
        epsilon_char="_",
    ) -> None:
        _delta: Dict[Tuple[str, str], Set[str]] = delta  # type: ignore
        self._states = Q
        self._input_symbols = Sigma
        self._transition_table = _delta
        self._start_state = q0
        self._final_states = F
        self._epsilon = epsilon_char
        super().__init__()

    def verify_legality(self) -> None:
        if self._epsilon in self._input_symbols:
            raise DetailedError(
                "Sigma contains epsilon",
                "You have declared your alphabet to include the symbol epsilon. This is not allowed. Either remove epsilon from the string or redefine the epsilon character by passing it as a kwarg to the NFA constructor",
            )

        transitions_valid = True
        seen_transition: set[Tuple[str, str]] = set()
        err_str = ""
        for k, v in self._transition_table.items():
            transition_current_is_state = k[0] in self._states
            transition_final_is_state = v.issubset(self._states)
            transition_input_is_alphabet = (
                k[1] in self._input_symbols or k[1] == self._epsilon
            )
            if k in seen_transition:
                err_str += f"delta has an attempt to redefine the transition from {quote_str(k[0])} while reading input {quote_str(k[1])}. If you meant to use nondeterminism, please combine the definitions\n"
            else:
                seen_transition.add(k)
            if not transition_current_is_state:
                err_str += f"delta has a transition from {quote_str(k[0])} on input {quote_str(k[1])} to the state(s) {set_str(v)}, but {quote_str(k[0])} is not in Q\n"
            if not transition_input_is_alphabet:
                err_str += f"delta has a transition from {quote_str(k[0])} on input {quote_str(k[1])} to the state(s) {set_str(v)}, but {quote_str(k[1])} is not in Sigma. If you are trying to use an epsilon transition use the character {quote_str(self._epsilon)} instead\n"
            if not transition_final_is_state:
                err_str += f"delta has a transition from {quote_str(k[0])} on input {quote_str(k[1])} to the state(s) {set_str(v)}, but the state(s) {set_str(v)} is not a subset of Q\n"

            transitions_valid &= (
                transition_current_is_state
                and transition_input_is_alphabet
                and transition_final_is_state
            )

        if not transitions_valid:
            raise DetailedError(
                "(Q, Sigma and/or delta are incorrectly defined)", err_str
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
        if not final_valid:
            raise DetailedError("(F is not a subset of Q)", err_str)

    def evaluate_one_step_no_closure(
        self, input_char, current_states: Set[str], enable_trace=0
    ) -> Set[str]:
        if not (input_char in self._input_symbols):
            raise DetailedError(
                "Encountered input character not in alphabet",
                f'The charachter "{input_char}" is not found in the alphabet {set_str(self._input_symbols)}',
            )
        if enable_trace >= 2:
            print(
                f"Now calculating new states after reading input {quote_str(input_char)}..."
            )
            print("    Mark Set: {}")
        new_states = set()
        for state in current_states:
            # Perform normal transition update
            input_char_new_states = (
                self._transition_table.get((state, input_char)) or set()
            )
            if enable_trace >= 2:
                print(
                    f"      By reading {quote_str(input_char)} from state {quote_str(state)}, we can transition to the following states: {set_str(input_char_new_states)}, so we add them to our mark set"
                )
            new_states.update(input_char_new_states)
        if enable_trace >= 2:
            print(f"    Mark Set: {set_str(new_states)}")
            print(
                f"After considering all transitions from states in {set_str(current_states)} involving {quote_str(input_char)}, the states we are in are {set_str(new_states)}"
            )
        elif enable_trace >= 1:
            print(
                f"Reading input {quote_str(input_char)}:\n    Current possible states: {set_str(new_states)}"
            )
        return new_states

    def epsilon_closure(self, current_states: Set[str], enable_trace=0) -> Set[str]:
        if enable_trace >= 2:
            print(
                "Now calculating epsilon closure of the above. We start by marking the above states"
            )
        closure = current_states
        previous_closure = set()
        first = True
        while closure != previous_closure:
            previous_closure = closure
            closure_comp = closure.copy()
            closure_iter = closure.copy()
            if enable_trace >= 2 and first:
                print(f"    Marked states: {set_str(previous_closure)}")
                first = False
            elif enable_trace >= 2:
                print(
                    "    We just marked some new states, so we must perform the closure again"
                )
            for state in closure_iter:
                epsilon_closure = (
                    self._transition_table.get((state, self._epsilon)) or set()
                )
                if enable_trace >= 2:
                    if len(epsilon_closure) == 0:
                        print(
                            f"      There are no epsilon transitions from {quote_str(state)}, so we dont add anything to our mark set"
                        )
                    else:
                        print(
                            f"      By taking an epsilon transition from state {quote_str(state)}, we can get to any of the following states: {set_str(epsilon_closure)}, so we mark them all if we haven't already"
                        )
                closure_comp.update(epsilon_closure)
            closure = closure_comp
            if enable_trace >= 2 and (not first):
                print(f"    Marked states: {set_str(closure)}")

        if enable_trace >= 2 and first:
            print(
                "We were in no states (empty set), so the closure is also the empty set"
            )
        elif enable_trace >= 2:
            print(
                "Our previous two marked state sets are equal, so we know we are done with calculating the closure"
            )
            print(f"The states we are now in are: {set_str(closure)}")
        elif enable_trace >= 1:
            print(f"    After epsilon closure: {set_str(closure)}")
        return closure

    def evaluate(self, input_str: str, enable_trace=0) -> Union[bool, None]:
        if enable_trace > 0:
            print(f"Starting in states {set_str(set([self._start_state]))}")

        current_states: Set[str] = self.epsilon_closure(
            set([self._start_state]), enable_trace
        )
        if enable_trace >= 2:
            print()
        for input_char in input_str:
            one_step = self.evaluate_one_step_no_closure(
                input_char, current_states, enable_trace
            )
            current_states = self.epsilon_closure(one_step, enable_trace)
            if enable_trace >= 2:
                print()

        is_final = False
        if enable_trace > 0:
            print("Done reading input...\n")
            print("Checking if at least one of the current states is a final state...")
        for state in current_states:
            if state in self._final_states:
                if enable_trace > 0:
                    print(
                        f"{quote_str(state)} IS a final state. We can stop looking and accept"
                    )
                is_final = True
                break
            elif enable_trace >= 2:
                print(f"{quote_str(state)} IS NOT a final state")

        if (not is_final) and enable_trace > 0:
            print(
                "None of the states we were in after computation are final states, so we reject."
            )
        return is_final

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

        for (
            from_state,
            input_symbol,
        ), to_state_collection in self._transition_table.items():
            for to_state in to_state_collection:
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

    def submit_as_answer(self, question_number: int) -> None:
        registry.add_to_registry("nfa", question_number, self)

    def export_to_dict(self) -> dict:
        rep = {
            "type": "nfa",
            "states": list(self._states),
            "input_symbols": list(self._input_symbols),
            "finals": list(self._final_states),
            "start_state": self._start_state,
            "delta": list(
                map(
                    lambda p: {"from": p[0][0], "to": list(p[1]), "input": p[0][1]},
                    self._transition_table.items(),
                )
            ),
            "epsilon_char": self._epsilon,
        }
        return rep
