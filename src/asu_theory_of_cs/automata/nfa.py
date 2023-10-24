from collections import defaultdict
from typing import DefaultDict, Dict, Set, Tuple, Union
from asu_theory_of_cs.automata.base import _Automata
from itertools import product
from asu_theory_of_cs.errors import DetailedError
from asu_theory_of_cs import registry
from asu_theory_of_cs.util import quote_str, set_str


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
        pass

    def evaluate_one_step_no_closure(
        self, input_char, current_states: Set[str], enable_trace=False
    ) -> Set[str]:
        if enable_trace:
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
            if enable_trace:
                print(
                    f"      By reading {quote_str(input_char)} from state {quote_str(state)}, we can transition to the following states: {set_str(input_char_new_states)}, so we add them to our mark set"
                )
            new_states.update(input_char_new_states)
        if enable_trace:
            print(f"    Mark Set: {set_str(new_states)}")
            print(
                f"After considering all transitions from states in {set_str(current_states)} involving {quote_str(input_char)}, the states we are in are {set_str(new_states)}"
            )
        return new_states

    def epsilon_closure(self, current_states: Set[str], enable_trace=False) -> Set[str]:
        if enable_trace:
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
            if enable_trace and first:
                print(f"    Marked states: {set_str(previous_closure)}")
                first = False
            elif enable_trace:
                print(
                    "    We just marked some new states, so we must perform the closure again"
                )
            for state in closure_iter:
                epsilon_closure = (
                    self._transition_table.get((state, self._epsilon)) or set()
                )
                if enable_trace:
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
            if enable_trace and (not first):
                print(f"    Marked states: {set_str(closure)}")

        if enable_trace and first:
            print(
                "We were in no states (empty set), so the closure is also the empty set"
            )
        elif enable_trace:
            print(
                "Our previous two marked state sets are equal, so we know we are done with calculating the closure"
            )
            print(f"The states we are now in are: {set_str(closure)}")
        return closure

    def evaluate(self, input_str: str, enable_trace=False) -> Union[bool, None]:
        if enable_trace:
            print(f"Starting in states {set_str(set([self._start_state]))}")

        current_states: Set[str] = self.epsilon_closure(
            set([self._start_state]), enable_trace
        )
        if enable_trace:
            print()
        for input_char in input_str:
            one_step = self.evaluate_one_step_no_closure(
                input_char, current_states, enable_trace
            )
            current_states = self.epsilon_closure(one_step, enable_trace)
            if enable_trace:
                print()

        is_final = False
        if enable_trace:
            print("Done reading input...\n")
            print("Checking if at least one of the current states is a final state...")
        for state in current_states:
            if state in self._final_states:
                if enable_trace:
                    print(
                        f"{quote_str(state)} IS a final state. We can stop looking and accept"
                    )
                is_final = True
                break
            elif enable_trace:
                print(f"{quote_str(state)} IS NOT a final state")

        if (not is_final) and enable_trace:
            print(
                "None of the states we were in after computation are final states, so we reject."
            )
        return is_final

    def _generate_dot_string(self) -> str:
        return super()._generate_dot_string()

    def submit_as_answer(self, question_number: int) -> None:
        registry.add_to_registry("nfa", question_number, self)
