from cse355_machine_design.automata.base import _Automaton, State
from cse355_machine_design.errors import DetailedError

from collections import defaultdict
from itertools import product


class _DFA(_Automaton):
    """
    A deterministic finite automaton (DFA).
    """

    # Beyond the base automaton variables, a DFA defines a transition function
    # that maps the current state and an input symbol to the next state.
    _transitions: dict[tuple[State, str], State]

    def __init__(
        self,
        Q: set[State],
        Sigma: set[str],
        delta: dict[tuple[State, str], State],
        q0: State,
        F: set[State],
    ) -> None:
        """
        Create a new DFA and then validate it.
        """
        super().__init__("DFA", Q, Sigma, q0, F)
        self._transitions = delta
        self.validate()

    def validate(self) -> None:
        """
        Validate this DFA according to its formal definition.
        """
        # Validate the DFA's base automaton variables.
        super().validate()

        # For each transition delta(q, x) = r in the DFA:
        # - q should be in the state set
        # - x should be in the alphabet
        # - r should be in the set
        err = ""
        for (q, x), r in self._transitions.items():
            qxr_err = ""
            if q not in self._states:
                qxr_err += f"\n- '{q}' is not in the state set {self._states}"
            if x not in self._alphabet:
                qxr_err += f"\n- '{x}' is not in the alphabet {self._alphabet}"
            if r not in self._states:
                qxr_err += f"\n- '{r}' is not in the state set {self._states}"

            if qxr_err != "":
                err += f"\nFor transition delta({q}, {x}) = {r}, {qxr_err}"

        # A DFA's transition function must also cover every state-symbol pair.
        for q, x in product(self._states, self._alphabet):
            if not self._transitions.get((q, x)):
                err += f"\nMissing transition delta({q}, {x})"

        if err != "":
            raise DetailedError("Invalid transition function", err)

    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this DFA.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the DFA accepts the input string.
        """
        # Validate the input string.
        if not set(input_str) <= self._alphabet:
            raise DetailedError(
                "Invalid input string",
                f"The symbols {set(input_str) - self._alphabet} in the input "
                + f"string '{input_str}' are not in the DFA's alphabet.",
            )

        # Computation starts from the start state.
        if trace:
            print(
                f"Evaluating input '{input_str}' from start state "
                + f"'{self._start_state}'..."
            )
        current_state = self._start_state

        # Trace through the input string one symbol at a time.
        for input_sym in input_str:
            next_state = self._transitions[(current_state, input_sym)]
            if trace:
                print(
                    f"Read input '{input_sym}'; transition states "
                    + f"{current_state} -> {next_state}"
                )
            current_state = next_state

        # Input exhausted; determine accept/reject decision.
        if trace:
            print(f"Done reading input; currently in state {current_state}")
        if current_state in self._accept_states:
            if trace:
                print(f"State {current_state} is accepting, so ACCEPT")
            return True
        else:
            if trace:
                print(f"State {current_state} is not accepting, so REJECT")
            return False

    def as_dict(self) -> dict:
        """
        Get a dict representation of this DFA.

        :return: A dict representation of this DFA.
        """
        dict_rep = super().as_dict()
        dict_rep["transitions"] = [
            {"from": q, "input": x, "to": r} for (q, x), r in self._transitions.items()
        ]

        return dict_rep

    def _as_dot_string(self) -> str:
        """
        Get a DOT string representation of this DFA for use in graphviz
        visualization.

        :return: A DOT string representation of this DFA.
        """
        # Define all states as DOT nodes.
        states_str = ""
        for state in self._states:
            shape = "doublecircle" if state in self._accept_states else "circle"
            states_str += f"{state} [shape={shape}]\n"

        # Define all transitions as DOT edges, combining transitions with the
        # same endpoints into one edge.
        combined: defaultdict[tuple[State, State], list[str]] = defaultdict(list)
        for (from_state, input_sym), to_state in self._transitions.items():
            combined[(from_state, to_state)].append(input_sym)
        edges_str = ""
        for (from_state, to_state), input_syms in combined.items():
            label = "".join([input_sym + ", " for input_sym in input_syms])[:-2]
            edges_str += f"{from_state} -> {to_state} [label={label}];\n"

        return f"""\
            strict digraph {{
                rankdir="LR";    // Direct graph from left to right.
                ranksep=0.2;     // Minimum distance between ranks.
                edge [minlen=3]; // Minimum edge length.

                // State nodes.
                {states_str.strip()}

                // Incoming arrow for the start state.
                nowhere [label="", shape=none, width=0, height=0];
                nowhere -> {self._start_state} [minlen=default];

                // Transition edges.
                {edges_str.strip()}
            }}\
        """
