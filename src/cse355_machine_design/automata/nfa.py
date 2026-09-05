from cse355_machine_design.automata.base import _Automaton, State
from cse355_machine_design.errors import DetailedError

from collections import defaultdict


class _NFA(_Automaton):
    """
    A nondeterministic finite automaton (NFA).
    """

    # Beyond the base automaton variables, an NFA defines an empty input symbol
    # (epsilon) for transitions that do not consume real input symbols and a
    # transition function that maps the current state and a (possibly empty)
    # input symbol to a (possibly empty) set of next states.
    _epsilon: str
    _transitions: dict[tuple[State, str], set[State]]

    def __init__(
        self,
        Q: set[State],
        Sigma: set[str],
        delta: dict[tuple[State, str], set[State]],
        q0: State,
        F: set[State],
        epsilon: str = "_",
    ) -> None:
        """
        Create a new NFA and then validate it.
        """
        super().__init__("NFA", Q, Sigma, q0, F)
        self._transitions = delta
        self._epsilon = epsilon
        self.validate()

    def validate(self) -> None:
        """
        Validate this NFA according to its formal definition.
        """
        # Validate the NFA's base automaton variables.
        super().validate()

        # The empty symbol should be a length-one string.
        if len(self._epsilon) != 1:
            raise DetailedError(
                "Invalid epsilon symbol",
                "The epsilon symbol should be an individual character, but "
                + f"'{self._epsilon}' is not.",
            )

        # The empty symbol should be outside the alphabet.
        if self._epsilon in self._alphabet:
            raise DetailedError(
                "Epsilon symbol in alphabet",
                f"The epsilon symbol '{self._epsilon}' should not be in the "
                + f"NFA's alphabet, {self._alphabet}.",
            )

        # For each transition delta(q, x) = s in the NFA:
        # - q should be in the state set
        # - x should be in the alphabet or epsilon
        # - s should be a (possibly empty) subset of the state set
        err = ""
        for (q, x), s in self._transitions.items():
            qxs_err = ""
            if q not in self._states:
                qxs_err += f"\n- '{q}' is not in the state set {self._states}"
            if x not in self._alphabet and x != self._epsilon:
                qxs_err += f"\n- '{x}' is not in the alphabet {self._alphabet}"
                qxs_err += f" nor is it the epsilon symbol '{self._epsilon}'"
            if not s <= self._states:
                qxs_err += f"\n- {s - self._states} are not in the state set"

            if qxs_err != "":
                err += f"\nFor transition delta({q}, {x}) = {s}, {qxs_err}"

        if err != "":
            raise DetailedError("Invalid transition function", err)

    def epsilon_closure(self, states: set[State], trace: bool = False) -> set[State]:
        """
        Compute the epsilon closure of the given set of states.
        """
        # Initialize the closure, the set of states to check for epsilon
        # transitions, and the set of states already checked.
        closure: set[State] = states.copy()
        states_to_check: set[State] = states.copy()
        states_checked: set[State] = set()

        # While there is still a state q to check, add delta(q, epsilon) to the
        # closure and delta(q, epsilon) \ states_checked to states_to_check.
        while len(states_to_check) > 0:
            q = states_to_check.pop()
            states_checked.add(q)
            epsilon_nbrs = self._transitions.get((q, self._epsilon)) or set()
            closure |= epsilon_nbrs
            states_to_check |= epsilon_nbrs - states_checked

        if trace:
            print(
                f"Transition states {states} -> {closure} following zero or "
                + "more epsilon transitions"
            )

        return closure

    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this NFA.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the NFA accepts the input string.
        """
        # Validate the input string.
        if not set(input_str) <= self._alphabet:
            raise DetailedError(
                "Invalid input string",
                f"The symbols {set(input_str) - self._alphabet} in the input "
                + f"string '{input_str}' are not in the NFA's alphabet.",
            )

        # Computation starts from the epsilon closure of the start state.
        if trace:
            print(
                f"Evaluating input '{input_str}' from start state "
                + f"'{self._start_state}'..."
            )
        current_states = self.epsilon_closure({self._start_state}, trace)

        # Trace through the input string one symbol at a time.
        for input_char in input_str:
            next_states = set()
            for q in current_states:
                next_states |= self._transitions.get((q, input_char)) or set()
            if trace:
                print(
                    f"Transition states {current_states} -> {next_states} "
                    + f"following exactly one '{input_char}' transition"
                )
            current_states = self.epsilon_closure(next_states, trace)

        # Input exhausted; determine accept/reject decision.
        if trace:
            print(f"Done reading input; currently in states {current_states}")
        reached_accept_states = current_states & self._accept_states
        if len(reached_accept_states) > 0:
            if trace:
                print(f"Reached accepting states {reached_accept_states}, so ACCEPT")
            return True
        else:
            if trace:
                print(f"None of {current_states} are accepting, so REJECT")
            return False

    def as_dict(self) -> dict:
        """
        Get a dict representation of this NFA.

        :return: A dict representation of this NFA.
        """
        dict_rep = super().as_dict()
        dict_rep["transitions"] = [
            {"from": q, "input": x, "to": s} for (q, x), s in self._transitions.items()
        ]
        dict_rep["epsilon"] = self._epsilon

        return dict_rep

    def _as_dot_string(self) -> str:
        """
        Get a DOT string representation of this NFA for use in graphviz
        visualization.

        :return: A DOT string representation of this NFA.
        """
        # Define all states as DOT nodes.
        states_str = ""
        for state in self._states:
            shape = "doublecircle" if state in self._accept_states else "circle"
            states_str += f"{state} [shape={shape}]\n"

        # Define all transitions as DOT edges, combining transitions with the
        # same endpoints into one edge.
        combined: defaultdict[tuple[State, State], list[str]] = defaultdict(list)
        for (from_state, input_char), to_states in self._transitions.items():
            for to_state in to_states:
                combined[(from_state, to_state)].append(input_char)
        edges_str = ""
        for (from_state, to_state), input_chars in combined.items():
            label = "".join([input_char + ", " for input_char in input_chars])[:-2]
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
