from cse355_machine_design.automata.base import _Automaton, State
from cse355_machine_design.errors import DetailedError

from collections import defaultdict


class _PDA(_Automaton):
    """
    A pushdown automaton (PDA).
    """

    # Beyond the base automaton variables, a PDA defines a stack alphabet; an
    # empty input/stack symbol (epsilon); and a transition function that maps
    # the current state, a (possibly empty) input symbol, and a (possibly
    # empty) popped stack symbol to a (possibly empty) set of next (state,
    # pushed stack symbol) pairs.
    _stack_alphabet: set[str]
    _epsilon: str
    _transitions: dict[tuple[State, str, str], set[tuple[State, str]]]

    def __init__(
        self,
        Q: set[State],
        Sigma: set[str],
        Gamma: set[str],
        delta: dict[tuple[State, str, str], set[tuple[State, str]]],
        q0: State,
        F: set[State],
        epsilon: str = "_",
    ) -> None:
        """
        Create a new PDA and then validate it.
        """
        super().__init__("PDA", Q, Sigma, q0, F)
        self._stack_alphabet = Gamma
        self._transitions = delta
        self._epsilon = epsilon
        self.validate()

    def validate(self) -> None:
        """
        Validate this PDA according to its formal definition.
        """
        # Validate the PDA's base automaton variables.
        super().validate()

        # There should be at least one stack alphabet symbol.
        if len(self._stack_alphabet) == 0:
            raise DetailedError(
                "Empty stack alphabet",
                "Your finite automaton's stack alphabet should contain at "
                + "least one symbol, but yours is empty.",
            )

        # Symbols in the stack alphabet should be length-one strings.
        bad_symbols = [s for s in self._stack_alphabet if len(s) != 1]
        if len(bad_symbols) > 0:
            raise DetailedError(
                "Invalid stack alphabet symbol(s)",
                "Stack alphabet symbols should be individual characters, but "
                + f"these are not: {bad_symbols}.",
            )

        # The empty symbol should be a length-one string.
        if len(self._epsilon) != 1:
            raise DetailedError(
                "Invalid epsilon symbol",
                "The epsilon symbol should be an individual character, but "
                + f"'{self._epsilon}' is not.",
            )

        # The empty symbol should be outside the input alphabet.
        if self._epsilon in self._alphabet:
            raise DetailedError(
                "Epsilon symbol in input alphabet",
                f"The epsilon symbol '{self._epsilon}' should not be in the "
                + f"PDA's input alphabet, {self._alphabet}.",
            )

        # The empty symbol should be outside the stack alphabet.
        if self._epsilon in self._stack_alphabet:
            raise DetailedError(
                "Epsilon symbol in stack alphabet",
                f"The epsilon symbol '{self._epsilon}' should not be in the "
                + f"PDA's stack alphabet, {self._stack_alphabet}.",
            )

        # For each transition delta(q, a, b) = s in the PDA:
        # - q should be in the state set
        # - a should be in the input alphabet or epsilon
        # - b should be in the stack alphabet or epsilon
        # - s should be a (possibly empty) set of pairs (r, c) where:
        #   - r should be in the state set
        #   - c should be in the stack alphabet or epsilon
        err = ""
        for (q, a, b), s in self._transitions.items():
            t_err = ""
            if q not in self._states:
                t_err += f"\n- '{q}' is not in the state set {self._states}"
            if a not in self._alphabet and a != self._epsilon:
                t_err += (
                    f"\n- '{a}' is not in the input alphabet {self._alphabet} "
                    + f"nor is it the epsilon symbol '{self._epsilon}'"
                )
            if b not in self._stack_alphabet and b != self._epsilon:
                t_err += (
                    f"\n- '{b}' is not in the stack alphabet "
                    + f"{self._stack_alphabet} nor is it the epsilon symbol "
                    + f"'{self._epsilon}'"
                )
            for r, c in s:
                if r not in self._states:
                    t_err += f"\n- '{r}' is not in the state set {self._states}"
                if c not in self._stack_alphabet and c != self._epsilon:
                    t_err += (
                        f"\n- '{c}' is not in the stack alphabet "
                        + f"{self._stack_alphabet} nor is it the epsilon "
                        + f"symbol '{self._epsilon}'"
                    )

            if t_err != "":
                err += f"\nFor transition delta({q}, {a}, {b}) = {s}, {t_err}"

        if err != "":
            raise DetailedError("Invalid transition function", err)

    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this PDA.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the PDA accepts the input string.
        """
        # Validate the input string.
        if not set(input_str) <= self._alphabet:
            raise DetailedError(
                "Invalid input string",
                f"The symbols {set(input_str) - self._alphabet} in the input "
                + f"string '{input_str}' are not in the PDA's input alphabet.",
            )

        # TODO: Complete.

    def as_dict(self) -> dict:
        """
        Get a dict representation of this PDA.

        :return: A dict representation of this PDA.
        """
        dict_rep = super().as_dict()
        dict_rep["stack_alphabet"] = self._stack_alphabet
        dict_rep["transitions"] = [
            {"from": q, "input": a, "pop": b, "to": s}
            for (q, a, b), s in self._transitions.items()
        ]
        dict_rep["epsilon"] = self._epsilon

        return dict_rep

    def _as_dot_string(self) -> str:
        """
        Get a DOT string representation of this PDA for use in graphviz
        visualization.

        :return: A DOT string representation of this PDA.
        """
        # Define all states as DOT nodes.
        states_str = ""
        for state in self._states:
            shape = "doublecircle" if state in self._accept_states else "circle"
            states_str += f"{state} [shape={shape}]\n"

        # Define all transitions as DOT edges, combining transitions with the
        # same endpoints into one edge.
        combined: defaultdict[tuple[State, State], list[tuple[str, str, str]]] = (
            defaultdict(list)
        )
        for (from_state, input_sym, pop_sym), to_pairs in self._transitions.items():
            for to_state, push_sym in to_pairs:
                combined[(from_state, to_state)].append((input_sym, pop_sym, push_sym))
        edges_str = ""
        for (from_state, to_state), sym_triples in combined.items():
            label = "".join(
                [
                    f"{input_sym}, {pop_sym} -> {push_sym}\n"
                    for (input_sym, pop_sym, push_sym) in sym_triples
                ]
            ).strip()
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
