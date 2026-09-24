from cse355_machine_design.automata.base import _Automaton, State
from cse355_machine_design.automata import CFG
from cse355_machine_design.errors import DetailedError

from collections import defaultdict
from itertools import product

from typeguard import typechecked


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

    @typechecked
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
        if self._epsilon in self._input_alphabet:
            raise DetailedError(
                "Epsilon symbol in input alphabet",
                f"The epsilon symbol '{self._epsilon}' should not be in the "
                + f"PDA's input alphabet, {self._input_alphabet}.",
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
            if a not in self._input_alphabet and a != self._epsilon:
                t_err += (
                    f"\n- '{a}' is not in the input alphabet "
                    + f"{self._input_alphabet} nor is it the epsilon symbol "
                    + f"'{self._epsilon}'"
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

    @typechecked
    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this PDA.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the PDA accepts the input string.
        """
        # Validate the input string.
        if not set(input_str) <= self._input_alphabet:
            raise DetailedError(
                "Invalid input string",
                f"The symbols {set(input_str) - self._input_alphabet} in the "
                + f"input string '{input_str}' are not in the PDA's input "
                + "alphabet.",
            )

        # TODO: Honor tracing parameter.
        return self._as_cfg().generates_string(input_str)

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

    def _as_cfg(self) -> CFG:
        """
        Transform this PDA into an equivalent CFG.

        Details of this transformation can be found in Sipser (3rd ed., 2013),
        Lemma 2.27. Note that this is somewhat different than the approach in
        Hopcroft, Motwani, and Ullman (3rd ed., 2006), Section 6.3.2.
        """
        # Verify that existing PDA states can't conflict with upcoming changes.
        conflict_states = [s for s in self._states if "cfg" in s]
        if len(conflict_states) > 0:
            raise DetailedError(
                "Conflicting state names",
                "The PDA -> CFG conversion modifies this PDA with new states "
                + "containing the string 'cfg', but this may conflict with "
                + f"existing states: {conflict_states}.",
            )

        # First, modify this PDA so it has a single accept state.
        mod_accept_states: set[State] = {"q_cfg_final"}
        mod_states = self._states | mod_accept_states
        mod_transitions: dict[tuple[State, str, str], set[tuple[State, str]]] = (
            defaultdict(set)
        )
        for accept_state in self._accept_states:
            mod_transitions[(accept_state, self._epsilon, self._epsilon)] = {
                ("q_cfg_final", self._epsilon)
            }

        # Next, ensure any computation empties its stack before accepting.
        for stack_sym in self._stack_alphabet:
            mod_transitions[("q_cfg_final", self._epsilon, stack_sym)] = {
                ("q_cfg_final", self._epsilon)
            }

        # Finally, ensure that every transition either pushes a symbol onto the
        # stack or pops a symbol from the stack, but not both simultaneously.
        aux_state_ix = 0
        dummy_sym = next(iter(self._stack_alphabet))
        for (q_from, input_sym, pop_sym), to_pairs in self._transitions.items():
            for q_to, push_sym in to_pairs:
                # Replace q_from (read a, pop b, push c) -> q_to with:
                # 1. q_from -- (read a, pop b, push nothing) -> q_new
                # 2. q_new -- (read nothing, pop nothing, push c) -> q_to
                if pop_sym != self._epsilon and push_sym != self._epsilon:
                    mod_transitions[(q_from, input_sym, pop_sym)].add(
                        (f"q_cfg_{aux_state_ix}", self._epsilon)
                    )
                    mod_transitions[
                        (f"q_cfg_{aux_state_ix}", self._epsilon, self._epsilon)
                    ].add((q_to, push_sym))
                    aux_state_ix += 1
                # Replace q_from (read a, pop nothing, push nothing) -> q_to with:
                # 1. q_from -- (read a, pop nothing, push dummy) -> q_new
                # 2. q_new -- (read nothing, pop dummy, push nothing) -> q_to
                elif pop_sym == self._epsilon and push_sym == self._epsilon:
                    mod_transitions[(q_from, input_sym, self._epsilon)].add(
                        (f"q_cfg_{aux_state_ix}", dummy_sym)
                    )
                    mod_transitions[
                        (f"q_cfg_{aux_state_ix}", self._epsilon, dummy_sym)
                    ].add((q_to, self._epsilon))
                    aux_state_ix += 1
                else:
                    mod_transitions[(q_from, input_sym, pop_sym)].add((q_to, push_sym))

        # The equivalent CFG's variables are PDA states pairs (p,q). Variable
        # (p,q) will generate all strings that take the PDA from state p to
        # state q while leaving the stack at state q exactly as it was in state
        # p. The CFG's start variable is thus (q_0,q_accept).
        cfg_variables = {f"({p},{q})" for (p, q) in product(self._states, repeat=2)}
        cfg_start_variable = f"({self._start_state},q_cfg_final)"

        # There are three types of rules in the equivalent CFG:
        # 1. For each state p in the PDA, (p,p) -> epsilon, i.e., it is always
        #    possible to transition from a state to itself with no input.
        # 2. For each triple of states p, q, r in the PDA, (p,q) -> (p,r)(r,q),
        #    i.e., one way of transitioning from p to q with the same start/end
        #    stack configuration is first transitioning from p to r with that
        #    stack configuration and then transitioning from r to q.
        # 3. For each (p, q, r, s) of PDA states, stack symbol u, and (possibly
        #    empty) input symbols a and b, if delta(p, a, epsilon) contains
        #    (r, u) and delta(s, b, u) contains (q, epsilon), (p,q) -> a(r,s)b.
        #    That is, one way of transitioning from p to q with the same start/
        #    end stack configuration is to (1) transition from p to r, reading
        #    a and pushing u; (2) transition from r to s with the same start/
        #    end stack configuration, and finally (3) transition from s to q,
        #    reading b and popping u.
        cfg_rules: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for p in mod_states:
            cfg_rules[f"({p},{p})"].add(tuple(self._epsilon))
            for q, r in product(mod_states, repeat=2):
                cfg_rules[f"({p},{q})"].add((f"({p},{r})", f"({r},{q})"))
                for s, u in product(mod_states, self._stack_alphabet):
                    for a, b in product(
                        self._input_alphabet | {self._epsilon}, repeat=2
                    ):
                        if (r, u) in mod_transitions[(p, a, self._epsilon)] and (
                            q,
                            self._epsilon,
                        ) in mod_transitions[(s, b, u)]:
                            cfg_rules[f"({p},{q})"].add((a, f"({r},{s})", b))

        return CFG(
            V=cfg_variables,
            Sigma=self._input_alphabet,
            R=cfg_rules,
            S=cfg_start_variable,
            epsilon=self._epsilon,
        )
