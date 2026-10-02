from cse355_machine_design.automata.base import _Automaton, State

from collections import defaultdict, deque
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

        # For each transition delta(q, a) = r in the DFA:
        # - q should be in the state set
        # - a should be in the input alphabet
        # - r should be in the set
        err = ""
        for (q, a), r in self._transitions.items():
            t_err = ""
            if q not in self._states:
                t_err += f"\n- '{q}' is not in the state set {self._states}"
            if a not in self._input_alphabet:
                t_err += (
                    f"\n- '{a}' is not in the input alphabet "
                    + f"{self._input_alphabet}"
                )
            if r not in self._states:
                t_err += f"\n- '{r}' is not in the state set {self._states}"

            if t_err != "":
                err += f"\nFor transition delta({q}, {a}) = {r}, {t_err}"

        # A DFA's transition function must also cover every state-symbol pair.
        for q, a in product(self._states, self._input_alphabet):
            if not self._transitions.get((q, a)):
                err += f"\nMissing transition delta({q}, {a})"

        if err != "":
            raise ValueError("Invalid transition function:" + err)

    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this DFA.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the DFA accepts the input string.
        """
        # Validate the input string.
        if not set(input_str) <= self._input_alphabet:
            raise ValueError(
                f"Invalid input string. The input '{input_str}' contains "
                + f"symbols {set(input_str) - self._input_alphabet} that are "
                + "not in the DFA's input alphabet."
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

    def _reachable_states(self) -> set[State]:
        """
        Collect all states that are reachable from the start state.
        """
        # For each state, find all states reachable by a single transition.
        out_states: dict[State, set[State]] = {
            from_state: {
                self._transitions[(from_state, input_sym)]
                for input_sym in self._input_alphabet
            }
            for from_state in self._states
        }

        # Explore reachable states by BFS from the start state.
        visited_states: set[State] = set()
        states_to_explore: deque[State] = deque([self._start_state])
        while len(states_to_explore) > 0:
            state = states_to_explore.popleft()
            visited_states.add(state)
            states_to_explore.extend(out_states[state] - visited_states)

        return visited_states

    def empty(self) -> bool:
        """
        Return True iff the language of this DFA is empty.
        """
        return self._accept_states.isdisjoint(self._reachable_states())

    def prune_unreachable(self) -> None:
        """
        Remove all states that are unreachable from the start state and the
        transitions containing them.
        """
        self._states = self._reachable_states()
        self._accept_states &= self._states
        for from_state, input_sym in list(self._transitions):
            if from_state not in self._states:
                del self._transitions[(from_state, input_sym)]

    def complement(self) -> "_DFA":
        """
        Construct a DFA recognizing the complement of this DFA's language.
        """
        # Note that because the transition function for a DFA maps (State, str)
        # tuples to States and States are strs, a shallow copy is safe.
        return _DFA(
            self._states.copy(),
            self._input_alphabet.copy(),
            self._transitions.copy(),
            self._start_state,
            self._states - self._accept_states,
        )

    def __invert__(self) -> "_DFA":
        """
        Operator override for complementation ~ DFA; see complement().
        """
        return self.complement()

    def _product(self, other: "_DFA") -> "_DFA":
        """
        Helper function for union() and intersection() that instruments the
        product construction of two DFAs, leaving the definition of the product
        DFA's accept states to the calling function.
        """
        # Validate the DFAs' input alphabets.
        if self._input_alphabet != other._input_alphabet:
            raise ValueError(
                "Mismatched input alphabets. Cannot apply the product "
                + "construction to two DFAs with different alphabets: "
                + f"{self._input_alphabet} != {other._input_alphabet}."
            )

        # Perform the product construction without defining the accept states.
        Q: set[State] = {
            f"({self_state},{other_state})"
            for self_state, other_state in product(self._states, other._states)
        }
        Sigma: set[str] = self._input_alphabet.copy()
        delta: dict[tuple[State, str], State] = {
            (
                f"({self_state},{other_state})",
                input_sym,
            ): f"({self._transitions[(self_state, input_sym)]}"
            + f",{other._transitions[(other_state, input_sym)]})"
            for self_state, other_state, input_sym in product(
                self._states, other._states, self._input_alphabet
            )
        }
        q0: State = f"({self._start_state},{other._start_state})"
        F: set[State] = set()

        return _DFA(Q, Sigma, delta, q0, F)

    def union(self, other: "_DFA", prune_unreachable: bool = True) -> "_DFA":
        """
        Construct a DFA recognizing the union of this and the other DFAs'
        languages using the product construction.
        """
        # Validate the DFAs' input alphabets.
        if self._input_alphabet != other._input_alphabet:
            raise ValueError(
                "Mismatched input alphabets. Cannot construct a union of two "
                + "DFAs with different alphabets: "
                + f"{self._input_alphabet} != {other._input_alphabet}."
            )

        # Construct the union via the product construction.
        D = self._product(other)
        for self_state, other_state in product(self._states, other._states):
            if self_state in self._accept_states or other_state in other._accept_states:
                D._accept_states.add(f"({self_state},{other_state})")

        # Prune unreachable states if requested.
        if prune_unreachable:
            D.prune_unreachable()

        return D

    def __or__(self, other: "_DFA") -> "_DFA":
        """
        Operator override for union DFA1 | DFA2. Note that using this operator
        automatically prunes unreachable states in the resulting DFA; to retain
        unreachable states, see union().
        """
        return self.union(other)

    def intersection(self, other: "_DFA", prune_unreachable: bool = True) -> "_DFA":
        """
        Construct a DFA recognizing the intersection of this and the other
        DFAs' languages using the product construction.
        """
        # Validate the DFAs' input alphabets.
        if self._input_alphabet != other._input_alphabet:
            raise ValueError(
                "Mismatched input alphabets. Cannot construct an intersection "
                + "of two DFAs with different alphabets: "
                + f"{self._input_alphabet} != {other._input_alphabet}."
            )

        # Construct the intersection via the product construction.
        D = self._product(other)
        D._accept_states = {
            f"({self_state},{other_state})"
            for self_state, other_state in product(
                self._accept_states, other._accept_states
            )
        }

        # Prune unreachable states if requested.
        if prune_unreachable:
            D.prune_unreachable()

        return D

    def __and__(self, other: "_DFA") -> "_DFA":
        """
        Operator override for intersection DFA1 & DFA2. Note that using this
        operator automatically prunes unreachable states in the resulting DFA;
        to retain unreachable states, see intersection().
        """
        return self.intersection(other)

    def difference(self, other: "_DFA", prune_unreachable: bool = True) -> "_DFA":
        """
        Construct a DFA recognizing the difference of this and the other DFAs'
        languages using complement and intersection.
        """
        # Validate the DFAs' input alphabets.
        if self._input_alphabet != other._input_alphabet:
            raise ValueError(
                "Mismatched input alphabets. Cannot construct a difference of "
                + "two DFAs with different alphabets: "
                + f"{self._input_alphabet} != {other._input_alphabet}."
            )

        return self.intersection(other.complement(), prune_unreachable)

    def __sub__(self, other: "_DFA") -> "_DFA":
        """
        Operator override for difference DFA1 - DFA2. Note that using this
        operator automatically prunes unreachable states in the resulting DFA;
        to retain unreachable states, see difference().
        """
        return self.difference(other)

    def symmetric_difference(
        self, other: "_DFA", prune_unreachable: bool = True
    ) -> "_DFA":
        """
        Construct a DFA recognizing the symmetric difference of this and the
        other DFAs' languages using difference and union.
        """
        # Validate the DFAs' input alphabets.
        if self._input_alphabet != other._input_alphabet:
            raise ValueError(
                "Mismatched input alphabets. Cannot construct a symmetric "
                + "difference of two DFAs with different alphabets: "
                + f"{self._input_alphabet} != {other._input_alphabet}."
            )

        self_minus_other = self.difference(other, prune_unreachable)
        other_minus_self = other.difference(self, prune_unreachable)
        return self_minus_other.union(other_minus_self, prune_unreachable)

    def __xor__(self, other: "_DFA") -> "_DFA":
        """
        Operator override for symmetric difference DFA1 ^ DFA2. Note that using
        this operator automatically prunes unreachable states in the resulting
        DFA; to retain unreachable states, see symmetric_difference().
        """
        return self.symmetric_difference(other)

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
