from cse355_machine_design.automata.base import _Automaton, AutomataComparison, State
from cse355_machine_design.automata.dfa import _DFA

from collections import defaultdict, deque


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
        self._transitions = {k: v.copy() for k, v in delta.items()}
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
            raise ValueError(
                "Invalid epsilon symbol. The epsilon symbol should be an "
                + f"individual character, but '{self._epsilon}' is not."
            )

        # The empty symbol should be outside the input alphabet.
        if self._epsilon in self._input_alphabet:
            raise ValueError(
                f"Invalid epsilon symbol. The epsilon symbol '{self._epsilon}'"
                + " should not be in the NFA's input alphabet, "
                + f"{self._input_alphabet}."
            )

        # For each transition delta(q, a) = s in the NFA:
        # - q should be in the state set
        # - a should be in the input alphabet or epsilon
        # - s should be a (possibly empty) subset of the state set
        err = ""
        for (q, a), s in self._transitions.items():
            t_err = ""
            if q not in self._states:
                t_err += f"\n- '{q}' is not in the state set {self._states}"
            if a not in self._input_alphabet and a != self._epsilon:
                t_err += (
                    f"\n- '{a}' is not in the input alphabet "
                    + f"{self._input_alphabet} nor is it the epsilon symbol "
                    + f"'{self._epsilon}'"
                )
            if not s <= self._states:
                t_err += f"\n- {s - self._states} are not in the state set"

            if t_err != "":
                err += f"\nFor transition delta({q}, {a}) = {s}, {t_err}"

        if err != "":
            raise ValueError("Invalid transition function:" + err)

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
        if not set(input_str) <= self._input_alphabet:
            raise ValueError(
                f"Invalid input string. The input '{input_str}' contains "
                + f"symbols {set(input_str) - self._input_alphabet} that are "
                + "not in the NFA's input alphabet."
            )

        # Computation starts from the epsilon closure of the start state.
        if trace:
            print(
                f"Evaluating input '{input_str}' from start state "
                + f"'{self._start_state}'..."
            )
        current_states = self.epsilon_closure({self._start_state}, trace)

        # Trace through the input string one symbol at a time.
        for input_sym in input_str:
            next_states = set()
            for q in current_states:
                next_states |= self._transitions.get((q, input_sym)) or set()
            if trace:
                print(
                    f"Transition states {current_states} -> {next_states} "
                    + f"following exactly one '{input_sym}' transition"
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

    def compare(self, other: "_Automaton") -> AutomataComparison:
        """
        Compare this and the other finite automaton's languages by converting
        both into DFAs and then using DFA.compare().

        :param other: The other DFA or NFA to compare against.
        :return: An AutomataComparison capturing the languages' relationship.
        """
        # Validate the other automaton's type.
        if not isinstance(other, _DFA | _NFA):
            raise TypeError(
                "Invalid comparison type. Cannot directly compare languages of"
                + f" a NFA and a {type(other)}."
            )

        # Perform the necessary conversions and then compare languages.
        if isinstance(other, _DFA):
            return self.as_dfa().compare(other)
        else:
            return self.as_dfa().compare(other.as_dfa())

    def as_dfa(self) -> _DFA:
        """
        Convert this NFA into an equivalent DFA using an efficient BFS version
        of the powerset construction. Note that unreachable states are omitted
        automatically by this method.

        :return: A DFA equivalent to this NFA.
        """
        # Initialize the BFS through the NFA's powerset representation with the
        # epsilon closure of the NFA's start state.
        nfa_q0 = self.epsilon_closure({self._start_state})
        nfa_state_sets_to_explore: deque[set[State]] = deque([nfa_q0])

        def to_dfa_state(nfa_state_set: set[State]) -> State:
            """
            Canonize a set of NFA states as a single DFA state.
            """
            if len(nfa_state_set) == 0:
                return "{}"
            else:
                return f"{{{str(sorted(nfa_state_set))[1:-1]}}}"

        # Set up the corresponding DFA's elements.
        Q: set[State] = set()
        Sigma: set[str] = self._input_alphabet.copy()
        delta: dict[tuple[State, str], State] = {}
        q0: State = to_dfa_state(nfa_q0)
        F: set[State] = set()

        # Perform the BFS.
        while len(nfa_state_sets_to_explore) > 0:
            # Create a canonical representation of this subset of NFA states to
            # use as a DFA state, and if the subset contains an accept state of
            # the NFA, also add it to the DFA's accept states.
            nfa_state_set = nfa_state_sets_to_explore.popleft()
            dfa_state = to_dfa_state(nfa_state_set)
            Q.add(dfa_state)
            if not self._accept_states.isdisjoint(nfa_state_set):
                F.add(dfa_state)

            # Let S be the subset of NFA states and x be any input symbol.
            for input_sym in self._input_alphabet:
                # Compute the epsilon closure of the union over all s in S of
                # delta_NFA(s, x).
                next_nfa_state_set: set[State] = set()
                for nfa_state in nfa_state_set:
                    next_nfa_state_set |= (
                        self._transitions.get((nfa_state, input_sym)) or set()
                    )
                next_nfa_state_set = self.epsilon_closure(next_nfa_state_set)

                # Create the corresponding DFA transition.
                next_dfa_state = to_dfa_state(next_nfa_state_set)
                delta[(dfa_state, input_sym)] = next_dfa_state

                if next_dfa_state not in Q:
                    nfa_state_sets_to_explore.append(next_nfa_state_set)

        return _DFA(Q, Sigma, delta, q0, F)

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
        for (from_state, input_sym), to_states in self._transitions.items():
            for to_state in to_states:
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
