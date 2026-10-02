from cse355_machine_design.automata.base import _Automaton, AutomataComparison, State
from cse355_machine_design.automata.cfg import _CFG

from collections import defaultdict, deque
from itertools import product
import math


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
            raise ValueError(
                "Empty stack alphabet. A PDA's stack alphabet should contain "
                + "at least one symbol, but yours is empty."
            )

        # Symbols in the stack alphabet should be length-one strings.
        bad_symbols = [s for s in self._stack_alphabet if len(s) != 1]
        if len(bad_symbols) > 0:
            raise ValueError(
                "Invalid stack alphabet symbol(s). Stack symbols should be "
                + f"individual characters, but these are not: {bad_symbols}."
            )

        # The empty symbol should be a length-one string.
        if len(self._epsilon) != 1:
            raise ValueError(
                "Invalid epsilon symbol. The epsilon symbol should be an "
                f"individual character, but '{self._epsilon}' is not."
            )

        # The empty symbol should be outside the input alphabet.
        if self._epsilon in self._input_alphabet:
            raise ValueError(
                f"Invalid epsilon symbol. The epsilon symbol '{self._epsilon}'"
                + " should not be in the PDA's input alphabet, "
                + f"{self._input_alphabet}."
            )

        # The empty symbol should be outside the stack alphabet.
        if self._epsilon in self._stack_alphabet:
            raise ValueError(
                f"Invalid epsilon symbol. The epsilon symbol '{self._epsilon}'"
                + " should not be in the PDA's stack alphabet, "
                + f"{self._stack_alphabet}."
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
            raise ValueError("Invalid transition function:" + err)

    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this PDA.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the PDA accepts the input string.
        """
        # Validate the input string.
        if not set(input_str) <= self._input_alphabet:
            raise ValueError(
                f"Invalid input string. The input '{input_str}' contains "
                + f"symbols {set(input_str) - self._input_alphabet} that are "
                + "not in the PDA's input alphabet."
            )

        # Determine whether the input string is accepted using the equivalent
        # CFG. If tracing is not required, simply return that result; if it is,
        # but the string is rejected, simply say so.
        accepts_input_str = self._as_cfg().generates_string(input_str)
        if not trace:
            return accepts_input_str
        elif not accepts_input_str:
            print(
                "No computation ever reaches an accept state after consuming "
                + f"the input string '{input_str}', so REJECT"
            )
            return accepts_input_str

        # Otherwise, the CFG computation guarantees that there is an accepting
        # computation on this input string, so find it via breadth-first search
        # over PDA configurations, i.e., triples of (current state, reamining
        # input, and current stack contents).
        type PDAConfig = tuple[State, str, deque[str]]

        # Start by setting up some data structures. The first maps all explored
        # PDA configurations to their unique integer indices. The second stores
        # configurations in index order. The third is a directed graph with the
        # indices as nodes and edges storing transition information. The last
        # is the next layer of configurations to explore according to BFS.
        start_config: PDAConfig = (self._start_state, input_str, deque())
        configs: dict[PDAConfig, int] = {start_config: 0}
        ix_to_config: list[PDAConfig] = [start_config]
        config_graph: dict[int, dict[int, tuple[str, str, str]]] = {0: {}}
        configs_to_extend: set[PDAConfig] = {start_config}

        # Explore in BFS order until finding an accepting configuration, i.e.,
        # one with an accept state that has consumed the whole input string.
        accepting_ix: int | None = None
        while accepting_ix is None:
            next_configs_to_extend: set[PDAConfig] = set()
            for current_config in configs_to_extend:
                # Consider all transitions that this configuration enables.
                current_state, current_input, current_stack = current_config
                reads = [self._epsilon, current_input[0]]
                pops = [self._epsilon]
                if len(current_stack) > 0:
                    pops.append(current_stack[-1])
                for read, pop in product(reads, pops):
                    if (current_state, read, pop) in self._transitions:
                        for next_state, push in self._transitions[
                            (current_state, read, pop)
                        ]:
                            # Create the configuration this transition goes to.
                            next_input = (
                                current_input
                                if read == self._epsilon
                                else current_input[1:]
                            )
                            next_stack = current_stack.copy()
                            if pop != self._epsilon:
                                next_stack.pop()
                            if push != self._epsilon:
                                next_stack.append(push)
                            next_config: PDAConfig = (
                                next_state,
                                next_input,
                                next_stack,
                            )

                            # If this is the first time this configuration has
                            # been reached, add it to all the data structures.
                            if next_config not in configs:
                                ix_to_config.append(next_config)
                                configs[next_config] = len(ix_to_config) - 1
                                config_graph[configs[next_config]] = {}
                                next_configs_to_extend.add(next_config)

                            # Then update the configurations graph, noting that
                            # this configuration is reachable from its parent
                            # by reading/popping/pushing the given symbols.
                            config_graph[configs[current_config]][
                                configs[next_config]
                            ] = (read, pop, push)

                            # If this next configuration is accepting, mark it
                            # so the BFS will stop before its next iteration.
                            if (
                                next_state in self._accept_states
                                and len(next_input) == 0
                            ):
                                accepting_ix = configs[next_config]

            # Instantiate the next layer of the BFS.
            configs_to_extend = next_configs_to_extend.copy()

        # Compute a shortest path in the configuration graph from the starting
        # configuration to all other configurations. Even though we already did
        # a BFS through reachable configurations, we need a shortest path to
        # avoid falling into cycles consisting only of stack operations.
        visited_configs: set[int] = set()
        predecessor: dict[int, int] = {}
        dist_to_config: list[float] = [0] + [math.inf] * (len(configs) - 1)
        while len(visited_configs) < len(configs):
            # Get any unexplored node adjacent to an explored one; this is the
            # starting configuration for the first iteration.
            for u in range(len(configs)):
                if u not in visited_configs and dist_to_config[u] < math.inf:
                    break

            # Add this node to the explored nodes.
            visited_configs.add(u)

            # Update this node's neighbors' distances and predecessors.
            for v in config_graph[u].keys() - visited_configs:
                if dist_to_config[u] + 1 < dist_to_config[v]:
                    dist_to_config[v] = dist_to_config[u] + 1
                    predecessor[v] = u

        # Gather the full accepting trace by backtracing the shortest path from
        # the accepting configuration to the starting configuration.
        tracing_strs = []
        config_ix = accepting_ix
        while config_ix != 0:
            read, pop, push = config_graph[predecessor[config_ix]][config_ix]
            state, remaining_input, stack = ix_to_config[config_ix]
            tracing_strs.append(
                f"Read '{read}', pop '{pop}', and push '{push}'\n"
                + f"-> State: {state}\t"
                + f"Remaining Input: {remaining_input}\t"
                + f"Stack: {list(stack)}"
            )
            config_ix = predecessor[config_ix]
        tracing_strs.reverse()

        # Finally, print the tracing information and return.
        print("Evaluating PDA from starting configuration:")
        print(f"-> State: {self._start_state}\tInput: {input_str}\tStack: []")
        for tracing_str in tracing_strs:
            print(tracing_str)
        print(
            "This configuration is accepting (i.e., its state is an accept "
            + "state and the input string is consumed), so ACCEPT"
        )
        return accepts_input_str

    def generate_strings(
        self, max_str_len: int, max_strs: int | None = None
    ) -> set[str]:
        """
        Generate all strings in this PDA's language that are at most the given
        length, or the first `max_strs` such strings if not None.

        :param max_str_len: The maximum length of strings to generate.
        :param max_str_len: The maximum number of strings to generate.
        :returns: A list of generated strings from this PDA's language.
        """
        raise NotImplementedError("Not implemented yet!")

    def compare(self, other: "_Automaton") -> AutomataComparison:
        """
        Compare this and the other PDA's languages.

        :param other: The other PDA to compare against.
        :return: An AutomataComparison capturing the languages' relationship.
        """
        raise NotImplementedError("Not implemented yet!")

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

    def _as_cfg(self) -> _CFG:
        """
        Transform this PDA into an equivalent CFG.

        Details of this transformation can be found in Sipser (3rd ed., 2013),
        Lemma 2.27. Note that this is somewhat different than the approach in
        Hopcroft, Motwani, and Ullman (3rd ed., 2006), Section 6.3.2.
        """
        # Verify that existing PDA states can't conflict with upcoming changes.
        conflict_states = [s for s in self._states if "cfg" in s]
        if len(conflict_states) > 0:
            raise ValueError(
                "Conflicting state names. The PDA -> CFG conversion adds new "
                + "states containing the string 'cfg', but this may conflict "
                + f"with existing states: {conflict_states}."
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

        return _CFG(
            V=cfg_variables,
            Sigma=self._input_alphabet,
            R=cfg_rules,
            S=cfg_start_variable,
            epsilon=self._epsilon,
        )
