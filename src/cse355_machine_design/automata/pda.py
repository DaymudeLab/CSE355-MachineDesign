from collections import defaultdict, deque
from typing import Any, Dict, Set, Tuple, Union, NamedTuple
from cse355_machine_design.automata.base import _Automata
from cse355_machine_design import registry
from cse355_machine_design.errors import DetailedError
from cse355_machine_design.util import set_str, quote_str
import math
import time
import uuid
import itertools


class CFG:
    variables: set[str]
    terminals: set[str]
    rules: dict[str, set[tuple[str, ...]]]
    start_variable: str
    epsilon: str

    def __init__(self, pda: "_PDA") -> None:
        pda = self.transform_pda(pda)

        self.epsilon = pda._epsilon
        self.terminals = pda._input_symbols.copy()
        self.start_variable = (
            f"A_[{pda._start_state}]_[{pda._final_states.copy().pop()}]"
        )
        self.variables = set()

        for p, q in itertools.product(pda._states, repeat=2):
            new_var = f"A_[{p}]_[{q}]"
            self.variables.add(new_var)

        sigma_e = pda._input_symbols.union([pda._epsilon])

        self.rules = defaultdict(set)

        for p in pda._states:
            self.rules[f"A_[{p}]_[{p}]"].add((self.epsilon,))
            for q, r in itertools.product(pda._states, repeat=2):
                self.rules[f"A_[{p}]_[{q}]"].add((f"A_[{p}]_[{r}]", f"A_[{r}]_[{q}]"))
                for s in pda._states:
                    for u in pda._stack_alphabet:
                        for a, b in itertools.product(sigma_e, repeat=2):
                            if (r, u) in pda._transition_table[
                                (p, a, pda._epsilon)
                            ] and (q, pda._epsilon) in pda._transition_table[(s, b, u)]:
                                self.rules[f"A_[{p}]_[{q}]"].add(
                                    (a, f"A_[{r}]_[{s}]", b)
                                )
        return

    def minimized(self) -> "CFG":
        useful = set()
        for variable, rule_set in self.rules.items():
            for rule in rule_set:
                if all([x in self.terminals or x == self.epsilon for x in rule]):
                    useful.add(variable)
        changed = True
        while changed:
            changed = False
            for variable, rule_set in self.rules.items():
                for rule in rule_set:
                    if all(
                        [
                            x in useful or x in self.terminals or x == self.epsilon
                            for x in rule
                        ]
                    ):
                        if variable not in useful:
                            changed = True
                            useful.add(variable)

        useless = self.variables.difference(useful)

        for variable in useless:
            del self.rules[variable]

        self.variables = useful

        for variable, rule_set in self.rules.items():
            new_rules = set()
            for rule in rule_set:
                if any([x in useless for x in rule]):
                    continue
                new_rules.add(rule)
            self.rules[variable] = new_rules

        # Remove rules unreachable from start

        reachable = set([self.start_variable])
        changed = True
        while changed:
            changed = False
            new_reach = set()
            for variable in reachable:
                for rule in self.rules[variable]:
                    for item in rule:
                        if item in self.variables:
                            if item not in reachable:
                                changed = True
                                new_reach.add(item)
            reachable.update(new_reach)

        unreachable = self.variables.difference(reachable)

        for variable in unreachable:
            del self.rules[variable]

        self.variables = reachable

        return self

    def cnf(self) -> "CFG":
        """ONLY WORKS IF OF THE FORM DECIDED BY CONSTRUCTOR"""
        # New start variable
        self.rules["S_0"].add((self.start_variable,))
        self.start_variable = "S_0"
        self.variables.add("S_0")

        # Eliminate rules with nonsolitary terminals
        for terminal in self.terminals:
            name = f"RULE_[{terminal}]"
            self.rules[name].add((terminal,))
            self.variables.add(name)
        self.variables.add(f"RULE_[{self.epsilon}]")
        self.rules[f"RULE_[{self.epsilon}]"].add((self.epsilon,))

        for variable, rule_set in self.rules.items():
            new_rules = set()
            for rule in rule_set:
                if len(rule) != 3:
                    new_rules.add(rule)
                else:
                    new_rules.add((f"RULE_[{rule[0]}]", rule[1], f"RULE_[{rule[2]}]"))
            self.rules[variable] = new_rules

        # Eliminate right-hand sides with more than 2 nonterminals
        rules_to_add: list[tuple[str, tuple[str, ...]]] = []
        for variable, rule_set in self.rules.items():
            new_rules = set()
            for rule in rule_set:
                if len(rule) != 3:
                    new_rules.add(rule)
                else:
                    nr_name = f"BIN_[{rule[1]}]_[{rule[2]}]"
                    self.variables.add(nr_name)
                    new_rules.add((rule[0], nr_name))
                    rules_to_add.append((nr_name, (rule[1], rule[2])))
            self.rules[variable] = new_rules

        for variable, new_rule in rules_to_add:
            self.rules[variable].add(new_rule)

        # Eliminate epsilon-rules (PAIN)
        map_to_nullable = lambda x: x not in self.terminals and x in nullable_vars

        nullable_vars: set[str] = set()
        changed = True
        for variable, rule_set in self.rules.items():
            for rule in rule_set:
                if len(rule) == 1 and rule[0] == self.epsilon:
                    nullable_vars.add(variable)
        while changed:
            changed = False
            new_nullables = set()
            for variable, rule_set in self.rules.items():
                if variable in nullable_vars:
                    continue
                for rule in rule_set:
                    all_nullable = all(map(map_to_nullable, rule))
                    if all_nullable:
                        new_nullables.add(variable)
            if len(new_nullables) != 0:
                changed = True
                nullable_vars.update(new_nullables)

        nullable_vars.discard(self.start_variable)

        for variable, rule_set in self.rules.items():
            new_rules = set()
            for rule in rule_set:
                is_nullable = list(map(map_to_nullable, rule))
                list_of_null_idx = list(
                    itertools.compress(range(len(is_nullable)), is_nullable)
                )
                new_rules.add(rule)
                for r in range(1, len(list_of_null_idx) + 1):
                    for permutation in itertools.permutations(list_of_null_idx, r):
                        new_rule = []
                        for i in range(len(rule)):
                            if i in permutation:
                                continue
                            new_rule.append(rule[i])
                        new_rule = tuple(new_rule)
                        if len(new_rule) == 0:
                            new_rules.add((self.epsilon,))
                        else:
                            new_rules.add(new_rule)

            self.rules[variable] = new_rules

        for variable in nullable_vars:
            new_rules = set()
            for rule in self.rules[variable]:
                if self.epsilon in rule:
                    continue
                new_rules.add(rule)
            self.rules[variable] = new_rules

        # Remove non-producing rules that arise from eps rem procedure
        blank_variables = set()
        changed = True
        while changed:
            changed = False
            for variable, rule_set in self.rules.items():
                new_rules = set()
                if len(rule_set) == 0:
                    if variable not in blank_variables:
                        changed = True
                        blank_variables.add(variable)
                    continue
                for rule in rule_set:
                    if any([x in blank_variables for x in rule]):
                        continue
                    new_rules.add(rule)
                if len(new_rules) == 0:
                    if variable not in blank_variables:
                        changed = True
                        blank_variables.add(variable)
                self.rules[variable] = new_rules

        self.variables.difference_update(blank_variables)
        for variable in blank_variables:
            del self.rules[variable]

        # Eliminate unit rules
        removed_units: set[tuple[str, str]] = set()
        changed = True
        while changed:
            changed = False
            for variable, rule_set in self.rules.items():
                new_rules = set()
                for rule in rule_set:
                    B = rule[0]
                    if len(rule) == 1 and B in self.variables:
                        for B_rule in self.rules[B]:
                            if len(B_rule) == 1 and B_rule[0] in self.variables:
                                if (variable, B_rule[0]) in removed_units:
                                    continue
                                else:
                                    new_rules.add(B_rule)
                                    removed_units.add((variable, B_rule[0]))
                                    changed = True
                            else:
                                new_rules.add(B_rule)
                    else:
                        new_rules.add(rule)
                self.rules[variable] = new_rules

        # remove all empty variables
        removed = set()
        for variable in self.variables:
            if len(self.rules[variable]) == 0:
                del self.rules[variable]
                removed.add(variable)
        self.variables.difference_update(removed)

        self.minimized()

        return self

    def cyk(self, input_str) -> bool:
        """MUST CALL CNF FIRST"""
        if input_str == "":
            return (self.epsilon,) in self.rules[self.start_variable]

        n = len(input_str)
        r = len(self.variables)

        bool_array: list[list[list[bool]]] = [
            [[False for _ in range(r)] for _ in range(n)] for _ in range(n)
        ]

        index_map: dict[str, int] = dict()
        rule_iter = self.rules.items()
        seen_start = False
        for i, (variable, _) in enumerate(rule_iter):
            if variable == self.start_variable:
                seen_start = True
                index_map[variable] = 0
            else:
                index_map[variable] = i if seen_start else i + 1

        for i in range(n):
            for v, rule_set in rule_iter:
                for rule in rule_set:
                    if (
                        len(rule) == 1
                        and rule[0] in self.terminals
                        and rule[0] == input_str[i]
                    ):
                        bool_array[0][i][index_map[v]] = True

        two_rule = dict()
        for v, rule_set in rule_iter:
            new_rules = set()
            for rule in rule_set:
                if len(rule) != 2:
                    continue
                new_rules.add(rule)
            two_rule[v] = new_rules

        two_rule_iter = two_rule.items()

        for l in range(2, n + 1):
            for s in range(n - l + 1):
                for p in range(1, l):
                    for v, rule_set in two_rule_iter:
                        for rule in rule_set:
                            b = index_map[rule[0]]
                            c = index_map[rule[1]]
                            if (
                                bool_array[p - 1][s][b]
                                and bool_array[l - p - 1][s + p][c]
                            ):
                                bool_array[l - 1][s][index_map[v]] = True

        return bool_array[n - 1][0][0]

    def transform_pda(self, pda: "_PDA") -> "_PDA":
        new_q = pda._states.copy()
        new_sigma = pda._input_symbols.copy()
        new_gamma = pda._stack_alphabet.copy()
        # SET UNIQUE ACCEPTING
        new_final = set(["Q_CFG_FINAL"])
        new_q.add("Q_CFG_FINAL")
        # SET STACK DIE DOWN BEFORE ACCEPTING
        new_q.add("Q_CFG_PRE_EMPTYING")
        new_q.add("Q_CFG_EMPTYING")
        new_q.add("Q_CFG_NEW_START")
        new_gamma.add("NEW_BOTTOM")
        new_gamma.add("ARB")

        new_delta: Dict[Tuple[str, str, str], Set[Tuple[str, str]]] = defaultdict(set)

        new_delta[("Q_CFG_NEW_START", pda._epsilon, pda._epsilon)] = {
            (pda._start_state, "NEW_BOTTOM")
        }
        new_delta[("Q_CFG_EMPTYING", pda._epsilon, "NEW_BOTTOM")] = {
            ("Q_CFG_FINAL", pda._epsilon)
        }
        new_delta[("Q_CFG_PRE_EMPTYING", pda._epsilon, "ARB")] = {
            ("Q_CFG_EMPTYING", pda._epsilon)
        }
        for symbol in pda._stack_alphabet:
            new_delta[("Q_CFG_EMPTYING", pda._epsilon, symbol)] = {
                ("Q_CFG_EMPTYING", pda._epsilon)
            }

        for final in pda._final_states:
            new_delta[(final, pda._epsilon, pda._epsilon)] = {
                ("Q_CFG_PRE_EMPTYING", "ARB")
            }

        for (
            state,
            input_char,
            pop,
        ), transition_set in pda._transition_table.items():
            for to_state, push in transition_set:
                if pop == pda._epsilon and push == pda._epsilon:
                    new_state_name = f"Q_{uuid.uuid4()}"
                    new_q.add(new_state_name)
                    new_delta[((state, input_char, pop))].add((new_state_name, "ARB"))
                    new_delta[((new_state_name, pda._epsilon, "ARB"))].add(
                        (to_state, push)
                    )
                elif pop != pda._epsilon and push != pda._epsilon:
                    new_state_name = f"Q_{uuid.uuid4()}"
                    new_q.add(new_state_name)
                    new_delta[((state, input_char, pop))].add(
                        (new_state_name, pda._epsilon)
                    )
                    new_delta[((new_state_name, pda._epsilon, pda._epsilon))].add(
                        (to_state, push)
                    )
                else:
                    # good to leave the transition unmodified
                    new_delta[(state, input_char, pop)].add((to_state, push))

        res = _PDA(
            new_q,
            new_sigma,
            new_gamma,
            new_delta,
            "Q_CFG_NEW_START",
            new_final,
            pda._epsilon,
            generate_cfg=False,
        )
        return res


class PDAState(NamedTuple):
    pda: "_PDA"
    """
    The parent pda that controls how state progresses.
    """
    parent_config: Union[tuple["PDAState", str, str, str], None]
    """
    The parent config state that lead to this one
    """
    input_str: str
    """
    The string being computed on
    """
    input_str_idx: int
    """
    The index into the input string the PDA is currently at.
    If the index is 0 that means the PDA can read from index 0
    and advance to 1 or choose to read epsilon and not advance"""
    current_state: str
    """
    The current state the PDA is in
    """
    stack: tuple[str, ...]
    """
    The current stack. Create a copy first then do: append start to push, remove start to pop
    """

    def one_step(self) -> set["PDAState"]:
        """
        After taking one step, either by consuming input or using a single epsilon to transition
        what configurations are we in
        """
        return self._one_step_char().union(self._one_step_epsilon())

    def _one_step_char(self) -> set["PDAState"]:
        res: set["PDAState"] = set()

        if self.input_str_idx == len(self.input_str):
            return res

        for (
            state,
            input_char,
            pop,
        ), transition_set in self.pda._transition_table.items():
            if state != self.current_state:
                continue
            if input_char != self.input_str[self.input_str_idx]:
                continue
            if pop == self.pda._epsilon:
                for to_state, push in transition_set:
                    new_stack = self.stack
                    if push != self.pda._epsilon:
                        new_stack = (push,) + self.stack
                    res.add(
                        PDAState(
                            self.pda,
                            (self, self.input_str[self.input_str_idx], pop, push),
                            self.input_str,
                            self.input_str_idx + 1,
                            to_state,
                            new_stack,
                        )
                    )
            elif len(self.stack) > 0 and self.stack[0] == pop:
                for to_state, push in transition_set:
                    new_stack = self.stack[1:]
                    if push != self.pda._epsilon:
                        new_stack = (push,) + self.stack[1:]
                    res.add(
                        PDAState(
                            self.pda,
                            (self, self.input_str[self.input_str_idx], pop, push),
                            self.input_str,
                            self.input_str_idx + 1,
                            to_state,
                            new_stack,
                        )
                    )
        return res

    def _one_step_epsilon(self) -> set["PDAState"]:
        res: set["PDAState"] = set()

        for (
            state,
            input_char,
            pop,
        ), transition_set in self.pda._transition_table.items():
            if state != self.current_state:
                continue
            if input_char != self.pda._epsilon:
                continue
            if pop == self.pda._epsilon:
                for to_state, push in transition_set:
                    new_stack = self.stack
                    if push != self.pda._epsilon:
                        new_stack = (push,) + self.stack
                    res.add(
                        PDAState(
                            self.pda,
                            (self, self.pda._epsilon, pop, push),
                            self.input_str,
                            self.input_str_idx,
                            to_state,
                            new_stack,
                        )
                    )
            elif len(self.stack) > 0 and self.stack[0] == pop:
                for to_state, push in transition_set:
                    new_stack = self.stack[1:]
                    if push != self.pda._epsilon:
                        new_stack = (push,) + self.stack[1:]
                    res.add(
                        PDAState(
                            self.pda,
                            (self, self.pda._epsilon, pop, push),
                            self.input_str,
                            self.input_str_idx,
                            to_state,
                            new_stack,
                        )
                    )
        return res

    def is_accepting(self) -> bool:
        """
        This is an accepting configuration if and only if we are in a final state and all input has been consumed
        """
        return (
            self.current_state in self.pda._final_states
            and self.input_str_idx == len(self.input_str)
        )

    def __hash__(self) -> int:
        return hash(
            (
                id(self.pda),
                self.input_str,
                self.input_str_idx,
                self.current_state,
                self.stack,
            )
        )

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PDAState):
            return False
        return (
            self.pda == __value.pda
            and self.input_str == __value.input_str
            and self.input_str_idx == __value.input_str_idx
            and self.current_state == __value.current_state
            and self.stack == __value.stack
        )


class _PDA(_Automata):
    _stack_alphabet: Set[str]
    _final_states: Set[str]
    _epsilon: str
    _transition_table: defaultdict[Tuple[str, str, str], Set[Tuple[str, str]]]
    _associated_cfg: CFG

    def __init__(
        self,
        Q: Set[str],
        Sigma: Set[str],
        Gamma: Set[str],
        delta: Dict[Tuple, Set[Tuple]],
        q0: str,
        F: Set[str],
        epsilon_char="_",
        generate_cfg=True,
    ) -> None:
        """
        Delta entries should be of the form (FROM_STATE, READ_INPUT, POP): {(TO_STATE, PUSH),...}

        """
        self._states = Q
        self._input_symbols = Sigma
        self._stack_alphabet = Gamma
        self._transition_table = defaultdict(set)
        for k, v in delta.items():
            self._transition_table[k] = v  # type: ignore
        self._start_state = q0
        self._final_states = F
        self._epsilon = epsilon_char
        super().__init__()
        if not generate_cfg:
            return
        self._associated_cfg = CFG(self).minimized().cnf()

    def verify_legality(self) -> None:
        # Verify that neither sigma nor gamma contains epsilon
        if self._epsilon in self._input_symbols:
            raise DetailedError(
                "Sigma contains epsilon",
                "You have declared your alphabet to include the symbol epsilon. This is not allowed. Either remove epsilon from the alphabet or redefine the epsilon character by passing it as a kwarg to the PDA constructor",
            )
        if self._epsilon in self._stack_alphabet:
            raise DetailedError(
                "Gamma contains epsilon",
                "You have declared your stack alphabet to include the symbol epsilon. This is not allowed. Either remove epsilon from the stack alphabet or redefine the epsilon character by passing it as a kwarg to the PDA constructor",
            )

        # Verify that states are strings and not tuples
        for state in self._states:
            if not isinstance(state, str):
                raise DetailedError(
                    "States are not strings",
                    "Please define you states as strings. At this time the library does not support tuple or any other form of state. So if you have ('q1','q2'), you may rewrite it as '(q1,q2)'.",
                )

        # Verify that the start state is valid
        start_valid = self._start_state in self._states
        if not start_valid:
            raise DetailedError(
                "(Start state q_0 not found in Q)",
                f'"{self._start_state}" was not found in Q',
            )

        # Verify that F is a subset of Q
        err_str = ""
        final_valid = True
        for potential_state in self._final_states:
            potential_state_valid = potential_state in self._states
            if not potential_state_valid:
                err_str += f'"{potential_state}" was declared as a final state, but it is not found in Q\n'
            final_valid &= potential_state_valid
        if not final_valid:
            raise DetailedError("(F is not a subset of Q)", err_str)

        # Verify that the transitions are valid. That is:
        # - all tuples in the form (Q, Sigma_e, Gamma_e) -> Set[(Q,Gamma_e)]
        transitions_valid = True
        err_str = ""
        for (from_state, input_symbol, pop), to_set in self._transition_table.items():
            if from_state not in self._states:
                transitions_valid = False
                err_str += f"{quote_str(from_state)} is not a valid state\n"
            if (
                input_symbol not in self._input_symbols
                and input_symbol != self._epsilon
            ):
                transitions_valid = False
                err_str += f"{quote_str(input_symbol)} not a valid alphabet symbol\n"
            if pop not in self._stack_alphabet and pop != self._epsilon:
                transitions_valid = False
                err_str += f"{quote_str(pop)} not a valid stack alphabet symbol\n"
            for to_state, push in to_set:
                if to_state not in self._states:
                    transitions_valid = False
                    err_str += f"{quote_str(to_state)} is not a valid state\n"
                if push not in self._stack_alphabet and push != self._epsilon:
                    transitions_valid = False
                    err_str += f"{quote_str(push)} not a valid stack alphabet symbol\n"
        if not transitions_valid:
            raise DetailedError("One or more transitions are invalid", err_str)

    def eval_cyk(self, input_str: str) -> bool:
        # Run CYK
        return self._associated_cfg.cyk(input_str)

    def evaluate(self, input_str: str, enable_trace=0) -> Union[bool, None]:
        """
        Evaluate the automata on a string

        This is the hard part. Due to nondeterminism and stack baloney I dont
        really know what the upper bound is on PDA path length. So we do the
        following:
        - Convert PDA to CFG
        - Strip CFG to minimal form
        - Convert to Chomsky Normal Form
        - Run Cocke-Younger-Kasami Algorithm to check if PDA would accept using the CFG (end here if log lvl 0)
        - If accept: do a BFS over PDA and log (lvl 1) the singular valid path
        - If not accept: do a BFS with a limit on
        """
        does_accept = self.eval_cyk(input_str)  # FIX WITH CYK

        if enable_trace == 1:
            print("Input accepted" if does_accept else "Input rejected")
        if enable_trace <= 1:
            return does_accept

        bfs_limit = (
            math.inf if does_accept else 100 * len(self._states)
        )  # This should limit the depth of the BFS (idk any sensible limit for the false case)

        current_configs: set[PDAState] = set()

        current_configs.add(PDAState(self, None, input_str, 0, self._start_state, ()))

        depth = 0
        accepting_config = None

        while depth < bfs_limit:
            depth += 1
            if len(current_configs) == 0:
                break
            new_configs = set()
            for config in current_configs:
                if config.is_accepting():
                    accepting_config = config
                    break
                new_configs.update(config.one_step())
            if accepting_config is not None:
                break
            current_configs = new_configs

        if does_accept and enable_trace > 1:
            path_str_list = []
            if accepting_config is None:
                raise DetailedError(
                    "Invalid Application State (Not User Error)",
                    "The PDA should have accepted the string provided according to the CYK algorithm but did not reach an accepting state",
                )
            backtrack = accepting_config
            sep = "." if "." not in self._input_symbols else "[READ_LOCATION]"
            while backtrack is not None:
                step = ""
                if backtrack.parent_config is not None:
                    step += f"Read: {quote_str(backtrack.parent_config[1])}\tPop: {quote_str(backtrack.parent_config[2])}\tPush: {quote_str(backtrack.parent_config[3])}\n"
                step += f'State: {quote_str(backtrack.current_state)}\tInput: "{input_str[:backtrack.input_str_idx]}{sep}{input_str[backtrack.input_str_idx:]}"\tStack: {list(backtrack.stack)}'
                if backtrack == accepting_config:
                    step += "\nAccepting configuration reached, accepting string"
                path_str_list.append(step)
                backtrack = (
                    backtrack.parent_config[0]
                    if backtrack.parent_config is not None
                    else None
                )
            print("\n".join(list(reversed(path_str_list))))
        elif not does_accept and enable_trace > 1:
            print(
                "Simulation results in no accepting configurations after a large number of steps."
            )
            print("Using fallback non-feedback method", end=None)
            time.sleep(0.5)
            print(".", end=None)
            time.sleep(0.5)
            print(".", end=None)
            time.sleep(0.5)
            print(".")
            time.sleep(0.5)
            print("This PDA does not accept the given string")

        return does_accept

    def submit_as_answer(self, question_number: int) -> None:
        registry.add_to_registry("pda", question_number, self)

    def export_to_dict(self) -> dict:
        rep = {
            "type": "pda",
            "states": list(self._states),
            "input_symbols": list(self._input_symbols),
            "stack_alphabet": list(self._stack_alphabet),
            "finals": list(self._final_states),
            "start_state": self._start_state,
            "delta": list(
                map(
                    lambda p: {
                        "from": p[0][0],
                        "to": list(p[1])[0],
                        "input": p[0][1],
                        "pop": p[0][2],
                        "push": list(p[1])[1],
                    },
                    self._transition_table.items(),
                )
            ),
            "epsilon_char": self._epsilon,
        }
        return rep

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
        tt_restructured: defaultdict[
            Tuple[str, str], list[Tuple[str, str, str]]
        ] = defaultdict(list)

        for (
            from_state,
            input_symbol,
            stack_pop_symbol,
        ), to_state_collection in self._transition_table.items():
            for to_state, stack_push_symbol in to_state_collection:
                tt_restructured[(from_state, to_state)].append(
                    (input_symbol, stack_pop_symbol, stack_push_symbol)
                )

        for (from_state, to_state), transition_list in tt_restructured.items():
            label = "{}, {} → {}".format(
                transition_list[0][0], transition_list[0][1], transition_list[0][2]
            )
            for i, (input_symbol, pop, push) in enumerate(transition_list):
                if i == 0:
                    continue
                label += "\\n"
                label += "{}, {} → {}".format(input_symbol, pop, push)
            res += '"{}" -> "{}" [label="{}"];'.format(from_state, to_state, label)

        res += "}"
        return res
