from cse355_machine_design.errors import DetailedError

from collections import defaultdict
from itertools import chain, combinations

from typeguard import typechecked


class _CFG:
    """
    A context-free grammar (CFG).
    """

    # CFG variables.
    _variables: set[str]
    _terminals: set[str]
    _rules: dict[str, set[tuple[str, ...]]]
    _start_variable: str
    _epsilon: str

    @typechecked
    def __init__(
        self,
        V: set[str],
        Sigma: set[str],
        R: dict[str, set[tuple[str, ...]]],
        S: str,
        epsilon: str = "_",
    ) -> None:
        """
        Create a new CFG and then validate it.
        """
        self._variables = V
        self._terminals = Sigma
        self._rules = R
        self._start_variable = S
        self._epsilon = epsilon
        self.validate()

    def validate(self) -> None:
        """
        Validate this CFG according to its formal definition.
        """
        # There should be at least one variable.
        if len(self._variables) == 0:
            raise DetailedError(
                "Empty variables set",
                "A CFG should have at least one variable, but yours doesn't "
                + "have any.",
            )

        # There should be at least one terminal symbol.
        if len(self._terminals) == 0:
            raise DetailedError(
                "Empty terminals set",
                "A CFG should have at least one terminal, but yours doesn't "
                + "have any.",
            )

        # Each terminal symbol should be a length-one string.
        bad_terminals = [t for t in self._terminals if len(t) != 1]
        if len(bad_terminals) > 0:
            raise DetailedError(
                "Invalid terminal symbol(s)",
                "Terminal symbols should be individual characters, but these "
                + f"are not: {bad_terminals}.",
            )

        # The sets of variables and terminals should be disjoint.
        overlap = self._variables & self._terminals
        if len(overlap) > 0:
            raise DetailedError(
                "Overlapping variables and terminals",
                "Variables and terminals should be distinct, but these are "
                + f"being used as both: {overlap}.",
            )

        # The start variable should be in the variables set.
        if not self._start_variable in self._variables:
            raise DetailedError(
                "Invalid start variable",
                f"The start variable '{self._start_variable}' must be one of "
                + f"the CFG's variables, {self._variables}.",
            )

        # The empty terminal should be outside the variables set.
        if self._epsilon in self._variables:
            raise DetailedError(
                "Epsilon terminal in variables set",
                f"The epsilon terminal '{self._epsilon}' should not be in the "
                + f"CFG's set of variables, {self._variables}.",
            )

        # The empty terminal should be outside the terminals set.
        if self._epsilon in self._terminals:
            raise DetailedError(
                "Epsilon terminal in terminals set",
                f"The epsilon terminal '{self._epsilon}' should not be in the "
                + f"CFG's set of terminals, {self._terminals}.",
            )

        # Strip unnecessary epsilon terminals from rules' right-hand sides.
        def strip_epsilon(rhs: tuple[str, ...]) -> tuple[str, ...]:
            if self._epsilon not in rhs or len(rhs) == 1:
                return rhs
            else:
                return tuple([x for x in rhs if x != self._epsilon])

        for lhs in self._rules:
            self._rules[lhs] = {strip_epsilon(rhs) for rhs in self._rules[lhs]}

        # Each rule should have a single variable on its left-hand side and one
        # or more variables or terminals on its right-hand side.
        err = ""
        for lhs in self._rules:
            for rhs in self._rules[lhs]:
                r_err = ""
                if lhs not in self._variables:
                    r_err += (
                        f"\n- '{lhs}' is not in the variables set '{self._variables}'"
                    )
                if len(rhs) == 0:
                    r_err += "\n- right-hand side has no variables or terminals"
                else:
                    bad_elements = (
                        set(rhs) - self._variables - self._terminals - {self._epsilon}
                    )
                    if len(bad_elements) != 0:
                        r_err += (
                            f"{bad_elements} appear on the right-hand side but"
                            + " are neither variables nor terminals"
                        )

                if r_err != "":
                    err += f"\n For rule '{lhs} -> {rhs}', {r_err}"

        if err != "":
            raise DetailedError("Invalid rules", err)

        # Some rule should have the start variable on its left-hand side.
        if self._start_variable not in self._rules:
            raise DetailedError(
                "Missing start rule(s)",
                f"The start variable '{self._start_variable}' does not appear "
                + "on the left-hand side of any rule.",
            )

    def normalize(self) -> None:
        """
        Convert this CFG to Chomsky normal form by:
        1. creating terminal rules X_i -> x_i and replacing non-singleton
           instances of x_i on rules' right-hand sides with X_i
        2. breaking rules A -> x_1x_2...x_k with k >= 3 into rules A -> x_1A_1,
           A_1 -> x_2A_2, ..., A_{k-2} -> x_{k-1}x_k
        3. removing rules of the form A -> epsilon where A != S
        4. removing unit rules of the form A -> B (and thus cycles A =>* A)
        5. removing unproductive variables, or variables A with no derivation
           A =>* w, where w is a string of terminals
        6. removing unreachable variables, or variables A with no derivation
           S =>* xAy, where S is the start variable and x and y are possibly
           empty strings of variables and terminals

        Details of these procedures can be found in Hopcroft, Motwani, and
        Ullman (3rd ed., 2006), Sections 7.1 and 7.4.2. In particular, the long
        rules are broken up before removing the epsilon-rules so that the whole
        conversion takes only quadratic time (instead of exponential time).
        """
        # Create a dedicated rule T -> t for each terminal t. Then, replace all
        # instances of t on non-singleton right-hand sides of rules with T.
        nonterminal_rules: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for t in self._terminals:
            nonterminal_rules[f"CNF_TERM_{t}"] = {tuple(t)}
        for lhs in self._rules:
            for rhs in self._rules[lhs]:
                if len(rhs) == 1:
                    nonterminal_rules[lhs].add(rhs)
                else:
                    nonterminal_rules[lhs].add(
                        tuple(
                            f"CNF_TERM_{x}" if x in self._terminals else x for x in rhs
                        )
                    )
        self._rules = dict(nonterminal_rules)

        # Break rules A -> x_1x_2...x_k with k >= 3 into chains of smaller
        # rules A -> x_1A_1, A_1 -> x_2A_2, ..., A_{k-2} -> x_{k-1}x_k.
        short_rules: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for lhs in self._rules:
            for rhs_ix, rhs in enumerate(self._rules[lhs]):
                if len(rhs) <= 2:
                    short_rules[lhs].add(rhs)
                else:
                    short_rules[lhs].add((rhs[0], f"CNF_{lhs}_{rhs_ix}_0"))
                    for i in range(len(rhs) - 3):
                        short_rules[f"CNF_{lhs}_{rhs_ix}_{i}"].add(
                            (rhs[i + 1], f"CNF_{lhs}_{rhs_ix}_{i + 1}")
                        )
                    short_rules[f"CNF_{lhs}_{rhs_ix}_{len(rhs) - 3}"].add(
                        (rhs[-2], rhs[-1])
                    )
        self._rules = dict(short_rules)

        # Identify all "nullable" variables A such that A =>* epsilon.
        nullable: set[str] = {self._epsilon}
        added_new_nullable = True
        while added_new_nullable:
            added_new_nullable = False
            for lhs in self._rules.keys() - nullable:
                for rhs in self._rules[lhs]:
                    if set(rhs) <= nullable:
                        nullable.add(lhs)
                        added_new_nullable = True
                        break

        # Remove all epsilon rules except S -> epsilon, if it exists.
        nonepsilon_rules: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        if tuple(self._epsilon) in self._rules[self._start_variable]:
            nonepsilon_rules[self._start_variable].add(tuple(self._epsilon))
        for lhs in self._rules:
            for rhs in self._rules[lhs]:
                # For each possible subset of the nullable variables in this
                # rule's right-hand side, create a new right-hand side with
                # those variables as null. Drop totally null right-hand sides.
                nullable_ixs: set[int] = {
                    ix for (ix, s) in enumerate(rhs) if s in nullable
                }
                for nulled_ixs in chain.from_iterable(
                    combinations(nullable_ixs, r) for r in range(len(nullable_ixs) + 1)
                ):
                    nulled_rhs = tuple(
                        s for (ix, s) in enumerate(rhs) if ix not in nulled_ixs
                    )
                    if len(nulled_rhs) > 0:
                        nonepsilon_rules[lhs].add(nulled_rhs)
        self._rules = dict(nonepsilon_rules)

        # Identify "unit" rules of the form A -> B where B is a variable.
        unit_rules: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for lhs in self._rules:
            for rhs in self._rules[lhs]:
                if len(rhs) == 1 and rhs[0] in self._variables:
                    unit_rules[lhs].add(rhs)
        unit_rules = dict(unit_rules)

        # Identify all "unit pairs" of variables (A, B) such that A =>* B using
        # only unit rules. In the base case, A =>* A using zero rules.
        unit_pairs: set[tuple[str, str]] = {(A, A) for A in self._variables}
        unit_pairs_to_explore = unit_pairs.copy()
        while len(unit_pairs_to_explore) > 0:
            # In the inductive case, if (A, B) is a unit pair and B -> C is a
            # unit rule, then (A, C) is a unit pair.
            new_unit_pairs: set[tuple[str, str]] = set()
            for A, B in unit_pairs_to_explore:
                if B in unit_rules:
                    new_unit_pairs |= {(A, rhs[0]) for rhs in unit_rules[B]}

            unit_pairs_to_explore = new_unit_pairs - unit_pairs
            unit_pairs |= new_unit_pairs

        # Remove unit rules: for each unit pair (A, B), if B -> alpha is a
        # non-unit rule, then include A -> alpha.
        nonunit_rules: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for A, B in unit_pairs:
            nonunit_rules[A] |= {rhs for rhs in self._rules[B] - unit_rules[B]}
        self._rules = dict(nonunit_rules)

        # Identify all productive elements (terminals and variables). In the
        # base case, terminal symbols and epsilon are self-producing.
        productive_elements: set[str] = self._terminals | {self._epsilon}
        while True:
            # In the inductive case, if A -> alpha is a rule and every element
            # of alpha is productive, then A is productive.
            new_productive_elements: set[str] = set()
            for lhs in self._rules.keys() - productive_elements:
                for rhs in self._rules[lhs]:
                    if set(rhs) <= productive_elements:
                        new_productive_elements.add(lhs)
                        break

            if len(new_productive_elements) == 0:
                break
            else:
                productive_elements |= new_productive_elements

        # Remove unproductive variables and rules that contain them.
        unproductive_variables = self._variables - productive_elements
        self._variables &= productive_elements
        for lhs in self._rules.keys() & unproductive_variables:
            del self._rules[lhs]
        for lhs in self._rules:
            self._rules[lhs] = {
                rhs
                for rhs in self._rules[lhs]
                if len(set(rhs) & unproductive_variables) == 0
            }

        # Identify all reachable variables, beginning with the start variable.
        reachable_variables: set[str] = {self._start_variable}
        variables_to_explore: set[str] = {self._start_variable}
        variables_explored: set[str] = set()
        while len(variables_to_explore) > 0:
            lhs = variables_to_explore.pop()
            variables_explored.add(lhs)
            for rhs in self._rules[lhs]:
                rhs_variables = set(rhs) & self._variables
                reachable_variables |= rhs_variables
                variables_to_explore |= rhs_variables - variables_explored

        # Drop unreachable variables and rules that contain them.
        self._variables = reachable_variables
        for lhs in self._rules.keys() - reachable_variables:
            del self._rules[lhs]
