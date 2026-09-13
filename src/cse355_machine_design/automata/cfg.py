from cse355_machine_design.errors import DetailedError

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

        # Each rule should have a single variable on its left-hand side and one
        # or more variables or terminals on its right-hand side.
        err = ""
        for lhs, rhss in self._rules.items():
            for rhs in rhss:
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
