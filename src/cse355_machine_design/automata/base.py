from cse355_machine_design import registry
from cse355_machine_design.errors import DetailedError

from abc import ABC, abstractmethod
from pathlib import Path

import render_html
from typeguard import typechecked


# Define a type alias for automata states (to avoid confusion with usual strs).
type State = str


class _Automaton(ABC):
    """
    An abstract base class for automata.
    """

    # Automata variables; note that the transition function is not included
    # here because it must be defined by a specific derived class.
    _automaton_type: str  # Type of automata (DFA, NFA, PDA, etc.).
    _states: set[State]  # State set Q.
    _input_alphabet: set[str]  # Input alphabet Sigma.
    _start_state: State  # Start state q0.
    _accept_states: set[State]  # Accept states F.

    @typechecked
    def __init__(
        self,
        automaton_type: str,
        Q: set[State],
        Sigma: set[str],
        q0: State,
        F: set[State],
    ) -> None:
        """
        Create a new base automaton.

        :param automaton_type: A string type of automaton (e.g., "DFA").
        :param Q: The automaton's state set.
        :param Sigma: The automaton's input alphabet.
        :param q0: The automaton's start state.
        :param F: The automaton's accepting/final states.
        """
        self._automaton_type = automaton_type
        self._states = Q
        self._input_alphabet = Sigma
        self._start_state = q0
        self._accept_states = F

    def validate(self) -> None:
        """
        Validate this automaton according to its formal definition.
        """
        # There should be at least one input alphabet symbol.
        if len(self._input_alphabet) == 0:
            raise DetailedError(
                "Empty input alphabet",
                "Your finite automaton's input alphabet should contain at "
                + "least one symbol, but yours is empty.",
            )

        # Symbols in the input alphabet should be length-one strings.
        bad_symbols = [s for s in self._input_alphabet if len(s) != 1]
        if len(bad_symbols) > 0:
            raise DetailedError(
                "Invalid input alphabet symbol(s)",
                "Input alphabet symbols should be individual characters, but "
                + f"these are not: {bad_symbols}.",
            )

        # There should be at least one state.
        if len(self._states) == 0:
            raise DetailedError(
                "Empty state set",
                "Your finite automaton should have at least one state, but "
                + "yours doesn't have any.",
            )

        # The start state should be in the state set.
        if self._start_state not in self._states:
            raise DetailedError(
                "Invalid start state",
                f"The start state '{self._start_state}' must be one of the "
                + f"finite automaton's states, {self._states}.",
            )

        # The accept states must all be in the state set.
        if not self._accept_states <= self._states:
            raise DetailedError(
                "Invalid accept state(s)",
                "Each accept state must be a state of the finite automaton, "
                + f"but these are not: {self._accept_states - self._states}.",
            )

    @abstractmethod
    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate the given input string with this automaton.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the automaton accepts the input string.
        """
        raise NotImplementedError("Abstract method not callable")

    @typechecked
    def submit_as_answer(self, problem_number: int) -> None:
        """
        Submit this automaton to the registry as the answer to a given problem.
        Overwrites any previously submitted automaton for the same problem.

        :param problem_number: The problem to submit this as an answer for.
        """
        registry.add_to_registry(self._automaton_type, problem_number, self)

    def as_dict(self) -> dict:
        """
        Get a dict representation of this automaton.

        :return: A dict representation of this automaton.
        """
        return {
            "type": self._automaton_type,
            "states": self._states,
            "input_alphabet": self._input_alphabet,
            "start_state": self._start_state,
            "accept_states": self._accept_states,
        }

    @abstractmethod
    def _as_dot_string(self) -> str:
        """
        Get a DOT string representation of this automaton for use in graphviz
        visualization.

        :return: A DOT string representation of this automaton.
        """
        raise NotImplementedError("Abstract method not callable")

    def display_state_diagram(self) -> None:
        """
        Show the automaton's state diagram in browser.
        """
        print("Opening state diagram in default browser...")
        html_contents = f"""\
            <!doctypehtml>
            <title>Automaton Visualization</title>
            <meta content="View your automaton in the browser"name=description>
            <script src=https://cdn.jsdelivr.net/npm/@viz-js/viz@3.2.4/lib/viz-standalone.min.js></script>
            <script>Viz.instance().then(function(e){{var n=e.renderSVGElement('{self._as_dot_string()}');document.getElementById("graph").appendChild(n)}})</script>
            <div id=graph></div>\
        """
        render_html.render_in_browser(html_contents, str(Path.cwd() / "preview.html"))
