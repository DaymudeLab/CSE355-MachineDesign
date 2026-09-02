from cse355_machine_design import registry

from abc import ABC, abstractmethod
from pathlib import Path

import render_html


# Define a type alias for automata states (to avoid confusion with usual strs).
type State = str


class _Automaton(ABC):
    """
    An abstract base class for automata.
    """

    # Automata variables; note that the transition function is not included
    # here because it must be defined by a specific derived class.
    _type: str  # Type of automata (DFA, NFA, PDA, etc.).
    _states: set[State]  # State set Q.
    _alphabet: set[str]  # Alphabet Sigma.
    _start_state: State  # Start state q0.
    _accept_states: set[State]  # Accept states F.

    def __init__(self) -> None:
        """
        Create a new automaton and then validate it.
        """
        self.validate()

    @abstractmethod
    def validate(self) -> None:
        """
        Validate this automaton according to its formal definition.
        """
        raise NotImplementedError("Abstract method not callable")

    @abstractmethod
    def evaluate(self, input_str: str, trace: bool = False) -> bool:
        """
        Evaluate this automaton on the given input string.

        :param input_str: An input string to evaluate.
        :param trace: True iff tracing information should be printed.
        :return: True iff the automaton accepts the input string.
        """
        raise NotImplementedError("Abstract method not callable")

    def submit_as_answer(self, problem_number: int) -> None:
        """
        Submit this automaton to the registry as the answer to a given problem.
        Overwrites any previously submitted automaton for the same problem.

        :param problem_number: The problem to submit this as an answer for.
        """
        registry.add_to_registry(self._type, problem_number, self)

    @abstractmethod
    def as_dict(self) -> dict:
        """
        Get a dict representation of this automaton.

        :return: A dict representation of this automaton.
        """
        raise NotImplementedError("Abstract method not callable")

    @abstractmethod
    def _as_dot_string(self) -> str:
        """
        Get a DOT string representation of this automaton for use in graphviz
        visualization.

        :return: A DOT string representation of this automaton.
        """
        raise NotImplementedError("Abstract method not callable")

    def display_state_diagram(self) -> None:
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
