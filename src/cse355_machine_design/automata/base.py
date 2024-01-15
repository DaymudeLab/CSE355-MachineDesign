from abc import ABC, abstractmethod
from typing import Set, Union

# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# import numpy as np
# import urllib.request
# from urllib.parse import quote
# import io
import render_html
import pathlib


class _Automata(ABC):
    _states: Set[str]
    _input_symbols: Set[str]
    _start_state: str

    def __init__(self) -> None:
        self.verify_legality()

    @abstractmethod
    def verify_legality(self) -> None:
        """
        Verify the basic n-tuple requirements
        """
        raise NotImplementedError("Abstract method not callable")

    @abstractmethod
    def evaluate(self, input_str: str, enable_trace=0) -> Union[bool, None]:
        """
        Evaluate the automata on a string
        """
        raise NotImplementedError("Abstract method not callable")

    @abstractmethod
    def submit_as_answer(self, question_number: int) -> None:
        """
        Submit this automata as the answer to a given question.

        #### Enter the question number exactly as seen on the homework assignment.

        This overrides any prior calls to this function for the given question.
        """
        raise NotImplementedError("Abstract method not callable")

    @abstractmethod
    def export_to_dict(self) -> dict:
        """
        In the following Python Dict spec, describe the machine:
        {
            "type": "dfa",
            "states": str[],
            "input_symbols": str[]
            "delta": {
                        "from": str,
                        "to": str,
                        "input": str[] //<-- singleton array
                    }[]
            "start_state": str,
            "finals": str[]
        } |
        {
            "type": "nfa",
            "states": str[],
            "input_symbols": str[]
            "delta": {
                        "from": str,
                        "to": str[],
                        "input": str
                    }[]
            "start_state": str,
            "finals": str[],
            "epsilon_char": str
        }
        """
        raise NotImplementedError("Abstract method not callable")

    @abstractmethod
    def _generate_dot_string(self) -> str:
        """
        Display the machine state diagram as a graphviz plot
        """
        raise NotImplementedError("Abstract method not callable")

    def display_state_diagram(self) -> None:
        html_contents = (
            '<!doctypehtml><title>Automata Visualization</title><meta content="View your automata in the browser"name=description><script src=https://cdn.jsdelivr.net/npm/@viz-js/viz@3.2.4/lib/viz-standalone.min.js></script><script>Viz.instance().then(function(e){var n=e.renderSVGElement(\''
            + self._generate_dot_string()
            + '\');document.getElementById("graph").appendChild(n)})</script><div id=graph></div>'
        )
        render_html.render_in_browser(
            html_contents, str(pathlib.Path.cwd() / "preview.html")
        )
