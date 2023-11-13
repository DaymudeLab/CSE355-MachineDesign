from abc import ABC, abstractmethod
from typing import Set, Union
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from urllib.parse import quote
import io


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
        req_url = "https://image-charts.com/chart?chl={}&cht=gv".format(
            quote(self._generate_dot_string())
        )
        print(req_url)
        with urllib.request.urlopen(req_url) as url:
            img_raw = io.BytesIO(url.read())
        pillow_img = Image.open(img_raw)
        gray = ImageOps.grayscale(pillow_img)
        img = np.asarray(gray)
        plt.grid(False)
        plt.axis("off")
        plt.imshow(img, cmap="gray")
        plt.show()
