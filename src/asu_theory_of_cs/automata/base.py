from abc import ABC, abstractmethod
from typing import Set
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from urllib.parse import quote
import io

class _Automata(ABC):
    states: Set[str]
    input_symbols: Set[str]
    start_state: str

    def __init__(self) -> None:
        self.verify()

    @abstractmethod
    def display(self) -> str:
        """
        Display the machine state diagram as a graphviz plot
        """
        raise NotImplementedError("Abstract method not callable")
    
    def render(self) -> None:
        req_url = "https://image-charts.com/chart?chl={}&cht=gv".format(quote(self.display()))
        with urllib.request.urlopen(req_url) as url:
            img_raw = io.BytesIO(url.read())
        pillow_img = Image.open(img_raw)
        gray = ImageOps.grayscale(pillow_img)
        img = np.asarray(gray)
        print(img.shape)
        plt.grid(False)
        plt.axis('off') 
        plt.imshow(img, cmap="gray")
        plt.show()

    @abstractmethod
    def verify(self) -> None:
        """
        Verify the basic n-tuple requirements
        """
        raise NotImplementedError("Abstract method not callable")
