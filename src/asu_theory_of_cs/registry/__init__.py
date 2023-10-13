from typing import Dict

from ..automata.dfa import _DFA


dfa_registry:Dict[str,_DFA] = {}