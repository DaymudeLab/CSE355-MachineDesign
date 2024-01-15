from __future__ import annotations
from typing import TYPE_CHECKING, Union
import json

if TYPE_CHECKING:
    from typing import Dict, TypedDict, KeysView, ValuesView

    from .. import automata
    from ..automata import base as automata_base

    class _Registry(TypedDict):
        dfa: Dict[int, automata.DFA]
        nfa: Dict[int, automata.NFA]

else:
    from .. import automata


_registry: _Registry = {"dfa": {}, "nfa": {}}


def add_to_registry(
    automata_type: str, question_number: int, submission: automata_base._Automata
) -> None:
    _registry[automata_type][question_number] = submission


def get_dfa(question_number: int) -> Union[automata.DFA, None]:
    return _registry["dfa"].get(question_number)


def get_nfa(question_number: int) -> Union[automata.NFA, None]:
    return _registry["nfa"].get(question_number)


def export_submissions() -> None:
    mod_registry = {}
    for outer_k, outer_v in _registry.items():
        mod_registry[outer_k] = {}
        for inner_k, inner_v in outer_v.items():  # type: ignore
            mod_registry[outer_k][inner_k] = inner_v.export_to_dict()
    out_str = json.dumps(mod_registry)
    with open("submissions.json", "w") as out:
        out.write(out_str)


def import_submissions(path: str = "submissions.json") -> None:
    with open(path, "r") as in_file:
        mod_registry = json.load(in_file)

    dfas: Dict[str, Dict] = mod_registry["dfa"]
    for submission_num, dfa_desc in dfas.items():
        Q = set(dfa_desc["states"])
        Sigma = set(dfa_desc["input_symbols"])
        q0 = dfa_desc["start_state"]
        F = set(dfa_desc["finals"])
        delta = {}
        for transition in dfa_desc["delta"]:
            delta[(transition["from"], transition["input"])] = transition["to"][0]
        dfa = automata.DFA(Q, Sigma, delta, q0, F)
        add_to_registry(dfa_desc["type"], int(submission_num), dfa)

    nfas: Dict[str, Dict] = mod_registry["nfa"]
    for submission_num, nfa_desc in nfas.items():
        Q = set(nfa_desc["states"])
        Sigma = set(nfa_desc["input_symbols"])
        q0 = nfa_desc["start_state"]
        F = set(nfa_desc["finals"])
        delta = {}
        for transition in nfa_desc["delta"]:
            delta[(transition["from"], transition["input"])] = set(transition["to"])
        nfa = automata.NFA(Q, Sigma, delta, q0, F, nfa_desc["epsilon_char"])
        add_to_registry(nfa_desc["type"], int(submission_num), nfa)
