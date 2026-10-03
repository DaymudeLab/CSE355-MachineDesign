from cse355_machine_design.automata import DFA, NFA, PDA

import json
from pathlib import Path
from typing import Any


class AutomataRegistry:
    """
    Registry of finalized automata.
    """

    _dfas: dict[int, DFA]
    _nfas: dict[int, NFA]
    _pdas: dict[int, PDA]

    def __init__(self) -> None:
        """
        Construct a new (empty) automata registry.
        """
        self._dfas: dict[int, DFA] = {}
        self._nfas: dict[int, NFA] = {}
        self._pdas: dict[int, PDA] = {}

    def add(self, automaton: DFA | NFA | PDA, id: int) -> None:
        """
        Add an automaton to the registry with the given identifier. Overwrites
        any existing automaton of the same type with the same identifier.

        :param automaton: A DFA, NFA, or PDA to add to the registry.
        :param id: An int identifier for the automaton.
        """
        if isinstance(automaton, DFA):
            self._dfas[id] = automaton
        elif isinstance(automaton, NFA):
            self._nfas[id] = automaton
        elif isinstance(automaton, PDA):
            self._pdas[id] = automaton

    def export_to_json(self, export_path: Path = Path("registry.json")) -> None:
        """
        Export the automata registry as a JSON file, leveraging the automata
        dictionary representations.

        :param export_path: The filepath to export to.
        """
        export_data: dict[str, dict[int, dict[str, Any]]] = {
            "dfas": {},
            "nfas": {},
            "pdas": {},
        }

        # Store automata as dictionary representations.
        for id, dfa in self._dfas.items():
            export_data["dfas"][id] = dfa.as_dict()
        for id, nfa in self._nfas.items():
            export_data["nfas"][id] = nfa.as_dict()
        for id, pda in self._pdas.items():
            export_data["pdas"][id] = pda.as_dict()

        # Write data to file.
        with open(export_path) as f:
            json.dump(export_data, f)

        print(f"Exported all registered automata as {export_path}.")

    def import_from_json(self, import_path: Path = Path("registry.json")) -> None:
        """
        Import the automata registry from a JSON file, leveraging the automata
        dictionary parsing functions. Overwrites any existing automata in the
        registry using the same identifiers.

        :param import_path: The filepath to export from.
        """
        # Read data from file.
        with open(import_path, "r") as f:
            import_data: dict[str, dict[int, dict[str, Any]]] = json.load(f)

        # Parse automata from dictionary representations.
        for id, dfa_as_dict in import_data["dfas"].items():
            self.add(DFA.from_dict(dfa_as_dict), id)
        for id, nfa_as_dict in import_data["nfas"].items():
            self.add(NFA.from_dict(nfa_as_dict), id)
        for id, pda_as_dict in import_data["pdas"].items():
            self.add(PDA.from_dict(pda_as_dict), id)

        print(f"Added all automata from {import_path} to the registry.")
