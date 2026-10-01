# Automatically instrument runtime type-checking for all type-annotated
# functions in the automata module, configured to type-check all elements in a
# collection instead of just the first one (default).
from typeguard import config, install_import_hook, CollectionCheckStrategy

config.collection_check_strategy = CollectionCheckStrategy.ALL_ITEMS
install_import_hook("cse355_machine_design.automata")

# Import public-facing package contents.
from .automata import DFA, NFA, PDA, CFG, BBTM
from . import registry
