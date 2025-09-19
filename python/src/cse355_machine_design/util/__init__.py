from typing import Set


def quote_str(string: str) -> str:
    return f'"{string}"'


def set_str(input: Set[str]) -> str:
    if len(input) == 0:
        return "{}"
    else:
        return str(input)
