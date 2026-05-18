def quote_str(s: str) -> str:
    """
    Return the given string in quotes.

    :param s: An input string to wrap in quotes.
    :return: The input string wrapped in quotes.
    """
    return f'"{s}"'


def set_str(s: set[str]) -> str:
    """
    Return a string representation of a Set, specifically designed to use "{}"
    for an empty set instead of "set()".

    :param s: A Set to represent as a string.
    :return: The string representation of the input Set.
    """
    if len(s) == 0:
        return "{}"
    else:
        return str(s)
