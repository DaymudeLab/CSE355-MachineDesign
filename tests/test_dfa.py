from cse355_machine_design import DFA, NFA, State

from itertools import chain, product

import pytest
from typeguard import TypeCheckError


@pytest.fixture
def dfa_unary_empty() -> DFA:
    """
    A DFA with a unary alphabet and an empty language.
    """
    Q: set[State] = {"q_0"}
    Sigma: set[str] = {"a"}
    delta: dict[tuple[State, str], State] = {("q_0", "a"): "q_0"}
    q0: State = "q_0"
    F: set[State] = set()

    return DFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def dfa_only_ab() -> DFA:
    """
    A DFA recognizing {"ab"}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2", "q_3"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], State] = {
        ("q_0", "a"): "q_1",
        ("q_0", "b"): "q_3",
        ("q_1", "a"): "q_3",
        ("q_1", "b"): "q_2",
        ("q_2", "a"): "q_3",
        ("q_2", "b"): "q_3",
        ("q_3", "a"): "q_3",
        ("q_3", "b"): "q_3",
    }
    q0: State = "q_0"
    F: set[State] = {"q_2"}

    return DFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def dfa_substring_ab() -> DFA:
    """
    A DFA recognizing {w in {a,b}* | "ab" is a substring of w}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], State] = {
        ("q_0", "a"): "q_1",
        ("q_0", "b"): "q_0",
        ("q_1", "a"): "q_1",
        ("q_1", "b"): "q_2",
        ("q_2", "a"): "q_2",
        ("q_2", "b"): "q_2",
    }
    q0: State = "q_0"
    F: set[State] = {"q_2"}

    return DFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def dfa_exactly_two_a() -> DFA:
    """
    A DFA recognizing {w in {a,b}* | w contains exactly two a's}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2", "q_3"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], State] = {
        ("q_0", "a"): "q_1",
        ("q_0", "b"): "q_0",
        ("q_1", "a"): "q_2",
        ("q_1", "b"): "q_1",
        ("q_2", "a"): "q_3",
        ("q_2", "b"): "q_2",
        ("q_3", "a"): "q_3",
        ("q_3", "b"): "q_3",
    }
    q0: State = "q_0"
    F: set[State] = {"q_2"}

    return DFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def dfa_even_number_a() -> DFA:
    """
    A DFA recognizing {w in {a,b}* | w contains an even number of a's}.
    """
    Q: set[State] = {"q_0", "q_1"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], State] = {
        ("q_0", "a"): "q_1",
        ("q_0", "b"): "q_0",
        ("q_1", "a"): "q_0",
        ("q_1", "b"): "q_1",
    }
    q0: State = "q_0"
    F: set[State] = {"q_0"}

    return DFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def dfa_even_number_a_alt() -> DFA:
    """
    A DFA recognizing {w in {a,b}* | w contains an even number of a's}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2", "q_3"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], State] = {
        ("q_0", "a"): "q_1",
        ("q_0", "b"): "q_0",
        ("q_1", "a"): "q_2",
        ("q_1", "b"): "q_1",
        ("q_2", "a"): "q_3",
        ("q_2", "b"): "q_2",
        ("q_3", "a"): "q_0",
        ("q_3", "b"): "q_3",
    }
    q0: State = "q_0"
    F: set[State] = {"q_0", "q_2"}

    return DFA(Q, Sigma, delta, q0, F)


class TestDFAInitValidate:
    """
    Test DFA initialization and validation.
    """

    @pytest.fixture(autouse=True)
    def init_dfa_params(self) -> None:
        self.Q: set[State] = {"q_0", "q_1", "q_2"}
        self.Sigma: set[str] = {"a", "b"}
        self.delta: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_0",
            ("q_1", "b"): "q_2",
            ("q_2", "a"): "q_0",
            ("q_2", "b"): "q_1",
        }
        self.q0: State = "q_0"
        self.F: set[State] = {"q_0", "q_2"}
        self.D = DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

    def test_good_init(self) -> None:
        assert self.D._automaton_type == "DFA"
        assert self.D._states == self.Q
        assert self.D._input_alphabet == self.Sigma
        assert self.D._transitions == self.delta
        assert self.D._start_state == self.q0
        assert self.D._accept_states == self.F

    def test_copy_safety_states(self) -> None:
        self.Q.pop()
        assert self.D._states != self.Q

    def test_copy_safety_input_alphabet(self) -> None:
        self.Sigma.pop()
        assert self.D._input_alphabet != self.Sigma

    def test_copy_safety_transitions_keys(self) -> None:
        del self.delta[("q_2", "b")]
        assert self.D._transitions != self.delta

    def test_copy_safety_transitions_values(self) -> None:
        self.delta[("q_2", "b")] = "q_2"
        assert self.D._transitions != self.delta

    def test_copy_safety_start_state(self) -> None:
        self.q0 = "q_1"
        assert self.D._start_state != self.q0

    def test_copy_safety_accept_states(self) -> None:
        self.F.pop()
        assert self.D._accept_states != self.F

    def test_bad_init_types(self) -> None:
        with pytest.raises(TypeCheckError):
            DFA({0}, self.Sigma, self.delta, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            DFA({"q_0", 1}, self.Sigma, self.delta, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            DFA(self.Q, {0, 1}, self.delta, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            DFA(self.Q, self.Sigma, {"q_0": "q_1"}, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            DFA(self.Q, self.Sigma, self.delta, 0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            DFA(self.Q, self.Sigma, self.delta, self.q0, {0, 1})  # type: ignore

    def test_empty_states(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DFA(set(), self.Sigma, self.delta, self.q0, self.F)
        assert "Empty state set." in str(excinfo.value)

    def test_empty_input_alphabet(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, set(), self.delta, self.q0, self.F)
        assert "Empty input alphabet." in str(excinfo.value)

    def test_multichar_input_symbol(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, {"a", "bc"}, self.delta, self.q0, self.F)
        assert "Invalid input alphabet symbol(s)." in str(excinfo.value)

    def test_invalid_start_state(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, "q_invalid", self.F)
        assert "Invalid start state." in str(excinfo.value)

    def test_invalid_accept_states(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, {"q_0", "q_invalid"})
        assert "Invalid accept state(s)." in str(excinfo.value)

    def test_bad_transition_from_state(self) -> None:
        self.delta[("q_invalid", "a")] = "q_0"
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "For transition delta(q_invalid, a) = q_0," in exc_str
        assert "- 'q_invalid' is not in the state set" in exc_str

    def test_bad_transition_input_symbol(self) -> None:
        self.delta[("q_0", "x")] = "q_1"
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "For transition delta(q_0, x) = q_1," in exc_str
        assert "- 'x' is not in the input alphabet" in exc_str

    def test_bad_transition_to_state(self) -> None:
        self.delta[("q_0", "a")] = "q_invalid"
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "For transition delta(q_0, a) = q_invalid," in exc_str
        assert "- 'q_invalid' is not in the state set" in exc_str

    def test_missing_transition(self) -> None:
        del self.delta[("q_0", "a")]
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "Missing transition delta(q_0, a)" in exc_str


class TestDFAEvaluate:
    """
    Test DFA string evaluation.
    """

    def test_bad_input_str(self, dfa_unary_empty) -> None:
        with pytest.raises(ValueError) as excinfo:
            dfa_unary_empty.evaluate("abb")
        assert "Invalid input string." in str(excinfo.value)


class TestDFAGenerateStrings:
    """
    Test DFA string generation.
    """

    def test_generate_strs_1(self, dfa_only_ab) -> None:
        assert dfa_only_ab.generate_strings(max_str_len=4) == {"ab"}

    def test_generate_strs_2(self, dfa_substring_ab) -> None:
        assert dfa_substring_ab.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(
                product(["a", "b"], repeat=r) for r in range(5)
            )
            if "ab" in "".join(p)
        }

    def test_generate_strs_3(self, dfa_exactly_two_a) -> None:
        assert dfa_exactly_two_a.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(
                product(["a", "b"], repeat=r) for r in range(5)
            )
            if "".join(p).count("a") == 2
        }

    def test_generate_strs_4(self, dfa_even_number_a) -> None:
        assert dfa_even_number_a.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(
                product(["a", "b"], repeat=r) for r in range(5)
            )
            if "".join(p).count("a") % 2 == 0
        }

    def test_generate_strs_with_limit(self, dfa_exactly_two_a) -> None:
        assert dfa_exactly_two_a.generate_strings(max_str_len=4, max_strs=5) == {
            "aa",
            "aab",
            "aba",
            "baa",
            "aabb",
        }


class TestDFAEmptyLanguage:
    """
    Test DFA empty language checking.
    """

    def test_no_accept_states(self, dfa_unary_empty) -> None:
        assert dfa_unary_empty.empty()

    def test_unreachable_accept_states(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_0",
            ("q_1", "b"): "q_1",
            ("q_2", "a"): "q_2",
            ("q_2", "b"): "q_0",
        }
        q0: State = "q_0"
        F: set[State] = {"q_2"}

        D = DFA(Q, Sigma, delta, q0, F)
        assert D.empty()

    def test_not_empty(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_0",
            ("q_1", "b"): "q_2",
            ("q_2", "a"): "q_0",
            ("q_2", "b"): "q_1",
        }
        q0: State = "q_0"
        F: set[State] = {"q_2"}

        D = DFA(Q, Sigma, delta, q0, F)
        assert not D.empty()


class TestDFAPruneUnreachable:
    """
    Test DFA removal of unreachable states.
    """

    def test_all_reachable(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_0",
            ("q_1", "b"): "q_2",
            ("q_2", "a"): "q_0",
            ("q_2", "b"): "q_1",
        }
        q0: State = "q_0"
        F: set[State] = {"q_0", "q_2"}

        D = DFA(Q, Sigma, delta, q0, F)
        D.prune_unreachable()

        assert D._states == Q
        assert D._input_alphabet == Sigma
        assert D._transitions == delta
        assert D._start_state == q0
        assert D._accept_states == F

    def test_disconnected_components(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2", "q_3", "q_4"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_1",
            ("q_1", "b"): "q_0",
            ("q_2", "a"): "q_2",
            ("q_2", "b"): "q_2",
            ("q_3", "a"): "q_4",
            ("q_3", "b"): "q_4",
            ("q_4", "a"): "q_3",
            ("q_4", "b"): "q_4",
        }
        q0: State = "q_0"
        F: set[State] = {"q_0", "q_2"}

        D = DFA(Q, Sigma, delta, q0, F)
        D.prune_unreachable()

        assert D._states == {"q_0", "q_1"}
        assert D._input_alphabet == Sigma
        assert D._transitions == {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_1",
            ("q_1", "b"): "q_0",
        }
        assert D._start_state == q0
        assert D._accept_states == {"q_0"}

    def test_backreferencing_components(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2", "q_3", "q_4"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_1",
            ("q_1", "b"): "q_0",
            ("q_2", "a"): "q_2",
            ("q_2", "b"): "q_0",
            ("q_3", "a"): "q_4",
            ("q_3", "b"): "q_0",
            ("q_4", "a"): "q_3",
            ("q_4", "b"): "q_0",
        }
        q0: State = "q_0"
        F: set[State] = {"q_0", "q_2"}

        D = DFA(Q, Sigma, delta, q0, F)
        D.prune_unreachable()

        assert D._states == {"q_0", "q_1"}
        assert D._input_alphabet == Sigma
        assert D._transitions == {
            ("q_0", "a"): "q_0",
            ("q_0", "b"): "q_1",
            ("q_1", "a"): "q_1",
            ("q_1", "b"): "q_0",
        }
        assert D._start_state == q0
        assert D._accept_states == {"q_0"}


class TestDFAOperations:
    """
    Test DFA operations including complement, union, intersection, difference,
    and symmetric difference.
    """

    @pytest.fixture(autouse=True)
    def setup_dfas(self, dfa_substring_ab, dfa_exactly_two_a) -> None:
        self.D_1 = dfa_substring_ab
        self.D_2 = dfa_exactly_two_a

        # Invariant product DFA information.
        self.target_states: set[State] = {
            "(q_0,q_0)",
            "(q_0,q_1)",
            "(q_0,q_2)",
            "(q_0,q_3)",
            "(q_1,q_0)",
            "(q_1,q_1)",
            "(q_1,q_2)",
            "(q_1,q_3)",
            "(q_2,q_0)",
            "(q_2,q_1)",
            "(q_2,q_2)",
            "(q_2,q_3)",
        }
        self.target_states_pruned: set[State] = {
            "(q_0,q_0)",
            "(q_1,q_1)",
            "(q_1,q_2)",
            "(q_1,q_3)",
            "(q_2,q_1)",
            "(q_2,q_2)",
            "(q_2,q_3)",
        }
        self.target_transitions: dict[tuple[State, str], State] = {
            ("(q_0,q_0)", "a"): "(q_1,q_1)",
            ("(q_0,q_0)", "b"): "(q_0,q_0)",
            ("(q_0,q_1)", "a"): "(q_1,q_2)",
            ("(q_0,q_1)", "b"): "(q_0,q_1)",
            ("(q_0,q_2)", "a"): "(q_1,q_3)",
            ("(q_0,q_2)", "b"): "(q_0,q_2)",
            ("(q_0,q_3)", "a"): "(q_1,q_3)",
            ("(q_0,q_3)", "b"): "(q_0,q_3)",
            ("(q_1,q_0)", "a"): "(q_1,q_1)",
            ("(q_1,q_0)", "b"): "(q_2,q_0)",
            ("(q_1,q_1)", "a"): "(q_1,q_2)",
            ("(q_1,q_1)", "b"): "(q_2,q_1)",
            ("(q_1,q_2)", "a"): "(q_1,q_3)",
            ("(q_1,q_2)", "b"): "(q_2,q_2)",
            ("(q_1,q_3)", "a"): "(q_1,q_3)",
            ("(q_1,q_3)", "b"): "(q_2,q_3)",
            ("(q_2,q_0)", "a"): "(q_2,q_1)",
            ("(q_2,q_0)", "b"): "(q_2,q_0)",
            ("(q_2,q_1)", "a"): "(q_2,q_2)",
            ("(q_2,q_1)", "b"): "(q_2,q_1)",
            ("(q_2,q_2)", "a"): "(q_2,q_3)",
            ("(q_2,q_2)", "b"): "(q_2,q_2)",
            ("(q_2,q_3)", "a"): "(q_2,q_3)",
            ("(q_2,q_3)", "b"): "(q_2,q_3)",
        }
        self.target_transitions_pruned: dict[tuple[State, str], State] = {
            ("(q_0,q_0)", "a"): "(q_1,q_1)",
            ("(q_0,q_0)", "b"): "(q_0,q_0)",
            ("(q_1,q_1)", "a"): "(q_1,q_2)",
            ("(q_1,q_1)", "b"): "(q_2,q_1)",
            ("(q_1,q_2)", "a"): "(q_1,q_3)",
            ("(q_1,q_2)", "b"): "(q_2,q_2)",
            ("(q_1,q_3)", "a"): "(q_1,q_3)",
            ("(q_1,q_3)", "b"): "(q_2,q_3)",
            ("(q_2,q_1)", "a"): "(q_2,q_2)",
            ("(q_2,q_1)", "b"): "(q_2,q_1)",
            ("(q_2,q_2)", "a"): "(q_2,q_3)",
            ("(q_2,q_2)", "b"): "(q_2,q_2)",
            ("(q_2,q_3)", "a"): "(q_2,q_3)",
            ("(q_2,q_3)", "b"): "(q_2,q_3)",
        }
        self.target_start_state: State = "(q_0,q_0)"

        # Test strings.
        self.test_strs: dict[str, set[str]] = {
            "D_1_only": {"ab", "aaab", "abbbb", "abababab", "abbbabbba"},
            "D_2_only": {"aa", "baa", "bbbaa", "bbbbbbbaa"},
            "both": {"aab", "aba", "abbbbbbba", "ababbbb"},
            "neither": {"", "a", "b", "ba", "bb", "aaaaa", "baaaaaaa"},
        }

    def test_mismatched_alphabets(self) -> None:
        self.D_2._input_alphabet = {"0", "1"}

        with pytest.raises(ValueError) as excinfo:
            self.D_1.union(self.D_2)
        assert "Mismatched input alphabets." in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            self.D_1 | self.D_2
        assert "Mismatched input alphabets." in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            self.D_1.intersection(self.D_2)
        assert "Mismatched input alphabets." in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            self.D_1 & self.D_2
        assert "Mismatched input alphabets." in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            self.D_1.difference(self.D_2)
        assert "Mismatched input alphabets." in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            self.D_1 - self.D_2
        assert "Mismatched input alphabets." in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            self.D_1.symmetric_difference(self.D_2)
        assert "Mismatched input alphabets." in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            self.D_1 ^ self.D_2
        assert "Mismatched input alphabets." in str(excinfo.value)

    def test_complement(self) -> None:
        D = ~self.D_1

        assert D._states == self.D_1._states
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.D_1._transitions
        assert D._start_state == self.D_1._start_state
        assert D._accept_states == {"q_0", "q_1"}

        assert all(not D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(D.evaluate(w) for w in self.test_strs["neither"])

    def test_union_with_prune(self) -> None:
        D = self.D_1 | self.D_2

        assert D._states == self.target_states_pruned
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.target_transitions_pruned
        assert D._start_state == self.target_start_state
        assert D._accept_states == {
            "(q_1,q_2)",
            "(q_2,q_1)",
            "(q_2,q_2)",
            "(q_2,q_3)",
        }

        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_union_without_prune(self) -> None:
        D = self.D_1.union(self.D_2, prune_unreachable=False)

        assert D._states == self.target_states
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.target_transitions
        assert D._start_state == self.target_start_state
        assert D._accept_states == {
            "(q_0,q_2)",
            "(q_1,q_2)",
            "(q_2,q_0)",
            "(q_2,q_1)",
            "(q_2,q_2)",
            "(q_2,q_3)",
        }

        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_intersection_with_prune(self) -> None:
        D = self.D_1 & self.D_2

        assert D._states == self.target_states_pruned
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.target_transitions_pruned
        assert D._start_state == self.target_start_state
        assert D._accept_states == {"(q_2,q_2)"}

        assert all(not D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_intersection_without_prune(self) -> None:
        D = self.D_1.intersection(self.D_2, prune_unreachable=False)

        assert D._states == self.target_states
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.target_transitions
        assert D._start_state == self.target_start_state
        assert D._accept_states == {"(q_2,q_2)"}

        assert all(not D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_difference_with_prune(self) -> None:
        D = self.D_1 - self.D_2

        assert D._states == self.target_states_pruned
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.target_transitions_pruned
        assert D._start_state == self.target_start_state
        assert D._accept_states == {"(q_2,q_1)", "(q_2,q_3)"}

        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_difference_without_prune(self) -> None:
        D = self.D_1.difference(self.D_2, prune_unreachable=False)

        assert D._states == self.target_states
        assert D._input_alphabet == self.D_1._input_alphabet
        assert D._transitions == self.target_transitions
        assert D._start_state == self.target_start_state
        assert D._accept_states == {"(q_2,q_0)", "(q_2,q_1)", "(q_2,q_3)"}

        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_symmetric_difference_with_prune(self) -> None:
        D = self.D_1 ^ self.D_2

        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_symmetric_difference_without_prune(self) -> None:
        D = self.D_1.symmetric_difference(self.D_2, prune_unreachable=False)

        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])


class TestDFAComparison:
    """
    Test DFA language comparisons.
    """

    def test_invalid_comparison(self, dfa_unary_empty) -> None:
        N = NFA({"q_0"}, {"a"}, {("q_0", "a"): {"q_0"}}, "q_0", {"q_0"})
        with pytest.raises(TypeError) as excinfo:
            dfa_unary_empty.compare(N)
        assert "Invalid comparison type." in str(excinfo.value)

    def test_mismatched_alphabets(self, dfa_unary_empty, dfa_only_ab) -> None:
        with pytest.raises(ValueError) as excinfo:
            dfa_unary_empty.compare(dfa_only_ab)
        assert "Mismatched input alphabets." in str(excinfo.value)

    def test_disjoint(self, dfa_only_ab, dfa_exactly_two_a) -> None:
        assert dfa_only_ab.is_disjoint(dfa_exactly_two_a)

    def test_partial_intersection(self, dfa_substring_ab, dfa_exactly_two_a) -> None:
        assert dfa_substring_ab.partially_intersects(dfa_exactly_two_a)

    def test_subset_1(self, dfa_only_ab, dfa_substring_ab) -> None:
        assert dfa_only_ab < dfa_substring_ab
        assert dfa_only_ab <= dfa_substring_ab
        assert dfa_only_ab != dfa_substring_ab

    def test_superset_1(self, dfa_only_ab, dfa_substring_ab) -> None:
        assert dfa_substring_ab > dfa_only_ab
        assert dfa_substring_ab >= dfa_only_ab
        assert dfa_substring_ab != dfa_only_ab

    def test_subset_2(self, dfa_exactly_two_a, dfa_even_number_a) -> None:
        assert dfa_exactly_two_a < dfa_even_number_a
        assert dfa_exactly_two_a <= dfa_even_number_a
        assert dfa_exactly_two_a != dfa_even_number_a

    def test_superset_2(self, dfa_exactly_two_a, dfa_even_number_a) -> None:
        assert dfa_even_number_a > dfa_exactly_two_a
        assert dfa_even_number_a >= dfa_exactly_two_a
        assert dfa_even_number_a != dfa_exactly_two_a

    def test_equality(self, dfa_even_number_a, dfa_even_number_a_alt) -> None:
        assert dfa_even_number_a == dfa_even_number_a_alt
        assert dfa_even_number_a >= dfa_even_number_a_alt
        assert dfa_even_number_a <= dfa_even_number_a_alt
