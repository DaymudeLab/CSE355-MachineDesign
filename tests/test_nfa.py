from cse355_machine_design import NFA, PDA, State

from itertools import chain, product
from typing import Any

import pytest
from typeguard import TypeCheckError


@pytest.fixture
def nfa_only_eps_a() -> NFA:
    """
    An NFA recognizing {"", "a"}.
    """
    Q: set[State] = {"q_0", "q_1"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], set[State]] = {
        ("q_0", "a"): {"q_1"},
    }
    q0: State = "q_0"
    F: set[State] = {"q_0", "q_1"}

    return NFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def nfa_ends_with_aa() -> NFA:
    """
    An NFA recognizing {w in {a,b}* | w ends with "aa"}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], set[State]] = {
        ("q_0", "a"): {"q_0", "q_1"},
        ("q_0", "b"): {"q_0"},
        ("q_1", "a"): {"q_2"},
    }
    q0: State = "q_0"
    F: set[State] = {"q_2"}

    return NFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def nfa_substring_aa_aba() -> NFA:
    """
    An NFA recognizing {w in {a,b}* | "aa" or "aba" is a substring of w}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2", "q_3"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], set[State]] = {
        ("q_0", "a"): {"q_0", "q_1"},
        ("q_0", "b"): {"q_0"},
        ("q_1", "b"): {"q_2"},
        ("q_1", "_"): {"q_2"},
        ("q_2", "a"): {"q_3"},
        ("q_3", "a"): {"q_3"},
        ("q_3", "b"): {"q_3"},
    }
    q0: State = "q_0"
    F: set[State] = {"q_3"}

    return NFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def nfa_one_or_even_b() -> NFA:
    """
    An NFA recognizing {w in {a,b}* | w contains one or an even number of b's}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2", "q_3"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], set[State]] = {
        ("q_0", "a"): {"q_0"},
        ("q_0", "b"): {"q_1"},
        ("q_1", "a"): {"q_1"},
        ("q_1", "b"): {"q_2"},
        ("q_2", "a"): {"q_2"},
        ("q_2", "b"): {"q_3"},
        ("q_3", "a"): {"q_3"},
        ("q_3", "b"): {"q_2"},
    }
    q0: State = "q_0"
    F: set[State] = {"q_0", "q_1", "q_2"}

    return NFA(Q, Sigma, delta, q0, F)


@pytest.fixture
def nfa_one_or_even_b_alt() -> NFA:
    """
    An NFA recognizing {w in {a,b}* | w contains one or an even number of b's}.
    """
    Q: set[State] = {"q_0", "q_1", "q_2", "q_3", "q_4"}
    Sigma: set[str] = {"a", "b"}
    delta: dict[tuple[State, str], set[State]] = {
        ("q_0", "_"): {"q_1", "q_3"},
        ("q_1", "a"): {"q_1"},
        ("q_1", "b"): {"q_2"},
        ("q_2", "a"): {"q_2"},
        ("q_3", "a"): {"q_3"},
        ("q_3", "b"): {"q_4"},
        ("q_4", "a"): {"q_4"},
        ("q_4", "b"): {"q_3"},
    }
    q0: State = "q_0"
    F: set[State] = {"q_2", "q_3"}

    return NFA(Q, Sigma, delta, q0, F)


class TestNFAInitValidate:
    """
    Test NFA initialization and validation.
    """

    @pytest.fixture(autouse=True)
    def init_nfa_params(self) -> None:
        self.Q: set[State] = {"q_0", "q_1", "q_2"}
        self.Sigma: set[str] = {"a", "b"}
        self.delta: dict[tuple[State, str], set[State]] = {
            ("q_0", "b"): {"q_1"},
            ("q_0", "e"): {"q_2"},
            ("q_1", "a"): {"q_1", "q_2"},
            ("q_1", "b"): {"q_2"},
            ("q_2", "a"): {"q_0"},
        }
        self.q0: State = "q_0"
        self.F: set[State] = {"q_0"}
        self.epsilon: str = "e"
        self.N = NFA(self.Q, self.Sigma, self.delta, self.q0, self.F, self.epsilon)

    def test_good_init(self) -> None:
        assert self.N._automaton_type == "NFA"
        assert self.N._states == self.Q
        assert self.N._input_alphabet == self.Sigma
        assert self.N._transitions == self.delta
        assert self.N._start_state == self.q0
        assert self.N._accept_states == self.F
        assert self.N._epsilon == self.epsilon

    def test_copy_safety_states(self) -> None:
        self.Q.pop()
        assert self.N._states != self.Q

    def test_copy_safety_input_alphabet(self) -> None:
        self.Sigma.pop()
        assert self.N._input_alphabet != self.Sigma

    def test_copy_safety_transitions_keys(self) -> None:
        self.delta[("q_2", "b")] = {"q_1", "q_2"}
        assert self.N._transitions != self.delta

    def test_copy_safety_transitions_values(self) -> None:
        self.delta[("q_2", "a")].pop()
        assert self.N._transitions != self.delta

    def test_copy_safety_start_state(self) -> None:
        self.q0 = "q_1"
        assert self.N._start_state != self.q0

    def test_copy_safety_accept_states(self) -> None:
        self.F.pop()
        assert self.N._accept_states != self.F

    def test_copy_safety_epsilon(self) -> None:
        self.epsilon = "_"
        assert self.N._epsilon != self.epsilon

    def test_bad_init_types(self) -> None:
        with pytest.raises(TypeCheckError):
            NFA({0}, self.Sigma, self.delta, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA({"q_0", 1}, self.Sigma, self.delta, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA(self.Q, {0, 1}, self.delta, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA(self.Q, self.Sigma, {"q_0": "q_1"}, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA(self.Q, self.Sigma, {("q_0", "a"): "q_1"}, self.q0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA(self.Q, self.Sigma, self.delta, 0, self.F)  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA(self.Q, self.Sigma, self.delta, self.q0, {0, 1})  # type: ignore
        with pytest.raises(TypeCheckError):
            NFA(self.Q, self.Sigma, self.delta, self.q0, self.F, None)  # type: ignore

    def test_empty_states(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(set(), self.Sigma, self.delta, self.q0, self.F)
        assert "Empty state set." in str(excinfo.value)

    def test_empty_input_alphabet(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, set(), self.delta, self.q0, self.F)
        assert "Empty input alphabet." in str(excinfo.value)

    def test_multichar_input_symbol(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, {"a", "bc"}, self.delta, self.q0, self.F)
        assert "Invalid input alphabet symbol(s)." in str(excinfo.value)

    def test_invalid_start_state(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, "q_invalid", self.F)
        assert "Invalid start state." in str(excinfo.value)

    def test_invalid_accept_states(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, self.q0, {"q_0", "q_invalid"})
        assert "Invalid accept state(s)." in str(excinfo.value)

    def test_multichar_epsilon_symbol(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, self.q0, self.F, epsilon="eps")
        assert "Invalid epsilon symbol." in str(excinfo.value)

    def test_invalid_epsilon_symbol(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, self.q0, self.F, epsilon="a")
        assert "Invalid epsilon symbol." in str(excinfo.value)

    def test_bad_transition_from_state(self) -> None:
        self.delta[("q_invalid", "a")] = {"q_0"}
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "For transition delta(q_invalid, a) = {'q_0'}," in exc_str
        assert "- 'q_invalid' is not in the state set" in exc_str

    def test_bad_transition_input_symbol(self) -> None:
        self.delta[("q_0", "x")] = {"q_1"}
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "For transition delta(q_0, x) = {'q_1'}," in exc_str
        assert "- 'x' is not in the input alphabet" in exc_str

    def test_bad_transition_to_states(self) -> None:
        self.delta[("q_0", "a")] = {"q_invalid"}
        with pytest.raises(ValueError) as excinfo:
            NFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert "Invalid transition function:" in exc_str
        assert "For transition delta(q_0, a) = {'q_invalid'}," in exc_str
        assert "- {'q_invalid'} are not in the state set" in exc_str


class TestNFAEvaluate:
    """
    Test NFA string evaluation.
    """

    def test_bad_input_str(self, nfa_only_eps_a) -> None:
        with pytest.raises(ValueError) as excinfo:
            nfa_only_eps_a.evaluate("011")
        assert "Invalid input string." in str(excinfo.value)


class TestNFAGenerateStrings:
    """
    Test NFA string generation.
    """

    def test_generate_strs_1(self, nfa_only_eps_a) -> None:
        assert nfa_only_eps_a.generate_strings(max_str_len=4) == {"", "a"}

    def test_generate_strs_2(self, nfa_ends_with_aa) -> None:
        assert nfa_ends_with_aa.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(
                product(["a", "b"], repeat=r) for r in range(5)
            )
            if len("".join(p)) >= 2 and "".join(p)[-2:] == "aa"
        }

    def test_generate_strs_3(self, nfa_substring_aa_aba) -> None:
        assert nfa_substring_aa_aba.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(
                product(["a", "b"], repeat=r) for r in range(5)
            )
            if "aa" in "".join(p) or "aba" in "".join(p)
        }

    def test_generate_strs_4(self, nfa_one_or_even_b) -> None:
        assert nfa_one_or_even_b.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(
                product(["a", "b"], repeat=r) for r in range(5)
            )
            if "".join(p).count("b") == 1 or "".join(p).count("b") % 2 == 0
        }

    def test_generate_strs_with_limit(self, nfa_substring_aa_aba) -> None:
        assert nfa_substring_aa_aba.generate_strings(max_str_len=4, max_strs=5) == {
            "aa",
            "aaa",
            "aab",
            "aba",
            "baa",
        }


class TestNFAComparison:
    """
    Test NFA language comparisons.
    """

    def test_invalid_comparison(self, nfa_only_eps_a) -> None:
        P = PDA(
            Q={"q_0"},
            Sigma={"a"},
            Gamma={"a"},
            delta={("q_0", "a", "_"): {("q_0", "a")}},
            q0="q_0",
            F=set(),
        )
        with pytest.raises(TypeError) as excinfo:
            nfa_only_eps_a.compare(P)
        assert "Invalid comparison type." in str(excinfo.value)

    def test_mismatched_alphabets(self, nfa_only_eps_a, nfa_ends_with_aa) -> None:
        nfa_only_eps_a._input_alphabet = {"a"}
        with pytest.raises(ValueError) as excinfo:
            nfa_only_eps_a.compare(nfa_ends_with_aa)
        assert "Mismatched input alphabets." in str(excinfo.value)

    def test_disjoint(self, nfa_only_eps_a, nfa_ends_with_aa) -> None:
        assert nfa_only_eps_a.is_disjoint(nfa_ends_with_aa)

    def test_partial_intersection(self, nfa_ends_with_aa, nfa_one_or_even_b) -> None:
        assert nfa_ends_with_aa.partially_intersects(nfa_one_or_even_b)

    def test_subset_1(self, nfa_only_eps_a, nfa_one_or_even_b) -> None:
        assert nfa_only_eps_a < nfa_one_or_even_b
        assert nfa_only_eps_a <= nfa_one_or_even_b
        assert nfa_only_eps_a != nfa_one_or_even_b

    def test_superset_1(self, nfa_only_eps_a, nfa_one_or_even_b) -> None:
        assert nfa_one_or_even_b > nfa_only_eps_a
        assert nfa_one_or_even_b >= nfa_only_eps_a
        assert nfa_one_or_even_b != nfa_only_eps_a

    def test_subset_2(self, nfa_ends_with_aa, nfa_substring_aa_aba) -> None:
        assert nfa_ends_with_aa < nfa_substring_aa_aba
        assert nfa_ends_with_aa <= nfa_substring_aa_aba
        assert nfa_ends_with_aa != nfa_substring_aa_aba

    def test_superset_2(self, nfa_ends_with_aa, nfa_substring_aa_aba) -> None:
        assert nfa_substring_aa_aba > nfa_ends_with_aa
        assert nfa_substring_aa_aba >= nfa_ends_with_aa
        assert nfa_substring_aa_aba != nfa_ends_with_aa

    def test_equality(self, nfa_one_or_even_b, nfa_one_or_even_b_alt) -> None:
        assert nfa_one_or_even_b == nfa_one_or_even_b_alt
        assert nfa_one_or_even_b >= nfa_one_or_even_b_alt
        assert nfa_one_or_even_b <= nfa_one_or_even_b_alt


class TestNFAToDFAConversion:
    """
    Test NFA to DFA conversions.
    """

    def test_nfa_to_dfa_1(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], set[State]] = {
            ("q_0", "b"): {"q_1"},
            ("q_0", "_"): {"q_2"},
            ("q_1", "a"): {"q_1", "q_2"},
            ("q_1", "b"): {"q_2"},
            ("q_2", "a"): {"q_0"},
        }
        q0: State = "q_0"
        F: set[State] = {"q_0"}

        N = NFA(Q, Sigma, delta, q0, F)
        D = N.as_dfa()

        assert D._states == {
            "{}",
            "{'q_1'}",
            "{'q_2'}",
            "{'q_0', 'q_2'}",
            "{'q_1', 'q_2'}",
            "{'q_0', 'q_1', 'q_2'}",
        }
        assert D._input_alphabet == Sigma
        assert D._transitions == {
            ("{}", "a"): "{}",
            ("{}", "b"): "{}",
            ("{'q_1'}", "a"): "{'q_1', 'q_2'}",
            ("{'q_1'}", "b"): "{'q_2'}",
            ("{'q_2'}", "a"): "{'q_0', 'q_2'}",
            ("{'q_2'}", "b"): "{}",
            ("{'q_0', 'q_2'}", "a"): "{'q_0', 'q_2'}",
            ("{'q_0', 'q_2'}", "b"): "{'q_1'}",
            ("{'q_1', 'q_2'}", "a"): "{'q_0', 'q_1', 'q_2'}",
            ("{'q_1', 'q_2'}", "b"): "{'q_2'}",
            ("{'q_0', 'q_1', 'q_2'}", "a"): "{'q_0', 'q_1', 'q_2'}",
            ("{'q_0', 'q_1', 'q_2'}", "b"): "{'q_1', 'q_2'}",
        }
        assert D._start_state == "{'q_0', 'q_2'}"
        assert D._accept_states == {"{'q_0', 'q_2'}", "{'q_0', 'q_1', 'q_2'}"}

    def test_nfa_to_dfa_2(self) -> None:
        Q: set[State] = {"q_0", "q_1", "q_2"}
        Sigma: set[str] = {"a", "b"}
        delta: dict[tuple[State, str], set[State]] = {
            ("q_0", "a"): {"q_2"},
            ("q_0", "_"): {"q_1"},
            ("q_1", "a"): {"q_0"},
            ("q_2", "a"): {"q_1"},
            ("q_2", "b"): {"q_1", "q_2"},
        }
        q0: State = "q_0"
        F: set[State] = {"q_1"}

        N = NFA(Q, Sigma, delta, q0, F)
        D = N.as_dfa()

        assert D._states == {
            "{}",
            "{'q_0', 'q_1'}",
            "{'q_1', 'q_2'}",
            "{'q_0', 'q_1', 'q_2'}",
        }
        assert D._input_alphabet == Sigma
        assert D._transitions == {
            ("{}", "a"): "{}",
            ("{}", "b"): "{}",
            ("{'q_0', 'q_1'}", "a"): "{'q_0', 'q_1', 'q_2'}",
            ("{'q_0', 'q_1'}", "b"): "{}",
            ("{'q_1', 'q_2'}", "a"): "{'q_0', 'q_1'}",
            ("{'q_1', 'q_2'}", "b"): "{'q_1', 'q_2'}",
            ("{'q_0', 'q_1', 'q_2'}", "a"): "{'q_0', 'q_1', 'q_2'}",
            ("{'q_0', 'q_1', 'q_2'}", "b"): "{'q_1', 'q_2'}",
        }
        assert D._start_state == "{'q_0', 'q_1'}"
        assert D._accept_states == {
            "{'q_0', 'q_1'}",
            "{'q_1', 'q_2'}",
            "{'q_0', 'q_1', 'q_2'}",
        }


class TestNFAToFromDict:
    """
    Test NFA dictionary representation creation and parsing.
    """

    @pytest.fixture(autouse=True)
    def dict_substring_aa_aba(self) -> None:
        self.dict_rep: dict[str, Any] = {
            "type": "NFA",
            "states": {"q_0", "q_1", "q_2", "q_3"},
            "input_alphabet": {"a", "b"},
            "transitions": [
                {"from_state": "q_0", "input_sym": "a", "to_states": {"q_0", "q_1"}},
                {"from_state": "q_0", "input_sym": "b", "to_states": {"q_0"}},
                {"from_state": "q_1", "input_sym": "b", "to_states": {"q_2"}},
                {"from_state": "q_1", "input_sym": "_", "to_states": {"q_2"}},
                {"from_state": "q_2", "input_sym": "a", "to_states": {"q_3"}},
                {"from_state": "q_3", "input_sym": "a", "to_states": {"q_3"}},
                {"from_state": "q_3", "input_sym": "b", "to_states": {"q_3"}},
            ],
            "start_state": "q_0",
            "accept_states": {"q_3"},
            "epsilon": "_",
        }

    def test_nfa_to_dict(self, nfa_substring_aa_aba) -> None:
        assert nfa_substring_aa_aba.as_dict() == self.dict_rep

    def test_dfa_from_dict(self, nfa_substring_aa_aba) -> None:
        N = NFA.from_dict(self.dict_rep)

        assert N._states == nfa_substring_aa_aba._states
        assert N._input_alphabet == nfa_substring_aa_aba._input_alphabet
        assert all(
            k in nfa_substring_aa_aba._transitions
            and v == nfa_substring_aa_aba._transitions[k]
            for (k, v) in N._transitions.items()
        )
        assert N._start_state == nfa_substring_aa_aba._start_state
        assert N._accept_states == nfa_substring_aa_aba._accept_states
