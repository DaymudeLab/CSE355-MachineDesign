from cse355_machine_design import DFA, NFA, State

from itertools import chain, product

import pytest
from typeguard import TypeCheckError


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

    def test_bad_input_str(self) -> None:
        D = DFA({"q_0"}, {"0"}, {("q_0", "0"): "q_0"}, "q_0", {"q_0"})
        with pytest.raises(ValueError) as excinfo:
            D.evaluate("011")
        assert "Invalid input string." in str(excinfo.value)


class TestDFAGenerateStrings:
    """
    Test DFA string generation.
    """

    def test_generate_strs_1(self) -> None:
        # Recognizes {ab}, the language containing only the string "ab".
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

        D_1 = DFA(Q, Sigma, delta, q0, F)
        assert D_1.generate_strings(max_str_len=4) == {"ab"}

    def test_generate_strs_2(self) -> None:
        # Recognizes {w in {a, b}* | ab is a substring of w}.
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

        D_2 = DFA(Q, Sigma, delta, q0, F)
        assert D_2.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(product(Sigma, repeat=r) for r in range(5))
            if "ab" in "".join(p)
        }

    def test_generate_strs_3(self) -> None:
        # Recognizes {w in {a, b}* | w has exactly two a's}.
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

        D_3 = DFA(Q, Sigma, delta, q0, F)
        assert D_3.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(product(Sigma, repeat=r) for r in range(5))
            if "".join(p).count("a") == 2
        }

    def test_generate_strs_4(self) -> None:
        # Recognizes {w in {a, b}* | w has an even number of a's}.
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

        D_4 = DFA(Q, Sigma, delta, q0, F)
        assert D_4.generate_strings(max_str_len=4) == {
            "".join(p)
            for p in chain.from_iterable(product(Sigma, repeat=r) for r in range(5))
            if "".join(p).count("a") % 2 == 0
        }

    def test_generate_strs_with_limit(self) -> None:
        # Recognizes {w in {a, b}* | w has exactly two a's}.
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

        D = DFA(Q, Sigma, delta, q0, F)
        assert D.generate_strings(max_str_len=4, max_strs=5) == {
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

    def test_no_accept_states(self) -> None:
        Q: set[State] = {"q_0"}
        Sigma: set[str] = {"a"}
        delta: dict[tuple[State, str], State] = {("q_0", "a"): "q_0"}
        q0: State = "q_0"
        F: set[State] = set()

        D = DFA(Q, Sigma, delta, q0, F)
        assert D.empty()

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

        assert (
            D._states == Q
            and D._input_alphabet == Sigma
            and D._transitions == delta
            and D._start_state == q0
            and D._accept_states == F
        )

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

        assert (
            D._states == {"q_0", "q_1"}
            and D._input_alphabet == Sigma
            and D._transitions
            == {
                ("q_0", "a"): "q_0",
                ("q_0", "b"): "q_1",
                ("q_1", "a"): "q_1",
                ("q_1", "b"): "q_0",
            }
            and D._start_state == q0
            and D._accept_states == {"q_0"}
        )

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

        assert (
            D._states == {"q_0", "q_1"}
            and D._input_alphabet == Sigma
            and D._transitions
            == {
                ("q_0", "a"): "q_0",
                ("q_0", "b"): "q_1",
                ("q_1", "a"): "q_1",
                ("q_1", "b"): "q_0",
            }
            and D._start_state == q0
            and D._accept_states == {"q_0"}
        )


class TestDFAOperations:
    """
    Test DFA operations including complement, union, intersection, difference,
    and symmetric difference.
    """

    @pytest.fixture(autouse=True)
    def setup_dfas(self) -> None:
        # Recognizes {w in {a, b}* | ab is a substring of w}.
        Q_1: set[State] = {"q_0", "q_1", "q_2"}
        Sigma_1: set[str] = {"a", "b"}
        delta_1: dict[tuple[State, str], State] = {
            ("q_0", "a"): "q_1",
            ("q_0", "b"): "q_0",
            ("q_1", "a"): "q_1",
            ("q_1", "b"): "q_2",
            ("q_2", "a"): "q_2",
            ("q_2", "b"): "q_2",
        }
        q0_1: State = "q_0"
        F_1: set[State] = {"q_2"}
        self.D_1 = DFA(Q_1, Sigma_1, delta_1, q0_1, F_1)

        # Recognizes {w in {a, b}* | w has exactly two a's}.
        Q_2: set[State] = {"q_A", "q_B", "q_C", "q_D"}
        Sigma_2: set[str] = {"a", "b"}
        delta_2: dict[tuple[State, str], State] = {
            ("q_A", "a"): "q_B",
            ("q_A", "b"): "q_A",
            ("q_B", "a"): "q_C",
            ("q_B", "b"): "q_B",
            ("q_C", "a"): "q_D",
            ("q_C", "b"): "q_C",
            ("q_D", "a"): "q_D",
            ("q_D", "b"): "q_D",
        }
        q0_2: State = "q_A"
        F_2: set[State] = {"q_C"}
        self.D_2 = DFA(Q_2, Sigma_2, delta_2, q0_2, F_2)

        # Invariant product DFA information.
        self.target_states: set[State] = {
            "(q_0,q_A)",
            "(q_0,q_B)",
            "(q_0,q_C)",
            "(q_0,q_D)",
            "(q_1,q_A)",
            "(q_1,q_B)",
            "(q_1,q_C)",
            "(q_1,q_D)",
            "(q_2,q_A)",
            "(q_2,q_B)",
            "(q_2,q_C)",
            "(q_2,q_D)",
        }
        self.target_states_pruned: set[State] = {
            "(q_0,q_A)",
            "(q_1,q_B)",
            "(q_1,q_C)",
            "(q_1,q_D)",
            "(q_2,q_B)",
            "(q_2,q_C)",
            "(q_2,q_D)",
        }
        self.target_transitions: dict[tuple[State, str], State] = {
            ("(q_0,q_A)", "a"): "(q_1,q_B)",
            ("(q_0,q_A)", "b"): "(q_0,q_A)",
            ("(q_0,q_B)", "a"): "(q_1,q_C)",
            ("(q_0,q_B)", "b"): "(q_0,q_B)",
            ("(q_0,q_C)", "a"): "(q_1,q_D)",
            ("(q_0,q_C)", "b"): "(q_0,q_C)",
            ("(q_0,q_D)", "a"): "(q_1,q_D)",
            ("(q_0,q_D)", "b"): "(q_0,q_D)",
            ("(q_1,q_A)", "a"): "(q_1,q_B)",
            ("(q_1,q_A)", "b"): "(q_2,q_A)",
            ("(q_1,q_B)", "a"): "(q_1,q_C)",
            ("(q_1,q_B)", "b"): "(q_2,q_B)",
            ("(q_1,q_C)", "a"): "(q_1,q_D)",
            ("(q_1,q_C)", "b"): "(q_2,q_C)",
            ("(q_1,q_D)", "a"): "(q_1,q_D)",
            ("(q_1,q_D)", "b"): "(q_2,q_D)",
            ("(q_2,q_A)", "a"): "(q_2,q_B)",
            ("(q_2,q_A)", "b"): "(q_2,q_A)",
            ("(q_2,q_B)", "a"): "(q_2,q_C)",
            ("(q_2,q_B)", "b"): "(q_2,q_B)",
            ("(q_2,q_C)", "a"): "(q_2,q_D)",
            ("(q_2,q_C)", "b"): "(q_2,q_C)",
            ("(q_2,q_D)", "a"): "(q_2,q_D)",
            ("(q_2,q_D)", "b"): "(q_2,q_D)",
        }
        self.target_transitions_pruned: dict[tuple[State, str], State] = {
            ("(q_0,q_A)", "a"): "(q_1,q_B)",
            ("(q_0,q_A)", "b"): "(q_0,q_A)",
            ("(q_1,q_B)", "a"): "(q_1,q_C)",
            ("(q_1,q_B)", "b"): "(q_2,q_B)",
            ("(q_1,q_C)", "a"): "(q_1,q_D)",
            ("(q_1,q_C)", "b"): "(q_2,q_C)",
            ("(q_1,q_D)", "a"): "(q_1,q_D)",
            ("(q_1,q_D)", "b"): "(q_2,q_D)",
            ("(q_2,q_B)", "a"): "(q_2,q_C)",
            ("(q_2,q_B)", "b"): "(q_2,q_B)",
            ("(q_2,q_C)", "a"): "(q_2,q_D)",
            ("(q_2,q_C)", "b"): "(q_2,q_C)",
            ("(q_2,q_D)", "a"): "(q_2,q_D)",
            ("(q_2,q_D)", "b"): "(q_2,q_D)",
        }
        self.target_start_state: State = "(q_0,q_A)"

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
        assert (
            D._states == self.D_1._states
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.D_1._transitions
            and D._start_state == self.D_1._start_state
            and D._accept_states == {"q_0", "q_1"}
        )
        assert all(not D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(D.evaluate(w) for w in self.test_strs["neither"])

    def test_union_with_prune(self) -> None:
        D = self.D_1 | self.D_2
        assert (
            D._states == self.target_states_pruned
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.target_transitions_pruned
            and D._start_state == self.target_start_state
            and D._accept_states
            == {
                "(q_1,q_C)",
                "(q_2,q_B)",
                "(q_2,q_C)",
                "(q_2,q_D)",
            }
        )
        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_union_without_prune(self) -> None:
        D = self.D_1.union(self.D_2, prune_unreachable=False)
        assert (
            D._states == self.target_states
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.target_transitions
            and D._start_state == self.target_start_state
            and D._accept_states
            == {
                "(q_0,q_C)",
                "(q_1,q_C)",
                "(q_2,q_A)",
                "(q_2,q_B)",
                "(q_2,q_C)",
                "(q_2,q_D)",
            }
        )
        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_intersection_with_prune(self) -> None:
        D = self.D_1 & self.D_2
        assert (
            D._states == self.target_states_pruned
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.target_transitions_pruned
            and D._start_state == self.target_start_state
            and D._accept_states == {"(q_2,q_C)"}
        )
        assert all(not D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_intersection_without_prune(self) -> None:
        D = self.D_1.intersection(self.D_2, prune_unreachable=False)
        assert (
            D._states == self.target_states
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.target_transitions
            and D._start_state == self.target_start_state
            and D._accept_states == {"(q_2,q_C)"}
        )
        assert all(not D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_difference_with_prune(self) -> None:
        D = self.D_1 - self.D_2
        assert (
            D._states == self.target_states_pruned
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.target_transitions_pruned
            and D._start_state == self.target_start_state
            and D._accept_states == {"(q_2,q_B)", "(q_2,q_D)"}
        )
        assert all(D.evaluate(w) for w in self.test_strs["D_1_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["D_2_only"])
        assert all(not D.evaluate(w) for w in self.test_strs["both"])
        assert all(not D.evaluate(w) for w in self.test_strs["neither"])

    def test_difference_without_prune(self) -> None:
        D = self.D_1.difference(self.D_2, prune_unreachable=False)
        assert (
            D._states == self.target_states
            and D._input_alphabet == self.D_1._input_alphabet
            and D._transitions == self.target_transitions
            and D._start_state == self.target_start_state
            and D._accept_states == {"(q_2,q_A)", "(q_2,q_B)", "(q_2,q_D)"}
        )
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

    @pytest.fixture(autouse=True)
    def setup_dfas(self) -> None:
        # Recognizes {ab}, the language containing only the string "ab".
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
        self.D_1 = DFA(Q, Sigma, delta, q0, F)

        # Recognizes {w in {a, b}* | ab is a substring of w}.
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
        self.D_2 = DFA(Q, Sigma, delta, q0, F)

        # Recognizes {w in {a, b}* | w has exactly two a's}.
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
        self.D_3 = DFA(Q, Sigma, delta, q0, F)

        # Recognizes {w in {a, b}* | w has an even number of a's}.
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
        self.D_4 = DFA(Q, Sigma, delta, q0, F)

        # Also recognizes {w in {a, b}* | w has an even number of a's}.
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
        self.D_5 = DFA(Q, Sigma, delta, q0, F)

    def test_invalid_comparison(self) -> None:
        N = NFA({"q_0"}, {"a"}, {("q_0", "a"): {"q_0"}}, "q_0", {"q_0"})
        with pytest.raises(TypeError) as excinfo:
            self.D_1.compare(N)
        assert "Invalid comparison type." in str(excinfo.value)

    def test_mismatched_alphabets(self) -> None:
        self.D_2._input_alphabet = {"0", "1"}
        with pytest.raises(ValueError) as excinfo:
            self.D_1.compare(self.D_2)
        assert "Mismatched input alphabets." in str(excinfo.value)

    def test_disjoint(self) -> None:
        assert self.D_1.is_disjoint(self.D_3)

    def test_partial_intersection(self) -> None:
        assert self.D_2.partially_intersects(self.D_3)

    def test_subset_1(self) -> None:
        assert self.D_1 < self.D_2 and self.D_1 <= self.D_2 and self.D_1 != self.D_2

    def test_superset_1(self) -> None:
        assert self.D_2 > self.D_1 and self.D_2 >= self.D_1 and self.D_2 != self.D_1

    def test_subset_2(self) -> None:
        assert self.D_3 < self.D_4 and self.D_3 <= self.D_4 and self.D_3 != self.D_4

    def test_superset_2(self) -> None:
        assert self.D_4 > self.D_3 and self.D_4 >= self.D_3 and self.D_4 != self.D_3

    def test_equality(self) -> None:
        assert self.D_4 == self.D_5 and self.D_4 <= self.D_5 and self.D_5 >= self.D_4
