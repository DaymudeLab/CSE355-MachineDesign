from cse355_machine_design.automata import NFA, State

import pytest
from typeguard import TypeCheckError


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


class TestNFAToDFAConversion:
    """
    Test NFA to DFA conversions.
    """

    def test_nfa_to_dfa(self) -> None:
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


class TestNFAComparison:
    """
    Test NFA language comparisons.
    """

    @pytest.fixture(autouse=True)
    def setup_nfas(self) -> None:
        # Recognizes {w in {a, b}* | w has one or an even number of b's}.
        Q_1 = {"q_0", "q_1", "q_2", "q_3"}
        Sigma_1 = {"a", "b"}
        delta_1 = {
            ("q_0", "a"): {"q_0"},
            ("q_0", "b"): {"q_1"},
            ("q_1", "a"): {"q_1"},
            ("q_1", "b"): {"q_2"},
            ("q_2", "a"): {"q_2"},
            ("q_2", "b"): {"q_3"},
            ("q_3", "a"): {"q_3"},
            ("q_3", "b"): {"q_2"},
        }
        q0_1 = "q_0"
        F_1 = {"q_0", "q_1", "q_2"}
        self.N_1 = NFA(Q_1, Sigma_1, delta_1, q0_1, F_1)

        # Also recognizes {w in {a, b}* | w has one or an even number of b's}.
        Q_2 = {"q_0", "q_1", "q_2", "q_3", "q_4"}
        Sigma_2 = {"a", "b"}
        delta_2 = {
            ("q_0", "_"): {"q_1", "q_3"},
            ("q_1", "a"): {"q_1"},
            ("q_1", "b"): {"q_2"},
            ("q_2", "a"): {"q_2"},
            ("q_3", "a"): {"q_3"},
            ("q_3", "b"): {"q_4"},
            ("q_4", "a"): {"q_4"},
            ("q_4", "b"): {"q_3"},
        }
        q0_2 = "q_0"
        F_2 = {"q_2", "q_3"}
        self.N_2 = NFA(Q_2, Sigma_2, delta_2, q0_2, F_2)

    def test_equality(self) -> None:
        assert self.N_1 == self.N_2
