from cse355_machine_design.automata import DFA, State

import pytest


class TestDFAInit:
    """
    Tests for DFA initialization and validation.
    """

    @pytest.fixture(autouse=True)
    def init_dfa_params(self) -> None:
        """
        Return a basic set of valid DFA parameters.
        """
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
        self.q0 = "q_0"
        self.F = {"q_0", "q_2"}

    def test_good_init(self) -> None:
        D = DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)
        assert (
            D._automaton_type == "DFA"
            and D._states == self.Q
            and D._input_alphabet == self.Sigma
            and D._transitions == self.delta
            and D._start_state == self.q0
            and D._accept_states == self.F
        )

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
        assert (
            "Invalid transition function:" in exc_str
            and "For transition delta(q_invalid, a) = q_0," in exc_str
            and "- 'q_invalid' is not in the state set" in exc_str
        )

    def test_bad_transition_input_symbol(self) -> None:
        self.delta[("q_0", "x")] = "q_1"
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert (
            "Invalid transition function:" in exc_str
            and "For transition delta(q_0, x) = q_1," in exc_str
            and "- 'x' is not in the input alphabet" in exc_str
        )

    def test_bad_transition_to_state(self) -> None:
        self.delta[("q_0", "a")] = "q_invalid"
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert (
            "Invalid transition function:" in exc_str
            and "For transition delta(q_0, a) = q_invalid," in exc_str
            and "- 'q_invalid' is not in the state set" in exc_str
        )

    def test_missing_transition(self) -> None:
        del self.delta[("q_0", "a")]
        with pytest.raises(ValueError) as excinfo:
            DFA(self.Q, self.Sigma, self.delta, self.q0, self.F)

        exc_str = str(excinfo.value)
        assert (
            "Invalid transition function:" in exc_str
            and "Missing transition delta(q_0, a)" in exc_str
        )
