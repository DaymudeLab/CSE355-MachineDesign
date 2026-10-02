from cse355_machine_design.automata import NFA, State

import pytest


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
            ("q_3", "b"): {"q_2"}
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
