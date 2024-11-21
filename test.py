from src.cse355_machine_design import PDA, registry



def testPda():

    Q = {'q0', 'q1'}
    Sigma = {'0', '1', '#'}
    Gamma = {'0', '1', '$'}
    delta = {
        ('q0', '0', '0'): {('q1', '0')},
    }
    q0 = "q0"
    F = {}

    return PDA(Q, Sigma, Gamma, delta, q0, F)



if __name__ == "__main__":
    #problem1().submit_as_answer(1)
    testPda().display_state_diagram()

    #problem2().submit_as_answer(2)
    #problem3().submit_as_answer(3)
    #problem4().submit_as_answer(4)
    #problem5().submit_as_answer(5)
    #problem6().submit_as_answer(6)
    #problem7().submit_as_answer(7)

    registry.export_submissions()