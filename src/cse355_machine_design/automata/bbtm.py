from cse355_machine_design.util import dump_df, load_df

import numpy as np
import os.path as osp
import pandas as pd


class _BusyBeaverTM():
    def __init__(self, tmstr, results_dir='results', plots_dir='plots'):
        """
        Construct a BusyBeaverTM from its string representation. A busy beaver
        TM is a deterministic TM with a doubly-infinite tape initialized with
        all 0's. Its string representation lists the TM's transition table
        whose entries corresponds to the current state (the "row") and tape
        symbol (the "column"). Since we are dealing with the busy beavers, we
        express states as upper case letters (A .. Z) and symbols as digits
        (0 .. 9); TMs with more than 26 states or 10 symbols are far-in-the-
        future for busy beavers. Each entry is a triple comprising (1) the new
        symbol to write, (2) a direction in {L, R} to move the tape head, and
        (3) the next state to transition to or "-" to halt. Unused transitions
        are set to "---". The string representation lists these entries in row-
        column order, joining a row's entries across columns and separating
        rows by underscores.

        :param tmstr: a string representation of the TM to construct
        :param db_dir: a string directory for the TM database and histories
        :param plots_dir: a string directory for space-time diagrams
        """
        # Parse and validate transition table.
        rows = tmstr.split("_")
        Q, Gamma = list(range(len(rows))), list(range(len(rows[0]) // 3))
        delta = np.full((len(Q), len(Gamma), 3), -2, dtype=np.int8)
        print(f"Making a TM with {len(Q)} states and {len(Gamma)} symbols...")
        for i, row in enumerate(rows):
            assert len(row) % 3 == 0, (f"ERROR: Row {i} = \'{row}\' must have "
                                       "triples as entries.")
            for j, t in enumerate([row[s:s+3] for s in range(0, len(row), 3)]):
                # Parse the new symbol to write to the tape.
                if t[0] != '-':
                    assert int(t[0]) in Gamma, (f"ERROR: Symbol \'{t[0]}\' in "
                                                f"entry ({i},{j}) = \'{t}\' "
                                                f"is not in Gamma = {Gamma}.")
                    delta[i, j, 0] = int(t[0])
                # Parse the direction to shift the tape head.
                if t[1] != '-':
                    assert t[1] in ['L', 'R'], (f"ERROR: Direction \'{t[1]}\' "
                                                f"in entry ({i},{j}) = "
                                                f"\'{t}\' is not \'L\' or "
                                                "\'R\'.")
                    delta[i, j, 1] = -1 if t[1] == 'L' else 1
                # Parse the next state to transition to.
                if t[2] != '-':
                    state = ord(t[2]) - ord('A')
                    assert state in Q, (f"ERROR: State \'{t[2]}\' in entry "
                                        f"({i},{j}) = \'{t}\' is not in Q = "
                                        f"{[chr(x+ord('A')) for x in Q]}.")
                    delta[i, j, 2] = ord(t[2]) - ord('A')

        # Store member data.
        self.Q, self.Gamma, self.delta = Q, Gamma, delta
        self.id = self.to_str()
        self.results_dir, self.plots_dir = results_dir, plots_dir

    def to_str(self):
        """
        Returns the string representation of this BusyBeaverTM; the inverse of
        the parsing occuring in __init__. Assumes validation already occurred
        and the transition function self.delta is correctly formatted.

        :returns: a string representation of this BusyBeaverTM
        """
        chars = []
        num_states, num_symbols, _ = self.delta.shape
        for i in range(num_states):
            for j in range(num_symbols):
                if self.delta[i, j, 0] == -2:
                    chars.append('-')
                else:
                    chars.append(str(self.delta[i, j, 0]))
                if self.delta[i, j, 1] == -2:
                    chars.append('-')
                else:
                    chars.append('L' if self.delta[i, j, 1] == -1 else 'R')
                if self.delta[i, j, 2] == -2:
                    chars.append('-')
                else:
                    chars.append(chr(self.delta[i, j, 2] + ord('A')))
            chars.append('_')

        return "".join(chars[:-1])

    def run(self, extlen=10):
        """
        Runs the BusyBeaverTM from the tape of all 0's. If the TM halts, then
        the number of steps it ran for is printed out and recorded in a local
        database (to avoid recomputation if this one is ever run again).

        WARNING: This function may run forever if the TM is non-halting.

        :param extlen: optionally, an int number of cells to extend the tape by
        each time its boundaries are exceeded (default = 10)
        """
        # Load TM database from file, or create it if it does not exist.
        tmdb_fname = osp.join(self.results_dir, 'database.csv')
        try:
            tmdb = load_df(tmdb_fname, index_col='ID')
        except FileNotFoundError:
            tmdb = pd.DataFrame(columns=['ID', 'Steps'])
            tmdb = tmdb.set_index('ID')

        # If this TM is already in the database, skip re-running it.
        if self.id in tmdb.index:
            print((f"TM {self.id} runs for {tmdb.loc[self.id]['Steps']} steps "
                   "and halts. Skipping re-run.\nIf for some reason you really"
                   f" need to re-run this TM, delete it from {tmdb_fname}."))
            return

        # Initialize the tape as all zeroes and the tape head in the "middle".
        # Because this tape needs to be extended when boundaries are exceeded,
        # also track the index of the starting cell as it changes.
        tape = [0 for i in range(extlen)]
        head = len(tape) // 2
        start = head

        # Run until halting.
        print(f"Running TM {self.id} on the all-0 tape...")
        state, step = 0, 0
        while True:
            print(f"Steps Run: {step}", end='\r')

            # Get the transition information.
            write, shift, nextstate = self.delta[state, tape[head]]
            write, shift, nextstate = int(write), int(shift), int(nextstate)

            # Halt if this is an unused transition.
            if write == -2 or shift == -2:
                break

            # Extend the tape (to the left or right) if needed.
            if head == 0 and shift == -1:
                tape = [0 for i in range(extlen)] + tape
                head += extlen
                start += extlen
            elif head == len(tape) - 1 and shift == 1:
                tape = tape + [0 for i in range(extlen)]

            # Process the transition, halting if there is no next state.
            step += 1
            tape[head] = write
            head += shift
            if nextstate != -2:
                state = nextstate
            else:
                break

        print(f"TM {self.id} ran for {step} steps and halted.")

        # Write this result to the local database.
        new_row = pd.DataFrame([(self.id, step)], columns=['ID', 'Steps'])
        tmdb = pd.concat([tmdb, new_row.set_index('ID')])
        dump_df(tmdb_fname, tmdb)
        print(f"Added this result to {tmdb_fname}.")
