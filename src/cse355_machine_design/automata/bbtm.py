import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import pandas as pd


class _BusyBeaverTM():
    def __init__(self, tmstr, configs_dir='configs', plots_dir='plots'):
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
        :param configs_dir: a string directory for configuration histories
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
        self.configs_dir, self.plots_dir = configs_dir, plots_dir
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

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
        tmdb_fname = 'database.csv'
        try:
            tmdb = pd.read_csv(tmdb_fname, sep=',', index_col='ID')
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
        state, step = 0, 0
        tape = [0 for i in range(extlen)]
        head = len(tape) // 2
        start = head

        # Set up configuration history file for this TM.
        config_fname = osp.join(self.configs_dir, f"{self.id}.csv")
        with open(config_fname, 'w') as f:
            writer = csv.writer(f)

            # Run until halting.
            print(f"Running TM {self.id} on the all-0 tape...")
            while True:
                print(f"Steps Run: {step}", end='\r')

                # Get the transition information.
                symb, shift, tostate = self.delta[state, tape[head]]
                symb, shift, tostate = int(symb), int(shift), int(tostate)

                # Halt if this is an unused transition.
                if symb == -2 or shift == -2:
                    break

                # Extend the tape (to the left or right) if needed.
                if head == 0 and shift == -1:
                    tape = [0 for i in range(extlen)] + tape
                    head += extlen
                    start += extlen
                elif head == len(tape) - 1 and shift == 1:
                    tape = tape + [0 for i in range(extlen)]

                # Process the transition and log any tape changes.
                step += 1
                if tape[head] != symb:
                    tape[head] = symb
                    writer.writerow([step, start, head, symb])
                head += shift

                # Make the state transition, or halt if there is no next state.
                if tostate != -2:
                    state = tostate
                else:
                    break

            # Before closing the configuration history file, write the total
            # steps run and the tape length. This will be useful later.
            writer.writerow([step, len(tape), 0, 0])

        print(f"TM {self.id} ran for {step} steps and halted.")
        print(f"Wrote history of tape changes to {config_fname}.")

        # Write this result to the local database.
        new_row = pd.DataFrame([(self.id, step)], columns=['ID', 'Steps'])
        tmdb = pd.concat([tmdb, new_row.set_index('ID')])
        tmdb.sort_values(by=['Steps'], ascending=False, inplace=True)
        tmdb.to_csv(tmdb_fname, sep=',', index=True)
        print(f"Added this result to {tmdb_fname}.")

    def plot_spacetime(self, compress=True, limit=None, title=True,
                       colors=['#8C1D40', '#FFC627']):
        """
        Plots this BusyBeaverTM's space-time diagram.

        :param compress: True if the diagram should only show steps when the
        tape was updated; False if all steps' tapes should be shown
        :param limit: an int maximum number of rows to display, regardless of
        compression mode; None if no limit
        :param title: True iff the TM string representation should be shown
        :param colors: a list of colors recognized by matplotlib, where the ith
        color will be used to represent tape cells marked with symbol i
        """
        # Check that there are enough colors for this TM.
        assert len(colors) >= len(self.Gamma), ("ERROR: This TM has ",
                                                f"{len(self.Gamma)} symbols "
                                                f"but only {len(colors)} "
                                                "colors were provided.")

        # Attempt to load this TM's configuration history; if it is missing,
        # error out and ask the user to use BusyBeaverTM.run() first.
        config_fname = osp.join(self.configs_dir, f"{self.id}.csv")
        try:
            changes = np.genfromtxt(config_fname, dtype=int, delimiter=',')
            steps_run, tape_len = int(changes[-1, 0]), int(changes[-1, 1])
            changes = changes[:-1]
        except FileNotFoundError:
            print((f"ERROR: You are trying to plot TM {self.id} but its "
                   f"configuration history {config_fname} does not exist. "
                   f"Run this TM with the .run() function before plotting."))
            return

        # Align all changes to the final length tape.
        for i in range(len(changes) - 2, -1, -1):
            if changes[i, 1] < changes[i+1, 1]:  # Tape was left-extended.
                extlen = changes[i+1, 1] - changes[i, 1]
                changes[:i+1, 1:3] += extlen

        # Create the (possibly compressed) configuration history.
        rows_needed = len(changes) + 1 if compress else steps_run + 1
        limit = rows_needed if limit is None else min(limit, rows_needed)
        try:
            configs = np.zeros((limit, tape_len), dtype=np.uint8)
        except np._core._exceptions._ArrayMemoryError:
            print(("ERROR: Ran out of memory trying to reconstruct this TM's "
                   "configuration history. Try running .plot_spacetime() with "
                   "compress=True, or limit the visualization to a smaller "
                   "number of rows with limit=<max rows>."))
            return
        row = 1
        for step, start, head, symb in changes:
            # When not compressing, copy tape until change.
            if not compress:
                while row < step and row < limit:
                    configs[row] = configs[row-1]
                    row += 1
                if row >= limit:
                    break

            # Now, regardless of compression, visualize the next change.
            configs[row] = configs[row-1]
            configs[row][head] = symb
            row += 1
            if row >= limit:
                break
        while row < limit:
            configs[row] = configs[row-1]
            row += 1

        # Trim the configuration history of extra 0-space on the outsides.
        col_sums = configs.sum(axis=0)
        left, right = 0, len(col_sums)
        for i, col_sum in enumerate(col_sums):
            if col_sum != 0:
                left = i
                break
        for j, col_sum in enumerate(col_sums[::-1]):
            if col_sum != 0:
                right = len(col_sums) - j
                break
        configs = configs[:, left:right]

        # Plot the space-time diagram.
        fig, ax = plt.subplots(figsize=(2, 6), dpi=300)
        cmap = mpl.colors.ListedColormap(colors)
        ax.imshow(configs, cmap=cmap, aspect='auto')
        ax.set_axis_off()
        if title:
            ax.set_title(self.id)
        fig.savefig(osp.join(self.plots_dir, f"{self.id}.png"),
                    bbox_inches='tight', pad_inches=0)
