""" Constants used by other files """

import pathlib

# Algorithm knobs
GAP_OPENING = -7  # Penalty for opening a gap
GAP_EXTENSION = -3  # Penalty for extending an already open gap

# Parallel processing
PROCESSES = 8   # Number of alignment processes to run in parallel

# Paths
THISDIR = pathlib.Path(__file__).resolve().parent

DATADIR = THISDIR.parent / 'data'  # Location of all the raw data
ALIGNDIR = THISDIR.parent / 'alignments'  # Location of all the alignments
SCOREDIR = THISDIR.parent / 'scores'  # Location of all the score files
PLOTDIR = THISDIR.parent / 'plots'  # Location of all the plots
