"""
Configuration file for MASF experiments.
"""
import os

# Paths
BASE_DIR = r"d:\ResearchPaperPrepare\JX02_Teaching_Research_2"
DATA_DIR = os.path.join(BASE_DIR, "data", "mcminer", "dataset", "corrupted_codes_best")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

# Random seed
SEED = 42

# Classification settings
N_SPLITS = 5
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (2, 4)
SVM_C = 1.0

# KC filtering
MIN_SAMPLES_PER_KC = 15

# Feedback evaluation
N_FEEDBACK_SAMPLES = 300

# Create directories
for d in [PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)
