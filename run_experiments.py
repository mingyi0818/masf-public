"""
MASF Experiments - masf-public (Fixed Version)
E1: KC-aware Misconception Classification Benchmark (no data leakage)
E2: Scaffolded Feedback Generation & Evaluation
E3: KC Association Analysis
E4: Ablation Study
E5: Parameter Sensitivity Analysis
"""
import os
import json
import csv
import re
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = r"d:\ResearchPaperPrepare\masf-public"
DATA_DIR = os.path.join(BASE_DIR, "data", "mcminer", "dataset", "corrupted_codes_best")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

for d in [PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# KC Mapping (same as before)
# ============================================================
KC_MAP = {
    "for_loops": "Control_Flow", "while_loops": "Control_Flow", "iteration": "Control_Flow",
    "loop_conditions": "Control_Flow", "loop_execution": "Control_Flow", "loop_repetition": "Control_Flow",
    "loop_control": "Control_Flow", "break_statement": "Control_Flow", "if_statements": "Control_Flow",
    "elif_statements": "Control_Flow", "if_else_statements": "Control_Flow", "else": "Control_Flow",
    "conditionals": "Control_Flow", "conditional_logic": "Control_Flow", "mutually_exclusive_conditions": "Control_Flow",
    "control_flow": "Control_Flow", "loop_vs_conditional": "Control_Flow", "boolean_conditions": "Control_Flow",
    "redundant_conditionals": "Control_Flow", "conditional_expressions": "Control_Flow",
    "if_elif_else_chains": "Control_Flow", "if_else_statement": "Control_Flow",
    "variable_scope": "Variable_Scope", "local_variables": "Variable_Scope", "function_scope": "Variable_Scope",
    "variable_assignment": "Variable_Scope", "variable_initialization": "Variable_Scope",
    "variable_shadowing": "Variable_Scope", "variable_binding": "Variable_Scope",
    "value_vs_expression": "Variable_Scope", "variable_lifecycle": "Variable_Scope",
    "variable_deletion": "Variable_Scope", "variable_names": "Variable_Scope", "identifiers": "Variable_Scope",
    "naming_conventions": "Variable_Scope", "variable_types": "Variable_Scope", "parameter_overwriting": "Variable_Scope",
    "descriptive_names": "Variable_Scope", "code_readability": "Variable_Scope", "aliasing": "Variable_Scope",
    "memory_model": "Variable_Scope",
    "function_definition": "Function_Recursion", "def_keyword": "Function_Recursion",
    "function_calls": "Function_Recursion", "function_objects": "Function_Recursion",
    "function_parameters": "Function_Recursion", "parameter_passing": "Function_Recursion",
    "parameter_handling": "Function_Recursion", "function_return_values": "Function_Recursion",
    "return_statements": "Function_Recursion", "return_values": "Function_Recursion",
    "recursion": "Function_Recursion", "base_case": "Function_Recursion", "recursive_structure": "Function_Recursion",
    "nested_calls": "Function_Recursion", "call_stack": "Function_Recursion", "stack_unwinding": "Function_Recursion",
    "first_class_functions": "Function_Recursion", "anonymous_objects": "Function_Recursion",
    "implicit_return": "Function_Recursion", "multiple_return_values": "Function_Recursion",
    "tuple_return": "Function_Recursion", "tuple_packing": "Function_Recursion",
    "multiple_assignment": "Function_Recursion", "none_type": "Function_Recursion",
    "string_methods": "String_Operations", "str.upper": "String_Operations", "str.lower": "String_Operations",
    "str.replace": "String_Operations", "str.strip": "String_Operations", "str.split": "String_Operations",
    "string_literals": "String_Operations", "string_concatenation": "String_Operations",
    "string_to_int": "String_Operations", "immutability": "String_Operations", "method_chaining": "String_Operations",
    "variable_substitution": "String_Operations",
    "list_indexing": "List_Collection", "list_assignment": "List_Collection", "list_multiplication": "List_Collection",
    "list": "List_Collection", "list_comprehension": "List_Collection", "list_append": "List_Collection",
    "list_operations": "List_Collection", "lists": "List_Collection", "shallow_copy": "List_Collection",
    "mutable_objects": "List_Collection", "object_references": "List_Collection",
    "enumerate_function": "List_Collection", "filtering": "List_Collection", "data_structures": "List_Collection",
    "tuple_creation": "List_Collection",
    "boolean_values": "Boolean_Logic", "boolean_expressions": "Boolean_Logic",
    "comparison_operators": "Boolean_Logic", "logical_operators": "Boolean_Logic",
    "truthy_values": "Boolean_Logic", "short_circuit_evaluation": "Boolean_Logic",
    "and_operator": "Boolean_Logic", "or_operator": "Boolean_Logic", "equality_operator": "Boolean_Logic",
    "boolean_logic": "Boolean_Logic", "boolean_conditions": "Boolean_Logic",
    "syntax_errors": "Error_Debugging", "type_errors": "Error_Debugging", "name_errors": "Error_Debugging",
    "runtime": "Error_Debugging", "error_type": "Error_Debugging", "unreachable_code": "Error_Debugging",
    "side_effects": "Error_Debugging", "causes_error": "Error_Debugging", "indexerror": "Error_Debugging",
    "__init__method": "Object_Class", "object_creation": "Object_Class", "object_initialization": "Object_Class",
    "constructor_behavior": "Object_Class", "self_parameter": "Object_Class", "instance_methods": "Object_Class",
    "dot_operator": "Object_Class", "object_instantiation": "Object_Class", "constructor_invocation": "Object_Class",
    "method_bodies": "Object_Class", "empty_methods": "Object_Class", "pass_statement": "Object_Class",
    "class": "Object_Class",
    "arithmetic_operators": "Operator_Expression", "operator_precedence": "Operator_Expression",
    "arithmetic_expressions": "Operator_Expression", "assignment_operator": "Operator_Expression",
    "expression_evaluation": "Operator_Expression", "ternary_operator": "Operator_Expression",
    "modulo_operator": "Operator_Expression", "integer_operations": "Operator_Expression",
    "float_operations": "Operator_Expression", "addition_operation": "Operator_Expression",
    "mathematical_operations": "Operator_Expression", "arithmetic": "Operator_Expression",
    "division_with_remainder": "Operator_Expression", "floor_division": "Operator_Expression",
    "integer_division": "Operator_Expression", "percentage_calculations": "Operator_Expression",
    "rounding": "Operator_Expression", "weighted_average": "Operator_Expression",
    "range_function": "Range_Indexing", "zero_based_indexing": "Range_Indexing",
    "negative_indexing": "Range_Indexing", "index_values": "Range_Indexing",
    "sequence_access": "Range_Indexing", "square_brackets": "Range_Indexing", "indexing": "Range_Indexing",
    "manual_counters": "Range_Indexing", "loop_iteration": "Range_Indexing", "inclusive_ranges": "Range_Indexing",
    "boundary_conditions": "Range_Indexing", "step_parameter": "Range_Indexing", "negative_step": "Range_Indexing",
    "type_conversion": "Type_Conversion", "type_checking": "Type_Conversion",
    "type_annotations": "Type_Conversion", "integer_type": "Type_Conversion",
    "immutable_operations": "Type_Conversion", "sorted_function": "Type_Conversion",
    "built_in_functions": "Type_Conversion", "numeric_comparison": "Type_Conversion",
    "integer_conversion": "Type_Conversion",
}

KC_ORDER = [
    "Control_Flow", "Variable_Scope", "Function_Recursion", "String_Operations",
    "List_Collection", "Boolean_Logic", "Error_Debugging", "Object_Class",
    "Operator_Expression", "Range_Indexing", "Type_Conversion"
]

def load_misconception_bank():
    path = os.path.join(BASE_DIR, "data", "mcminer", "dataset", "misconception_bank.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_valid_samples():
    csv_path = os.path.join(PROCESSED_DIR, "valid_samples.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json') and f != 'filtering_report.json']
    rows = []
    for fname in json_files:
        with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for sol in data.get('solutions', []):
            fb = sol.get('feedback_loop', {})
            final_eval = fb.get('final_evaluation', {}) or {}
            if final_eval.get('exhibits_misconception', False):
                rows.append({
                    'problem_id': data.get('problem_id'),
                    'misconception_id': data.get('misconception_id'),
                    'misconception_desc': data.get('misconception_description', ''),
                    'code': sol.get('generated_code', ''),
                    'error_type': sol.get('metadata', {}).get('error_type', ''),
                    'misconception_type': sol.get('metadata', {}).get('misconception_type', '')
                })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    return df

def assign_kc(misc_id, misc_bank):
    for m in misc_bank:
        if m['id'] == misc_id:
            constructs = m.get('meta_data', {}).get('related_constructs', [])
            kcs = []
            for c in constructs:
                kc = KC_MAP.get(c)
                if kc and kc not in kcs:
                    kcs.append(kc)
            return kcs[0] if kcs else "Other"
    return "Other"

def preprocess_code(code):
    if pd.isna(code) or code == 'NONE':
        return ""
    code = str(code)
    code = re.sub(r'\s+', ' ', code).strip()
    return code

# ============================================================
# Classification Helpers
# ============================================================

def run_classifier(name, clf, vectorizer, X_train, X_test, y_train, y_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)
    return {
        'accuracy': accuracy_score(y_test, preds),
        'f1_macro': f1_score(y_test, preds, average='macro', zero_division=0),
        'f1_micro': f1_score(y_test, preds, average='micro', zero_division=0),
        'precision': precision_score(y_test, preds, average='macro', zero_division=0),
        'recall': recall_score(y_test, preds, average='macro', zero_division=0),
        'preds': preds,
        'trues': y_test.values
    }

def cross_validate(name, clf_factory, vectorizer_factory, X_text, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metrics = defaultdict(list)
    all_preds, all_trues = [], []
    for train_idx, test_idx in skf.split(X_text, y):
        X_train, X_test = X_text.iloc[train_idx], X_text.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        vec = vectorizer_factory()
        clf = clf_factory()
        res = run_classifier(name, clf, vec, X_train, X_test, y_train, y_test)
        for k in ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall']:
            metrics[k].append(res[k])
        all_preds.extend(res['preds'])
        all_trues.extend(res['trues'])
    return {
        'method': name,
        **{k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics.items()},
        'all_preds': all_preds,
        'all_trues': all_trues,
        'fold_scores': {k: v for k, v in metrics.items()}
    }

# ============================================================
# E1: Main Classification Benchmark
# ============================================================

def run_e1_benchmark(X_text, y):
    print("\n[E1] KC-aware Misconception Classification Benchmark")
    results = []

    # Baseline 1: TF-IDF + Linear SVM
    results.append(cross_validate('TF-IDF + LinearSVM',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_text, y, n_splits=5))

    # Baseline 2: TF-IDF + Logistic Regression
    results.append(cross_validate('TF-IDF + LogisticRegression',
        lambda: LogisticRegression(C=1.0, max_iter=5000, random_state=SEED, n_jobs=4),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_text, y, n_splits=5))

    # Baseline 3: TF-IDF + Random Forest
    results.append(cross_validate('TF-IDF + RandomForest',
        lambda: RandomForestClassifier(n_estimators=300, max_depth=25, random_state=SEED, n_jobs=4),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_text, y, n_splits=5))

    # Baseline 4: TF-IDF + Naive Bayes
    results.append(cross_validate('TF-IDF + NaiveBayes',
        lambda: MultinomialNB(),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_text, y, n_splits=5))

    # Baseline 5: TF-IDF (char) + SVM
    results.append(cross_validate('Char-TFIDF + LinearSVM',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(analyzer='char_wb', max_features=5000, ngram_range=(2,4), sublinear_tf=True),
        X_text, y, n_splits=5))

    return results

# ============================================================
# E4: Ablation Study
# ============================================================

def run_ablation_study(df):
    print("\n[E4] Ablation Study")
    ablation_results = []

    # Full model (code only)
    X_full = df['code_processed']
    y = df['kc']
    res_full = cross_validate('Full (Code Only)',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_full, y, n_splits=5)
    ablation_results.append(res_full)

    # Without keywords
    def remove_keywords(text):
        python_keywords = ['def', 'return', 'if', 'else', 'elif', 'for', 'while', 'in', 'not', 'and', 'or',
                          'True', 'False', 'None', 'class', 'import', 'from', 'as', 'try', 'except', 'with']
        words = text.split()
        filtered = [w for w in words if w not in python_keywords]
        return ' '.join(filtered)

    X_no_kw = df['code_processed'].apply(remove_keywords)
    res_no_kw = cross_validate('No Keywords',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_no_kw, y, n_splits=5)
    ablation_results.append(res_no_kw)

    # Without numbers
    X_no_num = df['code_processed'].str.replace(r'\d+', 'NUM', regex=True)
    res_no_num = cross_validate('No Numbers',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_no_num, y, n_splits=5)
    ablation_results.append(res_no_num)

    # Without comments (there aren't many but just in case)
    X_no_comment = df['code_processed'].str.replace(r'#.*', '', regex=True)
    res_no_comment = cross_validate('No Comments',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
        X_no_comment, y, n_splits=5)
    ablation_results.append(res_no_comment)

    # Char-level only
    res_char = cross_validate('Char-level Only',
        lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
        lambda: TfidfVectorizer(analyzer='char_wb', max_features=5000, ngram_range=(2,4), sublinear_tf=True),
        X_full, y, n_splits=5)
    ablation_results.append(res_char)

    return ablation_results

# ============================================================
# E5: Parameter Sensitivity
# ============================================================

def run_parameter_sensitivity(X_text, y):
    print("\n[E5] Parameter Sensitivity Analysis")
    sens_results = []

    # C parameter for LinearSVM
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        res = cross_validate(f'LinearSVM_C={C}',
            lambda C=C: LinearSVC(C=C, max_iter=10000, random_state=SEED),
            lambda: TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True),
            X_text, y, n_splits=5)
        sens_results.append({'parameter': 'C', 'value': C, 'f1_macro': res['f1_macro']['mean']})

    # max_features for TF-IDF
    for mf in [500, 1000, 2000, 5000, 10000]:
        res = cross_validate(f'TFIDF_maxfeat={mf}',
            lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
            lambda mf=mf: TfidfVectorizer(max_features=mf, ngram_range=(1,2), sublinear_tf=True),
            X_text, y, n_splits=5)
        sens_results.append({'parameter': 'max_features', 'value': mf, 'f1_macro': res['f1_macro']['mean']})

    # ngram_range
    for ngr in [(1,1), (1,2), (1,3), (2,2), (2,3)]:
        res = cross_validate(f'TFIDF_ngram={ngr}',
            lambda: LinearSVC(C=1.0, max_iter=10000, random_state=SEED),
            lambda ngr=ngr: TfidfVectorizer(max_features=5000, ngram_range=ngr, sublinear_tf=True),
            X_text, y, n_splits=5)
        sens_results.append({'parameter': 'ngram_range', 'value': str(ngr), 'f1_macro': res['f1_macro']['mean']})

    return pd.DataFrame(sens_results)

# ============================================================
# E2: Feedback Generation
# ============================================================

def generate_scaffolded_feedback(code, misconception_desc, level='L3'):
    if level == 'L1':
        return f"Concept Check: {misconception_desc}"
    elif level == 'L2':
        lines = [l.strip() for l in str(code).split('\n') if l.strip()][:3]
        code_hint = "; ".join(lines[:2])
        return f"Localized Hint: {misconception_desc} Review lines: {code_hint}"
    elif level == 'L3':
        return (
            f"Guided Plan:\n"
            f"1. Understand: {misconception_desc}\n"
            f"2. Locate: Find the relevant pattern in your code.\n"
            f"3. Fix: Modify the code to remove the misconception.\n"
            f"4. Verify: Test with sample inputs."
        )
    return ""

def evaluate_feedback(feedback, misconception_desc, code):
    fb_words = set(feedback.lower().split())
    misc_words = set(str(misconception_desc).lower().split())
    overlap = len(fb_words & misc_words) / max(len(misc_words), 1)
    relevance = min(overlap * 2.5, 1.0)

    code_tokens = set(re.findall(r'\b[a-zA-Z_]+\b', str(code)))
    fb_code_overlap = len(fb_words & code_tokens) / max(len(code_tokens), 1)
    specificity = 0.3 + min(fb_code_overlap * 3, 0.7)

    action_words = ['fix', 'rewrite', 'review', 'check', 'verify', 'locate', 'understand', 'try', 'modify', 'test']
    has_action = any(w in fb_words for w in action_words)
    actionability = 0.5 + (0.5 if has_action else 0)

    return {
        'relevance': relevance,
        'specificity': specificity,
        'actionability': actionability,
        'overall': (relevance + specificity + actionability) / 3
    }

def run_feedback_experiment(df, n_samples=300):
    print("\n[E2] Scaffolded Feedback Generation & Evaluation")
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=SEED)
    results = []
    for level in ['L1', 'L2', 'L3']:
        scores = defaultdict(list)
        for _, row in sample_df.iterrows():
            fb = generate_scaffolded_feedback(row['code'], row['misconception_desc'], level)
            eval_scores = evaluate_feedback(fb, row['misconception_desc'], row['code'])
            for k, v in eval_scores.items():
                scores[k].append(v)
        results.append({
            'level': level,
            'relevance': round(np.mean(scores['relevance']), 4),
            'specificity': round(np.mean(scores['specificity']), 4),
            'actionability': round(np.mean(scores['actionability']), 4),
            'overall': round(np.mean(scores['overall']), 4)
        })
    return pd.DataFrame(results)

# ============================================================
# E3: KC Association
# ============================================================

def run_kc_association(df):
    print("\n[E3] KC Association Analysis")
    misc_bank = load_misconception_bank()
    rows = []
    for _, row in df.iterrows():
        for m in misc_bank:
            if m['id'] == row['misconception_id']:
                constructs = m.get('meta_data', {}).get('related_constructs', [])
                kcs = list(set([KC_MAP.get(c, 'Other') for c in constructs]))
                rows.append({'misconception_id': row['misconception_id'], 'kcs': kcs, 'primary_kc': row['kc']})
                break
    kc_df = pd.DataFrame(rows)
    all_kcs = sorted(set().union(*kc_df['kcs']))
    cooccur = pd.DataFrame(0, index=all_kcs, columns=all_kcs)
    for kcs in kc_df['kcs']:
        for kc1 in kcs:
            for kc2 in kcs:
                cooccur.loc[kc1, kc2] += 1
    kc_counts = Counter(kc_df['primary_kc'])
    total = sum(kc_counts.values())
    probs = [c/total for c in kc_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return {
        'cooccurrence': cooccur,
        'kc_distribution': dict(kc_counts),
        'entropy': round(entropy, 4),
        'num_kcs': len(all_kcs)
    }

# ============================================================
# Statistical Tests
# ============================================================

def paired_ttest_stats(fold_scores_a, fold_scores_b):
    diffs = np.array(fold_scores_a) - np.array(fold_scores_b)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-1)) if se > 0 else 1.0
    ci_low = mean_diff - stats.t.ppf(0.975, n-1) * se
    ci_high = mean_diff + stats.t.ppf(0.975, n-1) * se
    return {
        'mean_diff': round(mean_diff, 4),
        'std_diff': round(std_diff, 4),
        't_statistic': round(t_stat, 4),
        'df': n - 1,
        'p_value': round(p_value, 4),
        'ci_95_low': round(ci_low, 4),
        'ci_95_high': round(ci_high, 4),
        'significant': bool(p_value < 0.05)
    }

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("MASF Experiment Pipeline v2 (No Data Leakage)")
    print("=" * 70)

    df = load_valid_samples()
    print(f"Loaded {len(df)} valid samples")

    misc_bank = load_misconception_bank()
    df['kc'] = df['misconception_id'].apply(lambda x: assign_kc(x, misc_bank))
    df['code_processed'] = df['code'].apply(preprocess_code)

    # Filter valid KCs
    kc_counts = df['kc'].value_counts()
    valid_kcs = kc_counts[kc_counts >= 15].index.tolist()
    df = df[df['kc'].isin(valid_kcs)].copy()
    print(f"After KC filtering: {len(df)} samples, {len(valid_kcs)} KCs")

    # IMPORTANT: Use code ONLY, no misconception description to prevent leakage
    X_text = df['code_processed']
    y = df['kc']

    # E1
    e1_results = run_e1_benchmark(X_text, y)
    print("\n--- E1 Results ---")
    for r in e1_results:
        print(f"{r['method']}: Acc={r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f}, F1-M={r['f1_macro']['mean']:.4f}±{r['f1_macro']['std']:.4f}")

    # Statistical test: best vs second best
    best = e1_results[0]  # LinearSVM
    second = e1_results[2]  # RandomForest
    stat_test = paired_ttest_stats(best['fold_scores']['f1_macro'], second['fold_scores']['f1_macro'])
    print(f"\nPaired t-test ({best['method']} vs {second['method']}):")
    print(f"  mean_diff={stat_test['mean_diff']}, t={stat_test['t_statistic']}, p={stat_test['p_value']}, sig={stat_test['significant']}")

    # E4: Ablation
    ablation_results = run_ablation_study(df)
    print("\n--- E4 Ablation Results ---")
    for r in ablation_results:
        print(f"{r['method']}: Acc={r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f}, F1-M={r['f1_macro']['mean']:.4f}±{r['f1_macro']['std']:.4f}")

    # E5: Parameter Sensitivity
    sens_df = run_parameter_sensitivity(X_text, y)
    print("\n--- E5 Parameter Sensitivity ---")
    print(sens_df.to_string(index=False))

    # E2: Feedback
    feedback_df = run_feedback_experiment(df, n_samples=300)
    print("\n--- E2 Feedback Evaluation ---")
    print(feedback_df.to_string(index=False))

    # E3: KC Association
    kc_analysis = run_kc_association(df)
    print(f"\n--- E3 KC Analysis ---")
    print(f"Entropy: {kc_analysis['entropy']}, KCs: {kc_analysis['num_kcs']}")

    # Save all results
    summary_rows = []
    for r in e1_results:
        summary_rows.append({
            'Experiment': 'E1_Classification',
            'Method': r['method'],
            'Accuracy_mean': round(r['accuracy']['mean'], 4),
            'Accuracy_std': round(r['accuracy']['std'], 4),
            'F1_Macro_mean': round(r['f1_macro']['mean'], 4),
            'F1_Macro_std': round(r['f1_macro']['std'], 4),
            'F1_Micro_mean': round(r['f1_micro']['mean'], 4),
            'F1_Micro_std': round(r['f1_micro']['std'], 4),
            'Precision_mean': round(r['precision']['mean'], 4),
            'Precision_std': round(r['precision']['std'], 4),
            'Recall_mean': round(r['recall']['mean'], 4),
            'Recall_std': round(r['recall']['std'], 4),
        })

    ablation_rows = []
    for r in ablation_results:
        ablation_rows.append({
            'Experiment': 'E4_Ablation',
            'Variant': r['method'],
            'Accuracy_mean': round(r['accuracy']['mean'], 4),
            'Accuracy_std': round(r['accuracy']['std'], 4),
            'F1_Macro_mean': round(r['f1_macro']['mean'], 4),
            'F1_Macro_std': round(r['f1_macro']['std'], 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLES_DIR, "e1_classification_results.csv"), index=False)
    pd.DataFrame(ablation_rows).to_csv(os.path.join(TABLES_DIR, "e4_ablation_results.csv"), index=False)
    feedback_df.to_csv(os.path.join(TABLES_DIR, "e2_feedback_evaluation.csv"), index=False)
    sens_df.to_csv(os.path.join(TABLES_DIR, "e5_parameter_sensitivity.csv"), index=False)

    # Save JSON
    final_results = {
        'e1_classification': {
            r['method']: {
                'accuracy_mean': round(r['accuracy']['mean'], 4),
                'accuracy_std': round(r['accuracy']['std'], 4),
                'f1_macro_mean': round(r['f1_macro']['mean'], 4),
                'f1_macro_std': round(r['f1_macro']['std'], 4),
                'f1_micro_mean': round(r['f1_micro']['mean'], 4),
                'f1_micro_std': round(r['f1_micro']['std'], 4),
                'precision_mean': round(r['precision']['mean'], 4),
                'precision_std': round(r['precision']['std'], 4),
                'recall_mean': round(r['recall']['mean'], 4),
                'recall_std': round(r['recall']['std'], 4),
            } for r in e1_results
        },
        'e1_statistical_test': stat_test,
        'e4_ablation': {
            r['method']: {
                'accuracy_mean': round(r['accuracy']['mean'], 4),
                'accuracy_std': round(r['accuracy']['std'], 4),
                'f1_macro_mean': round(r['f1_macro']['mean'], 4),
                'f1_macro_std': round(r['f1_macro']['std'], 4),
            } for r in ablation_results
        },
        'e2_feedback': feedback_df.to_dict(orient='records'),
        'e5_sensitivity': sens_df.to_dict(orient='records'),
        'e3_kc_analysis': {
            'num_kcs': kc_analysis['num_kcs'],
            'entropy': kc_analysis['entropy'],
            'kc_distribution': {k: int(v) for k, v in kc_analysis['kc_distribution'].items()}
        },
        'data_stats': {
            'total_valid_samples': int(len(df)),
            'num_kcs': int(len(valid_kcs)),
            'num_problems': int(df['problem_id'].nunique()),
            'num_misconceptions': int(df['misconception_id'].nunique()),
            'kc_distribution': {k: int(v) for k, v in kc_counts.items() if k in valid_kcs}
        }
    }

    with open(os.path.join(RESULTS_DIR, "all_experiments_results.json"), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x.tolist() if isinstance(x, np.ndarray) else x)

    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    main()
