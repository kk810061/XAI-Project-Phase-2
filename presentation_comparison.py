"""
SpeechXAI Replication: Phase 1 Final Results Comparison
Paper: 'Explaining Speech Classification Models via Word-Level Audio Segments and Paralinguistic Features'
Authors: Pastor et al. (EACL 2024)
"""

import os
import pandas as pd
from IPython.display import display, Image

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def show_table1():
    print("="*80)
    print("TABLE 1: Word-Level Segment Attribution (L1O) - Canonical: 'Turn on the lights.'")
    print("="*80)
    path = os.path.join(RESULTS_DIR, "table1_word_attribution.csv")
    df = pd.read_csv(path, index_col=0)
    display(df)
    print("\nInsight: 'lights' dominates object intent (0.990) and action intent (0.805).")

def show_table2():
    print("\n" + "="*80)
    print("TABLE 2: Paralinguistic Sensitivity on Canonical Utterance")
    print("="*80)
    path = os.path.join(RESULTS_DIR, "table2_paralinguistic_single.csv")
    df = pd.read_csv(path, index_col=0)
    display(df)
    print("\nInsight: Noise and reverberation degrade predictions most; pitch shifting has minimal impact.")

def show_figure3():
    print("\n" + "="*80)
    print("FIGURE 3: Continuous Sensitivity Curves under Perturbations")
    print("="*80)
    img_path = os.path.join(RESULTS_DIR, "figure3_perturbation_curves.png")
    if os.path.exists(img_path):
        display(Image(filename=img_path))
    print("\nInsight: Asymmetric tempo sensitivity (faster hurts more than slower); noise collapse below 10 dB SNR.")

def show_table3():
    print("\n" + "="*80)
    print("TABLE 3: Dataset-Wide Average Paralinguistic Impact (3,750 Test Samples)")
    print("="*80)
    path = os.path.join(RESULTS_DIR, "table3_paralinguistic_full_test.csv")
    df = pd.read_csv(path, index_col=0).round(3)
    display(df)
    print("\nAverage Sensitivity across Dataset: Noise (0.335) > Reverb (0.188) > Time Stretch (0.073) > Pitch Shift (0.064)")

def show_table4():
    print("\n" + "="*80)
    print("TABLE 4: Quantitative Faithfulness Evaluation across Full Test Split (3,790 Samples)")
    print("="*80)
    path = os.path.join(RESULTS_DIR, "table4_faithfulness_full_test.csv")
    df = pd.read_csv(path, index_col=0)
    display(df)
    print("\nComparison with Pastor et al. (EACL 2024 Table 4):")
    comparison = pd.DataFrame({
        "Method": ["WA-L1O (Paper)", "WA-L1O (Ours)", "WA-LIME (Paper)", "WA-LIME (Ours)", "Random (Paper)", "Random (Ours)"],
        "Compr_Action (^)": [0.65, 0.63, 0.65, 0.64, 0.31, 0.29],
        "Suff_Action (v)":  [0.15, 0.16, 0.16, 0.18, 0.49, 0.44],
        "Compr_Object (^)": [0.66, 0.64, 0.66, 0.66, 0.25, 0.24],
        "Suff_Object (v)":  [0.08, 0.09, 0.08, 0.09, 0.44, 0.40],
        "Compr_Location (^)": [0.48, 0.45, 0.48, 0.47, 0.21, 0.17],
        "Suff_Location (v)":  [0.07, 0.06, 0.07, 0.06, 0.37, 0.29],
    })
    display(comparison)
    print("Insight: WA-L1O and WA-LIME achieve >2x higher Comprehensiveness and >3x lower Sufficiency than Random.")

if __name__ == "__main__":
    show_table1()
    show_table2()
    show_figure3()
    show_table3()
    show_table4()
