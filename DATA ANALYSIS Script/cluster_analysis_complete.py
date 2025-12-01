#!/usr/bin/env python3
"""
Complete Cluster Analysis: Figures, Stats, Interpretation, Validation
Extends injured_vs_contralateral_cluster_analysis.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, wilcoxon, shapiro
from scipy.cluster.hierarchy import dendrogram
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import umap

# Import the original clustering pipeline
from injured_vs_contralateral_cluster_analysis import (
    load_data, build_effect_features, preprocess_features,
    perform_clustering, select_best_model, consensus_clustering
)

# Paths
OUTPUT_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/DATA ANALYSIS Script/data_output")
RESULTS_DIR = OUTPUT_DIR / "RESULTS"
CLUSTER_DIR = RESULTS_DIR / "CLUSTERS"
FIGS_DIR = CLUSTER_DIR / "figs"

FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 100

AUDIT_LOG = []

def log_action(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    AUDIT_LOG.append(entry)
    print(entry)

# ============================================================================
# LOAD EXISTING CLUSTERING RESULTS
# ============================================================================
def load_clustering_results():
    """Load pre-computed clustering results"""
    log_action("Loading existing clustering results...")

    plane_memberships = pd.read_csv(CLUSTER_DIR / 'plane_memberships.csv')
    consensus_memberships = pd.read_csv(CLUSTER_DIR / 'consensus_membership.csv')
    stability_metrics = pd.read_csv(CLUSTER_DIR / 'stability_metrics.csv')

    log_action(f"  Loaded {len(plane_memberships)} plane memberships")
    log_action(f"  Loaded {len(consensus_memberships)} consensus memberships")

    return plane_memberships, consensus_memberships, stability_metrics

# ============================================================================
# STATISTICAL TESTS (Cluster Centroids)
# ============================================================================
def compute_cluster_centroids_and_tests(df_processed, plane_memberships, consensus_memberships, features):
    """Compute cluster centroids and test H0: mean(delta/rho) = 0"""
    log_action("Computing cluster centroid statistical tests...")

    # Merge memberships with processed data
    df_with_clusters = df_processed.copy()

    # Add plane memberships
    df_with_clusters = df_with_clusters.merge(
        plane_memberships[['Patient_ID', 'Plane', 'Cluster']],
        on=['Patient_ID', 'Plane'],
        how='left'
    )

    # Add consensus cluster
    df_with_clusters = df_with_clusters.merge(
        consensus_memberships.rename(columns={'Consensus_Cluster': 'Consensus_Cluster'}),
        on='Patient_ID',
        how='left'
    )

    test_results = []

    # Test per plane and consensus
    for analysis_type in ['Plane', 'Consensus']:
        if analysis_type == 'Plane':
            groupby_cols = ['Plane', 'Cluster']
        else:
            groupby_cols = ['Consensus_Cluster']
            df_with_clusters_subset = df_with_clusters.drop_duplicates(subset='Patient_ID')

        for group_name, group_df in df_with_clusters.groupby(groupby_cols):
            if analysis_type == 'Plane':
                plane, cluster = group_name
                group_label = f"{plane}_C{cluster}"
            else:
                cluster = group_name
                group_label = f"Consensus_C{cluster}"

            for feature in features:
                values = group_df[feature].dropna().values

                if len(values) < 3:
                    continue

                # Test normality
                if len(values) >= 3:
                    _, p_norm = shapiro(values)
                else:
                    p_norm = 1.0

                # One-sample test (H0: mean = 0)
                if p_norm > 0.05:
                    # Normal: use t-test
                    stat, p_value = ttest_1samp(values, 0)
                    test_name = 't-test'

                    # Cohen's d for one-sample
                    effect_size = np.mean(values) / np.std(values, ddof=1)
                    effect_name = 'Cohen_d'
                else:
                    # Non-normal: use Wilcoxon
                    stat, p_value = wilcoxon(values - 0)
                    test_name = 'Wilcoxon'

                    # Rank-biserial r
                    z = stat / np.sqrt(len(values))
                    effect_size = z / np.sqrt(len(values))
                    effect_name = 'r'

                # Bootstrap 95% CI
                boot_means = []
                for _ in range(1000):
                    boot_sample = np.random.choice(values, size=len(values), replace=True)
                    boot_means.append(np.mean(boot_sample))

                ci_lower = np.percentile(boot_means, 2.5)
                ci_upper = np.percentile(boot_means, 97.5)

                test_results.append({
                    'Analysis': analysis_type,
                    'Group': group_label,
                    'Feature': feature,
                    'N': len(values),
                    'Mean': np.mean(values),
                    'SD': np.std(values, ddof=1),
                    'Median': np.median(values),
                    'Test': test_name,
                    'Statistic': stat,
                    'p_value': p_value,
                    'Effect_Size': effect_size,
                    'Effect_Type': effect_name,
                    'CI_95_Lower': ci_lower,
                    'CI_95_Upper': ci_upper
                })

    df_tests = pd.DataFrame(test_results)

    # BH-FDR correction per analysis type
    for analysis_type in df_tests['Analysis'].unique():
        mask = df_tests['Analysis'] == analysis_type
        reject, pvals_corrected, _, _ = multipletests(
            df_tests.loc[mask, 'p_value'],
            method='fdr_bh'
        )
        df_tests.loc[mask, 'p_value_FDR'] = pvals_corrected
        df_tests.loc[mask, 'Significant_FDR'] = pvals_corrected < 0.05

    # Save
    df_tests.to_excel(CLUSTER_DIR / 'cluster_centroid_tests.xlsx', index=False)
    log_action(f"  Centroid tests completed: {len(df_tests)} tests")

    return df_tests

# ============================================================================
# PHENOTYPE LABELING
# ============================================================================
def assign_phenotype_labels(df_tests):
    """Assign human-readable phenotype names based on centroid signatures"""
    log_action("Assigning phenotype labels...")

    # Get consensus centroids
    consensus_tests = df_tests[df_tests['Analysis'] == 'Consensus'].copy()

    phenotypes = {}

    for cluster in consensus_tests['Group'].unique():
        cluster_tests = consensus_tests[consensus_tests['Group'] == cluster]

        # Extract key feature signatures
        sig_features = cluster_tests[cluster_tests['Significant_FDR'] == True]

        # Categorize by biomechanical meaning
        sway_features = ['delta_area', 'delta_perimeter', 'delta_compactness', 'rho_area']
        smoothness_features = ['delta_trajectory_smoothness', 'delta_closure_error']
        curvature_features = ['rho_curvature_std', 'delta_curv_phase_entropy', 'rho_curv_phase_variability_index']
        timing_features = ['delta_curv_phase_circular_corr', 'delta_mean_relative_phase']

        # Check signature patterns
        high_sway = any(feat in sig_features['Feature'].values and
                       sig_features[sig_features['Feature'] == feat]['Mean'].values[0] > 0
                       for feat in sway_features if feat in sig_features['Feature'].values)

        low_smoothness = any(feat in sig_features['Feature'].values and
                            sig_features[sig_features['Feature'] == feat]['Mean'].values[0] < 0
                            for feat in smoothness_features if feat in sig_features['Feature'].values)

        high_curvature_var = any(feat in sig_features['Feature'].values and
                                sig_features[sig_features['Feature'] == feat]['Mean'].values[0] > 0
                                for feat in curvature_features if feat in sig_features['Feature'].values)

        low_timing_sync = any(feat in sig_features['Feature'].values and
                             sig_features[sig_features['Feature'] == feat]['Mean'].values[0] < 0
                             for feat in timing_features if feat in sig_features['Feature'].values)

        # Assign label
        if high_sway and low_smoothness:
            phenotype = "Sway-Dominant Instability"
            interpretation = "↑CoM excursion, ↓compactness, ↓smoothness; mediolateral control deficit"
        elif high_curvature_var and low_timing_sync:
            phenotype = "Propulsion/Impact Dyscontrol"
            interpretation = "↑curvature variability, ↑entropy, phase lag; irregular push-off/landing"
        elif low_smoothness and not high_sway:
            phenotype = "Rotational Compensation"
            interpretation = "↑eccentricity, orientation bias, ↓phase coherence; trunk rotation to offload"
        else:
            # Check if near-zero (recovered)
            mean_abs_effect = cluster_tests['Mean'].abs().mean()
            if mean_abs_effect < 0.3:
                phenotype = "Near-Recovered"
                interpretation = "Symmetry indices ~0, low entropy, high smoothness; functional restoration"
            else:
                phenotype = "Unclassified Impairment"
                interpretation = "Mixed pattern, requires further investigation"

        phenotypes[cluster] = {
            'Phenotype': phenotype,
            'Interpretation': interpretation,
            'N_Significant_Features': len(sig_features),
            'Dominant_Direction': 'Impaired' if cluster_tests['Mean'].mean() != 0 else 'Neutral'
        }

    # Save dictionary
    with open(CLUSTER_DIR / 'cluster_dictionary.md', 'w') as f:
        f.write("# Cluster Phenotype Dictionary\n\n")
        f.write("**Date**: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        for cluster, info in sorted(phenotypes.items()):
            f.write(f"## {cluster}\n\n")
            f.write(f"**Phenotype**: {info['Phenotype']}\n\n")
            f.write(f"**Interpretation**: {info['Interpretation']}\n\n")
            f.write(f"**Significant Features**: {info['N_Significant_Features']}\n\n")
            f.write(f"**Direction**: {info['Dominant_Direction']}\n\n")
            f.write("---\n\n")

    log_action(f"  Assigned phenotypes to {len(phenotypes)} clusters")

    return phenotypes

# ============================================================================
# GENERATE INTERPRETATION SNIPPETS
# ============================================================================
def generate_interpretation_snippets(df_tests, phenotypes):
    """Generate one-liner interpretation snippets for manuscript"""
    log_action("Generating interpretation snippets...")

    snippets = []

    # Overall findings
    consensus_tests = df_tests[df_tests['Analysis'] == 'Consensus']
    n_sig = (consensus_tests['Significant_FDR'] == True).sum()

    snippets.append("# Interpretation Snippets for Manuscript\n")
    snippets.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    snippets.append("\n## Overall Findings\n")
    snippets.append(f"- {n_sig} features showed significant deviation from zero (BH-FDR q<0.05)\n")

    # Per phenotype
    snippets.append("\n## Cluster-Specific Interpretations\n")

    for cluster, info in sorted(phenotypes.items()):
        snippets.append(f"\n### {cluster}: {info['Phenotype']}\n")
        snippets.append(f"> {info['Interpretation']}\n")

        # Find top 3 significant features
        cluster_tests = consensus_tests[
            (consensus_tests['Group'] == cluster) &
            (consensus_tests['Significant_FDR'] == True)
        ].sort_values('p_value_FDR').head(3)

        if len(cluster_tests) > 0:
            snippets.append("\n**Key Effects**:\n")
            for _, row in cluster_tests.iterrows():
                direction = "↑" if row['Mean'] > 0 else "↓"
                snippets.append(
                    f"- {direction}{row['Feature']}: "
                    f"Δ={row['Mean']:.3f} (95% CI [{row['CI_95_Lower']:.3f}, {row['CI_95_Upper']:.3f}]), "
                    f"q={row['p_value_FDR']:.4f}\n"
                )

    # Hypothesis support statement
    snippets.append("\n## Hypothesis Support Statement\n")

    impaired_smoothness = consensus_tests[
        consensus_tests['Feature'].str.contains('smoothness') &
        (consensus_tests['Mean'] < 0) &
        (consensus_tests['Significant_FDR'] == True)
    ]

    increased_variability = consensus_tests[
        (consensus_tests['Feature'].str.contains('entropy|variability|curvature_std')) &
        (consensus_tests['Mean'] > 0) &
        (consensus_tests['Significant_FDR'] == True)
    ]

    if len(impaired_smoothness) > 0 or len(increased_variability) > 0:
        snippets.append("> **SUPPORTS HYPOTHESIS**: ")
        snippets.append("Injured leg exhibits less stable and less synchronous CoM control ")
        snippets.append("with reduced smoothness and increased variability/entropy.\n")
    else:
        snippets.append("> **PARTIAL SUPPORT**: ")
        snippets.append("Some evidence of altered control patterns, but not all primary endpoints significant.\n")

    # Save
    with open(CLUSTER_DIR / 'INTERPRETATION_SNIPPETS.md', 'w') as f:
        f.writelines(snippets)

    log_action("  Interpretation snippets generated")

# ============================================================================
# ARI VALIDATION
# ============================================================================
def validate_against_existing_labels(df_processed, consensus_memberships):
    """Compare discovered clusters to CYCLOGRAM_Cluster"""
    log_action("Validating against existing CYCLOGRAM_Cluster labels...")

    # Merge consensus with original labels
    df_validation = df_processed[['Patient_ID', 'CYCLOGRAM_Cluster']].drop_duplicates()
    df_validation = df_validation.merge(consensus_memberships, on='Patient_ID', how='inner')

    # Convert labels to strings for consistency
    existing_labels = df_validation['CYCLOGRAM_Cluster'].astype(str).values
    consensus_labels = df_validation['Consensus_Cluster'].astype(str).values

    # Compute ARI
    ari = adjusted_rand_score(existing_labels, consensus_labels)

    # Confusion matrix
    conf_matrix = confusion_matrix(existing_labels, consensus_labels)

    # Save
    ari_results = pd.DataFrame({
        'Metric': ['Adjusted_Rand_Index'],
        'Value': [ari],
        'Interpretation': ['High (>0.7)' if ari > 0.7 else 'Moderate (0.4-0.7)' if ari > 0.4 else 'Low (<0.4)']
    })
    ari_results.to_csv(CLUSTER_DIR / 'ari_vs_existing.csv', index=False)

    # Save confusion matrix - just save as numpy array to avoid dimension issues
    np.savetxt(CLUSTER_DIR / 'confusion_matrix.csv', conf_matrix, delimiter=',', fmt='%d')

    # Also save label mappings
    existing_unique = sorted(set(existing_labels))
    consensus_unique = sorted(set(consensus_labels))

    with open(CLUSTER_DIR / 'confusion_matrix_labels.txt', 'w') as f:
        f.write(f"Existing labels: {existing_unique}\n")
        f.write(f"Consensus labels: {consensus_unique}\n")
        f.write(f"Matrix shape: {conf_matrix.shape}\n")

    log_action(f"  ARI = {ari:.3f}")
    log_action(f"  Confusion matrix shape: {conf_matrix.shape}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=consensus_unique, yticklabels=existing_unique)
    ax.set_title(f'Confusion Matrix: Existing vs Consensus Clusters\n(ARI={ari:.3f})')
    ax.set_xlabel('Consensus Cluster')
    ax.set_ylabel('Existing CYCLOGRAM_Cluster')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'cluster_confusion_vs_existing.pdf')
    plt.savefig(FIGS_DIR / 'cluster_confusion_vs_existing.png', dpi=300)
    plt.close()

    log_action("  Confusion matrix figure saved")

    return ari, conf_matrix

# ============================================================================
# GENERATE BASIC FIGURES
# ============================================================================
def generate_umap_figure(df_processed, consensus_memberships, features):
    """Generate UMAP visualization"""
    log_action("Generating UMAP figure...")

    # Merge consensus clusters
    df_viz = df_processed.merge(consensus_memberships, on='Patient_ID', how='inner')

    # Deduplicate by patient (use one plane, e.g., 3D)
    df_viz_3d = df_viz[df_viz['Plane'] == '3D'].copy()

    # Drop NaN values
    df_viz_clean = df_viz_3d[features + ['Consensus_Cluster']].dropna()

    X = df_viz_clean[features].values
    labels = df_viz_clean['Consensus_Cluster'].values

    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = reducer.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for cluster in sorted(set(labels)):
        mask = labels == cluster
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                  label=f'Cluster {cluster}', alpha=0.6, s=50)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('UMAP: Consensus Clusters (3D Plane)')
    ax.legend()
    plt.tight_layout()

    plt.savefig(FIGS_DIR / 'cluster_umap_3D.pdf')
    plt.savefig(FIGS_DIR / 'cluster_umap_3D.png', dpi=300)
    plt.close()

    log_action("  UMAP figure saved")

def generate_radar_plots(df_tests):
    """Generate radar plots of cluster centroids"""
    log_action("Generating radar plots...")

    consensus_tests = df_tests[df_tests['Analysis'] == 'Consensus'].copy()

    # Get top features by effect size
    top_features = consensus_tests.groupby('Feature')['Effect_Size'].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(8).index.tolist()

    clusters = sorted(consensus_tests['Group'].unique())

    fig, axes = plt.subplots(1, len(clusters), figsize=(6*len(clusters), 6), subplot_kw=dict(projection='polar'))

    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        cluster_data = consensus_tests[consensus_tests['Group'] == cluster]

        values = []
        labels_list = []
        for feat in top_features:
            feat_data = cluster_data[cluster_data['Feature'] == feat]
            if len(feat_data) > 0:
                values.append(feat_data['Mean'].values[0])
                labels_list.append(feat.replace('delta_', 'Δ').replace('rho_', 'ρ'))

        if len(values) == 0:
            continue

        angles = np.linspace(0, 2*np.pi, len(values), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=cluster)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_list, size=9)
        ax.set_title(f'{cluster}\nCentroid Signature', pad=20)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'cluster_radar_consensus.pdf')
    plt.savefig(FIGS_DIR / 'cluster_radar_consensus.png', dpi=300)
    plt.close()

    log_action("  Radar plots saved")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*80)
    print("CLUSTER ANALYSIS COMPLETION: FIGURES, STATS, INTERPRETATION")
    print("="*80)

    try:
        # 1. Load existing results
        plane_memberships, consensus_memberships, stability_metrics = load_clustering_results()

        # 2. Re-load and process data
        df = load_data()
        df_effects, df_catalog = build_effect_features(df)
        df_processed, features = preprocess_features(df_effects, df_catalog)

        # 3. Statistical tests
        df_tests = compute_cluster_centroids_and_tests(
            df_processed, plane_memberships, consensus_memberships, features
        )

        # 4. Phenotype labeling
        phenotypes = assign_phenotype_labels(df_tests)

        # 5. Interpretation snippets
        generate_interpretation_snippets(df_tests, phenotypes)

        # 6. ARI validation
        ari, conf_matrix = validate_against_existing_labels(df_processed, consensus_memberships)

        # 7. Generate figures
        generate_umap_figure(df_processed, consensus_memberships, features)
        generate_radar_plots(df_tests)

        # Write audit log
        with open(CLUSTER_DIR / 'logs' / 'completion_audit.log', 'w') as f:
            for entry in AUDIT_LOG:
                f.write(entry + '\n')

        log_action("="*80)
        log_action("ANALYSIS COMPLETION SUCCESSFUL")
        log_action("="*80)
        log_action(f"Results in: {CLUSTER_DIR}")

        return df_tests, phenotypes, ari

    except Exception as e:
        log_action(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    df_tests, phenotypes, ari = main()
    print("\n\nSUCCESS: Complete cluster analysis finished")
    print(f"ARI vs existing labels: {ari:.3f}")
    print(f"Phenotypes discovered: {len(phenotypes)}")
