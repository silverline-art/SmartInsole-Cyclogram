#!/usr/bin/env python3
"""
Injured vs. Contralateral Cluster Analysis Pipeline
Publication-ready phenotype discovery with delta/rho effect features
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_1samp, wilcoxon, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/DATA ANALYSIS Script/Data/Clustered_Data")
OUTPUT_DIR = Path("/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/DATA ANALYSIS Script/data_output")
RESULTS_DIR = OUTPUT_DIR / "RESULTS"
CLUSTER_DIR = RESULTS_DIR / "CLUSTERS"
LOGS_DIR = CLUSTER_DIR / "logs"
FIGS_DIR = CLUSTER_DIR / "figs"

# Create directories
for d in [CLUSTER_DIR, LOGS_DIR, FIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    'XY': DATA_DIR / 'ACC_XY_CLUSTERED.xlsx',
    'XZ': DATA_DIR / 'ACC_XZ_CLUSTERED.xlsx',
    'YZ': DATA_DIR / 'ACC_YZ_CLUSTERED.xlsx',
    '3D': DATA_DIR / 'ACC_3D_CLUSTERED.xlsx'
}

# Feature families for effect computation
FEATURE_FAMILIES = {
    'sway_envelope': ['area', 'perimeter', 'compactness', 'closure_error'],
    'smoothness': ['trajectory_smoothness'],
    'curvature': ['mean_curvature', 'curvature_std', 'curv_phase_entropy',
                   'curv_phase_variability_index', 'curv_phase_rms'],
    'timing_sync': ['mean_relative_phase', 'marp', 'coupling_angle_variability',
                     'phase_shift', 'curv_phase_circular_corr'],
    'asymmetry': ['bilateral_symmetry_index', 'bilateral_mirror_correlation',
                   'bilateral_rms_trajectory_diff_x', 'bilateral_rms_trajectory_diff_y']
}

# Circular angle features (need sin-cos transform)
CIRCULAR_FEATURES = ['curv_phase_peak_phase_%', 'orientation_angle']

# Clustering parameters
K_RANGE = range(2, 7)
BOOTSTRAP_ITERATIONS = 100
JACCARD_THRESHOLD = 0.75
SPEARMAN_THRESHOLD = 0.90

# Audit log
AUDIT_LOG = []

def log_action(message):
    """Log actions for audit trail"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    AUDIT_LOG.append(entry)
    print(entry)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def parse_mean_std_string(value):
    """Parse 'mean ± std' format to extract mean value"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    str_val = str(value).strip()
    if '±' in str_val:
        try:
            return float(str_val.split('±')[0].strip())
        except:
            return np.nan
    else:
        try:
            return float(str_val)
        except:
            return np.nan

def map_injured_side(row):
    """Map fracture site to injured side (L/R)"""
    fracture_site = str(row.get('Fracture Site', '')).lower()

    if 'left' in fracture_site or 'l' == fracture_site.strip():
        return 'L'
    elif 'right' in fracture_site or 'r' == fracture_site.strip():
        return 'R'
    elif 'bilateral' in fracture_site or 'both' in fracture_site:
        return 'Bilateral'
    else:
        return 'Unknown'

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================
def load_data():
    """Load all plane data with mean±std parsing"""
    log_action("Loading clustered data files...")

    all_data = []

    for plane, filepath in DATA_FILES.items():
        if not filepath.exists():
            log_action(f"  WARNING: {filepath} not found, skipping")
            continue

        df = pd.read_excel(filepath, sheet_name=0)
        df['plane'] = plane

        # Parse mean±std columns
        metadata_cols = ['Dataset', 'Patient Name', 'Patient ID', 'Fracture Site',
                         'Fracture Name', 'CYCLOGRAM_Cluster', 'plane']
        numeric_cols = [col for col in df.columns if col not in metadata_cols]

        for col in numeric_cols:
            df[col] = df[col].apply(parse_mean_std_string)

        all_data.append(df)
        log_action(f"  Loaded {plane}: {len(df)} rows")

    df_combined = pd.concat(all_data, ignore_index=True)

    # Map injured side
    df_combined['Injured_Side'] = df_combined.apply(map_injured_side, axis=1)

    log_action(f"Combined: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    log_action(f"  Injured side distribution: {df_combined['Injured_Side'].value_counts().to_dict()}")

    return df_combined

# ============================================================================
# 2. BUILD EFFECT FEATURES (Δ and ρ)
# ============================================================================
def build_effect_features(df):
    """Build injured - contralateral delta and log-ratio features"""
    log_action("Building delta/rho effect features...")

    effect_features = []
    feature_catalog = []

    for plane in df['plane'].unique():
        df_plane = df[df['plane'] == plane].copy()

        # Exclude bilateral/unknown for paired analysis
        df_paired = df_plane[df_plane['Injured_Side'].isin(['L', 'R'])].copy()

        log_action(f"  {plane}: {len(df_paired)} patients with clear injured side")

        if len(df_paired) == 0:
            continue

        plane_effects = []

        for idx, row in df_paired.iterrows():
            patient_id = row['Patient ID']
            injured_side = row['Injured_Side']

            effect_row = {
                'Patient_ID': patient_id,
                'Plane': plane,
                'Injured_Side': injured_side,
                'Fracture_Site': row['Fracture Site'],
                'Fracture_Name': row['Fracture Name'],
                'CYCLOGRAM_Cluster': row['CYCLOGRAM_Cluster']
            }

            # Iterate through feature families
            for family, base_features in FEATURE_FAMILIES.items():
                for base_feat in base_features:
                    # Skip if base feature doesn't exist
                    left_col = f'left_{base_feat}'
                    right_col = f'right_{base_feat}'

                    # Also check bilateral metrics (no left/right prefix)
                    if base_feat.startswith('bilateral_'):
                        # Bilateral metrics are already aggregated
                        if base_feat in row.index and pd.notna(row[base_feat]):
                            # For bilateral metrics, just use the value directly
                            effect_row[f'{base_feat}'] = row[base_feat]

                            if plane == 'XY':  # Log only once
                                feature_catalog.append({
                                    'Feature': base_feat,
                                    'Family': family,
                                    'Type': 'Bilateral',
                                    'Transform': 'None'
                                })
                        continue

                    if left_col not in row.index or right_col not in row.index:
                        continue

                    left_val = row[left_col]
                    right_val = row[right_col]

                    if pd.isna(left_val) or pd.isna(right_val):
                        continue

                    # Determine injured and control values
                    if injured_side == 'L':
                        injured_val = left_val
                        control_val = right_val
                    else:
                        injured_val = right_val
                        control_val = left_val

                    # Handle circular features
                    if base_feat in CIRCULAR_FEATURES:
                        # Convert to radians and compute sin/cos
                        injured_rad = np.deg2rad(injured_val)
                        control_rad = np.deg2rad(control_val)

                        # Circular difference (delta angle)
                        delta_angle = np.rad2deg(np.arctan2(
                            np.sin(injured_rad - control_rad),
                            np.cos(injured_rad - control_rad)
                        ))

                        effect_row[f'delta_{base_feat}'] = delta_angle

                        if plane == 'XY':
                            feature_catalog.append({
                                'Feature': f'delta_{base_feat}',
                                'Family': family,
                                'Type': 'Delta',
                                'Transform': 'Circular'
                            })

                    # Decide delta vs rho
                    elif base_feat in ['compactness', 'trajectory_smoothness', 'mirror_correlation',
                                        'symmetry_index', 'curv_phase_circular_corr']:
                        # Bounded metrics: use delta
                        effect_row[f'delta_{base_feat}'] = injured_val - control_val

                        if plane == 'XY':
                            feature_catalog.append({
                                'Feature': f'delta_{base_feat}',
                                'Family': family,
                                'Type': 'Delta',
                                'Transform': 'Difference'
                            })

                    else:
                        # Unbounded positive metrics: use log-ratio
                        if injured_val > 0 and control_val > 0:
                            effect_row[f'rho_{base_feat}'] = np.log(injured_val / control_val)

                            if plane == 'XY':
                                feature_catalog.append({
                                    'Feature': f'rho_{base_feat}',
                                    'Family': family,
                                    'Type': 'Rho',
                                    'Transform': 'Log-ratio'
                                })

            plane_effects.append(effect_row)

        effect_features.extend(plane_effects)

    df_effects = pd.DataFrame(effect_features)
    df_catalog = pd.DataFrame(feature_catalog).drop_duplicates()

    log_action(f"  Built {len(df_effects)} effect feature rows")
    log_action(f"  Feature catalog: {len(df_catalog)} unique features")

    # Save catalog
    df_catalog.to_csv(LOGS_DIR / 'feature_catalog_initial.csv', index=False)

    return df_effects, df_catalog

# ============================================================================
# 3. QC AND PREPROCESSING
# ============================================================================
def preprocess_features(df_effects, df_catalog):
    """QC, imputation, winsorization, robust scaling, collinearity removal"""
    log_action("Preprocessing effect features...")

    # Separate metadata from features
    metadata_cols = ['Patient_ID', 'Plane', 'Injured_Side', 'Fracture_Site',
                     'Fracture_Name', 'CYCLOGRAM_Cluster']

    feature_cols = [col for col in df_effects.columns if col not in metadata_cols]

    log_action(f"  Feature columns: {len(feature_cols)}")

    # Missingness report
    missingness = df_effects[feature_cols].isnull().mean()

    # Categorize features by missingness
    keep_features = []
    impute_median = []
    impute_knn = []
    drop_features = []

    for col in feature_cols:
        miss_frac = missingness[col]

        if miss_frac == 0:
            keep_features.append(col)
        elif miss_frac <= 0.05:
            impute_median.append(col)
        elif miss_frac <= 0.15:
            impute_knn.append(col)
        else:
            drop_features.append(col)

    log_action(f"  Complete features: {len(keep_features)}")
    log_action(f"  Impute (median): {len(impute_median)}")
    log_action(f"  Impute (kNN): {len(impute_knn)}")
    log_action(f"  Drop (>15% missing): {len(drop_features)}")

    # Drop high-missingness features
    retained_features = [col for col in feature_cols if col not in drop_features]

    df_processed = df_effects[metadata_cols + retained_features].copy()

    # Imputation per plane
    for plane in df_processed['Plane'].unique():
        plane_mask = df_processed['Plane'] == plane

        # Median imputation
        for col in impute_median:
            if col in df_processed.columns:
                median_val = df_processed.loc[plane_mask, col].median()
                df_processed.loc[plane_mask, col].fillna(median_val, inplace=True)

        # kNN imputation
        if len(impute_knn) > 0:
            knn_cols = [col for col in impute_knn if col in df_processed.columns]
            if len(knn_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df_processed.loc[plane_mask, knn_cols] = imputer.fit_transform(
                    df_processed.loc[plane_mask, knn_cols]
                )

    log_action(f"  Imputation complete")

    # Winsorize at 1-99%
    for col in retained_features:
        if col in df_processed.columns:
            q01 = df_processed[col].quantile(0.01)
            q99 = df_processed[col].quantile(0.99)
            df_processed[col] = df_processed[col].clip(lower=q01, upper=q99)

    log_action(f"  Winsorization complete (1-99%)")

    # Robust scaling per plane
    scaler_dict = {}

    for plane in df_processed['Plane'].unique():
        plane_mask = df_processed['Plane'] == plane

        scaler = RobustScaler()
        df_processed.loc[plane_mask, retained_features] = scaler.fit_transform(
            df_processed.loc[plane_mask, retained_features]
        )
        scaler_dict[plane] = scaler

    log_action(f"  Robust scaling complete")

    # Collinearity removal (Spearman > 0.90)
    log_action(f"  Checking collinearity...")

    final_features = retained_features.copy()

    for plane in df_processed['Plane'].unique():
        df_plane = df_processed[df_processed['Plane'] == plane][retained_features].dropna()

        if len(df_plane) < 3:
            continue

        # Compute Spearman correlation matrix
        corr_matrix = df_plane.corr(method='spearman').abs()

        # Find pairs with |r| > 0.90
        upper_tri = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        high_corr_pairs = np.where((corr_matrix.values > SPEARMAN_THRESHOLD) & upper_tri)

        to_drop = set()
        for i, j in zip(*high_corr_pairs):
            feat1 = corr_matrix.index[i]
            feat2 = corr_matrix.columns[j]

            # Drop the one with higher mean abs correlation with others
            mean_corr1 = corr_matrix[feat1].mean()
            mean_corr2 = corr_matrix[feat2].mean()

            if mean_corr1 > mean_corr2:
                to_drop.add(feat1)
            else:
                to_drop.add(feat2)

        if len(to_drop) > 0:
            log_action(f"    {plane}: dropping {len(to_drop)} collinear features")
            final_features = [f for f in final_features if f not in to_drop]

    df_final = df_processed[metadata_cols + final_features].copy()

    log_action(f"  Final feature count: {len(final_features)}")

    # Update catalog
    df_catalog_final = df_catalog[df_catalog['Feature'].isin(final_features)].copy()
    df_catalog_final['Kept'] = True

    dropped_catalog = df_catalog[~df_catalog['Feature'].isin(final_features)].copy()
    dropped_catalog['Kept'] = False
    dropped_catalog['Reason'] = 'Missingness or Collinearity'

    df_catalog_complete = pd.concat([df_catalog_final, dropped_catalog], ignore_index=True)
    df_catalog_complete.to_csv(LOGS_DIR / 'feature_catalog.csv', index=False)

    return df_final, final_features

# ============================================================================
# 4. CLUSTERING ALGORITHMS
# ============================================================================
def cluster_gmm(X, k_range):
    """Gaussian Mixture Model clustering"""
    results = []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(X)

        bic = gmm.bic(X)
        aic = gmm.aic(X)

        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil = dbi = ch = np.nan

        results.append({
            'Algorithm': 'GMM',
            'K': k,
            'BIC': bic,
            'AIC': aic,
            'Silhouette': sil,
            'DBI': dbi,
            'CH': ch,
            'Labels': labels
        })

    return results

def cluster_kmeans(X, k_range):
    """K-Means clustering"""
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)

        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil = dbi = ch = np.nan

        results.append({
            'Algorithm': 'KMeans',
            'K': k,
            'BIC': np.nan,
            'AIC': np.nan,
            'Silhouette': sil,
            'DBI': dbi,
            'CH': ch,
            'Labels': labels
        })

    return results

def cluster_hierarchical(X, k_range):
    """Hierarchical (Ward) clustering"""
    results = []

    linkage_matrix = linkage(X, method='ward')

    for k in k_range:
        labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1

        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil = dbi = ch = np.nan

        results.append({
            'Algorithm': 'Hierarchical',
            'K': k,
            'BIC': np.nan,
            'AIC': np.nan,
            'Silhouette': sil,
            'DBI': dbi,
            'CH': ch,
            'Labels': labels,
            'Linkage': linkage_matrix
        })

    return results

def bootstrap_stability(X, labels, n_iterations=100):
    """Bootstrap Jaccard index for cluster stability"""
    n_samples = len(X)
    n_clusters = len(set(labels))

    jaccard_scores = []

    for _ in range(n_iterations):
        # Bootstrap sample
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]

        # Re-cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_boot = kmeans.fit_predict(X_boot)

        # Map back to original indices and compute Jaccard
        # (simplified: use ARI as proxy)
        original_labels_boot = labels[idx]
        ari = adjusted_rand_score(original_labels_boot, labels_boot)
        jaccard_scores.append(max(0, ari))  # ARI can be negative

    return np.mean(jaccard_scores)

def perform_clustering(df, features):
    """Run all clustering algorithms per plane"""
    log_action("Running clustering algorithms...")

    all_results = []

    for plane in df['Plane'].unique():
        df_plane = df[df['Plane'] == plane].copy()

        # Keep track of valid indices after dropna
        df_plane_clean = df_plane[features].dropna()
        valid_indices = df_plane_clean.index
        X = df_plane_clean.values

        if len(X) < 10:
            log_action(f"  {plane}: insufficient data ({len(X)} samples)")
            continue

        log_action(f"  {plane}: {len(X)} samples, {X.shape[1]} features")

        # Run algorithms
        gmm_results = cluster_gmm(X, K_RANGE)
        kmeans_results = cluster_kmeans(X, K_RANGE)
        hierarchical_results = cluster_hierarchical(X, K_RANGE)

        # Combine and add plane info
        for result_set in [gmm_results, kmeans_results, hierarchical_results]:
            for result in result_set:
                result['Plane'] = plane
                result['N_samples'] = len(X)
                result['Valid_Indices'] = valid_indices  # Store valid indices

                # Bootstrap stability for best models
                if result['Silhouette'] == result['Silhouette']:  # not NaN
                    jaccard = bootstrap_stability(X, result['Labels'], n_iterations=50)
                    result['Bootstrap_Jaccard'] = jaccard
                else:
                    result['Bootstrap_Jaccard'] = np.nan

                all_results.append(result)

    df_clustering = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Labels', 'Linkage', 'Valid_Indices']}
                                    for r in all_results])

    log_action(f"  Clustering complete: {len(all_results)} models evaluated")

    return all_results, df_clustering

# ============================================================================
# 5. MODEL SELECTION
# ============================================================================
def select_best_model(clustering_results, df_clustering):
    """Select best K and algorithm per plane"""
    log_action("Selecting best models...")

    best_models = {}

    for plane in df_clustering['Plane'].unique():
        df_plane = df_clustering[df_clustering['Plane'] == plane].copy()

        # Filter valid models (Jaccard > threshold, Silhouette > 0)
        df_valid = df_plane[
            (df_plane['Bootstrap_Jaccard'] > JACCARD_THRESHOLD) &
            (df_plane['Silhouette'] > 0)
        ].copy()

        if len(df_valid) == 0:
            log_action(f"  {plane}: No stable models found, using best by Silhouette")
            df_valid = df_plane[df_plane['Silhouette'] > 0].copy()

        if len(df_valid) == 0:
            log_action(f"  {plane}: No valid models")
            continue

        # Rank by composite score: Silhouette (high) + DBI (low) + Jaccard (high)
        df_valid['Composite_Score'] = (
            df_valid['Silhouette'] -
            df_valid['DBI'] / df_valid['DBI'].max() +
            df_valid['Bootstrap_Jaccard']
        )

        best_idx = df_valid['Composite_Score'].idxmax()
        best_row = df_valid.loc[best_idx]

        # Find full result with labels
        matching_result = [r for r in clustering_results
                           if r['Plane'] == plane and
                           r['Algorithm'] == best_row['Algorithm'] and
                           r['K'] == best_row['K']][0]

        best_models[plane] = matching_result

        log_action(f"  {plane}: {best_row['Algorithm']} K={best_row['K']} "
                   f"(Sil={best_row['Silhouette']:.3f}, Jaccard={best_row['Bootstrap_Jaccard']:.3f})")

    return best_models

# ============================================================================
# 6. CONSENSUS CLUSTERING
# ============================================================================
def consensus_clustering(df, best_models):
    """Build consensus clusters across planes"""
    log_action("Building consensus clusters...")

    # Get unique patients across all planes (from valid indices only)
    all_valid_patients = set()
    for plane, model_result in best_models.items():
        df_plane = df[df['Plane'] == plane].copy()
        valid_indices = model_result['Valid_Indices']
        all_valid_patients.update(df_plane.loc[valid_indices, 'Patient_ID'].values)

    patients = sorted(list(all_valid_patients))

    # Build co-association matrix
    n_patients = len(patients)
    co_assoc = np.zeros((n_patients, n_patients))

    patient_to_idx = {p: i for i, p in enumerate(patients)}

    for plane, model_result in best_models.items():
        df_plane = df[df['Plane'] == plane].copy()
        valid_indices = model_result['Valid_Indices']
        df_plane_valid = df_plane.loc[valid_indices]

        labels = model_result['Labels']
        plane_patients = df_plane_valid['Patient_ID'].values

        # Build co-association for this plane
        for i in range(len(plane_patients)):
            for j in range(len(plane_patients)):
                if labels[i] == labels[j]:
                    p1 = plane_patients[i]
                    p2 = plane_patients[j]

                    if p1 in patient_to_idx and p2 in patient_to_idx:
                        idx1 = patient_to_idx[p1]
                        idx2 = patient_to_idx[p2]
                        co_assoc[idx1, idx2] += 1

    # Normalize by number of planes
    co_assoc /= len(best_models)

    # Convert to distance and cluster
    # Ensure co_assoc is in [0, 1] and symmetric
    co_assoc = np.clip(co_assoc, 0, 1)
    co_assoc = (co_assoc + co_assoc.T) / 2

    distance_matrix = 1 - co_assoc

    # Ensure distance is symmetric and non-negative
    distance_matrix = np.clip(distance_matrix, 0, None)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)  # Zero diagonal for silhouette

    # Hierarchical clustering on consensus
    try:
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')
    except:
        # Fallback: use complete linkage if average fails
        linkage_matrix = linkage(distance_matrix, method='complete')

    # Try different K values
    best_k = 3
    best_score = -1

    for k in range(2, min(7, n_patients)):
        consensus_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1

        if len(set(consensus_labels)) > 1:
            # Use distance matrix for silhouette
            sil = silhouette_score(distance_matrix, consensus_labels, metric='precomputed')

            if sil > best_score:
                best_score = sil
                best_k = k

    consensus_labels = fcluster(linkage_matrix, best_k, criterion='maxclust') - 1

    log_action(f"  Consensus: K={best_k}, Silhouette={best_score:.3f}")

    # Create consensus membership DataFrame
    consensus_df = pd.DataFrame({
        'Patient_ID': patients,
        'Consensus_Cluster': consensus_labels
    })

    return consensus_df, linkage_matrix

# ============================================================================
# 7. SAVE MEMBERSHIPS AND METRICS
# ============================================================================
def save_clustering_outputs(df, best_models, consensus_df, df_clustering):
    """Save cluster memberships and stability metrics"""
    log_action("Saving clustering outputs...")

    # Plane-wise memberships
    plane_memberships = []

    for plane, model_result in best_models.items():
        df_plane = df[df['Plane'] == plane].copy()

        # Use valid indices to align labels
        valid_indices = model_result['Valid_Indices']
        labels = model_result['Labels']

        # Create membership dataframe only for valid indices
        df_plane_valid = df_plane.loc[valid_indices]

        membership_df = pd.DataFrame({
            'Patient_ID': df_plane_valid['Patient_ID'].values,
            'Plane': plane,
            'Cluster': labels,
            'Algorithm': model_result['Algorithm'],
            'K': model_result['K']
        })

        plane_memberships.append(membership_df)

    df_plane_memberships = pd.concat(plane_memberships, ignore_index=True)
    df_plane_memberships.to_csv(CLUSTER_DIR / 'plane_memberships.csv', index=False)

    # Consensus membership
    consensus_df.to_csv(CLUSTER_DIR / 'consensus_membership.csv', index=False)

    # Stability metrics
    df_clustering.to_csv(CLUSTER_DIR / 'stability_metrics.csv', index=False)

    log_action(f"  Saved plane memberships, consensus, and stability metrics")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution pipeline"""

    print("="*80)
    print("INJURED VS. CONTRALATERAL CLUSTER ANALYSIS")
    print("="*80)

    try:
        # 1. Load data
        df = load_data()

        # 2. Build effect features
        df_effects, df_catalog = build_effect_features(df)

        # 3. Preprocess
        df_processed, features = preprocess_features(df_effects, df_catalog)

        # 4. Clustering
        clustering_results, df_clustering = perform_clustering(df_processed, features)

        # 5. Model selection
        best_models = select_best_model(clustering_results, df_clustering)

        # 6. Consensus clustering
        consensus_df, consensus_linkage = consensus_clustering(df_processed, best_models)

        # 7. Save outputs
        save_clustering_outputs(df_processed, best_models, consensus_df, df_clustering)

        log_action("="*80)
        log_action("CLUSTERING ANALYSIS COMPLETE")
        log_action("="*80)

        # Write audit log
        with open(LOGS_DIR / 'clustering_audit.log', 'w') as f:
            for entry in AUDIT_LOG:
                f.write(entry + '\n')

        return df_processed, best_models, consensus_df, features

    except Exception as e:
        log_action(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    df_processed, best_models, consensus_df, features = main()
    print("\n\nSUCCESS: Clustering analysis complete")
    print(f"Results saved to: {CLUSTER_DIR}")
