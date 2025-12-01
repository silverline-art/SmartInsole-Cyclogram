#!/usr/bin/env python3
"""
Generate Missing Figures for Cluster Analysis - Simplified Version
==================================================================

Generates missing figures using available test results and memberships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

# Try plotly for Sankey
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: plotly not installed, will skip Sankey diagram")

# Paths
BASE_DIR = Path('/home/shivam/Desktop/Human_Pose/PROJECT CYCLOGRAM/DATA ANALYSIS Script')
CLUSTER_DIR = BASE_DIR / 'data_output' / 'RESULTS' / 'CLUSTERS'
FIG_DIR = CLUSTER_DIR / 'figs'

# Matplotlib settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("=" * 80)
print("GENERATING MISSING FIGURES - SIMPLIFIED APPROACH")
print("=" * 80)

# Load data
print("\nLoading data...")
df_plane_memberships = pd.read_csv(CLUSTER_DIR / 'plane_memberships.csv').drop_duplicates()
df_consensus = pd.read_csv(CLUSTER_DIR / 'consensus_membership.csv')
df_tests = pd.read_excel(CLUSTER_DIR / 'cluster_centroid_tests.xlsx')

print(f"  - Plane memberships: {df_plane_memberships.shape}")
print(f"  - Consensus: {df_consensus.shape}")
print(f"  - Tests: {df_tests.shape}")

planes = ['XY', 'XZ', 'YZ', '3D']

# ============================================================================
# 1. PER-PLANE RADAR PLOTS (from test results)
# ============================================================================

print("\n[1/4] Generating per-plane radar plots...")

for plane in planes:
    # Filter by plane prefix in Group column
    df_plane_tests = df_tests[df_tests['Group'].str.startswith(f'{plane}_', na=False)]

    if len(df_plane_tests) == 0:
        print(f"  - No tests for {plane}, skipping")
        continue

    # Get top 8 features by effect size
    df_top = df_plane_tests.loc[df_plane_tests.groupby('Feature')['Effect_Size'].transform('max').abs().nlargest(8).index]
    features = df_top['Feature'].unique()[:8]

    # Get unique clusters
    clusters = sorted(df_plane_tests['Group'].unique(), key=str)

    if len(features) == 0 or len(clusters) == 0:
        print(f"  - No data for {plane}, skipping")
        continue

    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    for cluster in clusters:
        df_cluster = df_plane_tests[
            (df_plane_tests['Group'] == cluster) &
            (df_plane_tests['Feature'].isin(features))
        ]

        values = []
        for feat in features:
            feat_data = df_cluster[df_cluster['Feature'] == feat]
            if len(feat_data) > 0:
                values.append(feat_data.iloc[0]['Mean'])
            else:
                values.append(0)

        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=f'{cluster}', markersize=6)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('delta_', 'Δ').replace('rho_', 'ρ') for f in features], fontsize=9)
    ax.set_title(f'{plane} Plane: Cluster Centroid Signatures\n(Top 8 Features)', pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f'cluster_radar_{plane}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'cluster_radar_{plane}.png', bbox_inches='tight')
    plt.close()

    print(f"  ✓ Radar plot for {plane}")

# ============================================================================
# 2. PER-PLANE VOLCANO PLOTS
# ============================================================================

print("\n[2/4] Generating per-plane volcano plots...")

for plane in planes:
    df_plane_tests = df_tests[df_tests['Group'].str.startswith(f'{plane}_', na=False)]

    if len(df_plane_tests) == 0:
        print(f"  - No tests for {plane}, skipping")
        continue

    effect_sizes = df_plane_tests['Effect_Size'].values
    q_values = df_plane_tests['p_value_FDR'].values
    features = df_plane_tests['Feature'].values

    neg_log_q = -np.log10(q_values + 1e-300)

    fig, ax = plt.subplots(figsize=(12, 8))

    significant = q_values < 0.05
    ax.scatter(effect_sizes[~significant], neg_log_q[~significant],
               alpha=0.5, s=50, color='gray', label='Not significant')
    ax.scatter(effect_sizes[significant], neg_log_q[significant],
               alpha=0.7, s=80, color='red', label='Significant (q<0.05)')

    ax.axhline(-np.log10(0.05), color='blue', linestyle='--', linewidth=1, label='q=0.05 threshold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

    # Annotate top features
    if significant.sum() > 0:
        top_indices = np.argsort(neg_log_q)[-min(8, significant.sum()):]
        for idx in top_indices:
            if significant[idx]:
                ax.annotate(features[idx].replace('delta_', 'Δ').replace('rho_', 'ρ'),
                           xy=(effect_sizes[idx], neg_log_q[idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    ax.set_xlabel("Effect Size (Cohen's d or r)")
    ax.set_ylabel('-log₁₀(q-value)')
    ax.set_title(f'{plane} Plane: Volcano Plot\n(Effect Size vs Statistical Significance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / f'cluster_volcano_{plane}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'cluster_volcano_{plane}.png', bbox_inches='tight')
    plt.close()

    print(f"  ✓ Volcano plot for {plane}")

# ============================================================================
# 3. SANKEY DIAGRAM
# ============================================================================

print("\n[3/4] Generating Sankey diagram...")

if PLOTLY_AVAILABLE:
    # Get one row per patient per plane
    df_flow = df_plane_memberships.drop_duplicates(['Patient_ID', 'Plane'])[['Patient_ID', 'Plane', 'Cluster']]
    df_flow = df_flow.pivot(index='Patient_ID', columns='Plane', values='Cluster')
    df_flow = df_flow.merge(df_consensus, on='Patient_ID', how='inner')
    df_flow = df_flow.dropna()

    # Build node labels
    nodes = []
    node_map = {}
    node_idx = 0

    for plane in ['XY', 'XZ', 'YZ', '3D', 'Consensus']:
        col = plane if plane != 'Consensus' else 'Consensus_Cluster'

        unique_clusters = sorted(df_flow[col].unique())
        for cluster in unique_clusters:
            label = f"{plane}_C{int(cluster)}"
            nodes.append(label)
            node_map[(plane, cluster)] = node_idx
            node_idx += 1

    # Build links
    links = []
    flow_sequence = [('XY', 'XZ'), ('XZ', 'YZ'), ('YZ', '3D'), ('3D', 'Consensus')]

    for source_plane, target_plane in flow_sequence:
        source_col = source_plane
        target_col = target_plane if target_plane != 'Consensus' else 'Consensus_Cluster'

        flow_counts = df_flow.groupby([source_col, target_col]).size().reset_index(name='count')

        for _, row in flow_counts.iterrows():
            source_cluster = row[source_col]
            target_cluster = row[target_col]
            count = row['count']

            source_idx = node_map[(source_plane, source_cluster)]
            target_idx = node_map[(target_plane, target_cluster)]

            links.append({'source': source_idx, 'target': target_idx, 'value': count})

    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color='black', width=0.5), label=nodes),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links]
        )
    )])

    fig.update_layout(
        title_text="Cluster Flow Across Planes: XY → XZ → YZ → 3D → Consensus",
        font_size=12,
        width=1400,
        height=800
    )

    fig.write_html(str(FIG_DIR / 'cluster_sankey_planes_to_consensus.html'))
    try:
        fig.write_image(str(FIG_DIR / 'cluster_sankey_planes_to_consensus.pdf'))
        fig.write_image(str(FIG_DIR / 'cluster_sankey_planes_to_consensus.png'))
        print("  ✓ Sankey diagram (HTML, PDF, PNG)")
    except:
        print("  ✓ Sankey diagram (HTML only - install kaleido for PDF/PNG)")
else:
    print("  - Skipped (plotly not installed)")

# ============================================================================
# 4. SUMMARY HEATMAP (Centroid Z-scores from test results)
# ============================================================================

print("\n[4/4] Generating centroid heatmap summary...")

for plane in planes:
    df_plane_tests = df_tests[df_tests['Group'].str.startswith(f'{plane}_', na=False)]

    if len(df_plane_tests) == 0:
        continue

    # Pivot to create cluster × feature matrix
    features = sorted(df_plane_tests['Feature'].unique())
    clusters = sorted(df_plane_tests['Group'].unique(), key=str)

    matrix = np.zeros((len(clusters), len(features)))

    for i, cluster in enumerate(clusters):
        for j, feature in enumerate(features):
            data = df_plane_tests[
                (df_plane_tests['Group'] == cluster) &
                (df_plane_tests['Feature'] == feature)
            ]
            if len(data) > 0:
                matrix[i, j] = data.iloc[0]['Mean']

    # Z-score normalize features (columns)
    from scipy.stats import zscore
    matrix_zscore = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if np.std(col) > 0:
            matrix_zscore[:, j] = zscore(col)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(matrix_zscore, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                xticklabels=[f.replace('delta_', 'Δ').replace('rho_', 'ρ') for f in features],
                yticklabels=[f'Cluster {c}' for c in clusters],
                cbar_kws={'label': 'Z-score'},
                ax=ax)

    ax.set_title(f'{plane} Plane: Cluster Centroid Heatmap (Z-scored)')
    ax.set_xlabel('Features')
    ax.set_ylabel('Clusters')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig.savefig(FIG_DIR / f'cluster_centroid_heatmap_{plane}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'cluster_centroid_heatmap_{plane}.png', bbox_inches='tight')
    plt.close()

    print(f"  ✓ Centroid heatmap for {plane}")

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80)
print("✓ FIGURE GENERATION COMPLETE")
print("=" * 80)

print("\nGenerated:")
print(f"  - Per-plane radar plots: 4 figures")
print(f"  - Per-plane volcano plots: 4 figures")
print(f"  - Sankey diagram: 1 figure")
print(f"  - Per-plane centroid heatmaps: 4 figures")
print(f"\nAll saved to: {FIG_DIR}")
