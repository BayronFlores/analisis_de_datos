import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# 0. CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})
RANDOM_STATE = 42
DATASET_PATH = 'heart_cleveland_upload.csv'

# Crear carpeta de resultados si no existe
OUTPUT_DIR = 'resultados'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fig_path(filename):
    """Devuelve la ruta completa dentro de la carpeta resultados/."""
    return os.path.join(OUTPUT_DIR, filename)

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
print("=" * 60)
print("FASE 2 - COMPRENSIÓN DE LOS DATOS")
print("=" * 60)
print(f"\nDimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nValores nulos por columna:\n{df.isnull().sum()}")
print(f"\nDistribución de la variable 'condition':")
print(df['condition'].value_counts().to_string())
print(f"\nEstadísticas descriptivas:")
print(df.describe().round(2).to_string())


# variables excluidas (condition, sex, fbs, restecg)
FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak','cp', 'exang', 'slope', 'ca', 'thal']

X = df[FEATURES].copy()
y_real = df['condition'].copy()   # guardado solo para evaluación posterior

# -----------------------------------------------------------------------------
# 3. PREPROCESAMIENTO - Estandarización (StandardScaler)
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 3 - PREPROCESAMIENTO DE DATOS")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

print("\nMedia post-estandarización (deben ser ~0):")
print(X_scaled_df.mean().round(4).to_string())
print("\nDesviación estándar post-estandarización (deben ser ~1):")
print(X_scaled_df.std().round(4).to_string())

# -----------------------------------------------------------------------------
# 4. ANÁLISIS EXPLORATORIO
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 3 - ANÁLISIS EXPLORATORIO")
print("=" * 60)

# 4.1 Distribución de variables numéricas clave por condición
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
vars_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
labels = {0: 'Sin enfermedad', 1: 'Con enfermedad'}
colores = {0: '#4C9BE8', 1: '#E8654C'}

for ax, var in zip(axes.flatten(), vars_plot):
    for cond in [0, 1]:
        subset = df[df['condition'] == cond][var]
        ax.hist(subset, bins=20, alpha=0.6, label=labels[cond],
                color=colores[cond], edgecolor='white')
    ax.set_title(f'Distribución de {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Frecuencia')
    ax.legend(fontsize=8)

plt.suptitle('Fig. 1. Distribución de variables clínicas por condición cardíaca',
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(fig_path('fig1_distribuciones.png'), bbox_inches='tight')
plt.close()
print("  -> Guardado: resultados/fig1_distribuciones.png")

# 4.2 Mapa de correlación
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[FEATURES + ['condition']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Fig. 2. Mapa de correlación de variables clínicas', fontsize=11)
plt.tight_layout()
plt.savefig(fig_path('fig2_correlacion.png'), bbox_inches='tight')
plt.close()
print("  -> Guardado: resultados/fig2_correlacion.png")

# Correlaciones con condition
print("\nCorrelación de cada variable con 'condition':")
print(corr['condition'].drop('condition').sort_values(ascending=False).round(3).to_string())


# -----------------------------------------------------------------------------
# 5. SELECCIÓN DE K - Método del Codo + Silhouette Score
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 4 - MODELADO: SELECCIÓN DE K")
print("=" * 60)

K_RANGE = range(2, 11)
inercias = []
silhouettes = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels_k))
    print(f"  k={k} | Inercia: {km.inertia_:.2f} | Silhouette: {silhouette_score(X_scaled, labels_k):.4f}")

# Gráfico método del codo + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(list(K_RANGE), inercias, 'o-', color='#4C9BE8', linewidth=2, markersize=7)
ax1.set_title('Método del Codo (Elbow Method)')
ax1.set_xlabel('Número de clusters k')
ax1.set_ylabel('Inercia (J)')
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.plot(list(K_RANGE), silhouettes, 's-', color='#E8654C', linewidth=2, markersize=7)
ax2.set_title('Silhouette Score por k')
ax2.set_xlabel('Número de clusters k')
ax2.set_ylabel('Silhouette Score')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax2.grid(True, linestyle='--', alpha=0.5)

k_optimo = list(K_RANGE)[silhouettes.index(max(silhouettes))]
ax2.axvline(x=k_optimo, color='green', linestyle='--',
            label=f'k óptimo = {k_optimo}')
ax2.legend()

plt.suptitle('Fig. 3. Selección del número óptimo de clusters', fontsize=11)
plt.tight_layout()
plt.savefig(fig_path('fig3_seleccion_k.png'), bbox_inches='tight')
plt.close()
print(f"\n  -> k óptimo según Silhouette Score: {k_optimo}")
print("  -> Guardado: resultados/fig3_seleccion_k.png")

# -----------------------------------------------------------------------------
# 6. MODELO FINAL K-MEANS
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"FASE 4 - MODELADO: K-MEANS CON k={k_optimo}")
print("=" * 60)

km_final = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

sil_final = silhouette_score(X_scaled, df['cluster'])
print(f"\nSilhouette Score final (k={k_optimo}): {sil_final:.4f}")
print(f"\nDistribución de pacientes por cluster:")
print(df['cluster'].value_counts().sort_index().to_string())

# -----------------------------------------------------------------------------
# 7. ANÁLISIS DE RESULTADOS
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 4 - RESULTADOS DEL CLUSTERING")
print("=" * 60)

# 7.1 Perfil medio de cada cluster
perfil = df.groupby('cluster')[FEATURES + ['condition']].mean().round(3)
print("\nPerfil medio por cluster:")
print(perfil.to_string())

# 7.2 Relación clusters vs condition (Pregunta 3)
print("\nDistribución de 'condition' por cluster (Pregunta 3):")
cross = pd.crosstab(df['cluster'], df['condition'],
                    rownames=['Cluster'], colnames=['Condition'])
cross.columns = ['Sano (0)', 'Enfermo (1)']
cross['Total'] = cross.sum(axis=1)
cross['% Enfermo'] = (cross['Enfermo (1)'] / cross['Total'] * 100).round(1)
print(cross.to_string())

# 7.3 Visualización perfil de clusters (heatmap de centroides)
centroides_df = pd.DataFrame(
    scaler.inverse_transform(km_final.cluster_centers_),
    columns=FEATURES
)
centroides_norm = (centroides_df - centroides_df.min()) / \
                  (centroides_df.max() - centroides_df.min())

fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(centroides_norm.T, annot=centroides_df.T.round(2),
            fmt='.2f', cmap='YlOrRd', linewidths=0.5, ax=ax,
            xticklabels=[f'Cluster {i}' for i in range(k_optimo)])
ax.set_title(f'Fig. 4. Perfil de centroides por cluster (k={k_optimo})', fontsize=11)
ax.set_ylabel('Variable clínica')
plt.tight_layout()
plt.savefig(fig_path('fig4_centroides.png'), bbox_inches='tight')
plt.close()
print("\n  -> Guardado: resultados/fig4_centroides.png")

# 7.4 Distribución de condition por cluster (barras apiladas)
cross_pct = pd.crosstab(df['cluster'], df['condition'], normalize='index') * 100
fig, ax = plt.subplots(figsize=(8, 5))
cross_pct.plot(kind='bar', stacked=True, ax=ax,
               color=['#4C9BE8', '#E8654C'], edgecolor='white')
ax.set_title(f'Fig. 5. Proporción de diagnóstico por cluster (k={k_optimo})', fontsize=11)
ax.set_xlabel('Cluster')
ax.set_ylabel('Porcentaje (%)')
ax.legend(['Sano (0)', 'Enfermo (1)'], title='Condición')
ax.set_xticklabels([f'Cluster {i}' for i in range(k_optimo)], rotation=0)
plt.tight_layout()
plt.savefig(fig_path('fig5_condition_por_cluster.png'), bbox_inches='tight')
plt.close()
print("  -> Guardado: resultados/fig5_condition_por_cluster.png")


# -----------------------------------------------------------------------------
# 8. VISUALIZACIÓN PCA - Distribución espacial de clusters
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FASE 4 - VISUALIZACIÓN PCA DE CLUSTERS")
print("=" * 60)

# Reducción a 2 componentes principales
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100
var_total = var_pc1 + var_pc2

print(f"\n  Varianza explicada PC1: {var_pc1:.1f}%")
print(f"  Varianza explicada PC2: {var_pc2:.1f}%")
print(f"  Varianza total explicada: {var_total:.1f}%")

# Cargas de cada variable en PC1 y PC2
cargas = pd.DataFrame(
    pca.components_.T,
    index=FEATURES,
    columns=['PC1', 'PC2']
).round(3)
print(f"\n  Cargas de variables en componentes principales:")
print(cargas.to_string())

colores_pca = ['#4C9BE8', '#E8654C']
markers_cond = {0: 'o', 1: '^'}
labels_cluster = [f'Cluster {i}' for i in range(k_optimo)]

# Fig 6: Clusters en espacio PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: coloreado por cluster
ax = axes[0]
for i in range(k_optimo):
    mask = df['cluster'] == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colores_pca[i], label=labels_cluster[i],
               alpha=0.7, edgecolors='white', linewidths=0.4, s=50)

# Centroides en espacio PCA
centroides_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroides_pca[:, 0], centroides_pca[:, 1],
           c='black', marker='X', s=150, zorder=5, label='Centroides')

ax.set_title(f'Clusters K-Means en espacio PCA\n(k={k_optimo})')
ax.set_xlabel(f'PC1 ({var_pc1:.1f}% varianza)')
ax.set_ylabel(f'PC2 ({var_pc2:.1f}% varianza)')
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.3)

# Panel derecho: coloreado por condition real
ax = axes[1]
for cond, marker in markers_cond.items():
    mask = df['condition'] == cond
    label = 'Sano (0)' if cond == 0 else 'Enfermo (1)'
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c='#4C9BE8' if cond == 0 else '#E8654C',
               marker=marker, label=label,
               alpha=0.7, edgecolors='white', linewidths=0.4, s=50)

ax.set_title('Diagnóstico real en espacio PCA\n(comparación con clusters)')
ax.set_xlabel(f'PC1 ({var_pc1:.1f}% varianza)')
ax.set_ylabel(f'PC2 ({var_pc2:.1f}% varianza)')
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.3)

plt.suptitle(f'Fig. 6. Visualización PCA: clusters vs diagnóstico real '
             f'(varianza total explicada: {var_total:.1f}%)', fontsize=11)
plt.tight_layout()
plt.savefig(fig_path('fig6_pca_clusters.png'), bbox_inches='tight')
plt.close()
print("  -> Guardado: resultados/fig6_pca_clusters.png")


# -----------------------------------------------------------------------------
# 9. RESUMEN FINAL
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"  Dataset            : Heart Disease Cleveland UCI")
print(f"  Registros          : {df.shape[0]}")
print(f"  Variables usadas   : {len(FEATURES)} ({', '.join(FEATURES)})")
print(f"  k óptimo           : {k_optimo}")
print(f"  Silhouette Score   : {sil_final:.4f}")
print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
for i in range(1, 7):
    print(f"  fig{i}_*.png")
print("=" * 60)