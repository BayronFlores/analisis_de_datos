Identificación de Perfiles de Riesgo Cardiovascular mediante K-Means

**Curso:** INFO1184 Inteligencia de Negocios  
**Dataset:** Heart Disease Cleveland (UCI)

---

## Descripción

Script de Python que aplica el algoritmo de clustering **K-Means** sobre el dataset Heart Disease Cleveland para identificar perfiles de riesgo cardiovascular de forma no supervisada. El análisis incluye preprocesamiento, selección del número óptimo de clusters, caracterización de perfiles y visualización mediante PCA.

---

## Requisitos

- Python 3.8 o superior
- Las siguientes librerías:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Instalación rápida:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Estructura del proyecto

```
.
├── tarea5_kmeans.py              # Script principal
├── heart_cleveland_upload.csv    # Dataset (debe estar en la misma carpeta)
└── resultados/                   # Carpeta generada automáticamente con las figuras
    ├── fig1_distribuciones.png
    ├── fig2_correlacion.png
    ├── fig3_seleccion_k.png
    ├── fig4_centroides.png
    ├── fig5_condition_por_cluster.png
    └── fig6_pca_clusters.png
```

---

## Uso

Colocar el archivo `heart_cleveland_upload.csv` en la misma carpeta que el script y ejecutar:

```bash
python tarea5_kmeans.py
```

La carpeta `resultados/` se crea automáticamente si no existe.

---

## Variables utilizadas

El script excluye `condition` (etiqueta de diagnóstico), `sex`, `fbs` y `restecg` del clustering, y trabaja con las siguientes 10 variables clínicas:

| Variable   | Descripción                                      |
|------------|--------------------------------------------------|
| `age`      | Edad del paciente (años)                         |
| `trestbps` | Presión arterial en reposo (mm Hg)               |
| `chol`     | Colesterol sérico (mg/dl)                        |
| `thalach`  | Frecuencia cardíaca máxima alcanzada             |
| `oldpeak`  | Depresión del ST inducida por ejercicio          |
| `cp`       | Tipo de dolor en el pecho (0–3)                  |
| `exang`    | Angina inducida por ejercicio (0/1)              |
| `slope`    | Pendiente del segmento ST en ejercicio (0–2)     |
| `ca`       | Número de vasos principales coloreados (0–3)     |
| `thal`     | Tipo de talasemia (1 = normal, 2 = defecto fijo, 3 = defecto reversible) |

---

## Flujo del script

1. **Carga de datos** — lectura del CSV y estadísticas descriptivas.
2. **Preprocesamiento** — estandarización con `StandardScaler` (media 0, desv. estándar 1).
3. **Análisis exploratorio** — histogramas por condición y mapa de correlación.
4. **Selección de k** — Método del Codo e índice Silhouette para k = 2 … 10.
5. **Modelo final** — K-Means con el k óptimo encontrado en el paso anterior.
6. **Análisis de resultados** — perfil medio por cluster y tabla cruzada cluster × diagnóstico.
7. **Visualización PCA** — proyección 2D de clusters y diagnóstico real.

---

## Figuras generadas

| Archivo                          | Contenido                                               |
|----------------------------------|---------------------------------------------------------|
| `fig1_distribuciones.png`        | Histogramas de variables clave separados por condición  |
| `fig2_correlacion.png`           | Mapa de calor de correlaciones entre variables          |
| `fig3_seleccion_k.png`           | Método del Codo y Silhouette Score para k = 2 … 10      |
| `fig4_centroides.png`            | Heatmap del perfil de centroides por cluster            |
| `fig5_condition_por_cluster.png` | Barras apiladas: proporción sano/enfermo por cluster    |
| `fig6_pca_clusters.png`          | Scatter PCA: clusters K-Means vs diagnóstico real       |

---

## Parámetros configurables

Todos los parámetros principales están definidos al inicio del script:

| Parámetro      | Valor por defecto               | Descripción                        |
|----------------|---------------------------------|------------------------------------|
| `RANDOM_STATE` | `42`                            | Semilla de aleatoriedad            |
| `DATASET_PATH` | `heart_cleveland_upload.csv`    | Ruta al dataset                    |
| `OUTPUT_DIR`   | `resultados`                    | Carpeta de salida para las figuras |
| `K_RANGE`      | `range(2, 11)`                  | Rango de k evaluados               |
