---
id: 20260911113623
title: Ejercicio 1 - Carga de datasets
description: Resumen del ejercicio 1 de la práctica 1 de MAAAI (carga y preprocesado de datasets PMLB y Electricity/Elec2 en Julia sin bucles) y plan de acción para implementar las 5 funciones requeridas.
tags:
  - MAAAI
  - practica1
  - ejercicio1
---

# Ejercicio 1 — Carga de datasets

Nota de resumen y plan de acción para el ejercicio 1 de la práctica 1, basada en [[1 - Carga de datasets]] (enunciado), [[firmas.jl]] y [[autoevaluacion.jl]].

## 1. Contexto: qué hay en `datasets/`

El `.zip` (`datasets.zip`) contiene 63 elementos: la carpeta `datasets/` más 62 ficheros de datos.

- **60 datasets de PMLB** (*Penn Machine Learning Benchmarks*), en formato `.tsv`:
  - 30 de **clasificación binaria** (target en $\{0,1\}$): `sonar`, `pima`, `wdbc`, etc.
  - 30 de **regresión** (target continuo): `cpu`, `pollution`, etc.
- **Dataset de Stream Learning — Electricity / Elec2** (mercado eléctrico de Nueva Gales del Sur, Australia; 45.312 instancias temporales), repartido en 2 ficheros:
  - `elec2_data.dat`: matriz de atributos $45312 \times 8$.
  - `elec2_label.dat`: etiquetas $45312 \times 1$.

Total: $30 + 30 + 2 = 62$ ficheros (+1 entrada de directorio en el zip = 63).

Se incluyen tantos datasets porque en las prácticas 1-6 se implementan algoritmos avanzados (SVMs, Stream Learning incremental, árboles de regresión/Model Trees, KNN híbrido...) y se evalúa que los modelos funcionen de forma reproducible sobre problemas y dimensiones variadas, no solo sobre un "juguete" como Iris.

### Detalle del dataset Electricity (Elec2)

Variables independientes originales (8 columnas en `elec2_data.dat`):

| # | Nombre | Descripción |
| --- | --- | --- |
| 1 | date | Fecha y hora del registro |
| 2 | day | Día de la semana (1-7) — **cíclico** |
| 3 | period | Periodo del día (1-48, intervalos de media hora) |
| 4 | nswprice | Precio electricidad en Nueva Gales del Sur |
| 5 | nswdemand | Demanda electricidad en Nueva Gales del Sur |
| 6 | vicprice | Precio electricidad en Victoria |
| 7 | vicdemand | Demanda electricidad en Victoria |
| 8 | transfer | Electricidad transferida entre NSW y Victoria |

El artículo original recomienda **eliminar los atributos 1 (`date`) y 4 (`nswprice`)**. El resto de atributos, salvo `day`, ya están normalizados. `day` es cíclico y necesita codificación seno/coseno. La variable objetivo (`class`) indica si el precio sube (UP) o baja (DOWN) respecto a la hora anterior → booleano.

## 2. Regla de oro del ejercicio

> [!CAUTION]
> En **ninguna** de las 5 funciones de este ejercicio se permite usar bucles (`for`, `while`). Todo se resuelve con vectorización (broadcasting `.`), funciones de orden superior (`filter`, `map`), indexación de matrices/vectores y utilidades de Julia (`readdir`, `readdlm`, `findall`, `unique`, `setdiff`, `sortperm`, etc.).

## 3. Las 5 funciones a implementar

Plantilla en [[firmas.jl]].

### 3.1 `fileNamesFolder(folderName::String, extension::String)`

- **Entrada**: ruta de carpeta, extensión sin punto (ej. `"tsv"`).
- **Salida**: `Vector{String}` con los nombres de archivo encontrados, **sin extensión**.
- **Pista del enunciado**:
  ```julia
  extension = uppercase(extension)
  fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName))
  ```
  Después solo falta quitar la extensión de cada nombre (sin bucles → usar broadcasting sobre el vector filtrado, p. ej. cortando la cadena con índices o `replace`/`chop`).
- **Assert de referencia** (`autoevaluacion.jl`): debe encontrar exactamente los 60 `.tsv` (lista completa de 60 nombres, clasificación + regresión mezclados).

### 3.2 `loadDataset(datasetName::String, datasetFolder::String; datasetType::DataType=Float64)`

- **Entrada**: nombre del dataset (sin `.tsv`), carpeta, tipo de dato opcional para las entradas.
- **Pasos**:
  1. Construir la ruta añadiendo la extensión `.tsv` al nombre (`joinpath(datasetFolder, datasetName * ".tsv")`).
  2. Si el archivo no existe (`isfile`), devolver `nothing`.
  3. Leer con `readdlm(ruta, '\t')`. La primera fila son las cabeceras.
  4. Localizar la columna cuyo nombre sea exactamente `"target"` (`findall` sobre la fila de cabeceras).
  5. A partir de la fila 2 en adelante:
     - **`inputs`**: todas las columnas salvo `"target"`, convertidas a `datasetType`.
     - **`targets`**: la columna `"target"`. Si sus valores únicos ⊆ $\{0,1\}$ → vector `Bool` (clasificación). Si no → convertir a `datasetType` (regresión).
- **Salida**: tupla `(inputs, targets)`, o `nothing` si no existe el archivo.
- **Funciones útiles sugeridas**: `joinpath`, `abspath`, `readdlm`, `findall`, `unique`, `setdiff`.
- **Assert de referencia**: `loadDataset("sonar", ...; datasetType=Float32)` → `inputs` de tamaño $(208,60)$, `targets` `Bool` de 208 elementos.

### 3.3 `datasetsDescription(datasetFolder::String, classification::Bool)`

- **Entrada**: carpeta de datasets, booleano `classification`.
- **Pasos**:
  1. Obtener todos los `.tsv` con `fileNamesFolder`.
  2. Cargarlos todos con `loadDataset`.
  3. Filtrar: si `classification == true`, quedarse con los que tienen `targets::Bool` (clasificación); si `false`, con el resto (regresión).
  4. Construir matriz de 3 columnas: `[nombre_dataset, num_filas, num_columnas]`.
  5. Ordenar ascendentemente por tamaño total ($\text{filas} \times \text{columnas}$).
- **Salida**: matriz ordenada.
- **Assert de referencia**: suma de las columnas 2:end para clasificación = 11.415; para regresión = 8.866.

### 3.4 `cyclicalEncoding(data::AbstractArray{<:Real,1})`

- **Entrada**: vector 1D de una variable periódica/cíclica (ej. día de la semana 1..7).
- **Fundamento teórico** (enunciado, págs. 2-3):
  - Se normaliza entre mínimo y máximo, pasando a un intervalo no $[0,1]$ sino $[0, 2\pi]$.
  - Para valores **continuos**, la fórmula simple $rad = 2\pi \frac{x-\min}{\max-\min}$ es correcta porque el "último" valor equivale de verdad al primero (ej. 23:59:59.99 ≈ 00:00:00).
  - Para valores **discretos** (como los días de la semana, donde el día 7 ≠ día 1) esta fórmula fallaría, porque el valor máximo caería en el ángulo $2\pi \equiv 0$, coincidiendo con el primer valor. Solución: ampliar el intervalo con la magnitud de cambio entre valores contiguos, $m$:
    $$rad = 2\pi \frac{x - \min}{\max - \min + m}$$
  - $m$ ya viene calculado por la función dada `intervalDiscreteVector(data)` (devuelve el salto entre valores si son discretos, o `0.` si son continuos) — así una sola fórmula sirve para ambos casos.
  - Una vez obtenido `rad`, calcular `sin.(rad)` y `cos.(rad)`.
- **Salida**: tupla `(vector_senos, vector_cosenos)`.
- **Funciones útiles sugeridas**: `maximum`, `minimum`, `extrema`, `intervalDiscreteVector` (dada), `sin`, `cos`.
- **Assert de referencia**: con `[1,2,3,2,1,0,-1,-2,-3]` se comprueban senos y cosenos exactos (ver `autoevaluacion.jl` líneas 64-66).

### 3.5 `loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float64)`

- **Entrada**: carpeta donde están `elec2_data.dat` y `elec2_label.dat`.
- **Pasos**:
  1. Cargar `elec2_label.dat` con `readdlm`, convertir a `Bool` y aplanar a vector con `vec`.
  2. Cargar `elec2_data.dat` ($45312 \times 8$) con `readdlm`.
  3. Eliminar las columnas 1 (`date`) y 4 (`nswprice`).
  4. Sobre la columna resultante que contiene el día de la semana (ahora la nueva columna 1), aplicar `cyclicalEncoding`.
  5. Sustituir esa columna de día por las dos columnas generadas (seno y coseno), colocadas como las **primeras** columnas de la matriz de salida.
  6. Convertir la matriz final a `datasetType`.
- **Orden final de las 7 columnas**: `senos, cosenos, period, nswdemand, vicprice, vicdemand, transfer` (confirmado literalmente en el enunciado, pág. 7).
- **Salida**: tupla `(inputs, targets)`, con `inputs::Matrix{datasetType}` de tamaño $(45312, 7)$ y `targets::Vector{Bool}` de 45312 elementos.
- **Funciones útiles sugeridas**: `joinpath`, `abspath`, `readdlm`, `vec`, `setdiff`, `cyclicalEncoding`.
- **Nota del enunciado**: menciona `using FileIO, Images, JLD2` como imports generales de la asignatura, pero para esta función en concreto basta con `DelimitedFiles` (`readdlm`).
- **Assert de referencia**: `size(inputs) == (45312,7)`, `length(targets) == 45312`, `eltype(inputs) == Float64`, `eltype(targets) == Bool`.

## 4. Cómo autoevaluar

Los asserts exactos están en [[autoevaluacion.jl]] (líneas 52-72):

- `fileNamesFolder` debe encontrar los 60 datasets `.tsv` exactos (lista completa en el assert de la línea 53).
- `loadDataset("sonar", ...)` → `inputs` $(208,60)$, `targets::Bool` de 208 elementos, `eltype(inputs)==Float32` si se pide así.
- `datasetsDescription(..., true)` → suma columnas 2:end = 11.415; `datasetsDescription(..., false)` → 8.866.
- `cyclicalEncoding` con el vector de ejemplo → senos/cosenos exactos (tolerancia `rtol=1e-4`).
- `loadStreamLearningDataset` → `inputs` $(45312,7)$ `Float64`, `targets` `Bool` de 45312 elementos.

> Recordatorio del propio archivo: pasar estos asserts no garantiza corrección total (no cubren toda la casuística), pero sirven de apoyo antes de la entrega oficial (ver [[Criterios de correccion]]).

## 5. Plan de acción

- [ ] **Paso 1** — Implementar `fileNamesFolder`. Es la base de `datasetsDescription`. Validar con el assert de los 60 nombres.
- [ ] **Paso 2** — Implementar `loadDataset`. Depende de localizar bien la columna `"target"` y decidir `Bool` vs `datasetType`. Validar con el caso `sonar`.
- [ ] **Paso 3** — Implementar `datasetsDescription`, reutilizando 1 y 2. Validar sumas 11.415 / 8.866.
- [ ] **Paso 4** — Implementar `cyclicalEncoding` usando `intervalDiscreteVector` (ya dada). Validar con el vector de ejemplo del autoevaluación.
- [ ] **Paso 5** — Implementar `loadStreamLearningDataset`, reutilizando `cyclicalEncoding`. Validar dimensiones $(45312,7)$ y tipos.
- [ ] **Paso 6** — Ejecutar `autoevaluacion.jl` completo (sección Ejercicio 1) y confirmar que todos los asserts pasan sin errores.
- [ ] **Paso 7** — Revisar que ninguna de las 5 funciones usa `for`/`while` (repasar el código antes de entregar).

## Enlaces relacionados

- [[1 - Carga de datasets]] — enunciado en PDF
- [[firmas.jl]] — plantilla de funciones a completar
- [[autoevaluacion.jl]] — asserts de validación
- [[Criterios de correccion]] — criterios de evaluación de la práctica
