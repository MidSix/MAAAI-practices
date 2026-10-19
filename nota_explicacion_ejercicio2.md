# Ejercicio 2 (SVM incrementales): qué hay implementado y por qué

Archivo: `practica1/ejercicios1-3/firmas_modificado.jl`, sección **Ejercicio 2**.

## 1. Resumen

- **El código que ya había del Ejercicio 2 era correcto y no se ha cambiado.** Pasa todos los tests de `autoevaluacion.jl` y además unas comprobaciones extra de casos que la autoevaluación no cubre (sección 4).
- Se han añadido **36 notas NoteFold** que explican cada función y cada línea de su cuerpo. Se insertaron con un script que solo añade líneas de comentario, y se comprobó que el código sin notas es **idéntico byte a byte** al original.
- Las firmas **no se han tocado**.
- También se ha cambiado la skill `code-notes` para que en Julia no use títulos con `##` (sección 6).

## 2. Qué pide el enunciado y cómo lo cumple cada función

| Función | Qué hace | Restricción del enunciado | Cómo se cumple |
|---|---|---|---|
| `batchInputs` | Devuelve `batch[1]` (matriz de entradas) | Sin bucles | Un simple indexado |
| `batchTargets` | Devuelve `batch[2]` (vector de salidas) | Sin bucles | Un simple indexado |
| `batchLength` | Número de instancias = filas de las entradas | Sin bucles | `size(batchInputs(batch), 1)` |
| `selectInstances` | Sub-lote con las filas indicadas | Sin bucles | Indexado de entradas y *targets* con los mismos índices |
| `joinBatches` | Une dos lotes, `batch1` arriba | Sin bucles, usar `vcat` | `vcat` de entradas y `vcat` de *targets* |
| `divideBatches` | Parte el dataset en lotes de `batchSize` | Sin bucles; se sugieren `shuffle`, `partition` y *comprehension* | `[selectInstances(dataset, idx) for idx in partition(indices, batchSize)]` |
| `trainSVM(::Batch, …)` | Entrena con SVs previos + lote y devuelve `(modelo, SVs, (idxViejos, idxNuevos))` | Sin bucles; código de `SVMClassifier`/`machine` dado | Sigue al pie de la letra el código del enunciado |
| `trainSVM(::AbstractArray{<:Batch,1}, …)` | SVM incremental sobre una lista de lotes | Un único bucle | Un `for` que encadena llamadas al método anterior |

### Por qué cada decisión

**`batchInputs` / `batchTargets` devuelven el objeto original, no una copia.** La autoevaluación compara con `===` (identidad, no solo igualdad): `batchInputs(dataset) === inputs`. Si se copiara, fallaría.

**`selectInstances` envuelve un índice entero en un vector (`[indices]`).** El enunciado no lo pide, pero sin esto `selectInstances(batch, 5)` devolvería `(Vector, escalar)`, porque en Julia `M[5, :]` quita una dimensión. Eso ya **no sería un `Batch`** y rompería cualquier función que lo recibiera. Con `[5]` se obtiene `(Matrix 1×n, Vector)`. Los rangos, los vectores de índices y las máscaras booleanas se usan tal cual.

**`divideBatches` baraja índices, no datos.** Se baraja el vector `1:N` y luego `selectInstances` copia las filas una sola vez por lote. Así entradas y *targets* se mueven con la **misma** permutación y no se desemparejan, y el resultado depende del generador global de `Random`, que es lo que la autoevaluación fija con `seed!(1)`.

**`trainSVM(::Batch, …)`:**
1. `N = batchLength(supportVectors)`: cuántos vectores de soporte previos hay.
2. `trainingData = joinBatches(supportVectors, dataset)`: lo antiguo arriba y lo nuevo abajo, sin desordenar, como exige el criterio de la práctica.
3. El modelo y la `machine` se crean con el código exacto del enunciado. La única diferencia es que se escriben con prefijo (`MLJ.machine`, `MLJ.categorical`), que es la misma función. Se entrena con `trainingData`, **no** con `dataset`: esa es la idea del SVM incremental.
4. `sort(mach.fitresult[1].SVs.indices)`: está comprobado que LIBSVM devuelve los índices **agrupados por clase** y no ordenados. Se ordenan para conservar el orden cronológico, y porque la autoevaluación comprueba `issorted`.
5. Índices `≤ N` → vienen de `supportVectors` y el índice coincide. Índices `> N` → vienen de `dataset` y se les resta `N`.
6. Los nuevos SVs se montan con `joinBatches(selectInstances(supportVectors, viejos), selectInstances(dataset, nuevos))`, en ese orden.

**`trainSVM(::AbstractArray{<:Batch,1}, …)`:**
- Crea un lote de SVs **vacío** con el mismo tipo de elemento y número de columnas que el primer lote. Hace falta para que `vcat` pueda unirlo al primer lote.
- Declara `local model` **antes** del bucle. En Julia una variable creada por primera vez dentro de un `for` no existe fuera de él (lo avisa el propio enunciado).
- En cada iteración: `model, svs, _ = trainSVM(batch, …; supportVectors=svs)`. El `_` descarta los índices, que aquí no hacen falta.
- Devuelve el último `model`.

**Despacho múltiple.** Las dos `trainSVM` son dos *métodos* de la misma función. Julia elige uno u otro según el tipo del primer argumento: un `Batch` o un vector de `Batch`.

## 3. Qué se ha corregido

**En el código, nada.** Antes de comentar se ejecutó la autoevaluación sobre el código existente y pasó todo. Después se revisó cada función contra el enunciado y los criterios de corrección:

- Firmas exactas y una sola definición por firma: ✅
- Bucles: ninguno, salvo el único `for` permitido en la versión por lotes: ✅
- `divideBatches` usa `shuffle`, `partition` y una *comprehension*, como sugiere el enunciado: ✅
- `trainSVM` usa el código de `SVMClassifier`/`machine` y la expresión `sort(mach.fitresult[1].SVs.indices)` del enunciado: ✅
- No se llama a ninguna función de librería que "resuelva" el ejercicio por nosotros: ✅

## 4. Verificación

**Autoevaluación** (`julia autoevaluacion.jl`), antes y después de añadir las notas:

```
>>> ¡TODOS LOS TESTS DEL EJERCICIO 1 HAN PASADO CON ÉXITO! <<<
>>> ¡TODOS LOS TESTS DEL EJERCICIO 2 HAN PASADO CON ÉXITO! <<<
```

**Comprobaciones extra** (script aparte, no forma parte de la entrega):

| Caso | Resultado |
|---|---|
| `selectInstances(ds, 5)` sigue siendo un `Batch` (matriz 1×60) | ✅ |
| `divideBatches(ds, 50; shuffleRows=true)` da lotes de 50, 50, 50, 50 y 8, sin perder ni repetir instancias | ✅ |
| `trainSVM` con un dataset `Float32` y SVs por defecto: SVs `Float32` e índices viejos vacíos | ✅ |
| La versión por lotes predice **exactamente igual** que encadenar a mano el método de un lote | ✅ |

**Versión de Julia.** Los tests se ejecutaron con **Julia 1.12.4**, la única instalada. El profesor usó la 1.11.2. Con la 1.12.4 los tests aleatorios (`seed!`) dan los mismos valores, pero si queréis comprobarlo con la versión del profesor: `juliaup add 1.11.2` (ver `instalar_julia/Instalar Julia.md`).

## 5. Las notas NoteFold

| Nota | Color | Contenido destacado |
|---|---|---|
| `using MLJ…` + `SVMClassifier` | amarilla | Qué aporta cada paquete; `@load` devuelve un **tipo** |
| `predict` (del profesor) | azul | Conversión de `CategoricalVector` a `Bool` paso a paso |
| `using Base.Iterators / StatsBase / Random` | amarilla | Para qué sirve cada import |
| `Batch` | morada | Alias de tipo, invariantes, covarianza de tuplas |
| Cada función del ejercicio | azul | Descripción, API pública / quién la usa, tablas de entradas y salida, errores |
| Cada línea de cada cuerpo | amarilla | Qué hace y por qué |

Llevan **diagrama Mermaid** `divideBatches` (el flujo de `partition` → `selectInstances`) y las dos `trainSVM` (el flujo de índices y el esquema incremental del enunciado). Llevan **LaTeX** la tabla de kernels (fórmulas de `linear`, `rbf`, `poly` y `sigmoid`) y el desplazamiento de índices $i - N$. Todas las notas se validaron con `check_notes.py`: estructura, ids únicos, títulos de 60 caracteres como máximo, y LaTeX y Mermaid que se renderizan.

## 6. Cambio en la skill `code-notes`

En Julia, una línea `# ## título` coincide con el patrón `^\s?#\s#+\s` de `julia.cellDelimiters` de la extensión de VS Code, así que **parte el archivo en celdas de código independientes**. Por eso:

- `SKILL.md`: se añade una excepción para Julia. El título va **sin `##`** (`nombre: qué hace`), y dentro de la nota no se usan encabezados Markdown (se usa **negrita**). El ejemplo de Julia de la skill también se ha corregido.
- `scripts/check_notes.py`: en archivos `.jl` exige el título sin `#` y marca como error cualquier línea de la nota que sea delimitador de celda. En el resto de lenguajes sigue exigiendo `## título`.

Las notas nuevas del Ejercicio 2 ya siguen esta regla.

## 7. Pendiente

- **Notas del Ejercicio 1.** Con la nueva regla, el validador marca 12 errores en las notas del Ejercicio 1: varias notas amarillas usan `# ## …`, que son delimitadores de celda. Hay además un título de más de 60 caracteres y un título con LaTeX. No se han tocado.
- **Experimento del enunciado (opcional, no se entrega).** Queda por comparar un SVM normal con el ISVM usando distintos tamaños de lote en datasets grandes (`agaricus-lepiota`, `twonorm`, `ring`, `hypothyroid`), para responder qué tamaño de lote da una precisión más cercana al SVM clásico. Se haría en un script aparte que haga `include` de `firmas_modificado.jl`.
