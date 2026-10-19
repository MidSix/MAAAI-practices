

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles

# @note-start id=11dczj color=blue
# fileNamesFolder: nombres de archivos con una extensión
# Devuelve los nombres (sin extensión) de los archivos de `folderName` cuya
# extensión coincide con `extension`, sin distinguir mayúsculas/minúsculas.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `folderName` | `String` | Ruta hasta la carpeta que contiene los datasets. |
# | `extension` | `String` | Extensión a buscar en esa ruta (`"tsv"`, `"dat"`, `"csv"`…). Acepta también `".tsv"`. |
#
# La extensión tiene que ser la de los datasets: si pasas `"mp4"` no
# encontrará nada, porque ningún dataset tiene ese formato. Aquí no hay
# ningún `csv` (*comma-separated values*): en machine learning es buena
# práctica usar `tsv` (*tab-separated values*), útil por ejemplo cuando hay
# números decimales escritos con coma.
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector{String}` | Nombres de archivo sin la extensión. Vacío si la carpeta no existe. |
#
# **Errores:** ninguno propio; si la carpeta no existe devuelve `String[]`.
# @note-body-end
function fileNamesFolder(folderName::String, extension::String)
# @note-end
    # @note-start id=1xdfwt color=yellow
    # Path inexistente: devolver lista vacía
    # La autoevaluación ya maneja el caso en que la ruta `folderName` con los
    # datasets no existe: tiene un `assert` que no se captura y pararía la
    # ejecución. Pero este módulo no siempre se ejecutará junto a la
    # autoevaluación, así que conviene poner aquí esta condición.
    #
    # El `!` es simplemente el *not* en Julia.
    # @note-body-end
    if !isdir(folderName)
        return String[]
    end
    # @note-end
    # @note-start id=55omu9 color=yellow
    # Normalizar la extensión: sin punto y en mayúsculas
    # `condición ? si_true : si_false` es el operador ternario.
    #
    # Se comprueba si la extensión empieza por punto. Si es así, se devuelve
    # el *slice* que empieza en el índice 2 (Julia indexa desde 1, así que el
    # índice 1 sería el `"."` si alguien escribió `".tsv"` por error en lugar
    # de `"tsv"`). Si no empieza por punto, se usa tal cual. En ambos casos
    # el resultado se pasa a mayúsculas con `uppercase`.
    #
    # Queremos `normalizar` pasando a mayusculas ya que la funcion\
    # `endswith()` que usamos para comprobar si el string pasado termina\
    # con el mismo sufijo es case-sensitive. Ya sea que el usuario pase la\
    # extension por ejemplo `.TSV` o `.TsV` o cosas asi, o bien porque\
    # el propio archivo tenga escrita asi su extension
    #
    # > Ningún sistema de archivos común (NTFS, ext4, APFS...)
    # > impone que la extensión vaya en minúsculas, así que puedes
    # > escribirla en mayúsculas, minúsculas o mezcladas.
    # > -> por ello al disponer de todas estas posibilidades es necesario\
    # > normalizar la extension de algun modo, ya sea con upper o lower
    # > case, eso ya es lo de menos, aqui se decidio arbitrariamente
    # > ir por uppercase. De otro modo las comprobaciones que verifican\
    # > si un archivo termina en la extension que se busca pueden fallar.
    # @note-body-end
    ext = uppercase(startswith(extension, ".") ? extension[2:end] : extension)
    # @note-end
    # @note-start id=srhgmg color=yellow
    # filter: quedarse con los archivos de esa extensión
    # `filter(f, itr)` tiene dos parámetros, una función y un iterable:
    #
    # - **El iterable** es `readdir(folderName)`, que devuelve una lista con
    #   los nombres de todos los archivos de la carpeta.
    # - **La función** es la anónima `f -> endswith(uppercase(f), ".$ext")`,
    #   que devuelve `true` si el nombre (en mayúsculas) termina en `.EXT`.
    #   `filter` se queda solo con los elementos para los que es `true`.\
    #   o sea, aplicamos la funcion anonima sobre cada elemento del iterable\
    #   y lo eliminamos del iterable si la funcion devuelve false, que en\
    #   este caso seria si no tiene la extension que estamos buscando.
    # @note-body-end
    files = filter(f -> endswith(uppercase(f), ".$ext"), readdir(folderName))
    # @note-end
    # @note-start id=jzow96 color=yellow
    # return: quitar la extensión a cada nombre de archivo
    # Una vez ya habiendo filtrado correctamente los dataset para solo tener en files\
    # Aquellos con la extension que queremos, esta línea se encarga de quitarle\
    # dicha extensión y el punto final a cada archivo para devolver solo\
    # el nombre limpio (por ejemplo, convertir "iris.tsv" en "iris").
    #
    # **Explicación de `return String.(map(f -> f[1:end-(length(ext)+1)], files))`**
    #
    # El objetivo de esta línea es **quitarle la extensión y el punto final a cada archivo** de la lista `files` para devolver únicamente sus nombres limpios (por ejemplo, convertir `"iris.tsv"` en `"iris"`).
    #
    # ---
    #
    # **Desglose paso a paso (de dentro hacia afuera)**
    #
    # 1. **`length(ext) + 1` (caracteres a eliminar):**
    #    - Si la extensión es `"tsv"`, su longitud `length(ext)` es `3`.
    #    - Se le suma `1` para tener en cuenta el carácter del punto (`.`).
    #    - En total son **4 caracteres** a eliminar del final (`.`, `t`, `s`, `v`).
    #
    # 2. **`f[1:end-(length(ext)+1)]` (recorte del nombre):**
    #    - `end` representa el índice del último carácter del string.
    #    - En Julia, la sintaxis `cadena[inicio:fin]` extrae una subcadena (*slice*).
    #    - Ejemplo con `f = "iris.tsv"` (longitud total = 8):
    #      - `end = 8`
    #      - `length(ext) + 1 = 4`
    #      - `end - 4 = 8 - 4 = 4`
    #      - `f[1:4]` extrae desde la posición 1 hasta la 4 $\to$ **`"iris"`**.
    #    - La función anónima `f -> ...` aplica este recorte a un nombre de archivo.
    #
    # 3. **`map(..., files)`:**
    #    - Aplica la función de recorte a cada elemento del array `files`.
    #    - Si `files = ["iris.tsv", "wine.tsv"]`, produce `["iris", "wine"]`.
    #
    # 4. **`String.(...)` (conversión de tipos con broadcasting):**
    #    - Al recortar cadenas en Julia (`f[1:4]`), el resultado es de tipo `SubString` para ahorrar memoria.
    #    - Al anteponer `String.` con el punto (operador de *broadcasting*), Julia convierte cada `SubString` a `String` estándar.
    #    - Garantiza que el valor de retorno sea estrictamente un `Vector{String}`.
    #
    # ---
    #
    # **Resumen visual**
    #
    # ```text
    # " i  r  i  s  .  t  s  v "
    #   1  2  3  4  5  6  7  8   <-- end = 8
    #  [----------] [--------]
    #     1 a 4     extensión + punto (4 caracteres)
    #       ↓
    #    "iris"
    # ```
    #
    # Tanto maps como filter recorren el iterable y aplican la función a cada elemento. Lo que hacen con el resultado es distinto:
    #
    # - `map(f, itr)` usa lo que devuelve la función como nuevo valor del elemento. El resultado tiene el mismo número de elementos, pero transformados.
    # - `filter(f, itr)` espera que la función devuelva true o false, y lo usa para decidir si el elemento se queda. Los elementos que se quedan no se modifican. El resultado puede tener menos elementos.
    # @note-body-end
    return String.(map(f -> f[1:end-(length(ext)+1)], files))
    # @note-end
end;



# @note-start id=kqmy5v color=blue
# loadDataset: carga un dataset .tsv como (inputs, targets)
# Lee el archivo `datasetName` (con o sin `.tsv`) de `datasetFolder`, separa
# la columna `target` del resto y convierte las entradas a `datasetType`.
#
# **API pública**
#
# ```julia
# function loadDataset(datasetName::String, datasetFolder::String;
#     datasetType::DataType=Float64)
# ```
#
# ---
#
# **1. Resumen de los parámetros**
#
# | Parámetro | Tipo | Naturaleza | Descripción |
# |---|---|---|---|
# | `datasetName` | `String` | Posicional (obligatorio) | Nombre del archivo a cargar (admite tanto `"iris"` como `"iris.tsv"`). |
# | `datasetFolder` | `String` | Posicional (obligatorio) | Ruta al directorio donde se encuentra el archivo (ej. `"data/"`). |
# | `datasetType` | `DataType` | Por nombre / *keyword* (opcional) | Precisión numérica a la que se convertirá la matriz de características (`inputs`) y las etiquetas (`targets`). Por defecto es `Float64`, pero se puede cambiar a `Float32`, etc. |
#
# > **¿Por qué el punto y coma (`;`)?**\
# > En Julia, el `;` en la cabecera separa los argumentos **posicionales** (los obligatorios que van por orden a la izquierda) de los argumentos **nombrados** (*keyword arguments*, a la derecha).
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Tuple{Matrix{datasetType}, Vector}` | `(inputs, targets)`. `targets` es `Vector{Bool}` si todas las etiquetas son 0 o 1; si no, `Vector{datasetType}`. |
# | `Nothing` | Si el archivo no existe o no tiene columna `target`. |
#
# **Errores:** `MethodError`/`InexactError` si alguna entrada no se puede convertir a `datasetType`.
#
# ---
#
# **2. Los tipos como valores en Julia (`DataType`)**
#
# En Julia, los tipos de datos son **ciudadanos de primera clase** (*first-class citizens*). Esto significa que un tipo de dato (como `Float64`, `Int32` o `String`) se comporta exactamente igual que cualquier otro valor (como un número o una cadena):
#
# - **Se pueden guardar en variables:**
#   ```julia
#   mi_tipo = Float64
#   ```
# - **Se pueden pasar como argumentos a funciones:**
#   ```julia
#   # Se pasa Float32 como si fuera un número o un texto
#   loadDataset("iris", "data", datasetType=Float32)
#   ```
# - **Se pueden usar para construir otros tipos o realizar conversiones:**
#   ```julia
#   convert(Matrix{datasetType}, datos) # Crea Matrix{Float64} o Matrix{Float32} según la variable
#   ```
#
# El identificador `DataType` es el tipo del propio tipo (su meta-tipo), de modo que `datasetType::DataType` simplemente exige: *"esta variable debe contener un tipo de dato válido"*.
#
# ---
#
# **3. ¿Se puede hacer esto en otros lenguajes de programación?**
#
# La posibilidad de pasar tipos de datos en variables depende de la filosofía del lenguaje:
#
# 1. **Python (Sí, muy similar):**\
#    En Python los tipos también son objetos en tiempo de ejecución (`type`). Es muy común hacer:
#    ```python
#    def cargar_datos(ruta, tipo_dato=float):
#        return np.array(datos, dtype=tipo_dato)
#    ```
#
# 2. **C y C++ (No en tiempo de ejecución):**\
#    No existen variables normales que guarden tipos. En C++ esto se resuelve en tiempo de compilación usando **plantillas** (*templates*), no pasando variables en ejecución:
#    ```cpp
#    template <typename T = double>
#    void loadDataset(string name, string folder);
#    ```
#
# 3. **Java y C# (De forma indirecta / limitada):**\
#    Se usan genéricos en compilación (`<T>`) o el sistema de **reflexión** en tiempo de ejecución pasando objetos de clase (`Class<T> clazz = Double.class`), pero resulta mucho más verboso y tiene sobrecostes de rendimiento.
#
# 4. **TypeScript (No en tiempo de ejecución):**\
#    Los tipos de TypeScript solo existen mientras escribes el código. Al compilar a JavaScript se borran por completo (*type erasure*), por lo que no puedes guardar un tipo estático en una variable de tiempo de ejecución.
# @note-body-end
function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float64)
# @note-end
    filename = endswith(datasetName, ".tsv") ? datasetName : (datasetName * ".tsv")
    filePath = joinpath(datasetFolder, filename)
    if !isfile(filePath)
        return nothing
    end
    data = readdlm(filePath, '\t')
    headers = data[1, :]
    targetCol = findfirst(h -> lowercase(string(h)) == "target", headers)
    if isnothing(targetCol)
        return nothing
    end
    inputCols = setdiff(1:size(data, 2), targetCol)
    inputs = convert(Matrix{datasetType}, data[2:end, inputCols])
    targetsRaw = vec(data[2:end, targetCol])
    targets = all(t -> t == 0 || t == 1, targetsRaw) ? Bool.(targetsRaw) : convert(Vector{datasetType}, targetsRaw)
    return (inputs, targets)
end;



function datasetsDescription(datasetFolder::String, classification::Bool)
    names = fileNamesFolder(datasetFolder, "tsv")
    loaded = filter(d -> !isnothing(d[2]), map(n -> (n, loadDataset(n, datasetFolder)), names))
    filtered = filter(d -> (eltype(d[2][2]) == Bool) == classification, loaded)
    if isempty(filtered)
        return Matrix{Any}(undef, 0, 3)
    end
    colNames = map(d -> d[1], filtered)
    colInstances = map(d -> size(d[2][1], 1), filtered)
    colAttributes = map(d -> size(d[2][1], 2), filtered)
    sizes = colInstances .* colAttributes
    order = sortperm(sizes)
    return hcat(colNames[order], colInstances[order], colAttributes[order])
end;


# @note-start id=4ekz0y color=green
# intervalDiscreteVector: paso m de una variable cíclica
# Calcula el parámetro $m$ de la fórmula para preprocesar variables cíclicas,
# tanto discretas como continuas. Si los valores son discretos, $m$ es la
# separación mínima entre valores consecutivos; si son continuos, $m = 0$.
#
# **Auxiliar interna**
# **Usada por:** `cyclicalEncoding`: necesita $m$ para el denominador
# $\max - \min + m$, de forma que el primer y el último valor discretos no
# caigan en el mismo ángulo.
# **Papel:** paso previo a la codificación seno/coseno de las variables
# cíclicas (por ejemplo, el día de la semana en `elec2`).
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `data` | `AbstractArray{<:Real,1}` | Valores de la variable cíclica. Al menos 2 valores distintos. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Real` | $m$: diferencia mínima si el vector es discreto, `0.` si es continuo. |
#
# **Errores:** `BoundsError` si `data` tiene menos de 2 valores distintos\
# (`differences` queda vacío).\
#
# En Julia, AbstractArray{<:Real, 1} significa:\
#
# - AbstractArray: cualquier array abstracto, como Vector, Array, vistas, etc.\
# - <:Real: cuyos elementos sean de un tipo que sea subtipo de Real, por ejemplo Int, Float64 o Rational.\
# - 1: que tenga una sola dimensión, es decir, un vector.\
#
# @note-body-end
function intervalDiscreteVector(data::AbstractArray{<:Real,1})
# @note-end
    # @note-start id=lyeyt4 color=yellow
    # Ordenar los datos sin repeticiones
    # - `unique`: devuelve un array con los elementos únicos, sin repeticiones.
    # - `sort`: ordena los elementos en orden ascendente (por defecto, de
    #   menor a mayor).
    # @note-body-end
    uniqueData = sort(unique(data));
    # @note-end
    # @note-start id=17mjga color=yellow
    # Diferencias entre elementos consecutivos
    # `diff` devuelve un array con las diferencias entre elementos
    # consecutivos. Por ejemplo, `[1, 2, 4, 7]` da `[1, 2, 3]` (2-1, 4-2 y
    # 7-4), y `[1, 3, 5, 7]` da `[2, 2, 2]`.
    #
    # Con `sort` nos aseguramos de que el primer elemento del array de
    # diferencias sea el menor.
    # @note-body-end
    differences = sort(diff(uniqueData));
    # @note-end
    # @note-start id=xdht8g color=yellow
    # Tomar la diferencia menor
    # Equivale a `minimum(differences)`: como `differences` ya está ordenado,
    # su primer elemento es exactamente la diferencia menor.
    # @note-body-end
    minDifference = differences[1];
    # @note-end

    # @note-start id=ceuenb color=yellow
    # isInteger: ¿x es entero salvo una tolerancia?
    # Devuelve `true` si `x` es entero con tolerancia `tol`.
    #
    # - Esta es una funcion auxiliar que se declara usando una sintaxis rara\
    # de Julia, es una forma compacta de declarar una funcion en una linea\
    # Tambien hay que recordar que Julia automaticamente devuelve como valor de\
    # return la ultima linea de un bloque de codigo, o sea que esta funcion\
    # retorna un bool que resulta de hacer la comprobacion `abs(round(x)-x) < tol`
    #
    # **Ejemplo: `x = 3.005` con tolerancia `tol = 1e-3` (0.001)**
    # ```julia
    # isInteger(3.005, 1e-3)
    # ```
    # 1. **Redondeo:** `round(3.005) = 3.0`
    # 2. **Distancia al entero:** $|3.0 - 3.005| = 0.005$
    # 3. **Comparación con la tolerancia:**
    #    $$
    #    0.005 < 0.001 \implies \text{false}
    #    $$
    #
    # A pesar de que `3.005` está cerca del entero `3`, la distancia (`0.005`) supera el margen de error permitido (`0.001`). Por lo tanto, la función concluye que **no es un entero**.
    #
    # Se usa para el criterio: si todas las diferencias son múltiplos exactos
    # (valores enteros) de la diferencia menor, el vector es de valores
    # discretos.
    # @note-body-end
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    # @note-end
    # @note-start id=gduorn color=yellow
    # ¿Discreto o continuo? Devolver m
    # Se aplica el broadcasting (`.`) dos veces:
    #
    # 1. `./` divide cada elemento de `differences` entre `minDifference`,
    #    obteniendo cada diferencia expresada en "pasos" de la menor.
    # 2. `isInteger.` comprueba si cada cociente es entero.
    #
    # Si al menos un cociente no es entero, al menos una diferencia no es
    # múltiplo exacto de la diferencia menor, y el vector es **continuo**
    # (se devuelve `0.`). Si no, es **discreto** y se devuelve `minDifference`.
    #
    # $$
    # m = \begin{cases} d_{\min} & \text{si } d_i / d_{\min} \in \mathbb{Z} \;\; \forall i \\ 0 & \text{en otro caso} \end{cases}
    # $$
    #
    # **¿Por qué?** Imagina una cuadrícula: con valores discretos existe un
    # "paso" (la diferencia menor) y todas las diferencias entre valores
    # consecutivos deben ser múltiplos exactos de ese paso. Si no es así, los
    # valores no tienen una separación fija y rígida, así que pueden tomar
    # cualquier valor entre dos consecutivos: esa es la definición de
    # continuidad.
    # @note-body-end
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
    # @note-end
end


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    minVal, maxVal = extrema(data)
    m = intervalDiscreteVector(data)
    denom = maxVal - minVal + m
    rad = denom == 0 ? zeros(Float64, length(data)) : 2pi .* (data .- minVal) ./ denom
    return (sin.(rad), cos.(rad))
end;



function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float64)
    dataPath = joinpath(datasetFolder, "elec2_data.dat")
    labelPath = joinpath(datasetFolder, "elec2_label.dat")
    if !isfile(dataPath) || !isfile(labelPath)
        return nothing
    end
    rawInputs = readdlm(dataPath)
    colsToKeep = setdiff(1:size(rawInputs, 2), [1, 4])
    dataRemaining = rawInputs[:, colsToKeep]
    sinDay, cosDay = cyclicalEncoding(dataRemaining[:, 1])
    inputs = hcat(sinDay, cosDay, dataRemaining[:, 2:end])
    inputs = convert(Matrix{datasetType}, inputs)
    rawLabels = readdlm(labelPath)
    targets = vec(Bool.(rawLabels))
    return (inputs, targets)
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------
# @note-start id=1q2eir color=yellow
# Carga de MLJ, LIBSVM y del tipo SVMClassifier
# Estas dos líneas las da el profesor; no se tocan.
#
# - `using MLJ, LIBSVM, MLJLIBSVMInterface`:
#   - **MLJ** es el *framework* de machine learning (como scikit-learn en
#     Python): aporta `machine`, `fit!`, `predict`, `table`, `categorical`…
#   - **LIBSVM** es la librería (escrita en C) que de verdad resuelve el
#     problema de optimización del SVM.
#   - **MLJLIBSVMInterface** es el "puente" que permite usar LIBSVM con la
#     interfaz de MLJ.
# - `SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0`: la macro `@load`
#   busca en el registro de modelos de MLJ el modelo `SVC` del paquete LIBSVM y
#   devuelve su **tipo** (`MLJLIBSVMInterface.SVC`). Lo guardamos en la
#   variable `SVMClassifier`, y por eso luego se puede llamar
#   `SVMClassifier(kernel=..., cost=...)` como si fuera un constructor (en
#   Julia los tipos son valores, igual que vimos en `loadDataset`).
#   `verbosity=0` hace que no imprima nada al cargarlo.
# @note-body-end
using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
# @note-end
# @note-start id=129sn2 color=blue
# predict: salidas Bool de un SVM entrenado con MLJ
# *Wrapper* dado por el profesor. `MLJ.predict` de un `SVC` no devuelve
# booleanos sino un `CategoricalVector{Bool}` (valores categóricos). Esta
# función lo convierte en un vector de `Bool` normal para poder compararlo
# directamente con los *targets*.
#
# **API pública** (la usa la autoevaluación:
# `predict(model, batchInputs(...))`, y la usará el Ejercicio 3 para calcular
# la precisión).
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `model` | `Machine{MLJLIBSVMInterface.SVC, MLJLIBSVMInterface.SVC, true}` | Máquina ya entrenada (la que devuelve `trainSVM`). El `true` indica que la máquina guarda los datos con los que se entrenó. |
# | `inputs` | `AbstractArray` | Matriz de entradas, instancias en filas. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `AbstractVector{Bool}` | Clase predicha para cada fila (en la práctica sale un `BitVector`). |
#
# **Errores:** `MethodError` si `model` no es una `Machine` de un `SVC`.
#
# **Cómo funciona** (de dentro hacia fuera):
#
# 1. `MLJ.table(inputs)`: MLJ trabaja con tablas, no con matrices.
# 2. `MLJ.predict(model, ...)`: devuelve el vector categórico `outputs`.
# 3. `int(outputs)`: el código entero de cada categoría (1 para el primer
#    nivel, 2 para el segundo).
# 4. `levels(outputs)[...]`: los niveles son `[false, true]`; al indexarlos con
#    esos códigos se recupera el `Bool` real de cada predicción.
# 5. `convert(Vector, ...)`: se asegura de devolver un vector plano.
#
# La sintaxis `f(x) = (a = ...; return b; )` es una función de una línea cuyo
# cuerpo es un bloque entre paréntesis con varias expresiones separadas por `;`.
# @note-body-end
predict(model::Machine{MLJLIBSVMInterface.SVC, MLJLIBSVMInterface.SVC, true}, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return convert(Vector, levels(outputs)[int(outputs)]); )
# @note-end

# @note-start id=tdg2ys color=yellow
# Paquetes auxiliares del Ejercicio 2
# - `using Base.Iterators`: da acceso a `partition`, que usa `divideBatches`
#   para trocear el vector de índices en grupos de `batchSize`.
# - `using StatsBase`: funciones estadísticas como `mean`. Las funciones del
#   Ejercicio 2 no la usan, pero la autoevaluación sí
#   (`mean.(batchInputs.(...))`) y el Ejercicio 3 también la necesitará.
# - `using Random: shuffle`: importa **solo** `shuffle`, que devuelve una
#   permutación aleatoria de un vector; `divideBatches` la usa para desordenar
#   las instancias cuando `shuffleRows=true`.
# @note-body-end
using Base.Iterators
using StatsBase
using Random: shuffle
# @note-end

# @note-start id=xo5oby color=purple
# Batch: alias de tipo para un lote (entradas, salidas)
# No es un `struct` nuevo, es un **alias**: un nombre corto para el tipo
# `Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}`. Cualquier tupla
# `(matriz, vector)` es automáticamente un `Batch`, sin tener que construirla
# de ninguna forma especial. Lo da el profesor.
#
# **Campos (posiciones de la tupla)**
#
# | Posición | Tipo | Significado |
# |---|---|---|
# | `1` | `AbstractArray{<:Real,2}` | Matriz de entradas: una instancia por fila, un atributo por columna. `<:Real` admite `Float32`, `Float64`, `Int`… |
# | `2` | `AbstractArray{<:Any,1}` | Vector de salidas deseadas. `<:Any` admite `Vector{Bool}`, `BitVector`, `Vector{Float64}`… |
#
# **Invariantes**
#
# - `size(batch[1], 1) == length(batch[2])`: tantas filas como *targets*.
# - Criterio de orden de la práctica: al unir lotes, los datos **antiguos van
#   arriba** y los **recientes al final**. Los SVM en teoría no dependen del
#   orden, pero el optimizador de LIBSVM sí (redondeos), así que respetar el
#   orden hace los resultados repetibles.
#
# **¿Por qué funcionan cosas como `Vector{Tuple{Matrix{Float64}, BitVector}}`
# como `AbstractArray{<:Batch,1}`?** Porque las tuplas en Julia son
# **covariantes**: `Tuple{Matrix{Float64}, BitVector} <: Batch`. Y el `<:` de
# `AbstractArray{<:Batch,1}` acepta vectores cuyo tipo de elemento sea
# *cualquier subtipo* de `Batch`, no solo `Batch` exacto.
# @note-body-end
Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}
# @note-end


# @note-start id=kzhwp9 color=blue
# batchInputs: matriz de entradas de un lote
# Devuelve el primer elemento de la tupla. Sirve para **abstraer** al resto del
# código de cómo está guardado un `Batch` por dentro: si mañana el `Batch`
# cambiase de forma, solo habría que cambiar esta función.
#
# **API pública** (la llama directamente la autoevaluación). También la usan
# internamente `batchLength`, `selectInstances`, `joinBatches` y
# `trainSVM(::Batch, ...)`.
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `batch` | `Batch` | Lote de datos `(entradas, salidas)`. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `AbstractArray{<:Real,2}` | La matriz de entradas, **el mismo objeto** (no una copia). |
#
# **Errores:** ninguno propio (sin bucles, como exige el enunciado).
# @note-body-end
function batchInputs(batch::Batch)
# @note-end
    # @note-start id=y1psqk color=yellow
    # Devolver el primer elemento de la tupla
    # `batch[1]` indexa la tupla (Julia indexa desde 1). No copia nada: devuelve la
    # misma matriz. Por eso la autoevaluación puede comprobar
    # `batchInputs(dataset) === inputs` con `===` (identidad del objeto, no solo
    # igualdad de valores).
    # @note-body-end
    return batch[1]
    # @note-end
end;

# @note-start id=cpxkjt color=blue
# batchTargets: vector de salidas deseadas de un lote
# Devuelve el segundo elemento de la tupla. Misma idea que `batchInputs`:
# ocultar la estructura interna del `Batch`.
#
# **API pública** (la llama directamente la autoevaluación). También la usan
# internamente `selectInstances`, `joinBatches` y `trainSVM(::Batch, ...)`.
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `batch` | `Batch` | Lote de datos `(entradas, salidas)`. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `AbstractArray{<:Any,1}` | El vector de salidas deseadas, el mismo objeto (no una copia). |
#
# **Errores:** ninguno propio.
# @note-body-end
function batchTargets(batch::Batch)
# @note-end
    # @note-start id=jdx52a color=yellow
    # Devolver el segundo elemento de la tupla
    # `batch[2]` es el vector de *targets*. Igual que antes, sin copia, por eso
    # pasa el test `batchTargets(dataset) === targets`.
    # @note-body-end
    return batch[2]
    # @note-end
end;

# @note-start id=08ur7x color=blue
# batchLength: número de instancias de un lote
# Cuenta las instancias como el número de **filas** de la matriz de entradas,
# porque en esta práctica cada instancia es una fila.
#
# **API pública** (la llama la autoevaluación). También la usan internamente
# `divideBatches` (para saber cuántos índices generar) y
# `trainSVM(::Batch, ...)` (para saber cuántos vectores de soporte previos hay,
# el famoso `N`).
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `batch` | `Batch` | Lote de datos. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Int` | Número de instancias ($\geq 0$; 0 si el lote está vacío). |
#
# **Errores:** ninguno propio.
# @note-body-end
function batchLength(batch::Batch)
# @note-end
    # @note-start id=jc966c color=yellow
    # Número de filas de la matriz de entradas
    # `size(M, 1)` devuelve el tamaño de la dimensión 1 (filas); `size(M, 2)`
    # serían las columnas (atributos). Se usa `batchInputs` en lugar de `batch[1]`
    # para respetar la abstracción.
    #
    # Se podría haber usado `length(batchTargets(batch))`, que da lo mismo, pero
    # con la matriz también funciona un lote vacío creado con
    # `Array{T,2}(undef, 0, nAtributos)`: `size` devuelve 0 sin problema.
    # @note-body-end
    return size(batchInputs(batch), 1)
    # @note-end
end;

# @note-start id=pqxkk6 color=blue
# selectInstances: sub-lote con las instancias indicadas
# Devuelve un nuevo `Batch` con solo las filas de entradas y los elementos de
# *targets* que indique `indices`, en el orden en que aparecen en `indices`.
#
# **API pública** (la llama la autoevaluación). También la usan internamente
# `divideBatches` (para construir cada lote) y `trainSVM(::Batch, ...)` (para
# extraer los vectores de soporte).
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `batch` | `Batch` | Lote de datos original. |
# | `indices` | `Any` | Qué instancias tomar. Admite un entero (`5`), un rango (`1:100`), un vector de enteros (`[3,1,2]`) o una máscara booleana (`Vector{Bool}`/`BitVector` de longitud `batchLength(batch)`). |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Batch` | Nueva tupla `(entradas[indices, :], targets[indices])`. Son **copias**, no vistas. |
#
# **Errores:** `BoundsError` si algún índice está fuera de rango, o si la
# máscara booleana no tiene la longitud del lote.
# @note-body-end
function selectInstances(batch::Batch, indices::Any)
# @note-end
    # @note-start id=m0ux1j color=yellow
    # Si el índice es un único entero, envolverlo en un vector
    # **Esto no lo pide el enunciado explícitamente, pero evita un error sutil.**
    # En Julia, indexar con un entero **elimina esa dimensión**:
    #
    # | Expresión | Resultado |
    # |---|---|
    # | `M[5, :]` | `Vector` (una fila "aplanada") |
    # | `M[[5], :]` | `Matrix` de 1 fila |
    # | `v[5]` | un escalar |
    # | `v[[5]]` | `Vector` de 1 elemento |
    #
    # Si no se hiciera esto, `selectInstances(batch, 5)` devolvería
    # `(Vector, escalar)`, que **no es un `Batch`** y rompería cualquier función que
    # lo reciba. Con `[indices]` el resultado sigue siendo `(Matrix 1×n, Vector)`.
    #
    # `isa(x, Integer)` comprueba si `x` es de algún tipo entero (`Int`, `Int32`…).
    # En cualquier otro caso (rango, vector, máscara) los índices se usan tal cual.
    # @note-body-end
    idx = isa(indices, Integer) ? [indices] : indices
    # @note-end
    # @note-start id=tadom3 color=yellow
    # Indexar entradas y targets con los mismos índices
    # - `batchInputs(batch)[idx, :]`: las filas `idx`, **todas** las columnas (`:`).
    # - `batchTargets(batch)[idx]`: los mismos elementos del vector de salidas.
    #
    # Usar exactamente el mismo `idx` en ambos garantiza que cada fila sigue
    # emparejada con su etiqueta. Los paréntesis `( , )` crean la tupla que se
    # devuelve, es decir, un `Batch`.
    # @note-body-end
    return (batchInputs(batch)[idx, :], batchTargets(batch)[idx])
    # @note-end
end;

# @note-start id=rcnf32 color=blue
# joinBatches: concatena dos lotes (batch1 arriba)
# Une dos lotes en uno respetando el orden: `batch1` queda en las primeras
# filas y `batch2` al final. Con el criterio de la práctica, `batch1` son los
# datos **antiguos** y `batch2` los **recientes**.
#
# **API pública** (la llama la autoevaluación). También la usa internamente
# `trainSVM(::Batch, ...)`: para unir vectores de soporte previos + lote nuevo
# y para montar el lote de vectores de soporte que devuelve.
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `batch1` | `Batch` | Lote que va arriba / al principio. |
# | `batch2` | `Batch` | Lote que va abajo / al final. Mismo número de columnas que `batch1`. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Batch` | Lote con `batchLength(batch1) + batchLength(batch2)` instancias. |
#
# **Errores:** `DimensionMismatch` si las matrices tienen distinto número de
# columnas.
# @note-body-end
function joinBatches(batch1::Batch, batch2::Batch)
# @note-end
    # @note-start id=0zq5ti color=yellow
    # vcat de entradas y vcat de targets
    # `vcat` (*vertical concatenation*) apila uno debajo de otro:
    #
    # - Con matrices, añade las filas de la segunda debajo de la primera
    #   ($n_1 \times d$ y $n_2 \times d \;\to\; (n_1+n_2) \times d$).
    # - Con vectores, pone los elementos del segundo detrás del primero.
    #
    # Se hace por separado para entradas y *targets* y se vuelve a empaquetar en
    # una tupla. Si los tipos no coinciden (por ejemplo `Float32` y `Float64`, o
    # `Vector{Bool}` y `BitVector`), `vcat` los promociona a un tipo común.
    # @note-body-end
    return (vcat(batchInputs(batch1), batchInputs(batch2)), vcat(batchTargets(batch1), batchTargets(batch2)))
    # @note-end
end;


# @note-start id=kx8xa3 color=blue
# divideBatches: parte un dataset en lotes de batchSize
# Divide un conjunto de datos (que también es un `Batch`) en lotes
# consecutivos. Todos tienen `batchSize` instancias salvo el último, que tiene
# las que sobren. Opcionalmente desordena las instancias antes.
#
# **API pública** (la llama la autoevaluación y la usarán los experimentos y el
# Ejercicio 3).
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `dataset` | `Batch` | Conjunto de datos completo. |
# | `batchSize` | `Int` | Instancias por lote ($\geq 1$). |
# | `shuffleRows` | `Bool` (*keyword*, por defecto `false`) | Si `true`, baraja las instancias antes de dividir. Usa el generador global de `Random`, así que depende de `seed!`. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector{<:Batch}` | $\lceil N / \text{batchSize} \rceil$ lotes. Ninguna instancia se pierde ni se repite. |
#
# **Errores:** `ArgumentError` de `partition` si `batchSize < 1`.
#
# Ejemplo con $N = 208$ y `batchSize = 100`: lotes de 100, 100 y 8 instancias.
# @note-body-end
function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
# @note-end
    # @note-start id=ucc7y2 color=yellow
    # N: número total de instancias del dataset
    # Se guarda en una variable porque se usa para construir el rango `1:N` en la
    # línea siguiente.
    # @note-body-end
    N = batchLength(dataset)
    # @note-end
    # @note-start id=5v7hh0 color=yellow
    # Índices en orden o barajados
    # Operador ternario otra vez:
    #
    # - `shuffleRows == true` → `shuffle(1:N)`: una permutación aleatoria de
    #   `1, …, N` (un `Vector{Int}`).
    # - `shuffleRows == false` → el rango `1:N` tal cual (no se crea ningún
    #   vector, un rango es "perezoso").
    #
    # **Se barajan los índices, no los datos.** Así solo se mueve un vector de
    # enteros y la copia real de filas la hace `selectInstances` una única vez por
    # lote. Además, entradas y *targets* quedan barajados con **la misma**
    # permutación, así que no se desemparejan.
    # @note-body-end
    indices = shuffleRows ? shuffle(1:N) : (1:N)
    # @note-end
    # @note-start id=0yq3rf color=yellow
    # Trocear los índices y crear un lote por trozo
    # Esto es una *array comprehension* (el enunciado prohíbe bucles `for`
    # explícitos y propone justo esta forma):
    #
    # 1. `Base.Iterators.partition(indices, batchSize)` agrupa los índices en
    #    trozos consecutivos de `batchSize`. Por ejemplo
    #    `partition(1:10, 4)` → `1:4`, `5:8`, `9:10`.
    # 2. `for idx in ...` recorre cada trozo.
    # 3. `selectInstances(dataset, idx)` convierte cada trozo de índices en un
    #    `Batch`.
    # 4. Los corchetes `[ ... ]` recogen todos los lotes en un `Vector`.
    #
    # ```mermaid
    # flowchart LR
    #     A["indices (1:N o barajados)"] --> B["partition(indices, batchSize)"]
    #     B --> C["idx₁, idx₂, …, idxₖ"]
    #     C --> D["selectInstances(dataset, idxᵢ)"]
    #     D --> E["[batch₁, …, batchₖ]"]
    # ```
    # @note-body-end
    return [selectInstances(dataset, idx) for idx in Base.Iterators.partition(indices, batchSize)]
    # @note-end
end;

# @note-start id=regkin color=blue
# trainSVM: entrena un SVM con un lote + SVs previos
# Entrena un SVM con el lote `dataset` **precedido** de los vectores de soporte
# de un entrenamiento anterior. Es el "paso" del SVM incremental: los
# vectores de soporte hacen de **resumen** de todos los datos anteriores.
#
# Además del modelo devuelve cuáles de los datos de entrenamiento han acabado
# siendo vectores de soporte, separando los que venían del lote de vectores de
# soporte previo y los que venían del lote nuevo.
#
# **API pública** (la llama la autoevaluación). También la usa
# `trainSVM(::AbstractArray{<:Batch,1}, ...)` en cada iteración de su bucle, y
# la usará `streamLearning_ISVM` en el Ejercicio 3.
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `dataset` | `Batch` | Lote nuevo de entrenamiento. *Targets* booleanos. |
# | `kernel` | `String` | `"linear"`, `"rbf"`, `"poly"` o `"sigmoid"`. |
# | `C` | `Real` | Hiperparámetro de coste: cuánto se penaliza clasificar mal ($C > 0$). |
# | `degree` | `Real` (*kw*, `1`) | Grado del kernel polinómico. Solo afecta a `"poly"`. |
# | `gamma` | `Real` (*kw*, `2`) | $\gamma$ de los kernels `rbf`, `poly` y `sigmoid`. |
# | `coef0` | `Real` (*kw*, `0.`) | Término independiente de `poly` y `sigmoid`. |
# | `supportVectors` | `Batch` (*kw*) | Vectores de soporte previos. Por defecto un lote **vacío** (0 filas) con el mismo tipo de elemento y número de columnas que `dataset`, es decir, entrenamiento "normal" sin historia. |
#
# **Salida**: una tupla de 3 elementos
#
# | Posición | Tipo | Descripción |
# |---|---|---|
# | 1 | `Machine{MLJLIBSVMInterface.SVC, ...}` | Modelo entrenado (`mach`). |
# | 2 | `Batch` | Nuevos vectores de soporte: primero los que venían de `supportVectors`, después los que venían de `dataset`. |
# | 3 | `Tuple{Vector{Int32}, Vector{Int32}}` | `(índices en supportVectors, índices en dataset)`, ambos ordenados de menor a mayor. |
#
# **Errores:** si `kernel` no es ninguno de los 4, se pasa `nothing` como
# kernel y MLJ lanza un error al construir el modelo.
#
# **Cómo funciona**
#
# ```mermaid
# flowchart TD
#     A["supportVectors (N filas)"] --> C["joinBatches: trainingData"]
#     B["dataset (lote nuevo)"] --> C
#     C --> D["SVMClassifier + machine + fit!"]
#     D --> E["índices de los SVs en trainingData, ordenados"]
#     E -->|"≤ N"| F["svIndices_old"]
#     E -->|"> N, restando N"| G["svIndices_new"]
#     F --> H["joinBatches: newSupportVectors"]
#     G --> H
# ```
# @note-body-end
function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
# @note-end

    # @note-start id=iugwps color=yellow
    # N: cuántos vectores de soporte previos hay
    # Hace falta para, al final, saber de dónde viene cada vector de soporte: como
    # los previos se colocan arriba, las filas $1..N$ de `trainingData` son los
    # vectores de soporte antiguos y las filas $N+1..$ son el lote nuevo.
    #
    # En el primer entrenamiento (lote vacío por defecto) $N = 0$.
    # @note-body-end
    N = batchLength(supportVectors)
    # @note-end
    # @note-start id=idp5sy color=yellow
    # Unir vectores de soporte previos + lote nuevo
    # Siguiendo el criterio de la práctica, lo **antiguo arriba** (los vectores de
    # soporte) y lo **nuevo abajo** (`dataset`). No se desordena nada. Este es el
    # lote con el que se entrena realmente el SVM, **no** `dataset` a secas: esa es
    # la idea del SVM incremental.
    #
    # $$
    # \text{trainingData} = \begin{bmatrix} \text{supportVectors} \\ \text{dataset} \end{bmatrix}
    # $$
    # @note-body-end
    trainingData = joinBatches(supportVectors, dataset)
    # @note-end

    # @note-start id=nvx2t7 color=yellow
    # Crear el modelo SVM con sus hiperparámetros
    # Código indicado en el enunciado. `SVMClassifier(...)` crea el modelo **sin
    # entrenar** (solo guarda la configuración).
    #
    # - `kernel`: una cadena de ternarios anidados traduce el `String` al valor
    #   que espera LIBSVM:
    #
    #   | `kernel` | LIBSVM | $K(x, x')$ |
    #   |---|---|---|
    #   | `"linear"` | `Kernel.Linear` | $x^\top x'$ |
    #   | `"rbf"` | `Kernel.RadialBasis` | $e^{-\gamma \lVert x - x' \rVert^2}$ |
    #   | `"poly"` | `Kernel.Polynomial` | $(\gamma\, x^\top x' + c_0)^{d}$ |
    #   | `"sigmoid"` | `Kernel.Sigmoid` | $\tanh(\gamma\, x^\top x' + c_0)$ |
    #
    #   Si no es ninguno, queda `nothing` y MLJ dará error.
    # - `cost`, `gamma`, `coef0` se convierten a `Float64` y `degree` a `Int32`
    #   porque son los tipos exactos que exige `MLJLIBSVMInterface.SVC`; así se
    #   puede llamar a la función con `C=1` (`Int`) o `gamma=3` sin que falle.
    # @note-body-end
    model = SVMClassifier(
        kernel =
            kernel=="linear"  ? LIBSVM.Kernel.Linear :
            kernel=="rbf"     ? LIBSVM.Kernel.RadialBasis :
            kernel=="poly"    ? LIBSVM.Kernel.Polynomial :
            kernel=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
        cost   = Float64(C),
        gamma  = Float64(gamma),
        degree = Int32(  degree),
        coef0  = Float64(coef0))
    # @note-end

    # @note-start id=q1r3w5 color=yellow
    # Crear la machine: enlazar modelo y datos
    # En MLJ una `machine` une un modelo con sus datos (y más tarde guarda el
    # resultado del entrenamiento).
    #
    # - `MLJ.table(batchInputs(trainingData))`: MLJ espera las entradas como
    #   **tabla**, así que la matriz se envuelve en una tabla (cada columna pasa a
    #   ser una "columna" con nombre).
    # - `MLJ.categorical(batchTargets(trainingData); levels = [false, true])`: los
    #   *targets* de clasificación deben ser **categóricos**. Fijar
    #   `levels = [false, true]` asegura que siempre haya 2 niveles en ese orden,
    #   aunque un lote solo tuviese una clase.
    #
    # **Importante:** se usa `trainingData` (vectores de soporte + lote) y no
    # `dataset`, como pide el enunciado.
    # @note-body-end
    mach = MLJ.machine(model,
        MLJ.table(batchInputs(trainingData)),
        MLJ.categorical(batchTargets(trainingData) ; levels = [false, true]))
    # @note-end
    # @note-start id=2n3efk color=yellow
    # Entrenar
    # `fit!` entrena el SVM y guarda el resultado dentro de `mach` (por eso lleva
    # `!`: modifica su argumento). `verbosity=0` evita que imprima el progreso.
    # @note-body-end
    MLJ.fit!(mach, verbosity=0)
    # @note-end

    # @note-start id=wpejzu color=yellow
    # Índices (ordenados) de los vectores de soporte
    # Tras `fit!`, `mach.fitresult` es una tupla
    # `(modelo LIBSVM, decodificador de clases)`. En el modelo LIBSVM:
    #
    # - `.SVs` es la estructura con los vectores de soporte.
    # - `.SVs.indices` son las **filas de `trainingData`** (índices desde 1) que
    #   han quedado como vectores de soporte.
    #
    # LIBSVM los devuelve **agrupados por clase**, no en orden (por ejemplo primero
    # todos los de la clase `false` y luego los de `true`). Por eso se ordenan con
    # `sort`: así los vectores de soporte mantienen el orden original de los datos
    # (lo antiguo arriba) y los índices salen ordenados, como comprueba la
    # autoevaluación con `issorted`.
    # @note-body-end
    indicesNewSupportVectors = sort( mach.fitresult[1].SVs.indices )
    # @note-end

    # @note-start id=ou8yvt color=yellow
    # Índices que vienen de los vectores de soporte previos
    # `indicesNewSupportVectors .<= N` crea una máscara booleana (el `.` aplica la
    # comparación a cada elemento) y al indexar con ella se quedan solo los índices
    # $\leq N$, que son las filas de `supportVectors`. Como esas filas estaban
    # arriba del todo, el índice en `trainingData` **coincide** con el índice en
    # `supportVectors`: no hay que ajustar nada.
    # @note-body-end
    svIndices_old = indicesNewSupportVectors[indicesNewSupportVectors .<= N]
    # @note-end
    # @note-start id=1iyv1t color=yellow
    # Índices que vienen del lote nuevo (restando N)
    # Los índices $> N$ son filas del lote `dataset`, pero están desplazados $N$
    # posiciones porque encima tienen los $N$ vectores de soporte previos. Con
    # `.- N` (resta elemento a elemento) se pasan a índices **dentro de
    # `dataset`**:
    #
    # $$
    # i_{\text{dataset}} = i_{\text{trainingData}} - N \qquad \text{si } i_{\text{trainingData}} > N
    # $$
    #
    # Ejemplo con $N = 38$: la fila 39 de `trainingData` es la fila 1 de `dataset`.
    # @note-body-end
    svIndices_new = indicesNewSupportVectors[indicesNewSupportVectors .> N] .- N
    # @note-end

    # @note-start id=d2tiud color=yellow
    # Montar el lote de nuevos vectores de soporte
    # Se seleccionan los vectores de soporte de cada origen con `selectInstances` y
    # se unen con `joinBatches` en el orden del enunciado: **primero los que venían
    # de `supportVectors`, después los del lote nuevo**. Es exactamente lo que
    # comprueba la autoevaluación:
    #
    # `joinBatches(selectInstances(supportVectors, idx[1]), selectInstances(dataset, idx[2])) == newSupportVectors`
    #
    # Como ambos vectores de índices están ordenados, el resultado equivale a
    # `selectInstances(trainingData, indicesNewSupportVectors)`.
    # @note-body-end
    newSupportVectors = joinBatches(selectInstances(supportVectors, svIndices_old), selectInstances(dataset, svIndices_new))
    # @note-end

    # @note-start id=0g447r color=yellow
    # Devolver (modelo, vectores de soporte, índices)
    # Tupla de 3 elementos en el orden que pide el enunciado. El tercero es a su
    # vez una tupla `(svIndices_old, svIndices_new)`, que se usará en el
    # Ejercicio 3 (`streamLearning_ISVM`) para saber qué vectores de soporte son
    # antiguos y cuáles nuevos.
    # @note-body-end
    return (mach, newSupportVectors, (svIndices_old, svIndices_new))
    # @note-end
end;

# @note-start id=quh4j5 color=blue
# trainSVM: SVM incremental sobre un vector de lotes
# Entrena un SVM de forma **incremental**: recorre los lotes en orden y, en
# cada paso, entrena con el lote actual más los vectores de soporte del paso
# anterior. Devuelve solo el último modelo.
#
# Es un **método** distinto de la misma función `trainSVM`: Julia elige cuál
# ejecutar por el tipo del primer argumento (**despacho múltiple**). Si recibe
# un `Batch` se ejecuta el de arriba; si recibe un vector de `Batch`, este.
#
# **API pública** (la llama la autoevaluación).
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `batches` | `AbstractArray{<:Batch,1}` | Lotes en orden cronológico (el primero es el más antiguo). No vacío. |
# | `kernel` | `String` | `"linear"`, `"rbf"`, `"poly"` o `"sigmoid"`. |
# | `C` | `Real` | Hiperparámetro de coste. |
# | `degree`, `gamma`, `coef0` | `Real` (*kw*; `1`, `2`, `0.`) | Hiperparámetros del kernel; se pasan tal cual a cada entrenamiento. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Machine{MLJLIBSVMInterface.SVC, ...}` | El modelo entrenado en la **última** iteración. |
#
# **Errores:** `BoundsError` si `batches` está vacío (no hay `batches[1]`).
#
# **Cómo funciona** (el esquema de la figura del enunciado):
#
# ```mermaid
# flowchart LR
#     S0["SVs₀ = vacío"] --> T1["trainSVM(batch₁, SVs₀)"]
#     T1 -->|"SVs₁"| T2["trainSVM(batch₂, SVs₁)"]
#     T2 -->|"SVs₂"| T3["…"]
#     T3 -->|"SVsₖ₋₁"| Tk["trainSVM(batchₖ, SVsₖ₋₁)"]
#     Tk --> M["model"]
# ```
# @note-body-end
function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
# @note-end

    # @note-start id=itaqfy color=yellow
    # Tomar el primer lote como referencia de tipos
    # Solo se usa para saber el **tipo de elemento** y el **número de columnas**
    # con los que crear el lote vacío de la línea siguiente.
    # @note-body-end
    firstBatch = batches[1]
    # @note-end
    # @note-start id=lcl3c1 color=yellow
    # Lote de vectores de soporte vacío inicial
    # Es la misma expresión que el valor por defecto de `supportVectors` en el otro
    # método:
    #
    # - `Array{eltype(firstBatch[1]),2}(undef, 0, size(firstBatch[1],2))`: matriz
    #   de **0 filas** y tantas columnas como atributos, del mismo tipo que las
    #   entradas (`Float32`, `Float64`…). `undef` indica que no se inicializa
    #   (da igual, no tiene filas).
    # - `Array{eltype(firstBatch[2]),1}(undef, 0)`: vector vacío del tipo de los
    #   *targets*.
    #
    # Tener el número de columnas correcto es necesario para que `vcat` pueda unir
    # este lote vacío con el primer lote en `joinBatches`. `local` deja claro que
    # es una variable del ámbito de la función, que el bucle irá sobrescribiendo.
    # @note-body-end
    local svs = ( Array{eltype(firstBatch[1]),2}(undef,0,size(firstBatch[1],2)) , Array{eltype(firstBatch[2]),1}(undef,0) )
    # @note-end
    # @note-start id=nxur4m color=yellow
    # Declarar model antes del bucle
    # En Julia cada iteración de un `for` tiene **su propio ámbito**: una variable
    # creada por primera vez dentro del bucle deja de existir al salir de él y no se
    # podría devolver. Declarándola aquí con `local model`, las asignaciones de
    # dentro del bucle **sobrescriben esta** variable, y al terminar sigue
    # disponible con el último modelo (es justo lo que explica el enunciado).
    # @note-body-end
    local model
    # @note-end
    # @note-start id=3jpho4 color=yellow
    # Bucle incremental (el único permitido)
    # Por cada lote, en orden, se llama al otro método de `trainSVM` pasándole:
    #
    # - el lote actual `batch`,
    # - la misma configuración (`kernel`, `C`, `degree`, `gamma`, `coef0`),
    # - `supportVectors=svs`: los vectores de soporte que se tienen hasta ahora.
    #
    # La salida se desempaqueta en tres variables:
    #
    # - `model` ← el modelo entrenado (sobrescribe el anterior),
    # - `svs` ← los **nuevos** vectores de soporte, que serán el "resumen" que se
    #   pase en la siguiente iteración,
    # - `_` ← los índices, que aquí no hacen falta (el `_` en Julia significa
    #   "descarta este valor").
    # @note-body-end
    for batch in batches
        model, svs, _ = trainSVM(batch, kernel, C; degree=degree, gamma=gamma, coef0=coef0, supportVectors=svs)
    end
    # @note-end
    # @note-start id=ajz59d color=yellow
    # Devolver el último modelo
    # Tras el bucle, `model` contiene el SVM entrenado con el último lote más todos
    # los vectores de soporte acumulados, que es lo que pide el enunciado.
    # @note-body-end
    return model
    # @note-end
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# @note-start id=isld01 color=blue
# initializeStreamLearningData: memoria inicial y lotes
# Carga el dataset de Stream Learning y separa las primeras `windowSize` instancias
# como memoria inicial (ventana temporal), dividiendo las restantes en un vector de lotes
# (*batches*) de tamaño `batchSize` ordenados cronológicamente.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `datasetFolder` | `String` | Carpeta que contiene los archivos de datos (`elec2_data.dat` y `elec2_label.dat`). |
# | `windowSize` | `Int` | Número de instancias que contendrá la memoria temporal inicial. |
# | `batchSize` | `Int` | Número de instancias por cada lote (*batch*) que llega a través del flujo. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Tuple{Batch, Vector{<:Batch}}` | Tupla `(memoria, lotes)` donde `memoria` es la ventana temporal inicial y `lotes` contiene los lotes restantes para simular el flujo continuo. |
#
# **Restricciones:** no se permite el uso de bucles.
# @note-body-end
function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
# @note-end
    # @note-start id=isld02 color=yellow
    # Carga y partición cronológica
    # 1. Se cargan todos los datos con `loadStreamLearningDataset`.
    # 2. Las primeras `windowSize` instancias forman la memoria inicial (`selectInstances`).
    # 3. El resto de instancias se dividen en lotes de tamaño `batchSize` mediante `divideBatches`
    #    con `shuffleRows=false` para preservar el orden temporal de llegada.
    # @note-body-end
    data = loadStreamLearningDataset(datasetFolder)
    memory = selectInstances(data, 1:windowSize)
    remainingData = selectInstances(data, (windowSize + 1):batchLength(data))
    batchList = divideBatches(remainingData, batchSize; shuffleRows=false)
    return (memory, batchList)
    # @note-end
end;

# @note-start id=adbt01 color=blue
# addBatch!: actualización FIFO in-place de la memoria
# Actualiza la memoria con un lote recién llegado aplicando una política FIFO
# (*First In, First Out*): descarta las $K$ instancias más antiguas del inicio de la
# memoria, desplaza el resto hacia el principio y añade las $K$ nuevas al final.
#
# Se realiza **in-place** sobre los arrays de la tupla `memory` sin crear nuevas matrices.
#
# $$
# \text{Memoria: } [1 \dots K \text{ (descartados)}] \;+\; [(K+1) \dots M \to 1 \dots (M-K)] \;+\; [\text{nuevo } 1 \dots K \to (M-K+1) \dots M]
# $$
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `memory` | `Batch` | Tupla `(entradas, salidas)` que representa la memoria actual (se modifica in-place). |
# | `newBatch` | `Batch` | Nuevo lote de datos que acaba de recibirse por el flujo. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Batch` | La misma tupla `memory` modificada in-place. |
#
# **Restricciones:** no se permite el uso de bucles.
# @note-body-end
function addBatch!(memory::Batch, newBatch::Batch)
# @note-end
    # @note-start id=adbt02 color=yellow
    # Desplazamiento e inserción in-place
    # Sea $M$ la longitud de la memoria y $K$ la longitud del nuevo lote:
    # - Filas $1 \dots M-K$ reciben las instancias existentes de $(K+1) \dots M$.
    # - Filas $(M-K+1) \dots M$ reciben las instancias del nuevo lote.
    # Se aplica tanto a la matriz de entradas (`memory[1]`) como al vector de salidas (`memory[2]`).
    # @note-body-end
    M = batchLength(memory)
    K = batchLength(newBatch)
    memory[1][1:(M - K), :] = memory[1][(K + 1):M, :]
    memory[2][1:(M - K)]   = memory[2][(K + 1):M]
    memory[1][(M - K + 1):M, :] = batchInputs(newBatch)
    memory[2][(M - K + 1):M]   = batchTargets(newBatch)
    return memory
    # @note-end
end;

# @note-start id=slsm01 color=blue
# streamLearning_SVM: flujo con SVM y ventana deslizante
# Simula el procesamiento en flujo continuo con un clasificador SVM entrenado
# siempre con los datos contenidos en la ventana deslizante (`memory`).
#
# Para cada lote entrante:
# 1. Se evalúa el modelo actual sobre el lote (test antes de entrenar).
# 2. Se actualiza la memoria con `addBatch!`.
# 3. Se reentrena el SVM completo con la nueva memoria.
#
# ```mermaid
# flowchart LR
#     B["Lote entrante"] --> T["1. Test (precisión)"]
#     T --> M["2. addBatch! (memoria)"]
#     M --> R["3. trainSVM (reentrenar)"]
# ```
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `datasetFolder` | `String` | Carpeta del dataset. |
# | `windowSize` | `Int` | Tamaño de la ventana de memoria. |
# | `batchSize` | `Int` | Tamaño de cada lote. |
# | `kernel` | `String` | Kernel del SVM (`"linear"`, `"rbf"`, `"poly"`, `"sigmoid"`). |
# | `C` | `Real` | Parámetro de penalización C. |
# | `degree`, `gamma`, `coef0` | `Real` | Hiperparámetros opcionales del kernel. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector{<:Real}` | Precisiones en test de cada lote antes de incorporar dicho lote a la memoria. |
#
# **Restricciones:** se permite un único bucle que itere sobre el vector de batches.
# @note-body-end
function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
# @note-end
    # @note-start id=slsm02 color=yellow
    # Inicialización, entrenamiento inicial y ciclo de flujo
    # - Se obtiene la memoria inicial y la lista de batches.
    # - Se entrena el modelo base inicial con `trainSVM(memory, ...)`.
    # - En cada iteración: test del lote actual -> `addBatch!` -> reentrenamiento.
    # @note-body-end
    memory, batchList = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    model, _, _ = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    accuracies = zeros(Float64, length(batchList))
    for (i, batch) in enumerate(batchList)
        preds = predict(model, batchInputs(batch))
        accuracies[i] = mean(preds .== batchTargets(batch))
        addBatch!(memory, batch)
        model, _, _ = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    end
    return accuracies
    # @note-end
end;

# @note-start id=sism01 color=blue
# streamLearning_ISVM: simulación con SVM incremental (ISVM)
# Simula el procesamiento en flujo continuo mediante aprendizaje incremental con SVM.
# En lugar de mantener una memoria de datos brutos, se conservan los **vectores de soporte (SVs)**
# del entrenamiento previo, gestionando su **antigüedad (edad)**:
# - Cada nuevo lote incrementa la edad de los SVs existentes en la longitud de ese lote.
# - Se descartan los SVs cuya edad supere `windowSize`.
# - Se reentrena el SVM usando solo los SVs supervivientes más el nuevo lote.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `datasetFolder` | `String` | Carpeta del dataset. |
# | `windowSize` | `Int` | Antigüedad máxima permitida para los vectores de soporte. |
# | `batchSize` | `Int` | Tamaño de cada lote del flujo. |
# | `kernel` | `String` | Kernel del SVM (`"linear"`, `"rbf"`, `"poly"`, `"sigmoid"`). |
# | `C` | `Real` | Parámetro de penalización C. |
# | `degree`, `gamma`, `coef0` | `Real` | Hiperparámetros opcionales del kernel. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector{<:Real}` | Precisiones en test de cada lote en el momento de su llegada. |
#
# **Restricciones:** se permite un único bucle que itere sobre el vector de batches.
# @note-body-end
function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
# @note-end
    # @note-start id=sism02 color=yellow
    # Warm start y ciclo incremental con gestión de edad
    # 1. Warm start: se usa `initializeStreamLearningData` con tamaño `batchSize` para separar el primer lote.
    # 2. Las edades iniciales van de `batchSize` (más antiguo) a 1 (más reciente).
    # 3. En cada lote:
    #    - Test del modelo previo sobre el nuevo lote.
    #    - Sumar $K = \text{length(batch)}$ a las edades de los SVs.
    #    - Filtrar SVs con $\text{edad} \le \text{windowSize}$.
    #    - Entrenar nuevo SVM con el nuevo lote y los SVs filtrados.
    #    - Concatenar nuevos SVs y sus edades ($K \dots 1$).
    # @note-body-end
    firstBatch, batchList = initializeStreamLearningData(datasetFolder, batchSize, batchSize)
    model, supportVectors, (_, indicesSupportVectorsInFirstBatch) = trainSVM(firstBatch, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    initialAges = collect(batchSize:-1:1)
    svAges = initialAges[indicesSupportVectorsInFirstBatch]
    accuracies = zeros(Float64, length(batchList))
    for (i, batch) in enumerate(batchList)
        preds = predict(model, batchInputs(batch))
        accuracies[i] = mean(preds .== batchTargets(batch))
        K = batchLength(batch)
        svAges = svAges .+ K
        validIdx = findall(svAges .<= windowSize)
        supportVectors = selectInstances(supportVectors, validIdx)
        svAges = svAges[validIdx]
        model, _, (idxOld, idxNew) = trainSVM(batch, kernel, C; degree=degree, gamma=gamma, coef0=coef0, supportVectors=supportVectors)
        svOld = selectInstances(supportVectors, idxOld)
        svNew = selectInstances(batch, idxNew)
        supportVectors = joinBatches(svOld, svNew)
        agesOld = svAges[idxOld]
        batchAges = collect(K:-1:1)
        agesNew = batchAges[idxNew]
        svAges = vcat(agesOld, agesNew)
    end
    return accuracies
    # @note-end
end;

# @note-start id=ecds01 color=blue
# euclideanDistances: distancias euclídeas vectoriales
# Calcula la distancia euclídea entre una instancia dada y todas las instancias
# de un lote (`dataset`), implementado de forma completamente vectorial y sin bucles.
#
# $$
# d_i = \sqrt{\sum_{j=1}^D (X_{ij} - v_j)^2}
# $$
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `dataset` | `Batch` | Lote de datos con matriz de entradas $N \times D$. |
# | `instance` | `AbstractArray{<:Real,1}` | Vector $1$D de dimensión $D$ con la instancia a comparar. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector{<:Real}` | Vector de longitud $N$ con las distancias euclídeas a cada fila del dataset. |
#
# **Restricciones:** no se permite el uso de bucles.
# @note-body-end
function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
# @note-end
    # @note-start id=ecds02 color=yellow
    # Broadcasting vectorial con trasposición
    # `instance` es un vector columna ($D \times 1$). Al trasponerlo con `instance'` ($1 \times D$),
    # el broadcast `X .- instance'` resta la instancia a cada fila de $X$.
    # Luego elevamos al cuadrado `.^ 2`, sumamos por filas con `sum(..., dims=2)` ($N \times 1$),
    # aplicamos la raíz cuadrada `sqrt.` y aplanamos a vector con `vec`.
    # @note-body-end
    diff = batchInputs(dataset) .- instance'
    return vec(sqrt.(sum(diff .^ 2, dims=2)))
    # @note-end
end;

# @note-start id=nrel01 color=blue
# nearestElements: k vecinos más cercanos
# Devuelve un subconjunto (`Batch`) con las $k$ instancias de `dataset` más cercanas
# a la instancia especificada, utilizando sus distancias euclídeas.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `dataset` | `Batch` | Conjunto de datos de búsqueda. |
# | `instance` | `AbstractArray{<:Real,1}` | Vector con la instancia de consulta. |
# | `k` | `Int` | Número de vecinos más cercanos a seleccionar. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Batch` | Lote con las $k$ instancias más próximas y sus respectivas salidas. |
#
# **Restricciones:** no se permite el uso de bucles.
# @note-body-end
function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
# @note-end
    # @note-start id=nrel02 color=yellow
    # Selección rápida con partialsortperm
    # `partialsortperm(distances, 1:k)` devuelve los índices de las $k$ distancias mínimas
    # ordenadas de menor a mayor. Con esos índices, `selectInstances` devuelve el sub-Batch.
    # @note-body-end
    distances = euclideanDistances(dataset, instance)
    nearestIdx = partialsortperm(distances, 1:k)
    return selectInstances(dataset, nearestIdx)
    # @note-end
end;

# @note-start id=pknn01 color=blue
# predictKNN: predicción kNN para una instancia individual
# Realiza la clasificación kNN para una única instancia mediante votación mayoritaria
# de las clases de sus $k$ vecinos más cercanos.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `dataset` | `Batch` | Lote de datos de entrenamiento (memoria). |
# | `instance` | `AbstractArray{<:Real,1}` | Instancia a clasificar. |
# | `k` | `Int` | Número de vecinos para el voto. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | Coincide con salidas | Valor modal (la clase más frecuente entre los $k$ vecinos). |
#
# **Restricciones:** no se permite el uso de bucles.
# @note-body-end
function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
# @note-end
    # @note-start id=pknn02 color=yellow
    # Vecinos y moda
    # Se obtienen los $k$ vecinos más cercanos con `nearestElements`,
    # se extraen sus etiquetas con `batchTargets` y se devuelve la moda con `mode`.
    # @note-body-end
    nearest = nearestElements(dataset, instance, k)
    return mode(batchTargets(nearest))
    # @note-end
end;

# @note-start id=pknn03 color=blue
# predictKNN: predicción kNN para varias instancias
# Versión sobrecargada que recibe una matriz de instancias ($M \times D$) y predice
# la etiqueta para cada fila usando comprensión de listas sobre `eachrow`.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `dataset` | `Batch` | Lote de datos de entrenamiento. |
# | `instances` | `AbstractArray{<:Real,2}` | Matriz de instancias a clasificar ($M \times D$). |
# | `k` | `Int` | Número de vecinos para la predicción. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector` | Vector con la predicción para cada instancia (fila). |
#
# **Restricciones:** no se permite el uso de bucles (se usa comprensión de listas).
# @note-body-end
function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
# @note-end
    # @note-start id=pknn04 color=yellow
    # Comprensión sobre filas
    # `eachrow(instances)` itera fila a fila como vectores 1D, invocando
    # la versión para instancia individual de `predictKNN`.
    # @note-body-end
    return [predictKNN(dataset, row, k) for row in eachrow(instances)]
    # @note-end
end;

# @note-start id=slkn01 color=blue
# streamLearning_KNN: simulación con kNN y ventana deslizante
# Simula el flujo continuo con kNN. A diferencia de SVM, kNN es un aprendiz perezoso
# (*lazy learner*): no requiere fase de entrenamiento. La memoria actúa directamente
# como base de datos de consulta y se actualiza con `addBatch!` tras evaluar cada lote.
#
# **API pública**
#
# **Entradas**
#
# | Parámetro | Tipo | Descripción |
# |---|---|---|
# | `datasetFolder` | `String` | Carpeta del dataset. |
# | `windowSize` | `Int` | Tamaño de la memoria deslizante. |
# | `batchSize` | `Int` | Tamaño de cada lote del flujo. |
# | `k` | `Int` | Número de vecinos para kNN. |
#
# **Salida**
#
# | Tipo | Descripción |
# |---|---|
# | `Vector{<:Real}` | Precisiones en test de cada lote al llegar. |
#
# **Restricciones:** se permite un único bucle que itere sobre el vector de batches.
# @note-body-end
function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
# @note-end
    # @note-start id=slkn02 color=yellow
    # Evaluación y avance de memoria
    # En cada iteración:
    # 1. Se predice el lote actual usando la memoria con `predictKNN`.
    # 2. Se calcula la precisión media `mean(preds .== batchTargets(batch))`.
    # 3. Se añade el lote a la memoria deslizante con `addBatch!`.
    # @note-body-end
    memory, batchList = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    accuracies = zeros(Float64, length(batchList))
    for (i, batch) in enumerate(batchList)
        preds = predictKNN(memory, batchInputs(batch), k)
        accuracies[i] = mean(preds .== batchTargets(batch))
        addBatch!(memory, batch)
    end
    return accuracies
    # @note-end
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics: mean

abstract type ModelTree end

mutable struct ModelTreeNode <: ModelTree
    feature    :: Int
    threshold  :: Real
    model      :: Any
    leftChild  :: ModelTree
    rightChild :: ModelTree
end

struct ModelTreeLeaf <: ModelTree
    model      :: Any
    n          :: Int
end


# @note-start id=mse01 color=blue
# mse: error cuadrático medio (Mean Squared Error)
# Calcula el promedio de las diferencias al cuadrado entre las predicciones `outputs`
# y los valores reales `targets`:
#
# $$
# \text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
# $$
#
# **Entradas**
# - `outputs`: `Union{Real, AbstractVector{<:Real}}`, predicciones (escalar o vector).
# - `targets`: `AbstractVector{<:Real}`, valores reales deseadas.
#
# **Salida**
# - `Real`: valor numérico con el MSE, del mismo tipo que los datos.
#
# **Restricciones:** 0 bucles permitidos.
# @note-body-end
function mse(outputs::Union{Real,AbstractVector{<:Real}}, targets::AbstractVector{<:Real})
    return mean((outputs .- targets).^2)
end;

# @note-start id=ct01 color=blue
# candidateThresholds: generación de umbrales candidatos de división
# Calcula los puntos medios entre valores únicos consecutivos ordenados de un atributo `x`:
#
# $$
# t_i = \frac{u_i + u_{i+1}}{2}
# $$
#
# **Entradas**
# - `x`: `AbstractVector{<:Real}`, vector de valores de un atributo.
#
# **Salida**
# - `Vector{<:Real}`: vector ordenado con los umbrales candidatos, del mismo tipo que `x`.
#   Si hay 1 o 0 valores únicos, devuelve un vector vacío `eltype(x)[]`.
#
# **Restricciones:** 0 bucles permitidos.
# @note-body-end
function candidateThresholds(x::AbstractVector{<:Real})
    u = sort(unique(x))
    if length(u) <= 1
        return eltype(x)[]
    end
    return (u[1:end-1] .+ u[2:end]) ./ 2
end;

# @note-start id=sdm01 color=blue
# splitDataset: división de una matriz de datos por atributo y umbral
# Divide `inputs` en dos subconjuntos según $x_j \le t$ y $x_j > t$.
#
# **Entradas**
# - `inputs`: `AbstractMatrix{<:Real}`, matriz de características.
# - `numFeature`: `Int`, índice del atributo $j$.
# - `threshold`: `Real`, valor del umbral $t$.
#
# **Salida**
# - `Tuple`: `(leftInputs, rightInputs, mask)` donde `mask` es el vector booleano con los
#   índices del primer subconjunto.
#
# **Restricciones:** 0 bucles permitidos.
# @note-body-end
function splitDataset(inputs::AbstractMatrix{<:Real}, numFeature::Int, threshold::Real)
    mask = inputs[:, numFeature] .<= threshold
    return (inputs[mask, :], inputs[.!mask, :], mask)
end;

# @note-start id=sdb01 color=blue
# splitDataset: división de un Batch por atributo y umbral
# Divide `dataset` en dos subconjuntos de tipo `Batch` según $x_j \le t$ y $x_j > t$.
#
# **Entradas**
# - `dataset`: `Batch`, lote de datos a dividir.
# - `numFeature`: `Int`, índice del atributo $j$.
# - `threshold`: `Real`, valor del umbral $t$.
#
# **Salida**
# - `Tuple`: `(leftBatch, rightBatch, mask)`.
#
# **Restricciones:** 0 bucles permitidos.
# @note-body-end
function splitDataset(dataset::Batch, numFeature::Int, threshold::Real)
    _, _, mask = splitDataset(batchInputs(dataset), numFeature, threshold)
    return (selectInstances(dataset, mask), selectInstances(dataset, .!mask), mask)
end;

# @note-start id=ces01 color=blue
# calculateErrorSplit: cálculo del error ponderado tras una partición
# Divide el dataset y calcula la suma ponderada del MSE de cada subconjunto:
#
# $$
# E(j, t) = N_L \cdot \text{MSE}_L + N_R \cdot \text{MSE}_R
# $$
#
# **Entradas**
# - `dataset`: `Batch`, conjunto de entrenamiento.
# - `numFeature`: `Int`, índice del atributo $j$.
# - `threshold`: `Real`, umbral de división $t$.
#
# **Salida**
# - `Real`: error ponderado de la partición (del tipo del dataset).
#
# **Restricciones:** 0 bucles permitidos.
# @note-body-end
function calculateErrorSplit(dataset::Batch, numFeature::Int, threshold::Real)
    leftBatch, rightBatch, _ = splitDataset(dataset, numFeature, threshold)
    nLeft = batchLength(leftBatch)
    nRight = batchLength(rightBatch)
    if nLeft == 0 || nRight == 0
        return convert(eltype(batchInputs(dataset)), Inf)
    end
    errLeft = mse(mean(batchTargets(leftBatch)), batchTargets(leftBatch))
    errRight = mse(mean(batchTargets(rightBatch)), batchTargets(rightBatch))
    return convert(eltype(batchInputs(dataset)), nLeft * errLeft + nRight * errRight)
end;

# @note-start id=bs01 color=blue
# bestSplit: búsqueda exhaustiva de la mejor partición (atributo y umbral)
# Evalúa todos los atributos y sus umbrales candidatos minimizando el error de partición.
#
# **Entradas**
# - `dataset`: `Batch`, conjunto de entrenamiento.
#
# **Salida**
# - `Tuple{Int, Real}`: `(bestFeature, bestThreshold)`. Si no mejora el error o $N \le 1$, devuelve `(-1, NaN)`.
#
# **Restricciones:** máximo 2 bucles anidados permitidos (atributos y umbrales).
# @note-body-end
function bestSplit(dataset::Batch)
    n = batchLength(dataset)
    T = eltype(batchInputs(dataset))
    if n <= 1
        return (-1, convert(T, NaN))
    end

    bestError = n * mse(mean(batchTargets(dataset)), batchTargets(dataset))
    bestFeature = -1
    bestThreshold = convert(T, NaN)

    inputs = batchInputs(dataset)
    numFeatures = size(inputs, 2)

    for j in 1:numFeatures
        thresholds = candidateThresholds(inputs[:, j])
        for t in thresholds
            err = calculateErrorSplit(dataset, j, t)
            if err < bestError
                bestError = err
                bestFeature = j
                bestThreshold = t
            end
        end
    end

    return (bestFeature, bestThreshold)
end;

# @note-start id=brt01 color=blue
# buildRegressionTree: construcción recursiva de un árbol de regresión
# Construye un árbol de tipo `ModelTree` dividiendo recursivamente el dataset por `bestSplit`
# hasta cumplir criterios de parada (`max_depth == 0`, `n <= min_samples_leaf`, o no haber split de mejora).
#
# **Entradas**
# - `dataset`: `Batch`, conjunto de entrenamiento.
# - `max_depth`: `Int = -1`, profundidad máxima permitida.
# - `min_samples_leaf`: `Int = 1`, número mínimo de instancias por hoja.
#
# **Salida**
# - `ModelTree`: raíz del árbol (`ModelTreeNode` o `ModelTreeLeaf`).
#
# **Restricciones:** 0 bucles permitidos (algoritmo recursivo).
# @note-body-end
function buildRegressionTree(dataset::Batch;
    max_depth        :: Int = -1,
    min_samples_leaf :: Int =  1
)
    n = batchLength(dataset)
    meanTarget = mean(batchTargets(dataset))

    if n <= min_samples_leaf || max_depth == 0
        return ModelTreeLeaf(meanTarget, n)
    end

    feature, threshold = bestSplit(dataset)
    if feature == -1
        return ModelTreeLeaf(meanTarget, n)
    end

    leftDataset, rightDataset, _ = splitDataset(dataset, feature, threshold)
    leftChild = buildRegressionTree(leftDataset; max_depth = max_depth - 1, min_samples_leaf = min_samples_leaf)
    rightChild = buildRegressionTree(rightDataset; max_depth = max_depth - 1, min_samples_leaf = min_samples_leaf)

    return ModelTreeNode(feature, threshold, meanTarget, leftChild, rightChild)
end;

# @note-start id=prd01 color=blue
# predict: predicción con valor escalar
# Devuelve un vector constante con el valor `model` para cada fila de `inputs`.
# @note-body-end
predict(model::Real, inputs::AbstractMatrix{<:Real}) = fill(model, size(inputs, 1))

# @note-start id=prd02 color=blue
# predict: predicción en nodo hoja
# Delega la predicción en el modelo interno almacenado en la hoja `tree.model`.
# @note-body-end
predict(tree::ModelTreeLeaf, inputs::AbstractMatrix{<:Real}) = predict(tree.model, inputs)

# @note-start id=prd03 color=blue
# predict: predicción en nodo interno de decisión
# Divide las instancias por el umbral del nodo, predice recursivamente en cada subárbol
# y ensambla las predicciones vectorialmente en el orden original.
#
# **Restricciones:** 0 bucles permitidos.
# @note-body-end
function predict(tree::ModelTreeNode, inputs::AbstractMatrix{<:Real})
    leftInputs, rightInputs, mask = splitDataset(inputs, tree.feature, tree.threshold)
    if all(mask)
        return predict(tree.leftChild, inputs)
    elseif !any(mask)
        return predict(tree.rightChild, inputs)
    end
    leftPreds = predict(tree.leftChild, leftInputs)
    rightPreds = predict(tree.rightChild, rightInputs)
    preds = Vector{promote_type(eltype(leftPreds), eltype(rightPreds))}(undef, size(inputs, 1))
    preds[mask] = leftPreds
    preds[.!mask] = rightPreds
    return preds
end;

#=
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


EpsilonSVR = MLJ.@load EpsilonSVR pkg=LIBSVM verbosity=0
predict(model::Machine{MLJLIBSVMInterface.EpsilonSVR, MLJLIBSVMInterface.EpsilonSVR, true}, inputs::AbstractArray) = MLJ.predict(model, inputs)

using SymDoME
function predict(model::SymDoME.Tree, inputs::AbstractMatrix{<:Real})
    #
    # Codigo a desarrollar
    #
end;

LinearModel = Tuple{Real, AbstractArray{<:Real,1}}
function predict(model::LinearModel, inputs::AbstractMatrix{<:Real})
    #
    # Codigo a desarrollar
    #
end;


function trainRegressionModel(configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;

function trainRegressionModel(::Val{:constant}, configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;

function trainRegressionModel(::Val{:linear}, configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;

function trainRegressionModel(::Val{:DoME}, configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;

function trainRegressionModel(::Val{:SVR}, configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;



function calculateErrorSplit(dataset::Batch, numFeature::Int, threshold::Real, modelOutputs::AbstractVector{<:Real})
    #
    # Codigo a desarrollar
    #
end;



function bestSplit(dataset::Batch, modelOutputs::AbstractVector{<:Real})
    #
    # Codigo a desarrollar
    #
end;

function buildModelTree(configuration::Dict, dataset::Batch;
    max_depth        :: Int = -1,
    min_samples_leaf :: Int =  1
)
    #
    # Codigo a desarrollar
    #
end;

function trainRegressionModel(::Val{:ModelTree}, configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;



function numInstances(tree::ModelTreeLeaf)
    #
    # Codigo a desarrollar
    #
end;

function numInstances(tree::ModelTreeNode)
    #
    # Codigo a desarrollar
    #
end;

function predict(tree::ModelTreeLeaf, inputs::AbstractMatrix{<:Real}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predict(tree::ModelTreeNode, inputs::AbstractMatrix{<:Real}, k::Int)
    #
    # Codigo a desarrollar
    #
end;


function prune!(tree::ModelTreeLeaf, validationDataset::Batch)
    #
    # Codigo a desarrollar
    #
end;

function prune!(tree::ModelTreeNode, validationDataset::Batch)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


struct KNN
    dataset                 :: Batch
    k                       :: Int
    globalModel             :: Any # isnothing(globalModel) -> only local predictions (alpha is not used)
    localModelConfiguration :: Dict
    alpha                   :: AbstractFloat # isnan(alpha) -> residual correction
end


function trainRegressionModel(::Val{:KNN}, configuration::Dict, dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;


function predict(model::KNN, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;


function predict(model::KNN, inputs::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;
=#