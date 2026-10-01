

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles

# @note-start id=11dczj color=blue
# ## fileNamesFolder: nombres de archivos con una extensión
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
    # ## Carpeta inexistente: devolver lista vacía
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
    # ## Normalizar la extensión: sin punto y en mayúsculas
    # `condición ? si_true : si_false` es el operador ternario.
    #
    # Se comprueba si la extensión empieza por punto. Si es así, se devuelve
    # el *slice* que empieza en el índice 2 (Julia indexa desde 1, así que el
    # índice 1 sería el `"."` si alguien escribió `".tsv"` por error en lugar
    # de `"tsv"`). Si no empieza por punto, se usa tal cual. En ambos casos
    # el resultado se pasa a mayúsculas con `uppercase`.
    # @note-body-end
    ext = uppercase(startswith(extension, ".") ? extension[2:end] : extension)
    # @note-end
    # @note-start id=srhgmg color=yellow
    # ## filter: quedarse con los archivos de esa extensión
    # `filter(f, itr)` tiene dos parámetros, una función y un iterable:
    #
    # - **El iterable** es `readdir(folderName)`, que devuelve una lista con
    #   los nombres de todos los archivos de la carpeta.
    # - **La función** es la anónima `f -> endswith(uppercase(f), ".$ext")`,
    #   que devuelve `true` si el nombre (en mayúsculas) termina en `.EXT`.
    #   `filter` se queda solo con los elementos para los que es `true`.
    # @note-body-end
    files = filter(f -> endswith(uppercase(f), ".$ext"), readdir(folderName))
    # @note-end.
    return String.(map(f -> f[1:end-(length(ext)+1)], files))
end;



function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float64)
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
# **Errores:** `BoundsError` si `data` tiene menos de 2 valores distintos
# (`differences` queda vacío).
# @note-body-end
function intervalDiscreteVector(data::AbstractArray{<:Real,1})
# @note-end
    # @note-start id=lyeyt4 color=yellow
    # ## Ordenar los datos sin repeticiones
    # - `unique`: devuelve un array con los elementos únicos, sin repeticiones.
    # - `sort`: ordena los elementos en orden ascendente (por defecto, de
    #   menor a mayor).
    # @note-body-end
    uniqueData = sort(unique(data));
    # @note-end
    # @note-start id=17mjga color=yellow
    # ## Diferencias entre elementos consecutivos
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
    # ## Tomar la diferencia menor
    # Equivale a `minimum(differences)`: como `differences` ya está ordenado,
    # su primer elemento es exactamente la diferencia menor.
    # @note-body-end
    minDifference = differences[1];
    # @note-end

    # @note-start id=ceuenb color=yellow
    # ## isInteger: ¿x es entero salvo una tolerancia?
    # Devuelve `true` si `x` es entero con tolerancia `tol`.
    #
    # Se usa para el criterio: si todas las diferencias son múltiplos exactos
    # (valores enteros) de la diferencia menor, el vector es de valores
    # discretos.
    # @note-body-end
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    # @note-end
    # @note-start id=gduorn color=yellow
    # ## ¿Discreto o continuo? Devolver m
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
using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
predict(model::Machine{MLJLIBSVMInterface.SVC, MLJLIBSVMInterface.SVC, true}, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return convert(Vector, levels(outputs)[int(outputs)]); )

using Base.Iterators
using StatsBase
using Random: shuffle

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    return batch[1]
end;

function batchTargets(batch::Batch)
    return batch[2]
end;

function batchLength(batch::Batch)
    return size(batchInputs(batch), 1)
end;

function selectInstances(batch::Batch, indices::Any)
    idx = isa(indices, Integer) ? [indices] : indices
    return (batchInputs(batch)[idx, :], batchTargets(batch)[idx])
end;

function joinBatches(batch1::Batch, batch2::Batch)
    return (vcat(batchInputs(batch1), batchInputs(batch2)), vcat(batchTargets(batch1), batchTargets(batch2)))
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    N = batchLength(dataset)
    indices = shuffleRows ? shuffle(1:N) : (1:N)
    return [selectInstances(dataset, idx) for idx in Base.Iterators.partition(indices, batchSize)]
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    
    N = batchLength(supportVectors)
    trainingData = joinBatches(supportVectors, dataset)

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

    mach = MLJ.machine(model,  
        MLJ.table(batchInputs(trainingData)),  
        MLJ.categorical(batchTargets(trainingData) ; levels = [false, true]))
    MLJ.fit!(mach, verbosity=0)

    indicesNewSupportVectors = sort( mach.fitresult[1].SVs.indices )

    svIndices_old = indicesNewSupportVectors[indicesNewSupportVectors .<= N]
    svIndices_new = indicesNewSupportVectors[indicesNewSupportVectors .> N] .- N

    newSupportVectors = joinBatches(selectInstances(supportVectors, svIndices_old), selectInstances(dataset, svIndices_new))

    return (mach, newSupportVectors, (svIndices_old, svIndices_new))
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    firstBatch = batches[1]
    local svs = ( Array{eltype(firstBatch[1]),2}(undef,0,size(firstBatch[1],2)) , Array{eltype(firstBatch[2]),1}(undef,0) )
    local model
    for batch in batches
        model, svs, _ = trainSVM(batch, kernel, C; degree=degree, gamma=gamma, coef0=coef0, supportVectors=svs)
    end
    return model
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# @note-start id=isld01 color=blue
# ## initializeStreamLearningData: inicialización de la memoria y flujo de datos
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
    # ## Carga y partición cronológica
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
# ## addBatch!: actualización FIFO in-place de la memoria
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
    # ## Desplazamiento e inserción in-place
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
# ## streamLearning_SVM: simulación con SVM clásico y ventana deslizante
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
    # ## Inicialización, entrenamiento inicial y ciclo de flujo
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
# ## streamLearning_ISVM: simulación con SVM incremental (ISVM)
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
    # ## Warm start y ciclo incremental con gestión de edad
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
# ## euclideanDistances: cálculo vectorial de distancias euclídeas
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
    # ## Broadcasting vectorial con trasposición
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
# ## nearestElements: k vecinos más cercanos
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
    # ## Selección rápida con partialsortperm
    # `partialsortperm(distances, 1:k)` devuelve los índices de las $k$ distancias mínimas
    # ordenadas de menor a mayor. Con esos índices, `selectInstances` devuelve el sub-Batch.
    # @note-body-end
    distances = euclideanDistances(dataset, instance)
    nearestIdx = partialsortperm(distances, 1:k)
    return selectInstances(dataset, nearestIdx)
    # @note-end
end;

# @note-start id=pknn01 color=blue
# ## predictKNN: predicción kNN para una instancia individual
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
    # ## Vecinos y moda
    # Se obtienen los $k$ vecinos más cercanos con `nearestElements`,
    # se extraen sus etiquetas con `batchTargets` y se devuelve la moda con `mode`.
    # @note-body-end
    nearest = nearestElements(dataset, instance, k)
    return mode(batchTargets(nearest))
    # @note-end
end;

# @note-start id=pknn03 color=blue
# ## predictKNN: predicción kNN para múltiples instancias (sobrecargada)
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
    # ## Comprensión sobre filas
    # `eachrow(instances)` itera fila a fila como vectores 1D, invocando
    # la versión para instancia individual de `predictKNN`.
    # @note-body-end
    return [predictKNN(dataset, row, k) for row in eachrow(instances)]
    # @note-end
end;

# @note-start id=slkn01 color=blue
# ## streamLearning_KNN: simulación con kNN y ventana deslizante
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
    # ## Evaluación y avance de memoria
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


#=
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


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


function mse(outputs::Union{Real,AbstractVector{<:Real}}, targets::AbstractVector{<:Real})
    #
    # Codigo a desarrollar
    #
end;
  
function candidateThresholds(x::AbstractVector{<:Real})
    #
    # Codigo a desarrollar
    #
end;

function splitDataset(dataset::Batch, numFeature::Int, threshold::Real)
    #
    # Codigo a desarrollar
    #
end

function calculateErrorSplit(dataset::Batch, numFeature::Int, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;

function bestSplit(dataset::Batch)
    #
    # Codigo a desarrollar
    #
end;


function buildRegressionTree(dataset::Batch;
    max_depth        :: Int = -1,
    min_samples_leaf :: Int =  1
)
    #
    # Codigo a desarrollar
    #
end;

function predict(model::Real, inputs::AbstractMatrix{<:Real})
    #
    # Codigo a desarrollar
    #
end;

function predict(tree::ModelTreeLeaf, inputs::AbstractMatrix{<:Real})
    #
    # Codigo a desarrollar
    #
end;

function predict(tree::ModelTreeNode, inputs::AbstractMatrix{<:Real})
    #
    # Codigo a desarrollar
    #
end;



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