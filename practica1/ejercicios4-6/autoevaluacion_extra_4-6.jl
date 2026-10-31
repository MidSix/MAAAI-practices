# ----------------------------------------------------------------------------------------------
# Extra self-assessment for exercises 4-6: edge cases taken from the assignment statements
# (4 - Arboles de Regresion.pdf, 5 - Arboles de Modelos.pdf, 6 - Modelos hibridos.pdf).
#
# Unlike autoevaluacion_4-6.jl, a failing check does not stop the script: every check is run,
# and a score per exercise is printed at the end. Expected values are computed by hand from
# the statement, never from the implementation under test.
#
# Run it from any folder, in a fresh Julia session:
#     julia autoevaluacion_extra_4-6.jl
# ----------------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "firmas_modificado.jl"));

using Random: seed!
using Statistics: mean


# ----------------------------------------------------------------------------------------------
# Test harness
# ----------------------------------------------------------------------------------------------

# Each entry: (exercise, description, scored, status, message)
testResults = NamedTuple[]

# Runs one check. The do-block must return `true` to pass; any other value is a failure and any
# exception is reported as an error. `scored = false` marks robustness checks that are not
# required by the statement and do not count towards the score.
function runTest(check::Function, exercise::Int, description::String; scored::Bool = true)
    status, message = try
        check() === true ? (:pass, "") : (:fail, "la comprobación ha devuelto false")
    catch err
        (:error, first(Base.split(sprint(showerror, err), '\n')))
    end
    push!(testResults, (exercise = exercise, description = description, scored = scored,
                        status = status, message = message))
    mark = status == :pass ? "OK   " : status == :fail ? "FALLO" : "ERROR"
    println("  [", mark, "] ", description, isempty(message) ? "" : "  -> " * message)
    return nothing
end

approxVec(a, b; atol = 1e-8) = length(a) == length(b) && all(isapprox.(a, b; atol = atol))

# Two bestSplit results are equal if they have the same feature and the same threshold
# (NaN == NaN is false in Julia, so the "no split" case is handled separately).
sameSplit(a, b) = a[1] == b[1] && (a[1] == -1 ? (isnan(a[2]) && isnan(b[2])) : a[2] == b[2])

# Structural comparison of two trees: same splits, same node/leaf models and same leaf sizes.
sameTree(a::ModelTreeLeaf, b::ModelTreeLeaf) = isapprox(a.model, b.model) && a.n == b.n
sameTree(a::ModelTreeNode, b::ModelTreeNode) =
    a.feature == b.feature && a.threshold == b.threshold && isapprox(a.model, b.model) &&
    sameTree(a.leftChild, b.leftChild) && sameTree(a.rightChild, b.rightChild)
sameTree(::Any, ::Any) = false

# Depth = number of splits from the root to the deepest leaf (a single leaf has depth 0).
treeDepth(::ModelTreeLeaf) = 0
treeDepth(tree::ModelTreeNode) = 1 + max(treeDepth(tree.leftChild), treeDepth(tree.rightChild))

# Number of training instances, counted from the leaves (independent of numInstances).
leafInstances(tree::ModelTreeLeaf) = tree.n
leafInstances(tree::ModelTreeNode) = leafInstances(tree.leftChild) + leafInstances(tree.rightChild)

# Smallest number of instances among the internal nodes of a tree (Inf if there are none).
minNodeInstances(::ModelTreeLeaf) = Inf
minNodeInstances(tree::ModelTreeNode) =
    min(leafInstances(tree), minNodeInstances(tree.leftChild), minNodeInstances(tree.rightChild))

# Small hand-made datasets
toyX = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
toyDataset = (toyX, [1.0, 1.0, 5.0, 5.0])        # perfect split at x <= 2.5

seed!(7)
randomX = rand(40, 3)
randomDataset = (randomX, rand(40))              # distinct inputs, arbitrary targets

knnX = reshape([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], 6, 1)
knnDataset = (knnX, [1.0, 1.0, 1.0, 5.0, 5.0, 5.0])   # two well separated groups

# Real dataset, as in autoevaluacion_4-6.jl
swdDataset = loadDataset("SWD", joinpath(@__DIR__, "datasets/"); datasetType = Float64)
swdSmall = selectInstances(swdDataset, 1:100)


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

println("\nEjercicio 4: árboles de regresión")

runTest(4, "mse: vector de salidas, (0² + 0² + 2²)/3 = 4/3") do
    isapprox(mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]), 4 / 3)
end
runTest(4, "mse: salida escalar igual que repetirla en un vector") do
    isapprox(mse(2.0, [1.0, 2.0, 3.0]), 2 / 3) && mse(2.0, [1.0, 2.0, 3.0]) == mse(fill(2.0, 3), [1.0, 2.0, 3.0])
end
runTest(4, "mse: predicción perfecta da 0") do
    mse([0.3, -1.2, 7.0], [0.3, -1.2, 7.0]) == 0
end
runTest(4, "mse: con datos Float32 devuelve Float32") do
    isa(mse(Float32[1, 2], Float32[1, 3]), Float32) && isa(mse(1.0f0, Float32[1, 3]), Float32)
end

runTest(4, "candidateThresholds: entrada desordenada y con repetidos") do
    candidateThresholds([3.0, 1.0, 2.0, 1.0, 3.0]) == [1.5, 2.5]
end
runTest(4, "candidateThresholds: un único valor distinto da vector vacío del mismo tipo") do
    r64 = candidateThresholds([2.0, 2.0, 2.0])
    r32 = candidateThresholds(Float32[7, 7])
    isempty(r64) && eltype(r64) == Float64 && isempty(r32) && eltype(r32) == Float32
end
runTest(4, "candidateThresholds: con Float32 devuelve Float32") do
    r = candidateThresholds(Float32[4, 1, 2])
    eltype(r) == Float32 && r == Float32[1.5, 3.0]
end
runTest(4, "candidateThresholds: valores negativos") do
    candidateThresholds([1.0, -1.0, -3.0]) == [-2.0, 0.0]
end

runTest(4, "splitDataset (matriz): x_j == t va a la izquierda y se mantiene el orden") do
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 2.0 40.0]
    left, right, mask = splitDataset(X, 1, 2.0)
    left == [1.0 10.0; 2.0 20.0; 2.0 40.0] && right == [3.0 30.0] && mask == [true, true, false, true]
end
runTest(4, "splitDataset (matriz): conserva el tipo y la máscara es booleana") do
    X = Float32[1 10; 2 20; 3 30]
    left, right, mask = splitDataset(X, 2, 25)
    isa(left, Matrix{Float32}) && isa(right, Matrix{Float32}) && eltype(mask) == Bool && length(mask) == 3
end
runTest(4, "splitDataset (matriz): todo a un lado deja el otro con 0 filas") do
    X = [1.0 10.0; 2.0 20.0]
    left, right, mask = splitDataset(X, 1, 100.0)
    left == X && size(right) == (0, 2) && all(mask)
end
runTest(4, "splitDataset (Batch): las salidas se reparten igual que las entradas") do
    X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 2.0 40.0]
    leftBatch, rightBatch, mask = splitDataset((X, [1.0, 2.0, 3.0, 4.0]), 1, 2.0)
    batchTargets(leftBatch) == [1.0, 2.0, 4.0] && batchTargets(rightBatch) == [3.0] &&
        batchInputs(leftBatch) == X[[1, 2, 4], :] && mask == [true, true, false, true]
end

runTest(4, "calculateErrorSplit: partición perfecta da 0") do
    calculateErrorSplit(toyDataset, 1, 2.5) == 0
end
runTest(4, "calculateErrorSplit: calculado a mano, 1·0 + 3·(32/9) = 32/3") do
    isapprox(calculateErrorSplit(toyDataset, 1, 1.5), 32 / 3)
end
runTest(4, "calculateErrorSplit: con datos Float32 devuelve Float32") do
    ds32 = (Float32.(toyX), Float32[1, 1, 5, 5])
    isa(calculateErrorSplit(ds32, 1, 1.5f0), Float32)
end

runTest(4, "bestSplit: una sola instancia devuelve (-1, NaN) con tipos Int y Float64") do
    r = bestSplit((reshape([1.0], 1, 1), [3.0]))
    isa(r, Tuple) && r[1] == -1 && isa(r[1], Int) && isnan(r[2]) && isa(r[2], Float64)
end
runTest(4, "bestSplit: una sola instancia Float32 devuelve NaN de tipo Float32") do
    r = bestSplit((reshape(Float32[1], 1, 1), Float32[3]))
    r[1] == -1 && isnan(r[2]) && isa(r[2], Float32)
end
runTest(4, "bestSplit: salidas constantes, ninguna partición mejora: (-1, NaN)") do
    r = bestSplit((reshape([1.0, 2.0, 3.0], 3, 1), [4.0, 4.0, 4.0]))
    r[1] == -1 && isnan(r[2])
end
runTest(4, "bestSplit: entradas idénticas, no hay umbrales: (-1, NaN)") do
    r = bestSplit((ones(3, 2), [1.0, 2.0, 3.0]))
    r[1] == -1 && isnan(r[2])
end
runTest(4, "bestSplit: elige el atributo que separa perfectamente (atributo 2, t = 2.5)") do
    X = [5.0 1.0; 3.0 2.0; 4.0 3.0; 1.0 4.0]
    r = bestSplit((X, [0.0, 0.0, 10.0, 10.0]))
    r == (2, 2.5) && isa(r, Tuple{Int, Float64})
end

runTest(4, "buildRegressionTree: max_depth = 0 da una hoja con la media y n") do
    t = buildRegressionTree(toyDataset; max_depth = 0)
    isa(t, ModelTreeLeaf) && t.model == 3.0 && t.n == 4
end
runTest(4, "buildRegressionTree: min_samples_leaf >= n da una hoja") do
    t = buildRegressionTree(toyDataset; min_samples_leaf = 4)
    isa(t, ModelTreeLeaf) && t.n == 4
end
runTest(4, "buildRegressionTree: max_depth = 1 da un tocón con las hojas correctas") do
    t = buildRegressionTree(toyDataset; max_depth = 1)
    isa(t, ModelTreeNode) && t.feature == 1 && t.threshold == 2.5 && t.model == 3.0 &&
        isa(t.leftChild, ModelTreeLeaf) && t.leftChild.model == 1.0 && t.leftChild.n == 2 &&
        isa(t.rightChild, ModelTreeLeaf) && t.rightChild.model == 5.0 && t.rightChild.n == 2
end
runTest(4, "buildRegressionTree: sin límites, con entradas distintas, ajusta el entrenamiento") do
    t = buildRegressionTree(randomDataset)
    approxVec(predict(t, randomX), batchTargets(randomDataset))
end
runTest(4, "buildRegressionTree: respeta max_depth = 2") do
    treeDepth(buildRegressionTree(randomDataset; max_depth = 2)) <= 2
end
runTest(4, "buildRegressionTree: solo se divide un nodo con más de min_samples_leaf instancias") do
    t = buildRegressionTree(randomDataset; min_samples_leaf = 5)
    minNodeInstances(t) > 5 && leafInstances(t) == 40
end
runTest(4, "buildRegressionTree: el modelo de la raíz es la media de todas las salidas") do
    isapprox(buildRegressionTree(randomDataset).model, mean(batchTargets(randomDataset)))
end
runTest(4, "buildRegressionTree: entradas repetidas con salidas distintas acaban en una hoja con la media") do
    t = buildRegressionTree((reshape([1.0, 1.0, 2.0], 3, 1), [0.0, 1.0, 5.0]))
    predict(t, reshape([1.0, 2.0], 2, 1)) == [0.5, 5.0]
end

runTest(4, "predict (Real): 0 filas devuelve un vector vacío") do
    p = predict(3.8, zeros(0, 3))
    isa(p, AbstractVector) && length(p) == 0
end
runTest(4, "predict (Real): mantiene el tipo del escalar") do
    eltype(predict(2.0f0, rand(3, 2))) == Float32
end
runTest(4, "predict (nodo): x_j == t va al hijo izquierdo y se mantiene el orden") do
    t = ModelTreeNode(1, 0.5, 0.0, ModelTreeLeaf(10.0, 1), ModelTreeLeaf(20.0, 1))
    predict(t, reshape([0.5, 0.6, 0.4, 0.7], 4, 1)) == [10.0, 20.0, 10.0, 20.0]
end
runTest(4, "predict (nodo): árbol de dos niveles hecho a mano") do
    inner = ModelTreeNode(1, 5.0, 0.0, ModelTreeLeaf(1.0, 1), ModelTreeLeaf(2.0, 1))
    t = ModelTreeNode(2, 0.0, 0.0, ModelTreeLeaf(-1.0, 1), inner)
    predict(t, [9.0 -1.0; 3.0 1.0; 7.0 1.0]) == [-1.0, 1.0, 2.0]
end
runTest(4, "predict (nodo): 0 filas devuelve un vector vacío") do
    t = ModelTreeNode(1, 0.5, 0.0, ModelTreeLeaf(10.0, 1), ModelTreeLeaf(20.0, 1))
    length(predict(t, zeros(0, 1))) == 0
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

println("\nEjercicio 5: árboles de modelos")

runTest(5, "predict (LinearModel): X·w + b calculado a mano") do
    predict((1.0, [2.0, 3.0]), [1.0 1.0; 0.0 0.0; 2.0 -1.0]) == [6.0, 1.0, 2.0]
end
runTest(5, "predict (DoME): expresión constante devuelve un vector, no un escalar") do
    ds = (rand(10, 2), fill(3.0, 10))
    m = trainRegressionModel(Dict("modelType" => :DoME, "maximumNodes" => 5), ds)
    p = predict(m, rand(4, 2))
    isa(p, AbstractVector) && length(p) == 4 && all(isapprox.(p, 3.0))
end

runTest(5, "trainRegressionModel (:constant): media de las salidas, del mismo tipo") do
    m = trainRegressionModel(Dict("modelType" => :constant), (Float32.(toyX), Float32[1, 1, 5, 5]))
    isa(m, Float32) && m == 3.0f0
end
runTest(5, "trainRegressionModel (:linear): recupera y = 2 + 3·x1 - x2") do
    seed!(3); X = rand(20, 2)
    b, w = trainRegressionModel(Dict("modelType" => :linear), (X, 2 .+ 3 .* X[:, 1] .- X[:, 2]))
    isapprox(b, 2.0; atol = 1e-8) && approxVec(w, [3.0, -1.0])
end
runTest(5, "trainRegressionModel (:linear): devuelve un LinearModel con el tipo de los datos") do
    seed!(3); X = rand(Float32, 20, 2)
    m = trainRegressionModel(Dict("modelType" => :linear), (X, rand(Float32, 20)))
    isa(m, LinearModel) && isa(m[1], Float32) && eltype(m[2]) == Float32 && length(m[2]) == 2
end
runTest(5, "trainRegressionModel (:SVR): sin hiperparámetros usa los valores por defecto") do
    m = trainRegressionModel(Dict("modelType" => :SVR), randomDataset)
    svr = m.model
    svr.kernel == LIBSVM.Kernel.RadialBasis && svr.epsilon == 0.1 && svr.cost == 1.0 &&
        isapprox(svr.gamma, 1 / 3) && svr.degree == 3 && svr.coef0 == 0.0 &&
        length(predict(m, randomX)) == 40
end
runTest(5, "trainRegressionModel (:SVR): \"linear\" y \"sigmoid\" usan su kernel de LIBSVM") do
    lin = trainRegressionModel(Dict("modelType" => :SVR, "kernel" => "linear"), randomDataset)
    sig = trainRegressionModel(Dict("modelType" => :SVR, "kernel" => "sigmoid"), randomDataset)
    lin.model.kernel == LIBSVM.Kernel.Linear && sig.model.kernel == LIBSVM.Kernel.Sigmoid
end
runTest(5, "trainRegressionModel: un modelType que no existe da MethodError") do
    try
        trainRegressionModel(Dict("modelType" => :notAModel), toyDataset)
        false
    catch err
        isa(err, MethodError)
    end
end

runTest(5, "calculateErrorSplit (modelOutputs): salidas constantes usan la versión del ej. 4") do
    calculateErrorSplit(toyDataset, 1, 1.5, fill(100.0, 4)) == calculateErrorSplit(toyDataset, 1, 1.5)
end
runTest(5, "calculateErrorSplit (modelOutputs): salidas perfectas dan 0") do
    calculateErrorSplit(toyDataset, 1, 1.5, [1.0, 1.0, 5.0, 5.0]) == 0
end
runTest(5, "calculateErrorSplit (modelOutputs): calculado a mano, 2·0.5 + 2·0.5 = 2") do
    isapprox(calculateErrorSplit(toyDataset, 1, 2.5, [1.0, 2.0, 5.0, 6.0]), 2.0)
end
runTest(5, "calculateErrorSplit (modelOutputs): con datos Float32 devuelve Float32") do
    ds32 = (Float32.(toyX), Float32[1, 1, 5, 5])
    isa(calculateErrorSplit(ds32, 1, 2.5f0, Float32[1, 2, 5, 6]), Float32)
end

runTest(5, "bestSplit (modelOutputs): una sola instancia devuelve (-1, NaN)") do
    r = bestSplit((reshape([1.0], 1, 1), [3.0]), [2.0])
    r[1] == -1 && isnan(r[2])
end
runTest(5, "bestSplit (modelOutputs): salidas perfectas, nada mejora: (-1, NaN)") do
    r = bestSplit(toyDataset, [1.0, 1.0, 5.0, 5.0])
    r[1] == -1 && isnan(r[2])
end
runTest(5, "bestSplit (modelOutputs): con la media como salida coincide con el ej. 4") do
    y = batchTargets(randomDataset)
    sameSplit(bestSplit(randomDataset, fill(mean(y), length(y))), bestSplit(randomDataset))
end

runTest(5, "buildModelTree (:constant) construye el mismo árbol que buildRegressionTree") do
    c = Dict("modelType" => :constant)
    sameTree(buildModelTree(c, randomDataset), buildRegressionTree(randomDataset)) &&
        sameTree(buildModelTree(c, randomDataset; max_depth = 2, min_samples_leaf = 6),
                 buildRegressionTree(randomDataset; max_depth = 2, min_samples_leaf = 6))
end
runTest(5, "trainRegressionModel (:ModelTree): sin max_depth ni min_samples_leaf usa -1 y 1") do
    m = trainRegressionModel(Dict("modelType" => :ModelTree,
                                  "localModelConfiguration" => Dict("modelType" => :constant)), randomDataset)
    sameTree(m, buildRegressionTree(randomDataset))
end
runTest(5, "trainRegressionModel (:ModelTree): pasa max_depth y min_samples_leaf a buildModelTree") do
    m = trainRegressionModel(Dict("modelType" => :ModelTree, "max_depth" => 1, "min_samples_leaf" => 3,
                                  "localModelConfiguration" => Dict("modelType" => :constant)), randomDataset)
    sameTree(m, buildRegressionTree(randomDataset; max_depth = 1, min_samples_leaf = 3))
end
runTest(5, "trainRegressionModel (:ModelTree): las hojas usan la configuración local (:linear)") do
    m = trainRegressionModel(Dict("modelType" => :ModelTree, "max_depth" => 0,
                                  "localModelConfiguration" => Dict("modelType" => :linear)), randomDataset)
    isa(m, ModelTreeLeaf) && isa(m.model, LinearModel) && m.n == 40
end

runTest(5, "numInstances: árbol hecho a mano, 3 + (2 + 4) = 9, de tipo Int") do
    t = ModelTreeNode(1, 0.5, 0.0, ModelTreeLeaf(0.0, 3),
                      ModelTreeNode(1, 0.7, 0.0, ModelTreeLeaf(0.0, 2), ModelTreeLeaf(0.0, 4)))
    numInstances(t) == 9 && isa(numInstances(t), Int) && numInstances(t.leftChild) == 3
end
runTest(5, "numInstances: en un árbol entrenado es el número de instancias del dataset") do
    numInstances(buildRegressionTree(randomDataset; min_samples_leaf = 4)) == 40
end

smoothTree = ModelTreeNode(1, 0.5, 0.0, ModelTreeLeaf(10.0, 3), ModelTreeLeaf(20.0, 1))
runTest(5, "predict con k: k = 0 y k < 0 dan lo mismo que sin suavizado") do
    t = buildRegressionTree(randomDataset; max_depth = 3)
    predict(t, randomX, 0) == predict(t, randomX) && predict(t, randomX, -3) == predict(t, randomX)
end
runTest(5, "predict con k: en una hoja no hay suavizado") do
    leaf = ModelTreeLeaf(4.0, 7)
    predict(leaf, rand(3, 2), 5) == predict(leaf, rand(3, 2))
end
runTest(5, "predict con k: (n·p + k·q)/(n + k) calculado a mano") do
    # x = 0.4: (3·10 + 1·0)/(3 + 1) = 7.5;  x = 0.6: (1·20 + 1·0)/(1 + 1) = 10
    approxVec(predict(smoothTree, reshape([0.4, 0.6], 2, 1), 1), [7.5, 10.0])
end
runTest(5, "predict con k: suavizado en dos niveles calculado a mano") do
    inner = ModelTreeNode(1, 0.25, 4.0, ModelTreeLeaf(2.0, 1), ModelTreeLeaf(6.0, 3))
    t = ModelTreeNode(1, 0.5, 0.0, inner, ModelTreeLeaf(8.0, 4))
    # x = 0.1: inner = (1·2 + 2·4)/3 = 10/3, root = (4·10/3 + 2·0)/6 = 20/9;  x = 0.9: (4·8)/6 = 16/3
    approxVec(predict(t, reshape([0.1, 0.9], 2, 1), 2), [20 / 9, 16 / 3])
end
runTest(5, "predict con k: k muy grande tiende al modelo del nodo") do
    approxVec(predict(smoothTree, reshape([0.4, 0.6], 2, 1), 10^9), [0.0, 0.0]; atol = 1e-6)
end
runTest(5, "predict con k: 0 filas devuelve un vector vacío") do
    length(predict(smoothTree, zeros(0, 1), 2)) == 0
end

pruneTree() = ModelTreeNode(1, 0.5, 5.0, ModelTreeLeaf(0.0, 2), ModelTreeLeaf(10.0, 3))
runTest(5, "prune!: una hoja no se poda, se devuelve la misma") do
    leaf = ModelTreeLeaf(1.0, 3)
    prune!(leaf, toyDataset) === leaf
end
runTest(5, "prune!: con el conjunto de validación vacío se devuelve el mismo nodo") do
    t = pruneTree()
    prune!(t, (zeros(0, 1), Float64[])) === t && isa(t.leftChild, ModelTreeLeaf) && t.leftChild.model == 0.0
end
runTest(5, "prune!: si el modelo del nodo es mejor, se sustituye por una hoja") do
    # validation y = [5, 5]: node error 25, node model (5.0) error 0
    r = prune!(pruneTree(), (reshape([0.4, 0.6], 2, 1), [5.0, 5.0]))
    isa(r, ModelTreeLeaf) && r.model == 5.0 && r.n == 5
end
runTest(5, "prune!: con errores iguales también se poda (\"inferior o igual\")") do
    # validation y = [2.5]: node predicts 0 (error 6.25), node model predicts 5 (error 6.25)
    isa(prune!(pruneTree(), (reshape([0.4], 1, 1), [2.5])), ModelTreeLeaf)
end
runTest(5, "prune!: si el árbol es mejor, se devuelve el mismo nodo") do
    t = pruneTree()
    prune!(t, (reshape([0.4, 0.6], 2, 1), [0.0, 10.0])) === t
end
runTest(5, "prune!: modifica el árbol recibido (poda un hijo y conserva la raíz)") do
    inner = ModelTreeNode(1, 0.25, 1.0, ModelTreeLeaf(0.0, 1), ModelTreeLeaf(2.0, 1))
    root = ModelTreeNode(1, 0.5, 100.0, inner, ModelTreeLeaf(50.0, 2))
    r = prune!(root, (reshape([0.2, 0.3, 0.9], 3, 1), [1.0, 1.0, 50.0]))
    r === root && isa(root.leftChild, ModelTreeLeaf) && root.leftChild.model == 1.0 && root.leftChild.n == 2
end
runTest(5, "prune!: un hijo sin datos de validación se queda como está") do
    right = ModelTreeNode(1, 0.75, 20.0, ModelTreeLeaf(10.0, 1), ModelTreeLeaf(30.0, 1))
    root = ModelTreeNode(1, 0.5, 5.0, ModelTreeLeaf(0.0, 1), right)
    r = prune!(root, (reshape([0.1], 1, 1), [0.0]))
    r === root && root.rightChild === right
end


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

println("\nEjercicio 6: modelos híbridos (kNN)")

knnConfig(extra...) = Dict{String, Any}("modelType" => :KNN, extra...)

runTest(6, "trainRegressionModel (:KNN): valores por defecto (k = 5, sin global, :constant, α = NaN)") do
    m = trainRegressionModel(knnConfig(), knnDataset)
    isa(m, KNN) && m.k == 5 && isnothing(m.globalModel) &&
        m.localModelConfiguration["modelType"] == :constant && isnan(m.alpha) &&
        batchInputs(m.dataset) == knnX
end
runTest(6, "trainRegressionModel (:KNN): guarda k y α, y entrena el modelo global") do
    m = trainRegressionModel(knnConfig("k" => 3, "alpha" => 0.2,
                                       "globalModelConfiguration" => Dict("modelType" => :constant)), knnDataset)
    m.k == 3 && m.alpha == 0.2 && isapprox(m.globalModel, 3.0)
end
runTest(6, "trainRegressionModel (:KNN): α entero (\"alpha\" => 1) se guarda como AbstractFloat") do
    m = trainRegressionModel(knnConfig("alpha" => 1), knnDataset)
    isa(m.alpha, AbstractFloat) && m.alpha == 1.0
end

runTest(6, "predict (instancia): devuelve un número, no un vector") do
    p = predict(trainRegressionModel(knnConfig("k" => 3), knnDataset), [0.5])
    isa(p, Real)
end
runTest(6, "predict (instancia): kNN clásico es la media de los k vecinos") do
    m = trainRegressionModel(knnConfig("k" => 3), knnDataset)
    predict(m, [0.5]) == 1.0 && predict(m, [11.0]) == 5.0
end
runTest(6, "predict: con k = 1 sobre los datos de entrenamiento devuelve sus salidas") do
    m = trainRegressionModel(knnConfig("k" => 1), randomDataset)
    approxVec(predict(m, randomX), batchTargets(randomDataset))
end
runTest(6, "predict: con k = n devuelve siempre la media de todas las salidas") do
    m = trainRegressionModel(knnConfig("k" => 6), knnDataset)
    approxVec(predict(m, [0.0 ; 100.0 ;; ]), [3.0, 3.0])
end
runTest(6, "predict: global constante + corrección local constante = kNN clásico") do
    classic = trainRegressionModel(knnConfig("k" => 4), randomDataset)
    hybrid = trainRegressionModel(knnConfig("k" => 4, "globalModelConfiguration" => Dict("modelType" => :constant)), randomDataset)
    approxVec(predict(hybrid, randomX), predict(classic, randomX))
end
runTest(6, "predict: α = 1 devuelve solo la predicción local") do
    local1 = trainRegressionModel(knnConfig("k" => 4), randomDataset)
    hybrid = trainRegressionModel(knnConfig("k" => 4, "alpha" => 1.0,
                                            "globalModelConfiguration" => Dict("modelType" => :linear)), randomDataset)
    approxVec(predict(hybrid, randomX), predict(local1, randomX))
end
runTest(6, "predict: α = 0 devuelve solo la predicción global") do
    globalModel = trainRegressionModel(Dict("modelType" => :linear), randomDataset)
    hybrid = trainRegressionModel(knnConfig("k" => 4, "alpha" => 0.0,
                                            "globalModelConfiguration" => Dict("modelType" => :linear)), randomDataset)
    approxVec(predict(hybrid, randomX), predict(globalModel, randomX))
end
runTest(6, "predict: α·local + (1-α)·global calculado a mano, 0.25·1 + 0.75·3 = 2.5") do
    m = trainRegressionModel(knnConfig("k" => 3, "alpha" => 0.25,
                                       "globalModelConfiguration" => Dict("modelType" => :constant)), knnDataset)
    isapprox(predict(m, [0.5]), 2.5)
end
runTest(6, "predict: corrección local de un global lineal exacto no cambia nada (y = 2x + 1)") do
    X = reshape(collect(0.0:1.0:9.0), 10, 1)
    m = trainRegressionModel(knnConfig("k" => 3, "globalModelConfiguration" => Dict("modelType" => :linear)),
                             (X, 2 .* X[:, 1] .+ 1))
    isapprox(predict(m, [3.3]), 7.6; atol = 1e-8)
end
runTest(6, "predict (matriz): una predicción por fila, igual que llamar fila a fila") do
    m = trainRegressionModel(knnConfig("k" => 3), knnDataset)
    inputs = reshape([0.5, 11.0, 6.0], 3, 1)
    p = predict(m, inputs)
    length(p) == 3 && eltype(p) <: Real && p == [predict(m, inputs[i, :]) for i in 1:3]
end
runTest(6, "predict (matriz): instancias Float32 sobre un dataset Float64") do
    m = trainRegressionModel(knnConfig("k" => 3), knnDataset)
    approxVec(predict(m, Float32[0.5; 11.0;;]), [1.0, 5.0])
end
runTest(6, "kNN con un árbol de modelos como modelo global") do
    m = trainRegressionModel(knnConfig("k" => 5,
            "globalModelConfiguration" => Dict("modelType" => :ModelTree, "max_depth" => 2,
                                               "localModelConfiguration" => Dict("modelType" => :constant))), swdSmall)
    p = predict(m, batchInputs(swdSmall))
    length(p) == 100 && all(isfinite, p)
end
runTest(6, "kNN con otro kNN como modelo local") do
    m = trainRegressionModel(knnConfig("k" => 10,
            "localModelConfiguration" => knnConfig("k" => 3)), swdSmall)
    p = predict(m, batchInputs(selectInstances(swdSmall, 1:5)))
    length(p) == 5 && all(isfinite, p)
end


# ----------------------------------------------------------------------------------------------
# ----------------------- Robustez (casos que el enunciado no especifica) -----------------------
# ----------------------------------------------------------------------------------------------

println("\nRobustez (no puntúa)")

# The statement asks to fit the linear model with the `\` operator. When the design matrix
# hcat(ones, inputs) is square (instances == attributes + 1) and singular, `\` uses an LU
# factorization and throws SingularException; with more rows it falls back to least squares.
# In a model tree this happens in any node that ends up with exactly attributes + 1 instances
# with repeated values (e.g. SWD, whose inputs are small integers).
runTest(5, "Modelo lineal con tantas instancias como columnas y datos repetidos"; scored = false) do
    m = trainRegressionModel(Dict("modelType" => :linear),
                             (hcat([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]), [1.0, 2.0, 3.0]))
    isa(m, LinearModel)
end
runTest(5, "Árbol de modelos lineales sobre SWD (100 instancias, max_depth = 2)"; scored = false) do
    m = trainRegressionModel(Dict("modelType" => :ModelTree, "max_depth" => 2,
                                  "localModelConfiguration" => Dict("modelType" => :linear)), swdSmall)
    all(isfinite, predict(m, batchInputs(swdSmall)))
end

# nearestElements (exercise 3) asks partialsortperm for k positions, so it fails when the
# dataset has fewer than k instances. The expected value assumes k is capped at the dataset size.
runTest(6, "kNN con k mayor que el número de instancias del dataset"; scored = false) do
    m = trainRegressionModel(knnConfig("k" => 10), knnDataset)
    isapprox(predict(m, [0.5]), 3.0)
end
runTest(6, "Árbol de modelos con kNN (k = 3) en los nodos, como el ejemplo del enunciado"; scored = false) do
    m = trainRegressionModel(Dict("modelType" => :ModelTree, "max_depth" => 4, "min_samples_leaf" => 20,
            "localModelConfiguration" => knnConfig("k" => 3,
                "globalModelConfiguration" => Dict("modelType" => :linear))), swdSmall)
    p = predict(m, batchInputs(swdSmall))
    length(p) == 100 && all(isfinite, p)
end


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------

println("\n", "="^70)
println("RESUMEN")
println("="^70)
grades = Float64[]
for exercise in 4:6
    scoredResults = filter(r -> r.exercise == exercise && r.scored, testResults)
    passed = count(r -> r.status == :pass, scoredResults)
    grade = 10 * passed / length(scoredResults)
    push!(grades, grade)
    println("Ejercicio ", exercise, ": ", lpad(passed, 2), " / ", lpad(length(scoredResults), 2),
            " pruebas superadas  ->  nota ", round(grade, digits = 2), " / 10")
end
robustness = filter(r -> !r.scored, testResults)
println("Robustez:    ", count(r -> r.status == :pass, robustness), " / ", length(robustness),
        " (no puntúa)")
println("-"^70)
println("NOTA ORIENTATIVA (media de los 3 ejercicios): ", round(mean(grades), digits = 2), " / 10")

failed = filter(r -> r.status != :pass, testResults)
if !isempty(failed)
    println("\nPruebas no superadas:")
    for r in failed
        println("  - Ej. ", r.exercise, r.scored ? "" : " (robustez)", ": ", r.description,
                "\n      ", r.status == :fail ? "FALLO" : "ERROR", ": ", r.message)
    end
end
