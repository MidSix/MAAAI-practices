

# Archivo de pruebas para realizar autoevaluación de algunas funciones de los ejercicios

# Importamos el archivo con las soluciones a los ejercicios
include("soluciones.jl");
#   Cambiar "soluciones.jl" por el nombre del archivo que contenga las funciones desarrolladas

# Fichero de pruebas realizado con la versión 1.11.3 de Julia
println(VERSION)
#  y la 1.11.3 de Random
using Random; println(Random.VERSION)
#  y las versiones 0.14.25 de Flux y 0.20.0 de MLJ
import Pkg
Pkg.status("Flux")
Pkg.status("MLJ")

# Es posible que con otras versiones los resultados sean distintos, estando las funciones bien, sobre todo en la funciones que implican alguna componente aleatoria


# Para la correcta ejecución de este archivo, los datasets estarán en la siguiente carpeta:
datasetFolder = "../datasets";
# Cambiadla por la carpeta donde tengáis los datasets

@assert(isdir(datasetFolder))


# Cargamos el dataset
dataset = loadDataset("SWD", datasetFolder; datasetType=Float64);
inputs, targets = dataset

using Random:seed!

# Establecemos la semilla para que los resultados sean siempre los mismos
# Comprobamos que la generación de números aleatorios es la esperada:
seed!(1); @assert(isapprox(rand(), 0.07336635446929285))
#  Si fallase aquí, seguramente dara error al comprobar los resultados de la ejecución de la siguiente función porque depende de la generación de números aleatorios


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


seed!(42); @assert(isapprox(mse(randn(),   randn(10)), 1.836422079285861 ))
seed!(42); @assert(isapprox(mse(randn(10), randn(10)), 0.549799523801031 ))

@assert(candidateThresholds([0., 1, 2.5, 2, 1]) == [0.5, 1.5, 2.25])

@assert(isapprox(calculateErrorSplit(dataset, 3, 3.), 595.2137403355684 ))

@assert(bestSplit(dataset) == (3, 1.5))

@assert(predict(3.8, randn(5,3)) == fill(3.8, 5))

model = buildRegressionTree(dataset; max_depth = 0, min_samples_leaf =  1 )
@assert(isa(model, ModelTreeLeaf))
seed!(42); @assert(all(isapprox.(predict(model, rand(3, size(batchInputs(dataset),2))), fill(3.801, 3))))

model = buildRegressionTree(dataset; max_depth = -1, min_samples_leaf =  1 )
@assert(isa(model, ModelTreeNode))
seed!(42); @assert(all(isapprox.(predict(model, 3 .*rand(5, size(batchInputs(dataset),2))), [4.166666666666667, 3.0, 2.727272727272727, 3.4, 3.0])))
 

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


@assert(all(isapprox.(predict(trainRegressionModel(Dict("modelType" => :constant), dataset), batchInputs(selectInstances(dataset, 1:3))), fill(mean(batchTargets(dataset)), 3))))
@assert(all(isapprox.(predict(trainRegressionModel(Dict("modelType" => :linear),  dataset), batchInputs(selectInstances(dataset, 1:3))), [ 2.7977921396596357, 4.1756460431869336, 4.403562097821388 ])))
@assert(all(isapprox.(predict(trainRegressionModel(Dict("modelType" => :DoME, "maximumNodes" => 41), dataset), batchInputs(selectInstances(dataset, 1:3))), [ 3.2081728001735823, 4.140857954051513, 3.90768666558203 ])))
model = trainRegressionModel(Dict(
        "modelType" => :SVR,
        "epsilon"   => 0.1,
        "C"         => 0.01,
        "kernel"    => "poly",
        "gamma"     => 0.2,
        "degree"    => 3,
        "coef0"     => 1.
    ), dataset)
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 2.8015538995837637, 4.9003744708055645, 4.899861677464058 ])))

@assert(isapprox(calculateErrorSplit(dataset, 4, 2., predict(model, batchInputs(dataset))), 365.6887101464047 ))

split = bestSplit(dataset, predict(model, batchInputs(dataset)))
@assert(split[1]==-1 && isnan(split[2]))
split = bestSplit(selectInstances(dataset, 1:50), predict(model, batchInputs(selectInstances(dataset, 1:50))))
@assert(split == (3, 1.5))

model = trainRegressionModel(Dict(
    "modelType" => :ModelTree,
    "max_depth" => 0,
    "min_samples_leaf" => 10,
    "localModelConfiguration" => Dict(
        "modelType" => :DoME,
        "maximumNodes" => 45)
    ), dataset)
@assert(isa(model, ModelTreeLeaf));
@assert(numInstances(model) == batchLength(dataset))
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 2.7594752786711934, 4.221625572706246, 4.426749558276412 ])))
@assert(prune!(model, selectInstances(dataset, 1:20)) == model)



model = trainRegressionModel(Dict(
    "modelType" => :ModelTree,
    "min_samples_leaf" => 50,
    "localModelConfiguration" => Dict(
        "modelType" => :linear)
     ), dataset)
@assert(isa(model, ModelTreeNode));
@assert(numInstances(model) == batchLength(dataset))
@assert(all(isapprox.(predict(       model,                                  batchInputs(selectInstances(dataset, 1:3))), [ 2.888888888888887, 4.454545454545454, 4.62412524426133 ])))
@assert(all(isapprox.(predict(prune!(model, selectInstances(dataset, 1:30)), batchInputs(selectInstances(dataset, 1:3))), [ 2.656526083184641, 4.341382955404691, 4.405449803133104 ])))





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------



model = trainRegressionModel(Dict(
    "modelType" => :KNN,
    "globalModelConfiguration" => Dict(
        "modelType"    => :DoME,
        "maximumNodes" => 11)
    ), dataset)
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 2.8, 4.2, 4.2 ])))

model = trainRegressionModel(Dict(
    "modelType" => :KNN,
    "k"         => 11,
    "globalModelConfiguration" => Dict(
        "modelType"    => :DoME,
        "maximumNodes" => 41)
    ), dataset)
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 2.909090909090909, 4.454545454545454, 4.545454545454545 ])))

model = trainRegressionModel(Dict(
    "modelType" => :KNN,
    "k"         => 23,
    "localModelConfiguration" => Dict(
        "modelType" => :SVR,
        "epsilon"   => 0.2,
        "C"         => 1.,
        "kernel"    => "linear")
    ), dataset)
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 2.799560546875001, 4.8, 4.799560546874998 ])))

model = trainRegressionModel(Dict(
    "modelType" => :KNN,
    "k"         => 27,
    "globalModelConfiguration" => Dict(
        "modelType" => :constant),
    "localModelConfiguration" => Dict(
        "modelType" => :linear),
    "alpha" => 0.5
    ), dataset)
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 3.3449444444444443, 4.127772727272728, 4.173227272727273 ])))

model = trainRegressionModel(Dict(
    "modelType" => :KNN,
    "k"         => 27,
    "globalModelConfiguration" => Dict(
        "modelType" => :constant),
    "localModelConfiguration" => Dict(
        "modelType" => :linear),
    ), dataset)
@assert(all(isapprox.(predict(model, batchInputs(selectInstances(dataset, 1:3))), [ 2.8888888888888884, 4.454545454545455, 4.545454545454546 ])))

