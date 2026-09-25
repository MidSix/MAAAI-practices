
#=
Os dejamos en este archivo unas instrucciones de autoevaluación, 
para que podáis ir verificando que las funciones que programáis son correctas.

Tened en cuenta dos cosas. La primera, que hay varias funciones, 
como la creación de RR.NN.AA., que dependen de la generación de números
aleatorios. Esta generación de números aleatorios es dependiente de 
la versión de Julia que se tenga instalada, aunque parece que en las 
últimas versiones estas funciones se han mantenido sin modificar. 
Por ese motivo, veréis que en el script hay una línea para verificar 
que la generación de números aleatorios sea la utilizada para generar 
los resultados que veréis en este archivo.

La segunda cuestión es el hecho de que vuestras funciones den las
salidas correctas según este archivo no garantiza que sean totalmente 
correctas, puesto que no se evalúa toda la casuística.
Además, no todas las funciones se evalúan en este script. 
Por otra parte, también podría darse el caso de que una función
de error en este script pero sea correcta, aunque esto es muy 
poco probable. Por este motivo, este archivo simplemente es de apoyo, 
y no sustituye a la evaluación realizada en la entrega.
# Archivo de pruebas para realizar autoevaluación de algunas funciones de los ejercicios
=#

# Importamos el archivo con las soluciones a los ejercicios
archivo_con_soluciones = joinpath(@__DIR__, "firmas_modificado.jl");
include(archivo_con_soluciones);
#   Cambiar "firmas_modificado.jl" por el nombre del archivo que 
#   contenga las funciones desarrolladas

println("Comprobando la versión de Julia, Random y SymDoME")

# Fichero de pruebas realizado con la versión 1.11.2 de Julia
println(VERSION)
#  y la 1.11.3 de Random
using Random; println(Random.VERSION)
# ---> El profesor dice que ha usado una version de Julia y del paquete
# Random que discrepan, esto no es posible porque Random es un 
# paquete built-in que viene ligado con Julia, la version de todo
# paquete built-in debe ser la misma que la de Julia, por lo que
# ni idea de si escribio la version de Random mal o la de Julia mal xD.
# Pero sea como sea la discrepancia es 1.11.x que simplemente son fixes
# por lo que no debe afectar a los asserts, gemini tambien dice lo mismo
# es seguro asumir que ya sea que se elija Julia 1.11.2 o 1.11.3
# no debe afectar a los asserts. Para este caso elegi la 1.11.2 pero
# la 1.11.3 deberia funcionar igual.


#  y la versión 1.0.4 de SymDoME
import Pkg
Pkg.status("SymDoME")

# Es posible que con otras versiones los resultados sean distintos, estando las funciones bien, sobre todo en la funciones que implican alguna componente aleatoria

# Para la correcta ejecución de este archivo, los datasets estarán en la siguiente carpeta:
# se puede pasar el dataset dentro del directorio ejercicios1-3 
# y simplemente escribir datasetFolder = "datasets" pero
# tambien se puede simplemente subir un nivel el path y ya esta.
# Simplemente tenemos que entregar el modulo de firmas, no la
# autoevaluacion asi que da bastante igual que escribamos aqui.
# datasetFolder = normpath(joinpath(@__DIR__, "..", "datasets"));
datasetFolder = joinpath(@__DIR__, "datasets");
# Cambiadla por la carpeta donde tengáis los datasets y las imágenes

# isdir simplemente comprueba si el valor pasado por parametro que debe 
# ser una ruta hacia un directorio existe, en caso que lo haga
# gucci: true, si no lo hace not gucci:false, este booleano se le 
# pasa al assert que si es true gucci y si es false lanza un excepcion 
# que no se captura y por tanto detiene la ejecucion del script, 
# esto es para comprobar que este modulo es capaz de usar los datasets.
@assert(isdir(datasetFolder))
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

datasetFileNames = fileNamesFolder(datasetFolder,"tsv")
@assert(sort(datasetFileNames) == ["ERA", "ESL", "FacultySalaries", "LEV", "SWD", "USCrime", "analcatdata_apnea1", "analcatdata_apnea2", "analcatdata_election2000", "analcatdata_neavote", "analcatdata_vehicle", "appendicitis", "australian", "backache", "biomed", "buggyCrx", "bupa", "chatfield_4", "chscase_geyser1", "cleve", "cloud", "colic", "corral", "cpu", "elusage", "german", "heart-c", "heart-statlog", "hungarian", "ionosphere", "labor", "lupus", "machine_cpu", "molecular_biology_promoters", "monk1", "monk2", "monk3", "mux6", "no2", "parity5", "pima", "pm10", "pollution", "prnn_crabs", "prnn_synth", "profb", "rabe_266", "rmftsa_ladata", "saheart", "sleuth_case1202", "sleuth_case2002", "sleuth_ex1605", "sleuth_ex1714", "sonar", "vineyard", "vinnie", "visualizing_environmental", "visualizing_galaxy", "vote", "wdbc"]);

inputs, targets = loadDataset("sonar", datasetFolder; datasetType=Float32);
@assert(size(inputs)==(208,60))
@assert(length(targets)==208)
@assert(eltype(inputs)==Float32)
@assert(eltype(targets)==Bool)

@assert(sum(datasetsDescription(datasetFolder, true)[:,2:end]) == 11415)
@assert(sum(datasetsDescription(datasetFolder, false)[:,2:end]) == 8866)

sinEncoding, cosEncoding = cyclicalEncoding([1, 2, 3, 2, 1, 0, -1, -2, -3]);
@assert(all(isapprox.(sinEncoding, [-0.433883739117558, -0.9749279121818236, -0.7818314824680299, -0.9749279121818236, -0.433883739117558, 0.43388373911755823, 0.9749279121818236, 0.7818314824680298, 0.0]; rtol=1e-4)))
@assert(all(isapprox.(cosEncoding, [-0.9009688679024191, -0.2225209339563146, 0.6234898018587334, -0.2225209339563146, -0.9009688679024191, -0.900968867902419, -0.22252093395631434, 0.6234898018587336, 1.0]; rtol=1e-4)))

inputs, targets = loadStreamLearningDataset(datasetFolder; datasetType=Float64)
@assert(size(inputs)==(45312,7))
@assert(length(targets)==45312)
@assert(eltype(inputs)==Float64)
@assert(eltype(targets)==Bool)

println(">>> ¡TODOS LOS TESTS DEL EJERCICIO 1 HAN PASADO CON ÉXITO! <<<")

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# Cargamos el dataset
dataset = loadDataset("sonar", datasetFolder; datasetType=Float64);
inputs, targets = dataset

using Random:seed!

# Establecemos la semilla para que los resultados sean siempre los mismos
# Comprobamos que la generación de números aleatorios es la esperada:
seed!(1); @assert(isapprox(rand(), 0.07336635446929285))
#  Si fallase aquí, seguramente dara error al comprobar los resultados de la ejecución de la siguiente función porque depende de la generación de números aleatorios


@assert(batchInputs(dataset) === inputs)
@assert(batchTargets(dataset) === targets)
@assert(batchLength(dataset) === 208)
i = rand(Bool, batchLength( dataset ));
@assert(batchTargets(selectInstances(dataset, i)) == targets[i]);
@assert(joinBatches(selectInstances(dataset, i), selectInstances(dataset, .!i)) == (vcat(inputs[i,:], inputs[.!i,:]), vcat(targets[i], targets[.!i])))
@assert(divideBatches(dataset, 100; shuffleRows=false) == [selectInstances(dataset, 1:100), selectInstances(dataset, 101:200), selectInstances(dataset, 201:208)])
seed!(1); @assert(all(isapprox.(mean.(batchInputs.(divideBatches(dataset, 100; shuffleRows=true))), [0.2812910166666666, 0.28142645, 0.2803852083333333])))
seed!(1); @assert(all(isapprox.(mean.(batchTargets.(divideBatches(dataset, 100; shuffleRows=true))), [0.43, 0.5, 0.5])))



model, supportVectors, supportVectorIndices = trainSVM(dataset, "rbf", 1; gamma=3)
@assert(batchLength(supportVectors) == 197)
@assert(isempty(supportVectorIndices[1]))
@assert(length(supportVectorIndices[2]) == 197)
@assert(issorted(supportVectorIndices[2]))
@assert(selectInstances(dataset, supportVectorIndices[2]) == supportVectors)

model, supportVectors, supportVectorIndices = trainSVM( selectInstances(dataset, 1:100), "poly", 100; degree=2, gamma=3, coef0=2)
@assert(batchLength(supportVectors) == 38)
@assert(isempty(supportVectorIndices[1]))
@assert(length(supportVectorIndices[2]) == 38)
@assert(issorted(supportVectorIndices[2]))
@assert(selectInstances(dataset, supportVectorIndices[2]) == supportVectors)

model, newSupportVectors, newSupportVectorIndices = trainSVM( selectInstances(dataset, 101:208), "poly", 100; degree=2, gamma=3, coef0=2, supportVectors=supportVectors)
@assert(batchLength(newSupportVectors) == 72)
@assert(length(newSupportVectorIndices[1]) == 30)
@assert(length(newSupportVectorIndices[2]) == 42)
@assert(issorted(newSupportVectorIndices[1]))
@assert(issorted(newSupportVectorIndices[2]))
@assert(joinBatches(
    selectInstances(supportVectors, newSupportVectorIndices[1]),
    selectInstances( selectInstances(dataset, 101:208 ), newSupportVectorIndices[2])) == newSupportVectors)


model = trainSVM(divideBatches(dataset, 100; shuffleRows=false), "rbf", 10; gamma=4)
@assert(findall(predict(model, batchInputs(selectInstances(dataset, 1:20)))) == 13:20)

println(">>> ¡TODOS LOS TESTS DEL EJERCICIO 2 HAN PASADO CON ÉXITO! <<<")

#=
# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------


memory, batchList = initializeStreamLearningData(datasetFolder, 1000, 100);
@assert(isa(memory, Batch))
@assert(isapprox(mean(batchInputs(memory)), 0.24773968142857153))
@assert(isapprox(mean(batchTargets(memory)), 0.506))
@assert(isa(batchList, Vector{<:Batch}))
@assert(length(batchList) == 444)
@assert(isapprox(mean(mean.(batchInputs.(batchList))), 0.2648037485224358))
@assert(isapprox(mean(mean.(batchTargets.(batchList))), 0.57753003003003))

addBatch!(memory, batchList[1])
@assert(all(isapprox.(mean.(memory), (0.25013414f0, 0.531))))



accuracies  = streamLearning_SVM(datasetFolder, 1000, 500, "rbf", 1.; gamma=2);
@assert(isa(accuracies, Vector{<:Real}))
@assert(length(accuracies) == 89);
@assert(all(isapprox.(accuracies, [0.498, 0.684, 0.616, 0.624, 0.65, 0.64, 0.608, 0.656, 0.558, 0.636, 0.656, 0.678, 0.546, 0.624, 0.714, 0.664, 0.568, 0.682, 0.688, 0.622, 0.592, 0.662, 0.608, 0.584, 0.68, 0.794, 0.696, 0.756, 0.586, 0.646, 0.606, 0.736, 0.756, 0.626, 0.686, 0.796, 0.654, 0.816, 0.668, 0.69, 0.644, 0.7, 0.696, 0.664, 0.76, 0.676, 0.74, 0.66, 0.656, 0.754, 0.842, 0.818, 0.696, 0.68, 0.668, 0.608, 0.682, 0.582, 0.666, 0.662, 0.602, 0.59, 0.618, 0.78, 0.664, 0.684, 0.578, 0.638, 0.7, 0.756, 0.678, 0.706, 0.664, 0.728, 0.658, 0.708, 0.702, 0.68, 0.658, 0.666, 0.718, 0.67, 0.73, 0.654, 0.688, 0.586, 0.704, 0.772, 0.7852564102564102])))


accuracies = streamLearning_ISVM(datasetFolder, 1000, 500, "rbf", 1.; gamma=2);
@assert(isa(accuracies, Vector{<:Real}))
@assert(length(accuracies) == 90);
@assert(all(isapprox.(accuracies, [0.632, 0.446, 0.64, 0.608, 0.566, 0.63, 0.604, 0.62, 0.662, 0.494, 0.662, 0.662, 0.63, 0.55, 0.556, 0.684, 0.604, 0.628, 0.646, 0.722, 0.572, 0.546, 0.696, 0.612, 0.558, 0.66, 0.782, 0.704, 0.748, 0.572, 0.622, 0.602, 0.732, 0.75, 0.612, 0.714, 0.79, 0.646, 0.814, 0.688, 0.674, 0.644, 0.708, 0.716, 0.69, 0.758, 0.66, 0.728, 0.66, 0.604, 0.748, 0.832, 0.812, 0.694, 0.682, 0.652, 0.58, 0.682, 0.578, 0.636, 0.63, 0.636, 0.582, 0.61, 0.752, 0.636, 0.7, 0.578, 0.586, 0.706, 0.752, 0.662, 0.666, 0.644, 0.71, 0.626, 0.722, 0.712, 0.684, 0.622, 0.678, 0.726, 0.696, 0.744, 0.648, 0.642, 0.59, 0.636, 0.796, 0.7884615384615384])))

distances = euclideanDistances(memory, batchList[1][1][1,:]);
@assert(isa(distances, Vector{<:Real}))
@assert(all(isapprox.(distances[1:3], [2.117765421038137, 2.117012336126041, 2.1175662466106786])))

nearestInstances =  nearestElements(memory, batchList[1][1][1,:], 3);
@assert(all(isapprox(batchInputs(nearestInstances), 
    [0.0  1.0  0.851064  0.582118  0.003467  0.422915  0.414912;
     0.0  1.0  0.829787  0.562928  0.003467  0.422915  0.414912;
     0.0  1.0  0.87234   0.557721  0.003467  0.422915  0.414912])))
@assert(all(batchTargets(nearestInstances) .== [false, true, true]))

output = predictKNN(memory, batchList[1][1][1,:], 10);
@assert(!output);

outputs = predictKNN(memory, batchList[1][1][1:10,:], 10);
@assert(isa(outputs, Vector{<:Bool}))
@assert(outputs == Bool[0, 0, 0, 1, 1, 0, 1, 1, 1, 1])

accuracies = streamLearning_KNN(datasetFolder, 1000, 500, 9);
@assert(isa(accuracies, Vector{<:Real}))
@assert(length(accuracies) == 89);
@assert(all(isapprox.(accuracies, [0.496, 0.652, 0.65, 0.638, 0.67, 0.644, 0.65, 0.738, 0.564, 0.676, 0.668, 0.636, 0.58, 0.656, 0.678, 0.718, 0.62, 0.73, 0.756, 0.706, 0.618, 0.648, 0.602, 0.602, 0.686, 0.786, 0.714, 0.77, 0.608, 0.598, 0.646, 0.748, 0.76, 0.694, 0.756, 0.784, 0.678, 0.79, 0.696, 0.7, 0.602, 0.75, 0.682, 0.662, 0.738, 0.642, 0.734, 0.682, 0.614, 0.672, 0.81, 0.818, 0.692, 0.67, 0.652, 0.594, 0.658, 0.622, 0.556, 0.69, 0.594, 0.676, 0.628, 0.71, 0.664, 0.646, 0.602, 0.588, 0.708, 0.752, 0.68, 0.662, 0.65, 0.732, 0.568, 0.72, 0.716, 0.67, 0.61, 0.662, 0.684, 0.75, 0.734, 0.642, 0.666, 0.568, 0.744, 0.79, 0.7660256410256411])))

=#
