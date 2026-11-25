include("scikitlearn.jl")


using ScikitLearn
using ScikitLearn.Pipelines: Pipeline, named_steps
using ScikitLearn.GridSearch: GridSearchCV

@sk_import decomposition: PCA
#@sk_import model_selection: GridSearchCV
@sk_import datasets: load_iris

# Cargar los datos
iris = load_iris()
X = iris["data"]
y = iris["target"]

# Definir la búsqueda en cuadrícula (hiperparámetros a probar)
param_grid = Dict("ann__maxEpochs" => [500, 1000], "ann__learningRate" => [0.01, 0.1])

# Definir los parámetros
topology = [3, 4]
functions = [σ, σ]
maxEpochs = 1000
minLoss = 0.0
learningRate = 0.01

# Crear una instancia de ClassANN
ann = ClassANN(topology=topology, transferFunctions=functions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)

estimators = [("pca",PCA()),("ann",ann)]
pipe = Pipeline(estimators)

# Configurar el GridSearchCV
grid_search = GridSearchCV(pipe, param_grid)

# Ajustar el modelo usando GridSearchCV
fit!(grid_search, X, y)

# Obtener el mejor modelo y sus hiperparámetros
println("Mejor modelo: ", grid_search.best_estimator_)
println("Mejores hiperparámetros: ", grid_search.best_params_)
