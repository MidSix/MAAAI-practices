import ScikitLearnBase: BaseClassifier, predict, score,
                        fit!, @declare_hyperparameters, is_classifier

using Flux
using Flux.Losses
using Statistics

export ClassANN, fit!, predict, score, is_classifier
################################################################################
# Classifier

mutable struct ClassANN <: BaseClassifier
    # Hiperparámetros del modelo (no aprendidos de los datos)
    topology::AbstractVector{Int}
    transferFunctions::AbstractVector{Function}
    maxEpochs::Int
    minLoss::Real
    learningRate::Real

    # Parámetros aprendidos (modelo de Flux y optimizador)
    model::Chain
    opt::ADAM

    # Constructor que acepta los hiperparámetros como argumentos con nombre
    ClassANN(; topology=[1], transferFunctions=fill(σ, 1), maxEpochs=1000, minLoss=0.0, learningRate=0.01) =
        new(topology, transferFunctions, maxEpochs, minLoss, learningRate, Chain(), ADAM(learningRate))
end

@declare_hyperparameters(ClassANN, [:topology, :transferFunctions, :maxEpochs, :minLoss, :learningRate])

# Indicar que ClassANN es un clasificador
is_classifier(::ClassANN) = true

#Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        # Cualquiera de estos dos tipos (Array{Bool,2} o BitArray{2}) vale perfectamente
        # oneHot = Array{Bool,2}(undef, length(targets), numClasses);
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado, simplemente lo convertimos a una matriz columna
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará

function buildClassANN(numInputs::Int, topology::AbstractVector{Int}, numOutputs::Int;
            transferFunctions::AbstractVector{Function}=fill(σ, length(topology)))
         ann = Chain()
         numInputsLayer = numInputs
         for numHiddenLayer in 1:length(topology)
             numNeurons = topology[numHiddenLayer]
             ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
             numInputsLayer = numNeurons
         end
         if numOutputs == 1
             ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
         else
             ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
             ann = Chain(ann..., softmax)
         end
         return ann
     end
     
# Implementar el ajuste (fit!) del modelo
function fit!(model::ClassANN, X, y)
    numInputs = size(X, 2)
    numOutputs = length(unique(y))

    # Construir el modelo usando Flux
    model.model = buildClassANN(numInputs, model.topology, numOutputs, transferFunctions=model.transferFunctions)
    model.opt = ADAM(model.learningRate)

    # Convertir las etiquetas a formato one-hot si es clasificación multiclase
    if numOutputs > 1
        y = oneHotEncoding(y, unique(y))
    end

    # Definir la función de pérdida
    loss(x, y) = Flux.crossentropy(model.model(x), y)

    # Entrenar el modelo
    for epoch in 1:model.maxEpochs
        Flux.train!(loss, Flux.params(model.model), [(X', y')], model.opt)
        current_loss = loss(X', y')
        println("Epoch: $epoch, Loss: $current_loss")
        if current_loss <= model.minLoss
            break
        end
    end
    return model
end

# Implementar la predicción (predict) del modelo
function predict(model::ClassANN, X)
    if size(model.model(X'), 1) > 1
        return Flux.onecold(model.model(X'), 1:size(model.model(X'), 1))
    else
        return round.(model.model(X'))
    end
end

# Función adicional para calcular el puntaje (score) del modelo
function score(model::ClassANN, X, y)
    predictions = predict(model, X)
    return mean(predictions .== y)
end

#= #Implementar set_params y get_params
function get_params(model::ClassANN)
    return Dict(:topology => model.topology,
                :transferFunctions => model.transferFunctions,
                :maxEpochs => model.maxEpochs,
                :minLoss => model.minLoss,
                :learningRate => model.learningRate)
end

function set_params!(model::ClassANN; kwargs...)
    for (key, value) in kwargs
        if hasfield(model, key)
            setfield!(model, key, value)
        else
            error("No such parameter: $key")
        end
    end
    return model
end =#