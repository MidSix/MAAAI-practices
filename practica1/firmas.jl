

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles

function fileNamesFolder(folderName::String, extension::String)
    #
    # Codigo a desarrollar
    #
end;



function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float64)
    #
    # Codigo a desarrollar
    #
end;



function datasetsDescription(datasetFolder::String, classification::Bool)
    #
    # Codigo a desarrollar
    #
end;



function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;



function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float64)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
predict(model::Machine{MLJLIBSVMInterface.SVC, MLJLIBSVMInterface.SVC, true}, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return convert(Vector, levels(outputs)[int(outputs)]); )



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;


    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;



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



