FROM jupyter/julia-notebook
LABEL author="Enrique Fernández Blanco"

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y htop && \
    pip install --no-cache-dir catboost lightgbm matplotlib plotly rich scikit-learn seaborn xgboost

# ========== Install Julia packages as application user ==========

RUN julia -e "using Pkg; Pkg.add.([ \
    \"IJulia\", \
    \"CSV\", \
    \"CategoricalArrays\", \
    \"CatBoost\", \
    \"DataFrames\", \
    \"DecisionTree\", \
    \"EvoTrees\", \
    \"FileIO\", \
    \"Flux\", \
    \"HypothesisTests\", \
    \"Images\", \
    \"JLD2\", \
    \"LIBSVM\", \
    \"LightGBM\", \
    \"MAT\", \
    \"MLBase\", \
    \"MLJ\", \
    \"MLJBase\", \
    \"MLJDecisionTreeInterface\", \
    \"MLJEnsembles\", \
    \"MLJLIBSVMInterface\", \
    \"MLJLinearModels\", \
    \"MLJModelInterface\", \
    \"MLJModels\", \
    \"MLJMultivariateStatsInterface\", \
    \"MLJNaiveBayesInterface\", \
    \"ManifoldLearning\", \
    \"MultivariateStats\", \
    \"NaiveBayes\", \
    \"NearestNeighborModels\", \
    \"Plots\", \
    \"PyCall\", \
    \"PythonCall\", \
    \"RDatasets\", \
    \"ScikitLearn\", \
    \"ScikitLearnBase\", \
    \"StatsPlots\", \
    \"TSne\", \
    \"Tables\", \
    \"UrlDownload\", \
    \"XGBoost\", \
    \"XLSX\", \
]); Pkg.precompile();"

# Pre compile
RUN julia -e "using IJulia, \
    CSV, \
    CategoricalArrays, \
    CatBoost, \
    DataFrames, \
    DecisionTree, \
    DelimitedFiles, \
    EvoTrees, \
    FileIO, \
    Flux, \
    HypothesisTests, \
    Images, \
    JLD2, \
    LIBSVM, \
    LightGBM, \
    MAT, \
    MLBase, \
    MLJ, \
    MLJBase, \
    MLJDecisionTreeInterface, \
    MLJEnsembles, \
    MLJLIBSVMInterface, \
    MLJLinearModels, \
    MLJModelInterface, \
    MLJModels, \
    MLJMultivariateStatsInterface, \
    MLJNaiveBayesInterface, \
    ManifoldLearning, \
    MultivariateStats, \
    NaiveBayes, \
    NearestNeighborModels, \
    Plots, \
    PyCall, \
    PythonCall, \
    RDatasets, \
    Random, \
    ScikitLearn, \
    ScikitLearnBase, \
    Statistics, \
    StatsPlots, \
    TSne, \
    Tables, \
    UrlDownload, \
    XGBoost, \
    XLSX"

USER $NB_UID
