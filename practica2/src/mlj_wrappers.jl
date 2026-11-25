# ==============================================================================
# Practica 2 - MAAAI (Curso 2026/27)
# Modulo: mlj_wrappers.jl
# Transformadores, filtros de caracteristicas, proyecciones, clasificadores base,
# ensembles (Bagging, SAMME AdaBoost, Hard Voting, Batch-Aware Stacking, EvoTree,
# RF, XGBoost, LightGBM, CatBoost) y wrappers 2D out-of-sample compatibles con MLJ.
# Todos los transformadores se ajustan EXCLUSIVAMENTE sobre entrenamiento (zero leakage).
# ==============================================================================

using MLJ
using MLJBase
using MLJModelInterface
using Tables
using DataFrames
using CategoricalArrays
using Statistics
using StatsBase
using LinearAlgebra
using Random
using Distances
using MultivariateStats
using MLJLinearModels
using Flux
using DecisionTree
using LIBSVM
using NearestNeighborModels
using MLJEnsembles
using EvoTrees
using XGBoost
using LightGBM
using ManifoldLearning
using TSne
using PyCall

const MMI = MLJModelInterface

# ------------------------------------------------------------------------------
# Utilidad interna: conversion segura Tabla/Matriz -> Matrix{Float64}
# ------------------------------------------------------------------------------
function to_float_matrix(X)::Matrix{Float64}
    if X isa AbstractMatrix
        return Matrix{Float64}(X)
    else
        return Matrix{Float64}(MLJBase.matrix(X))
    end
end

function get_col_names(X, p::Int)::Vector{Symbol}
    if X isa AbstractMatrix
        return [Symbol("x$j") for j in 1:p]
    else
        return collect(Symbol.(Tables.columnnames(X)))
    end
end

function to_int_labels(y)::Vector{Int}
    if y isa CategoricalArray
        return [parse(Int, string(v)) for v in y]
    else
        return Int.(y)
    end
end

# ==============================================================================
# 1. IMPUTADOR Y ESCALADORES LEAK-FREE (MLJModelInterface.Unsupervised)
# ==============================================================================

"""
    MedianImputer <: MMI.Unsupervised

Imputa valores `NaN` o no finitos usando la mediana por columna calculada EXCLUSIVAMENTE
sobre el conjunto de entrenamiento.
"""
mutable struct MedianImputer <: MMI.Unsupervised end

function MMI.fit(::MedianImputer, verbosity::Int, X)
    Xmat = to_float_matrix(X)
    p = size(Xmat, 2)
    medians = zeros(Float64, p)
    for j in 1:p
        vals = filter(isfinite, @view Xmat[:, j])
        medians[j] = isempty(vals) ? 0.0 : median(vals)
    end
    col_names = get_col_names(X, p)
    return (medians=medians, col_names=col_names), nothing, (n_features=p,)
end

function MMI.transform(::MedianImputer, fitresult, X)
    Xmat = copy(to_float_matrix(X))
    n, p = size(Xmat)
    meds = fitresult.medians
    @inbounds for j in 1:p
        m = meds[j]
        for i in 1:n
            if !isfinite(Xmat[i, j])
                Xmat[i, j] = m
            end
        end
    end
    return X isa AbstractMatrix ? Xmat : MLJBase.table(Xmat; names=fitresult.col_names)
end

"""
    MinMaxScaler <: MMI.Unsupervised

Escalado Min-Max ajustado unicamente sobre entrenamiento:
    x_scaled = (x - x_min) / (x_max - x_min)
Columnas constantes en train mantienen escala 1.0 para evitar division por cero.
No aplica clipping artificial en validacion/test para preservar la informacion real de drift.
"""
mutable struct MinMaxScaler <: MMI.Unsupervised
    method::Symbol  # :minmax (por defecto), :standard, :robust
end
MinMaxScaler(; method::Symbol=:minmax) = MinMaxScaler(method)

function MMI.fit(model::MinMaxScaler, verbosity::Int, X)
    Xmat = to_float_matrix(X)
    p = size(Xmat, 2)
    loc = zeros(Float64, p)
    scale = ones(Float64, p)
    for j in 1:p
        col = @view Xmat[:, j]
        if model.method == :minmax
            lo, hi = extrema(col)
            loc[j] = lo
            scale[j] = (hi - lo) > 1e-12 ? (hi - lo) : 1.0
        elseif model.method == :standard
            μ = mean(col)
            σ = std(col)
            loc[j] = μ
            scale[j] = σ > 1e-12 ? σ : 1.0
        elseif model.method == :robust
            q1, q2, q3 = quantile(col, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            loc[j] = q2
            scale[j] = iqr > 1e-12 ? iqr : (std(col) > 1e-12 ? std(col) : 1.0)
        else
            error("Metodo de escalado desconocido: $(model.method)")
        end
    end
    col_names = get_col_names(X, p)
    return (loc=loc, scale=scale, col_names=col_names), nothing, (method=model.method,)
end

function MMI.transform(::MinMaxScaler, fitresult, X)
    Xmat = to_float_matrix(X)
    n, p = size(Xmat)
    out = Matrix{Float64}(undef, n, p)
    loc = fitresult.loc
    sc = fitresult.scale
    @inbounds for j in 1:p
        l = loc[j]
        s = sc[j]
        for i in 1:n
            out[i, j] = (Xmat[i, j] - l) / s
        end
    end
    return X isa AbstractMatrix ? out : MLJBase.table(out; names=fitresult.col_names)
end

# ==============================================================================
# 2. FILTROS DE SELECCION DE CARACTERISTICAS (MLJModelInterface.Supervised)
#    Todos ajustados unicamente en el fold de entrenamiento.
# ==============================================================================

"""
    ANOVAFilter <: MMI.Supervised

Seleccion de las `k` mejores variables segun el estadistico F de ANOVA unifactorial.
"""
mutable struct ANOVAFilter <: MMI.Supervised
    k::Int
end
ANOVAFilter(; k::Int=32) = ANOVAFilter(k)

function MMI.fit(model::ANOVAFilter, verbosity::Int, X, y)
    Xmat = to_float_matrix(X)
    y_int = to_int_labels(y)
    n, p = size(Xmat)
    classes = sort(unique(y_int))
    K = length(classes)
    class_indices = [findall(==(c), y_int) for c in classes]
    class_counts = [length(idx) for idx in class_indices]

    fstats = zeros(Float64, p)
    for j in 1:p
        col = @view Xmat[:, j]
        grand_mean = mean(col)
        ssb = 0.0
        ssw = 0.0
        for (idx, nc) in zip(class_indices, class_counts)
            nc == 0 && continue
            grp = @view col[idx]
            μc = mean(grp)
            ssb += nc * (μc - grand_mean)^2
            ssw += sum(v -> (v - μc)^2, grp)
        end
        if ssw > 1e-15 && K > 1 && n > K
            msb = ssb / (K - 1)
            msw = ssw / (n - K)
            fstats[j] = msb / msw
        else
            fstats[j] = 0.0
        end
    end

    k_eff = min(model.k, p)
    idxs = sortperm(fstats; rev=true)[1:k_eff]
    col_names = get_col_names(X, p)
    sel_names = col_names[idxs]
    return (idxs=idxs, selected_names=sel_names), nothing, (scores=fstats, idxs=idxs, selected_features=sel_names)
end

function MMI.transform(::ANOVAFilter, fitresult, X)
    Xmat = to_float_matrix(X)
    Xsel = Xmat[:, fitresult.idxs]
    return X isa AbstractMatrix ? Xsel : MLJBase.table(Xsel; names=fitresult.selected_names)
end

"""
    fast_binary_kendall_tau_b(x::AbstractVector{Float64}, y_bin::BitVector) -> Float64

Calcula la correlacion exacta Kendall tau-b entre un vector continuo `x` y un indicador
binario `y_bin` en complejidad O(n log n) (identica a `StatsBase.corkendall` hasta 1e-14).
"""
function fast_binary_kendall_tau_b(x::AbstractVector{Float64}, y_bin::BitVector)::Float64
    n = length(x)
    n1 = count(y_bin)
    n0 = n - n1
    (n1 == 0 || n0 == 0) && return 0.0

    ranks = StatsBase.tiedrank(x)
    sum_r1 = 0.0
    @inbounds for i in 1:n
        if y_bin[i]
            sum_r1 += ranks[i]
        end
    end
    U = sum_r1 - n1 * (n1 + 1) / 2.0
    c_minus_d = 2.0 * U - Float64(n1) * Float64(n0)

    # Correccion exacta de empates en x
    sx = sort(x)
    tx = 0.0
    i = 1
    while i <= n
        j = i + 1
        while j <= n && sx[j] == sx[i]
            j += 1
        end
        t = j - i
        if t > 1
            tx += t * (t - 1) / 2.0
        end
        i = j
    end
    P = Float64(n) * Float64(n - 1) / 2.0
    denom = sqrt((P - tx) * Float64(n1) * Float64(n0))
    return denom > 0.0 ? c_minus_d / denom : 0.0
end

"""
    CorrelationFilter <: MMI.Supervised

Filtro por correlacion multiclase invariante a permutaciones de etiquetas mediante
agregacion One-vs-Rest (media de |rho(x_j, I(y == c))| sobre las clases):
- `method = :pearson`
- `method = :spearman`
- `method = :kendall` (Kendall tau-b exacto O(n log n))
"""
mutable struct CorrelationFilter <: MMI.Supervised
    k::Int
    method::Symbol
end
CorrelationFilter(; k::Int=32, method::Symbol=:pearson) = CorrelationFilter(k, method)

function MMI.fit(model::CorrelationFilter, verbosity::Int, X, y)
    Xmat = to_float_matrix(X)
    y_int = to_int_labels(y)
    n, p = size(Xmat)
    classes = sort(unique(y_int))
    y_bins = [BitVector(y_int .== c) for c in classes]
    y_floats = [Float64.(yb) for yb in y_bins]
    y_ranks = model.method == :spearman ? [StatsBase.tiedrank(yf) for yf in y_floats] : Vector{Vector{Float64}}()

    scores = zeros(Float64, p)
    for j in 1:p
        col = @view Xmat[:, j]
        if std(col) <= 1e-15
            scores[j] = 0.0
            continue
        end
        acc_corr = 0.0
        if model.method == :pearson
            for yf in y_floats
                r = cor(col, yf)
                acc_corr += isnan(r) ? 0.0 : abs(r)
            end
        elseif model.method == :spearman
            xr = StatsBase.tiedrank(col)
            for yr in y_ranks
                r = cor(xr, yr)
                acc_corr += isnan(r) ? 0.0 : abs(r)
            end
        elseif model.method == :kendall
            for yb in y_bins
                r = fast_binary_kendall_tau_b(col, yb)
                acc_corr += isnan(r) ? 0.0 : abs(r)
            end
        else
            error("Metodo de correlacion no soportado: $(model.method)")
        end
        scores[j] = acc_corr / length(classes)
    end

    k_eff = min(model.k, p)
    idxs = sortperm(scores; rev=true)[1:k_eff]
    col_names = get_col_names(X, p)
    sel_names = col_names[idxs]
    return (idxs=idxs, selected_names=sel_names), nothing, (scores=scores, idxs=idxs, selected_features=sel_names)
end

function MMI.transform(::CorrelationFilter, fitresult, X)
    Xmat = to_float_matrix(X)
    Xsel = Xmat[:, fitresult.idxs]
    return X isa AbstractMatrix ? Xsel : MLJBase.table(Xsel; names=fitresult.selected_names)
end

"""
    MutualInfoFilter <: MMI.Supervised

Filtro por Informacion Mutua I(X_j; Y) con discretizacion adaptativa por cuantiles
(calculados unicamente en entrenamiento) y correccion de sesgo de Miller-Madow.
"""
mutable struct MutualInfoFilter <: MMI.Supervised
    k::Int
    n_bins::Int
end
MutualInfoFilter(; k::Int=32, n_bins::Int=16) = MutualInfoFilter(k, n_bins)

function MMI.fit(model::MutualInfoFilter, verbosity::Int, X, y)
    Xmat = to_float_matrix(X)
    y_int = to_int_labels(y)
    n, p = size(Xmat)
    classes = sort(unique(y_int))
    K = length(classes)
    class_map = Dict(c => idx for (idx, c) in enumerate(classes))
    y_mapped = [class_map[v] for v in y_int]

    p_y = [count(==(c), y_mapped) / n for c in 1:K]
    probs_q = collect(range(0.0, 1.0; length=model.n_bins + 1))[2:end-1]

    scores = zeros(Float64, p)
    for j in 1:p
        col = @view Xmat[:, j]
        edges = unique(quantile(col, probs_q))
        nb = length(edges) + 1
        joint = zeros(Int, nb, K)
        @inbounds for i in 1:n
            b = searchsortedlast(edges, col[i]) + 1
            joint[b, y_mapped[i]] += 1
        end
        mi = 0.0
        nonzero_cells = 0
        for b in 1:nb
            row_sum = sum(@view joint[b, :])
            row_sum == 0 && continue
            pb = row_sum / n
            for c in 1:K
                cnt = joint[b, c]
                if cnt > 0
                    nonzero_cells += 1
                    pbc = cnt / n
                    mi += pbc * log(pbc / (pb * p_y[c]))
                end
            end
        end
        # Correccion de sesgo de Miller-Madow (no negativa)
        nb_eff = count(b -> sum(@view joint[b, :]) > 0, 1:nb)
        mm_corr = max(0, nonzero_cells - nb_eff - K + 1) / (2.0 * n)
        scores[j] = max(0.0, mi - mm_corr)
    end

    k_eff = min(model.k, p)
    idxs = sortperm(scores; rev=true)[1:k_eff]
    col_names = get_col_names(X, p)
    sel_names = col_names[idxs]
    return (idxs=idxs, selected_names=sel_names), nothing, (scores=scores, idxs=idxs, selected_features=sel_names)
end

function MMI.transform(::MutualInfoFilter, fitresult, X)
    Xmat = to_float_matrix(X)
    Xsel = Xmat[:, fitresult.idxs]
    return X isa AbstractMatrix ? Xsel : MLJBase.table(Xsel; names=fitresult.selected_names)
end

"""
    RFELogisticFilter <: MMI.Supervised

Eliminacion Recursiva de Caracteristicas (RFE) usando Regresion Logistica Multinomial L2
(`MLJLinearModels.MultinomialRegression`), eliminando el 50% de las variables en cada
iteracion (p. ej. 128 -> 64 -> 32) segun la norma L2 de los pesos de cada variable.
"""
mutable struct RFELogisticFilter <: MMI.Supervised
    k::Int
    eliminate_fraction::Float64
    lambda::Float64
end
RFELogisticFilter(; k::Int=32, eliminate_fraction::Float64=0.5, lambda::Float64=1.0) =
    RFELogisticFilter(k, eliminate_fraction, lambda)

function fit_multinomial_l2_nesterov(Xsub::Matrix{Float64}, y_mapped::Vector{Int}, K::Int;
                                     lambda::Float64=1.0, max_iter::Int=250, tol::Float64=1e-6)::Matrix{Float64}
    n, d = size(Xsub)
    # Matriz aumentada con columna de unos para el intercepto (no penalizado por L2)
    X_aug = hcat(Xsub, ones(Float64, n))  # n x (d + 1)
    Y_oh = zeros(Float64, n, K)
    @inbounds for i in 1:n
        Y_oh[i, y_mapped[i]] = 1.0
    end

    # Constante de Lipschitz exacta del gradiente: L = eigmax(X_aug' * X_aug) / (2n) + lambda
    G_sym = Symmetric((X_aug' * X_aug) ./ n)
    L = 0.5 * max(eigmax(G_sym), 1e-8) + lambda
    lr = 1.0 / L

    Θ = zeros(Float64, d + 1, K)
    Θ_prev = zeros(Float64, d + 1, K)
    V = zeros(Float64, d + 1, K)
    Z = Matrix{Float64}(undef, n, K)
    G = Matrix{Float64}(undef, d + 1, K)

    for t in 1:max_iter
        β = (t - 1.0) / (t + 2.0)
        @. V = Θ + β * (Θ - Θ_prev)
        mul!(Z, X_aug, V)
        # Softmax numericamente estable por fila -> Z almacena (P - Y_oh) / n
        inv_n = 1.0 / n
        @inbounds for i in 1:n
            m = Z[i, 1]
            for c in 2:K
                if Z[i, c] > m
                    m = Z[i, c]
                end
            end
            s = 0.0
            for c in 1:K
                e = exp(Z[i, c] - m)
                Z[i, c] = e
                s += e
            end
            inv_s = 1.0 / s
            for c in 1:K
                Z[i, c] = (Z[i, c] * inv_s - Y_oh[i, c]) * inv_n
            end
        end
        mul!(G, transpose(X_aug), Z)
        # Penalizacion L2 solo sobre los pesos W (filas 1:d), no sobre el intercepto (fila d+1)
        @views @. G[1:d, :] += lambda * V[1:d, :]

        if norm(G) < tol
            Θ .= V
            break
        end
        copyto!(Θ_prev, Θ)
        @. Θ = V - lr * G
    end
    return @views Matrix{Float64}(Θ[1:d, :])
end

function MMI.fit(model::RFELogisticFilter, verbosity::Int, X, y)
    Xmat = to_float_matrix(X)
    y_int = to_int_labels(y)
    n, p = size(Xmat)
    classes = sort(unique(y_int))
    K = length(classes)
    class_map = Dict(c => idx for (idx, c) in enumerate(classes))
    y_mapped = [class_map[v] for v in y_int]

    active = collect(1:p)
    target_k = min(model.k, p)
    history = Int[p]
    # Escalar lambda por 1/n de forma coherente con el objetivo promedio por muestra
    λ_eff = model.lambda / max(n, 1)

    while length(active) > target_k
        Xsub = Xmat[:, active]
        d = length(active)
        W = fit_multinomial_l2_nesterov(Xsub, y_mapped, K; lambda=λ_eff, max_iter=250, tol=1e-6)
        importances = [norm(@view W[j, :]) for j in 1:d]

        # Eliminar el 50% inferior en cada iteracion (respetando no bajar de target_k)
        n_keep = max(target_k, ceil(Int, d * (1.0 - model.eliminate_fraction)))
        keep_local = sort(sortperm(importances; rev=true)[1:n_keep])
        active = active[keep_local]
        push!(history, length(active))
    end

    col_names = get_col_names(X, p)
    sel_names = col_names[active]
    return (idxs=active, selected_names=sel_names), nothing, (idxs=active, selected_features=sel_names, rfe_steps=history)
end

function MMI.transform(::RFELogisticFilter, fitresult, X)
    Xmat = to_float_matrix(X)
    Xsel = Xmat[:, fitresult.idxs]
    return X isa AbstractMatrix ? Xsel : MLJBase.table(Xsel; names=fitresult.selected_names)
end

# ==============================================================================
# 3. PROYECCIONES LINEALES (PCA, LDA, ICA) - Ajustadas SOLO en entrenamiento
# ==============================================================================

mutable struct PCAProjector <: MMI.Unsupervised
    pratio::Float64
    maxoutdim::Int
end
PCAProjector(; pratio::Float64=0.95, maxoutdim::Int=128) = PCAProjector(pratio, maxoutdim)

function MMI.fit(model::PCAProjector, verbosity::Int, X)
    Xmat = to_float_matrix(X)
    p = size(Xmat, 2)
    maxd = min(model.maxoutdim, p, size(Xmat, 1) - 1)
    M = MultivariateStats.fit(MultivariateStats.PCA, Xmat'; pratio=model.pratio, maxoutdim=maxd)
    outd = MultivariateStats.outdim(M)
    if outd == 0
        M = MultivariateStats.fit(MultivariateStats.PCA, Xmat'; pratio=1.0, maxoutdim=1)
        outd = 1
    end
    names_out = [Symbol("PC$i") for i in 1:outd]
    var_ratio = MultivariateStats.tprincipalvar(M) / max(MultivariateStats.tvar(M), 1e-12)
    return (M=M, names_out=names_out), nothing, (outdim=outd, explained_variance_ratio=var_ratio)
end

function MMI.transform(::PCAProjector, fitresult, X)
    Xmat = to_float_matrix(X)
    Z = Matrix(MultivariateStats.predict(fitresult.M, Xmat')')
    return X isa AbstractMatrix ? Z : MLJBase.table(Z; names=fitresult.names_out)
end

mutable struct LDAProjector <: MMI.Supervised
    outdim::Int
    regcoef::Float64
end
LDAProjector(; outdim::Int=5, regcoef::Float64=1e-4) = LDAProjector(outdim, regcoef)

function MMI.fit(model::LDAProjector, verbosity::Int, X, y)
    Xmat = to_float_matrix(X)
    y_int = to_int_labels(y)
    classes = sort(unique(y_int))
    K = length(classes)
    cmap = Dict(c => i for (i, c) in enumerate(classes))
    y_mapped = [cmap[c] for c in y_int]
    p = size(Xmat, 2)
    outd = min(model.outdim, K - 1, p)
    X_t = Matrix{Float64}(Xmat')
    # MultivariateStats v0.9/v0.10 compatible: intentar firma con nc (v0.9) y sin nc (v0.10)
    M = try
        MultivariateStats.fit(MultivariateStats.MulticlassLDA, K, X_t, y_mapped; outdim=outd, regcoef=model.regcoef)
    catch
        MultivariateStats.fit(MultivariateStats.MulticlassLDA, X_t, y_mapped; outdim=outd, regcoef=model.regcoef)
    end
    actual_d = MultivariateStats.outdim(M)
    names_out = [Symbol("LD$i") for i in 1:actual_d]
    return (M=M, names_out=names_out), nothing, (outdim=actual_d,)
end

function MMI.transform(::LDAProjector, fitresult, X)
    Xmat = to_float_matrix(X)
    Z = Matrix(MultivariateStats.predict(fitresult.M, Matrix{Float64}(Xmat'))')
    return X isa AbstractMatrix ? Z : MLJBase.table(Z; names=fitresult.names_out)
end

mutable struct ICAProjector <: MMI.Unsupervised
    outdim::Int
    maxiter::Int
    tol::Float64
    seed::Int
end
ICAProjector(; outdim::Int=10, maxiter::Int=500, tol::Float64=1e-3, seed::Int=104) =
    ICAProjector(outdim, maxiter, tol, seed)

function fastica_polar_fallback(Z_std::Matrix{Float64}, W_init::Matrix{Float64};
                                maxiter::Int=400, tol::Float64=1e-4)::Matrix{Float64}
    n, d = size(Z_std)
    W = copy(W_init)
    S = Matrix{Float64}(undef, n, d)
    G = Matrix{Float64}(undef, n, d)
    inv_n = 1.0 / n
    for _ in 1:maxiter
        mul!(S, Z_std, W)
        @. G = tanh(S)
        gp_mean = vec(mean(x -> 1.0 - x * x, G; dims=1))
        W_new = (transpose(Z_std) * G) .* inv_n .- W .* transpose(gp_mean)
        F = svd(W_new)
        W_orth = F.U * F.Vt
        chg = maximum(1.0 .- abs.(diag(transpose(W_orth) * W)))
        W .= W_orth
        chg < tol && break
    end
    return W
end

function MMI.fit(model::ICAProjector, verbosity::Int, X)
    Xmat = to_float_matrix(X)
    p = size(Xmat, 2)
    outd = min(model.outdim, p)
    # Pre-reduccion PCA con blanqueamiento estable sobre el subespacio con varianza real
    pca_M = MultivariateStats.fit(MultivariateStats.PCA, Matrix{Float64}(Xmat'); pratio=1.0, maxoutdim=outd)
    Z_pca = Matrix(MultivariateStats.predict(pca_M, Matrix{Float64}(Xmat'))')
    d_eff = size(Z_pca, 2)
    # Blanqueamiento exacto de las componentes PCA (media 0, covarianza identidad)
    μ = vec(mean(Z_pca; dims=1))
    σ = [max(std(@view Z_pca[:, j]), 1e-8) for j in 1:d_eff]
    Z_std = (Z_pca .- μ') ./ σ'

    rng = MersenneTwister(model.seed)
    F_qr = qr(randn(rng, d_eff, d_eff))
    W0_orth = Matrix(F_qr.Q) * Diagonal(sign.(diag(F_qr.R)))

    ica_M = nothing
    W_fallback = nothing
    Z_t = Matrix{Float64}(Z_std')
    try
        ica_M = MultivariateStats.fit(MultivariateStats.ICA, Z_t, d_eff;
            alg=:deflation, maxiter=model.maxiter, tol=model.tol, do_whiten=false, mean=0.0, winit=W0_orth)
    catch
        W_fallback = fastica_polar_fallback(Z_std, W0_orth; maxiter=model.maxiter, tol=model.tol)
    end
    names_out = [Symbol("IC$i") for i in 1:d_eff]
    return (pca_M=pca_M, μ=μ, σ=σ, ica_M=ica_M, W_fallback=W_fallback, names_out=names_out),
           nothing, (outdim=d_eff, used_fallback=(ica_M === nothing))
end

function MMI.transform(::ICAProjector, fitresult, X)
    Xmat = to_float_matrix(X)
    Z_pca = Matrix(MultivariateStats.predict(fitresult.pca_M, Matrix{Float64}(Xmat'))')
    Z_std = (Z_pca .- fitresult.μ') ./ fitresult.σ'
    Z_ica = if fitresult.ica_M !== nothing
        Matrix(MultivariateStats.predict(fitresult.ica_M, Matrix{Float64}(Z_std'))')
    else
        Z_std * fitresult.W_fallback
    end
    return X isa AbstractMatrix ? Z_ica : MLJBase.table(Z_ica; names=fitresult.names_out)
end

# ==============================================================================
# 4. CLASIFICADORES BASE Y ENSEMBLES EN JULIA PURO / MLJ
# ==============================================================================

"""
    FluxMLPClassifier <: MMI.Probabilistic

Perceptron Multicapa (MLP) determinista implementado con `Flux.jl`, compatible con MLJ.
Soporta arquitecturas arbitrarias como `[50]`, `[100]` y `[100, 50]`, regularizacion L2
y mini-batching reproducible con semilla fija.
"""
mutable struct FluxMLPClassifier <: MMI.Probabilistic
    hidden::Vector{Int}
    epochs::Int
    batch_size::Int
    lr::Float64
    weight_decay::Float64
    seed::Int
end
FluxMLPClassifier(; hidden::Vector{Int}=[50], epochs::Int=80, batch_size::Int=128,
                  lr::Float64=2e-3, weight_decay::Float64=1e-4, seed::Int=104) =
    FluxMLPClassifier(hidden, epochs, batch_size, lr, weight_decay, seed)

function MMI.fit(model::FluxMLPClassifier, verbosity::Int, X, y)
    rng = MersenneTwister(model.seed)
    Xmat = Float32.(to_float_matrix(X))
    y_int = to_int_labels(y)
    n, d_in = size(Xmat)
    class_levels = collect(1:6)
    K = length(class_levels)

    # Construir arquitectura Dense + ReLU -> salida lineal (logits)
    layers = Any[]
    prev = d_in
    for h in model.hidden
        W = Float32.(randn(rng, h, prev) .* sqrt(2.0 / prev))
        b = zeros(Float32, h)
        push!(layers, Dense(W, b, relu))
        prev = h
    end
    W_out = Float32.(randn(rng, K, prev) .* sqrt(2.0 / prev))
    b_out = zeros(Float32, K)
    push!(layers, Dense(W_out, b_out))
    net = Chain(layers...)

    # Codificacion one-hot (K x n)
    Y_oh = zeros(Float32, K, n)
    @inbounds for i in 1:n
        Y_oh[y_int[i], i] = 1.0f0
    end
    Xt = permutedims(Xmat)  # d_in x n

    opt_rule = Flux.OptimiserChain(Flux.WeightDecay(Float32(model.weight_decay)), Flux.Adam(Float32(model.lr)))
    opt_state = Flux.setup(opt_rule, net)

    bs = min(model.batch_size, n)
    for ep in 1:model.epochs
        perm = randperm(rng, n)
        for start in 1:bs:n
            stop = min(start + bs - 1, n)
            idx = @view perm[start:stop]
            xb = Xt[:, idx]
            yb = Y_oh[:, idx]
            grads = Flux.gradient(net) do m
                Flux.logitcrossentropy(m(xb), yb)
            end
            Flux.update!(opt_state, net, grads[1])
        end
    end

    return (net=net, class_levels=class_levels), nothing, (hidden=model.hidden, epochs=model.epochs)
end

function predict_proba_matrix(model::FluxMLPClassifier, fitresult, X)::Matrix{Float64}
    Xmat = Float32.(to_float_matrix(X))
    logits = fitresult.net(permutedims(Xmat))
    probs = Float64.(Flux.softmax(logits; dims=1))
    return permutedims(probs)  # n x K
end

function MMI.predict(model::FluxMLPClassifier, fitresult, X)
    P = predict_proba_matrix(model, fitresult, X)
    lvls = fitresult.class_levels
    return [MLJBase.UnivariateFinite(lvls, P[i, :]) for i in 1:size(P, 1)]
end

function predict_class_int(model::FluxMLPClassifier, fitresult, X)::Vector{Int}
    P = predict_proba_matrix(model, fitresult, X)
    return [argmax(@view P[i, :]) for i in 1:size(P, 1)]
end

"""
    SAMMEAdaBoostStumpClassifier <: MMI.Probabilistic

Implementacion multiclase rigurosa de AdaBoost con Decision Stumps mediante el algoritmo
SAMME (Zhu et al., 2009: Stagewise Additive Modeling using a Multi-class Exponential loss):
- Soporta K >= 2 clases (a diferencia de `DecisionTree.build_adaboost_stumps`, limitado a 2 clases).
- Cada clasificador debil es un arbol de profundidad 1 (Stump Tree) optimizado sobre la
  impureza de Gini ponderada por los pesos w_i de las instancias.
"""
mutable struct SAMMEAdaBoostStumpClassifier <: MMI.Probabilistic
    n_iter::Int
    n_thresholds::Int
end
SAMMEAdaBoostStumpClassifier(; n_iter::Int=5, n_thresholds::Int=32) =
    SAMMEAdaBoostStumpClassifier(n_iter, n_thresholds)

struct WeightedDecisionStump
    feature::Int
    threshold::Float64
    left_class::Int
    right_class::Int
end

function fit_weighted_stump(Xmat::Matrix{Float64}, y_int::Vector{Int}, w::Vector{Float64}, K::Int, n_thresholds::Int)
    n, p = size(Xmat)
    best_gini = Inf
    best_feat = 1
    best_thr = 0.0
    best_left = 1
    best_right = 1

    total_w_cls = zeros(Float64, K)
    @inbounds for i in 1:n
        total_w_cls[y_int[i]] += w[i]
    end
    total_w = sum(total_w_cls)

    left_w_cls = zeros(Float64, K)
    right_w_cls = zeros(Float64, K)
    probs_q = collect(range(0.05, 0.95; length=n_thresholds))

    for j in 1:p
        col = @view Xmat[:, j]
        thrs = unique(quantile(col, probs_q))
        for thr in thrs
            fill!(left_w_cls, 0.0)
            @inbounds for i in 1:n
                if col[i] <= thr
                    left_w_cls[y_int[i]] += w[i]
                end
            end
            wl = sum(left_w_cls)
            wr = total_w - wl
            (wl <= 1e-12 || wr <= 1e-12) && continue

            g_l = 1.0
            g_r = 1.0
            @inbounds for c in 1:K
                rc = total_w_cls[c] - left_w_cls[c]
                right_w_cls[c] = rc
                pl = left_w_cls[c] / wl
                pr = rc / wr
                g_l -= pl * pl
                g_r -= pr * pr
            end
            gini = (wl * g_l + wr * g_r) / total_w
            if gini < best_gini
                best_gini = gini
                best_feat = j
                best_thr = thr
                best_left = argmax(left_w_cls)
                best_right = argmax(right_w_cls)
            end
        end
    end
    return WeightedDecisionStump(best_feat, best_thr, best_left, best_right)
end

function predict_stump(s::WeightedDecisionStump, Xmat::Matrix{Float64})::Vector{Int}
    n = size(Xmat, 1)
    out = Vector{Int}(undef, n)
    j = s.feature
    thr = s.threshold
    lc = s.left_class
    rc = s.right_class
    @inbounds for i in 1:n
        out[i] = Xmat[i, j] <= thr ? lc : rc
    end
    return out
end

function MMI.fit(model::SAMMEAdaBoostStumpClassifier, verbosity::Int, X, y)
    Xmat = to_float_matrix(X)
    y_int = to_int_labels(y)
    n = size(Xmat, 1)
    K = 6
    w = fill(1.0 / n, n)
    stumps = WeightedDecisionStump[]
    alphas = Float64[]

    for m in 1:model.n_iter
        stump = fit_weighted_stump(Xmat, y_int, w, K, model.n_thresholds)
        preds = predict_stump(stump, Xmat)
        err = 0.0
        @inbounds for i in 1:n
            if preds[i] != y_int[i]
                err += w[i]
            end
        end
        err = clamp(err, 1e-10, 1.0 - 1.0 / K - 1e-6)
        # Factor SAMME multiclase: alpha = log((1 - err)/err) + log(K - 1)
        alpha = log((1.0 - err) / err) + log(K - 1.0)
        push!(stumps, stump)
        push!(alphas, alpha)

        # Actualizar y normalizar pesos
        @inbounds for i in 1:n
            if preds[i] != y_int[i]
                w[i] *= exp(alpha)
            end
        end
        w ./= sum(w)
    end
    return (stumps=stumps, alphas=alphas, K=K), nothing, (n_estimators=length(stumps), alphas=alphas)
end

function predict_proba_matrix(::SAMMEAdaBoostStumpClassifier, fitresult, X)::Matrix{Float64}
    Xmat = to_float_matrix(X)
    n = size(Xmat, 1)
    K = fitresult.K
    scores = zeros(Float64, n, K)
    for (stump, alpha) in zip(fitresult.stumps, fitresult.alphas)
        preds = predict_stump(stump, Xmat)
        @inbounds for i in 1:n
            scores[i, preds[i]] += alpha
        end
    end
    # Softmax por fila para probabilidades calibradas
    P = Matrix{Float64}(undef, n, K)
    for i in 1:n
        m = maximum(@view scores[i, :])
        exps = exp.((@view scores[i, :]) .- m)
        P[i, :] .= exps ./ sum(exps)
    end
    return P
end

function predict_class_int(model::SAMMEAdaBoostStumpClassifier, fitresult, X)::Vector{Int}
    P = predict_proba_matrix(model, fitresult, X)
    return [argmax(@view P[i, :]) for i in 1:size(P, 1)]
end

# ==============================================================================
# 5. PIPELINE UNIFICADO LEAK-FREE (Imputacion -> Escalado -> Filtro -> Proyeccion -> Modelo)
# ==============================================================================

"""
    GasClassificationPipeline

Encapsula un flujo completo anti-leakage donde TODAS las transformaciones
(MedianImputer -> MinMaxScaler -> [Filtro opcional] -> [Proyeccion opcional] -> Clasificador)
se ajustan exclusivamente sobre `(X_train, y_train)` y se aplican despues sobre `X_eval`.
"""
struct GasClassificationPipeline
    name::String
    scaler_method::Symbol
    filter_spec::Symbol      # :none, :anova, :pearson, :spearman, :kendall, :mi, :rfe
    filter_k::Int
    proj_spec::Symbol        # :none, :pca, :lda, :ica
    proj_param::Any          # pratio para PCA, outdim para LDA/ICA
    clf_spec::Symbol         # :svm, :knn, :mlp, :bagging_knn, :adaboost, :evotree, :rf, :xgboost, :lightgbm, :catboost
    clf_params::Dict{Symbol, Any}
end

function build_filter(spec::Symbol, k::Int)
    if spec == :none
        return nothing
    elseif spec == :anova
        return ANOVAFilter(k=k)
    elseif spec == :pearson
        return CorrelationFilter(k=k, method=:pearson)
    elseif spec == :spearman
        return CorrelationFilter(k=k, method=:spearman)
    elseif spec == :kendall
        return CorrelationFilter(k=k, method=:kendall)
    elseif spec == :mi
        return MutualInfoFilter(k=k, n_bins=16)
    elseif spec == :rfe
        return RFELogisticFilter(k=k, eliminate_fraction=0.5, lambda=1.0)
    else
        error("Filtro desconocido: $spec")
    end
end

function build_projector(spec::Symbol, param)
    if spec == :none
        return nothing
    elseif spec == :pca
        pr = param isa AbstractFloat ? Float64(param) : 0.95
        return PCAProjector(pratio=pr, maxoutdim=128)
    elseif spec == :lda
        od = param isa Integer ? Int(param) : 5
        return LDAProjector(outdim=od, regcoef=1e-4)
    elseif spec == :ica
        od = param isa Integer ? Int(param) : 12
        return ICAProjector(outdim=od, maxiter=500, tol=1e-3, seed=104)
    else
        error("Proyeccion desconocida: $spec")
    end
end

# Exact BLAS-accelerated Euclidean k-NN probability predictor (avoids KD-Tree curse of dimensionality at d=128)
function knn_proba_blas(Xtr::Matrix{Float64}, ytr::Vector{Int}, Xev::Matrix{Float64}, k::Int; K::Int=6)::Matrix{Float64}
    n_tr = size(Xtr, 1)
    n_ev = size(Xev, 1)
    k_eff = min(k, n_tr)
    sq_tr = vec(sum(abs2, Xtr; dims=2))
    P = zeros(Float64, n_ev, K)
    inv_k = 1.0 / k_eff

    chunk = 1024
    for start in 1:chunk:n_ev
        stop = min(start + chunk - 1, n_ev)
        len = stop - start + 1
        Xblk = @view Xev[start:stop, :]
        sq_ev = vec(sum(abs2, Xblk; dims=2))
        D = Matrix{Float64}(undef, n_tr, len)
        mul!(D, Xtr, transpose(Xblk))
        @inbounds for j in 1:len
            se = sq_ev[j]
            for i in 1:n_tr
                D[i, j] = max(0.0, sq_tr[i] + se - 2.0 * D[i, j])
            end
        end
        for j in 1:len
            nn = partialsortperm(@view(D[:, j]), 1:k_eff)
            row_out = start + j - 1
            @inbounds for idx in nn
                P[row_out, ytr[idx]] += inv_k
            end
        end
    end
    return P
end

# Auxiliares de entrenamiento/prediccion uniforme para clasificadores
function fit_classifier(spec::Symbol, params::Dict{Symbol, Any}, Xtr::Matrix{Float64}, ytr::Vector{Int})
    seed = get(params, :seed, 104)
    if spec == :svm
        C = Float64(get(params, :C, 1.0))
        prob = Bool(get(params, :probability, false))
        # LIBSVM nativo espera matrices d x n (columnas = muestras), con gamma = 1/d por defecto
        m = LIBSVM.svmtrain(permutedims(Xtr), ytr;
            svmtype=LIBSVM.SVC, kernel=LIBSVM.Kernel.RadialBasis,
            cost=C, cachesize=800.0, probability=prob, verbose=false)
        return (spec=spec, model=m, probability=prob)

    elseif spec == :knn
        k = Int(get(params, :k, 10))
        return (spec=spec, Xtr=copy(Xtr), ytr=copy(ytr), k=k)

    elseif spec == :mlp
        hidden = Vector{Int}(get(params, :hidden, [50]))
        epochs = Int(get(params, :epochs, 80))
        lr = Float64(get(params, :lr, 2e-3))
        mlp = FluxMLPClassifier(hidden=hidden, epochs=epochs, batch_size=128, lr=lr, weight_decay=1e-4, seed=seed)
        fr, _, _ = MMI.fit(mlp, 0, Xtr, ytr)
        return (spec=spec, model=mlp, fitresult=fr)

    elseif spec == :bagging_knn
        k = Int(get(params, :k, 5))
        n_est = Int(get(params, :n_estimators, 10))
        bag_frac = Float64(get(params, :bagging_fraction, 0.8))
        rng = MersenneTwister(seed)
        n_tr = size(Xtr, 1)
        m_sub = max(k, round(Int, bag_frac * n_tr))
        bags = [rand(rng, 1:n_tr, m_sub) for _ in 1:n_est]
        return (spec=spec, Xtr=copy(Xtr), ytr=copy(ytr), k=k, bags=bags)

    elseif spec == :adaboost
        n_iter = Int(get(params, :n_iter, 5))
        ada = SAMMEAdaBoostStumpClassifier(n_iter=n_iter, n_thresholds=32)
        fr, _, _ = MMI.fit(ada, 0, Xtr, ytr)
        return (spec=spec, model=ada, fitresult=fr)

    elseif spec == :evotree
        nrounds = Int(get(params, :nrounds, 50))
        eta = Float64(get(params, :eta, 0.2))
        max_depth = Int(get(params, :max_depth, 6))
        Xtbl = MLJBase.table(Xtr)
        ycat = categorical(ytr; levels=1:6)
        evo = EvoTrees.EvoTreeClassifier(nrounds=nrounds, eta=eta, max_depth=max_depth, seed=seed)
        mach = MLJ.machine(evo, Xtbl, ycat)
        MLJ.fit!(mach; verbosity=0)
        return (spec=spec, mach=mach)

    elseif spec == :rf
        n_trees = Int(get(params, :n_trees, 500))
        max_depth = Int(get(params, :max_depth, 10))
        n_sub = Int(get(params, :n_subfeatures, -1))
        n_sub_eff = n_sub == -1 ? max(1, round(Int, sqrt(size(Xtr, 2)))) : n_sub
        forest = DecisionTree.build_forest(
            ytr, Xtr, n_sub_eff, n_trees, 0.7, max_depth;
            rng=MersenneTwister(seed)
        )
        imp = DecisionTree.impurity_importance(forest)
        return (spec=spec, forest=forest, importances=imp)

    elseif spec == :xgboost
        # Valores por defecto estandar de XGBoost multiclase (100 arboles, etiquetas 0..5)
        y0 = Float32.(ytr .- 1)
        dtrain = XGBoost.DMatrix(Float32.(Xtr); label=y0)
        bst = XGBoost.xgboost(dtrain; num_round=100, objective="multi:softprob", num_class=6, seed=seed, verbosity=0, watchlist=NamedTuple())
        return (spec=spec, bst=bst, n_features=size(Xtr, 2))

    elseif spec == :lightgbm
        # Valores por defecto de LightGBM multiclase (etiquetas 0..5)
        y0 = ytr .- 1
        est = LightGBM.LGBMClassification(
            objective="multiclass",
            num_class=6,
            seed=seed,
            data_random_seed=seed,
            feature_fraction_seed=seed,
            bagging_seed=seed,
            verbosity=-1
        )
        LightGBM.fit!(est, Xtr, y0; verbosity=-1)
        return (spec=spec, est=est)

    elseif spec == :catboost
        cb = pyimport("catboost")
        clf = cb.CatBoostClassifier(random_seed=seed, verbose=false, allow_writing_files=false)
        clf.fit(Xtr, ytr .- 1)
        return (spec=spec, clf=clf)

    else
        error("Clasificador desconocido: $spec")
    end
end

function predict_classifier(fitted, Xev::Matrix{Float64})::Tuple{Vector{Int}, Matrix{Float64}}
    spec = fitted.spec
    n = size(Xev, 1)
    K = 6

    if spec == :svm
        preds, dec = LIBSVM.svmpredict(fitted.model, permutedims(Xev))
        preds_int = Int.(preds)
        P = zeros(Float64, n, K)
        if fitted.probability && size(dec, 1) == K
            # LIBSVM devuelve probabilidades ordenadas segun fitted.model.labels
            lbls = Int.(fitted.model.labels)
            for (row_idx, cls_lbl) in enumerate(lbls)
                P[:, cls_lbl] .= dec[row_idx, :]
            end
        else
            for i in 1:n
                P[i, preds_int[i]] = 1.0
            end
        end
        return preds_int, P

    elseif spec == :knn
        P = knn_proba_blas(fitted.Xtr, fitted.ytr, Xev, fitted.k; K=K)
        preds_int = [argmax(@view P[i, :]) for i in 1:n]
        return preds_int, P

    elseif spec == :bagging_knn
        P = zeros(Float64, n, K)
        nb = length(fitted.bags)
        for bag_idx in fitted.bags
            P .+= knn_proba_blas(fitted.Xtr[bag_idx, :], fitted.ytr[bag_idx], Xev, fitted.k; K=K)
        end
        P ./= nb
        preds_int = [argmax(@view P[i, :]) for i in 1:n]
        return preds_int, P

    elseif spec == :evotree
        Xtbl = MLJBase.table(Xev)
        dists = MLJ.predict(fitted.mach, Xtbl)
        P = zeros(Float64, n, K)
        preds_int = Vector{Int}(undef, n)
        for i in 1:n
            d = dists[i]
            for c in 1:K
                P[i, c] = pdf(d, c)
            end
            preds_int[i] = argmax(@view P[i, :])
        end
        return preds_int, P

    elseif spec in (:mlp, :adaboost)
        P = predict_proba_matrix(fitted.model, fitted.fitresult, Xev)
        preds_int = [argmax(@view P[i, :]) for i in 1:n]
        return preds_int, P

    elseif spec == :rf
        preds_int = Int.(DecisionTree.apply_forest(fitted.forest, Xev))
        P = Float64.(DecisionTree.apply_forest_proba(fitted.forest, Xev, collect(1:K)))
        return preds_int, P

    elseif spec == :xgboost
        raw = XGBoost.predict(fitted.bst, Float32.(Xev))
        if raw isa AbstractMatrix
            # XGBoost.jl v2 devuelve matriz n x K para multi:softprob
            P = size(raw, 1) == n ? Matrix{Float64}(raw) : Matrix{Float64}(permutedims(raw))
        else
            P = permutedims(reshape(Float64.(raw), K, n))
        end
        preds_int = [argmax(@view P[i, :]) for i in 1:n]
        return preds_int, P

    elseif spec == :lightgbm
        P = Matrix{Float64}(LightGBM.predict(fitted.est, Xev; verbosity=-1))
        preds_int = [argmax(@view P[i, :]) for i in 1:n]
        return preds_int, P

    elseif spec == :catboost
        P = Matrix{Float64}(fitted.clf.predict_proba(Xev))
        preds_int = [argmax(@view P[i, :]) for i in 1:n]
        return preds_int, P

    else
        error("Clasificador desconocido: $spec")
    end
end

"""
    fit_pipeline(pipe::GasClassificationPipeline, Xtr::Matrix{Float64}, ytr::Vector{Int})

Ajusta el pipeline completo estrictamente sobre entrenamiento (zero data leakage).
"""
function fit_pipeline(pipe::GasClassificationPipeline, Xtr::Matrix{Float64}, ytr::Vector{Int})
    # 1. Imputacion
    imp = MedianImputer()
    imp_fr, _, _ = MMI.fit(imp, 0, Xtr)
    Z = MMI.transform(imp, imp_fr, Xtr)

    # 2. Escalado
    scl = MinMaxScaler(method=pipe.scaler_method)
    scl_fr, _, _ = MMI.fit(scl, 0, Z)
    Z = MMI.transform(scl, scl_fr, Z)

    # 3. Filtro de caracteristicas (opcional, supervisado)
    flt = build_filter(pipe.filter_spec, pipe.filter_k)
    flt_fr = nothing
    flt_rep = nothing
    if flt !== nothing
        flt_fr, _, flt_rep = MMI.fit(flt, 0, Z, ytr)
        Z = MMI.transform(flt, flt_fr, Z)
    end

    # 4. Proyeccion (opcional)
    prj = build_projector(pipe.proj_spec, pipe.proj_param)
    prj_fr = nothing
    prj_rep = nothing
    if prj !== nothing
        if prj isa LDAProjector
            prj_fr, _, prj_rep = MMI.fit(prj, 0, Z, ytr)
        else
            prj_fr, _, prj_rep = MMI.fit(prj, 0, Z)
        end
        Z = MMI.transform(prj, prj_fr, Z)
    end

    # 5. Clasificador
    clf_fitted = fit_classifier(pipe.clf_spec, pipe.clf_params, Z, ytr)

    return (
        pipe = pipe,
        imp = imp, imp_fr = imp_fr,
        scl = scl, scl_fr = scl_fr,
        flt = flt, flt_fr = flt_fr, flt_rep = flt_rep,
        prj = prj, prj_fr = prj_fr, prj_rep = prj_rep,
        clf_fitted = clf_fitted,
        n_features_final = size(Z, 2)
    )
end

function predict_pipeline(state, Xev::Matrix{Float64})::Tuple{Vector{Int}, Matrix{Float64}}
    Z = MMI.transform(state.imp, state.imp_fr, Xev)
    Z = MMI.transform(state.scl, state.scl_fr, Z)
    if state.flt !== nothing
        Z = MMI.transform(state.flt, state.flt_fr, Z)
    end
    if state.prj !== nothing
        Z = MMI.transform(state.prj, state.prj_fr, Z)
    end
    return predict_classifier(state.clf_fitted, Z)
end

# ==============================================================================
# 6. ENSEMBLES COMPUESTOS: HARD VOTING SVM (3 particiones, semilla 104) Y STACKING
# ==============================================================================

"""
    fit_predict_hard_voting_svm(best_svm_pipe, Xtr, ytr, Xte; seed=104)

Cumple exactamente el enunciado:
"Hard Voting: con SVM como modelo base, particione en 3 el conjunto de entrenamiento con
el valor 104 y ejecute la votacion por mayoria".
Cada una de las 3 SVM ajusta su propio pipeline completo (imputador + escalador + filtro +
proyeccion + SVM) unicamente sobre su particion 1/3. En caso de empate a 1 voto entre 3
clases, desempata usando la mayor probabilidad agregada de los 3 modelos.
"""
function fit_predict_hard_voting_svm(
    best_svm_pipe::GasClassificationPipeline,
    Xtr::Matrix{Float64},
    ytr::Vector{Int},
    Xte::Matrix{Float64};
    seed::Int=104
)
    rng = MersenneTwister(seed)
    n = length(ytr)
    # Particion estratificada en 3 subconjuntos disjuntos con semilla 104
    part_ids = zeros(Int, n)
    for c in sort(unique(ytr))
        c_idx = findall(==(c), ytr)
        perm = c_idx[randperm(rng, length(c_idx))]
        for (k, idx) in enumerate(perm)
            part_ids[idx] = mod1(k, 3)
        end
    end

    # Activar probabilidad para desempate limpio si los 3 votan clases distintas
    svm_params_prob = copy(best_svm_pipe.clf_params)
    svm_params_prob[:probability] = true
    pipe_p = GasClassificationPipeline(
        best_svm_pipe.name * "_hv",
        best_svm_pipe.scaler_method,
        best_svm_pipe.filter_spec,
        best_svm_pipe.filter_k,
        best_svm_pipe.proj_spec,
        best_svm_pipe.proj_param,
        :svm,
        svm_params_prob
    )

    votes = Matrix{Int}(undef, size(Xte, 1), 3)
    prob_sum = zeros(Float64, size(Xte, 1), 6)
    part_sizes = Int[]

    for p_id in 1:3
        idx = findall(==(p_id), part_ids)
        push!(part_sizes, length(idx))
        st = fit_pipeline(pipe_p, Xtr[idx, :], ytr[idx])
        pr, pr_prob = predict_pipeline(st, Xte)
        votes[:, p_id] .= pr
        prob_sum .+= pr_prob
    end

    # Votacion mayoritaria dura (Hard Voting)
    n_te = size(Xte, 1)
    final_preds = Vector{Int}(undef, n_te)
    ties = 0
    for i in 1:n_te
        v1, v2, v3 = votes[i, 1], votes[i, 2], votes[i, 3]
        if v1 == v2 || v1 == v3
            final_preds[i] = v1
        elseif v2 == v3
            final_preds[i] = v2
        else
            ties += 1
            # Empate 1-1-1: desempatar entre las 3 candidatas por suma de probabilidades
            cands = (v1, v2, v3)
            final_preds[i] = cands[argmax([prob_sum[i, c] for c in cands])]
        end
    end
    return final_preds, (part_sizes=part_sizes, ties=ties, individual_votes=votes)
end

"""
    fit_predict_batch_stacking(svm_pipe, knn_pipe, mlp_pipe, Xtr, ytr, folds, Xte; seed=104)

Stacking Ensemble libre de data leakage:
- Modelos base: mejor SVM (probabilistico) + mejor KNN + mejor MLP.
- Meta-clasificador final: MLP (`[50]`).
- Las meta-caracteristicas de entrenamiento (18 probabilidades: 3 modelos x 6 clases) se
  generan EXCLUSIVAMENTE mediante predicciones out-of-fold respetando la estructura temporal
  batch-wise (los 5 folds de validacion cruzada sobre Batches 1..9), evitando sobreajuste
  del meta-clasificador.
- Finalmente, los 3 modelos base se reentrenan sobre el conjunto completo de entrenamiento
  (Batches 1..9) para generar las meta-caracteristicas del conjunto de test (Batch 10).
"""
function fit_predict_batch_stacking(
    svm_pipe::GasClassificationPipeline,
    knn_pipe::GasClassificationPipeline,
    mlp_pipe::GasClassificationPipeline,
    Xtr::Matrix{Float64},
    ytr::Vector{Int},
    folds::Vector{Tuple{Vector{Int}, Vector{Int}}},
    Xte::Matrix{Float64};
    seed::Int=104
)
    # Asegurar salida probabilistica en SVM base
    svm_p = copy(svm_pipe.clf_params)
    svm_p[:probability] = true
    svm_prob_pipe = GasClassificationPipeline(
        svm_pipe.name * "_prob", svm_pipe.scaler_method,
        svm_pipe.filter_spec, svm_pipe.filter_k,
        svm_pipe.proj_spec, svm_pipe.proj_param, :svm, svm_p
    )
    base_pipes = [svm_prob_pipe, knn_pipe, mlp_pipe]

    # 1. Generar meta-features OOF en los folds batch-wise
    oof_X_blocks = Matrix{Float64}[]
    oof_y_blocks = Vector{Int}[]
    for (tr_idx, va_idx) in folds
        n_va = length(va_idx)
        meta_va = Matrix{Float64}(undef, n_va, 18)
        for (m_i, bp) in enumerate(base_pipes)
            st = fit_pipeline(bp, Xtr[tr_idx, :], ytr[tr_idx])
            _, P_va = predict_pipeline(st, Xtr[va_idx, :])
            meta_va[:, (m_i - 1) * 6 + 1 : m_i * 6] .= P_va
        end
        push!(oof_X_blocks, meta_va)
        push!(oof_y_blocks, ytr[va_idx])
    end
    Z_meta_tr = vcat(oof_X_blocks...)
    y_meta_tr = vcat(oof_y_blocks...)

    # 2. Entrenar meta-clasificador MLP sobre predicciones OOF
    meta_mlp = FluxMLPClassifier(hidden=[50], epochs=100, batch_size=128, lr=2e-3, weight_decay=1e-3, seed=seed)
    meta_fr, _, _ = MMI.fit(meta_mlp, 0, Z_meta_tr, y_meta_tr)

    # 3. Reentrenar los 3 modelos base en TODO el train (Batches 1..9) y predecir test
    n_te = size(Xte, 1)
    Z_meta_te = Matrix{Float64}(undef, n_te, 18)
    for (m_i, bp) in enumerate(base_pipes)
        st_full = fit_pipeline(bp, Xtr, ytr)
        _, P_te = predict_pipeline(st_full, Xte)
        Z_meta_te[:, (m_i - 1) * 6 + 1 : m_i * 6] .= P_te
    end

    P_final = predict_proba_matrix(meta_mlp, meta_fr, Z_meta_te)
    preds_final = [argmax(@view P_final[i, :]) for i in 1:n_te]
    return preds_final, P_final
end

# ==============================================================================
# 7. WRAPPERS 2D OUT-OF-SAMPLE PARA VISUALIZACION (t-SNE, Isomap, LLE)
#    Ajuste ESTRICTAMENTE sobre entrenamiento; proyeccion out-of-sample sobre test.
# ==============================================================================

"""
    stratified_subsample_indices(y::Vector{Int}, per_class::Int; seed::Int=104)

Selecciona un subconjunto estratificado balanceado de entrenamiento para metodos de
variedades O(n^2)/O(n^3) (t-SNE, Isomap, LLE), manteniendo reproducibilidad total.
"""
function stratified_subsample_indices(y::Vector{Int}, per_class::Int; seed::Int=104)::Vector{Int}
    rng = MersenneTwister(seed)
    chosen = Int[]
    for c in sort(unique(y))
        c_idx = findall(==(c), y)
        take = min(per_class, length(c_idx))
        append!(chosen, c_idx[randperm(rng, length(c_idx))[1:take]])
    end
    sort!(chosen)
    return chosen
end

"""
    oos_knn_project(X_ref, Y_ref, X_new; k=15, mode=:idw)

Proyeccion out-of-sample estricta (sin reajustar sobre test):
- `:idw`: regresion ponderada por distancia inversa / kernel gaussiano adaptativo sobre los
  `k` vecinos mas cercanos en `X_ref` (para Isomap y t-SNE).
- `:lle`: pesos de reconstruccion lineal local de Saul & Roweis (2003) resolviendo
  min_w ||x - X_N w||^2 + eps ||w||^2 sujeto a sum(w) = 1 (extension canonica OOS de LLE).
"""
function oos_knn_project(X_ref::Matrix{Float64}, Y_ref::Matrix{Float64}, X_new::Matrix{Float64};
                         k::Int=15, mode::Symbol=:idw)::Matrix{Float64}
    D = Distances.pairwise(Euclidean(), X_ref', X_new'; dims=2)  # n_ref x n_new
    n_new = size(X_new, 1)
    d_out = size(Y_ref, 2)
    Y_new = Matrix{Float64}(undef, n_new, d_out)

    for i in 1:n_new
        dists = @view D[:, i]
        nn_idx = partialsortperm(dists, 1:k)
        nn_d = dists[nn_idx]

        # Coincidencia exacta con punto de referencia
        if nn_d[1] < 1e-12
            Y_new[i, :] .= @view Y_ref[nn_idx[1], :]
            continue
        end

        if mode == :lle
            # Reconstruccion lineal local (Saul & Roweis 2003)
            # Z_k: k x p diferencias (x_j - x_new)
            Z = X_ref[nn_idx, :] .- (@view X_new[i, :])'
            G = Z * Z'  # matriz de Gram k x k
            tr_g = LinearAlgebra.tr(G)
            reg = max(1e-3 * tr_g, 1e-6)
            @inbounds for a in 1:k
                G[a, a] += reg
            end
            w = G \ ones(Float64, k)
            s = sum(w)
            w = abs(s) > 1e-12 ? (w ./ s) : fill(1.0 / k, k)
            Y_new[i, :] .= vec(w' * (@view Y_ref[nn_idx, :]))
        else
            # Kernel gaussiano adaptativo (escala local = mediana de distancias k-NN)
            σ = max(median(nn_d), 1e-6)
            w = exp.(-0.5 .* (nn_d ./ σ).^2)
            w ./= sum(w)
            Y_new[i, :] .= vec(w' * (@view Y_ref[nn_idx, :]))
        end
    end
    return Y_new
end

mutable struct OutOfSampleManifold2D <: MMI.Unsupervised
    method::Symbol     # :tsne, :isomap, :lle
    k_neighbors::Int
    perplexity::Float64
    max_iter::Int
    seed::Int
end
OutOfSampleManifold2D(; method::Symbol=:isomap, k_neighbors::Int=15,
                      perplexity::Float64=30.0, max_iter::Int=500, seed::Int=104) =
    OutOfSampleManifold2D(method, k_neighbors, perplexity, max_iter, seed)

function MMI.fit(model::OutOfSampleManifold2D, verbosity::Int, X_sub)
    Xmat = to_float_matrix(X_sub)
    # Pre-compresion suave con PCA (hasta 30 dims, >99.5% varianza) para estabilidad numerica
    p = size(Xmat, 2)
    pca_pre = MultivariateStats.fit(MultivariateStats.PCA, Xmat'; pratio=0.995, maxoutdim=min(30, p))
    X_pre = Matrix(MultivariateStats.predict(pca_pre, Xmat')')

    Y_emb, ref_idx = if model.method == :isomap
        M = ManifoldLearning.fit(ManifoldLearning.Isomap, X_pre'; k=model.k_neighbors, maxoutdim=2)
        Matrix{Float64}(MultivariateStats.predict(M)'), collect(Int, M.component)
    elseif model.method == :lle
        M = ManifoldLearning.fit(ManifoldLearning.LLE, X_pre'; k=model.k_neighbors, maxoutdim=2)
        Matrix{Float64}(MultivariateStats.predict(M)'), collect(Int, M.component)
    elseif model.method == :tsne
        Random.seed!(model.seed)
        Matrix{Float64}(TSne.tsne(X_pre, 2, 0, model.max_iter, model.perplexity; progress=false)), collect(1:size(X_pre, 1))
    else
        error("Metodo manifold desconocido: $(model.method)")
    end

    # Alinear exactamente las muestras retenidas por la componente conexa mayor
    X_ref = X_pre[ref_idx, :]
    return (pca_pre=pca_pre, X_ref=X_ref, Y_ref=Y_emb, ref_idx=ref_idx, method=model.method, k=model.k_neighbors),
           nothing, (n_ref=length(ref_idx),)
end

function MMI.transform(::OutOfSampleManifold2D, fitresult, X_new)
    Xmat = to_float_matrix(X_new)
    X_pre = Matrix(MultivariateStats.predict(fitresult.pca_pre, Xmat')')
    mode = fitresult.method == :lle ? :lle : :idw
    return oos_knn_project(fitresult.X_ref, fitresult.Y_ref, X_pre; k=fitresult.k, mode=mode)
end
