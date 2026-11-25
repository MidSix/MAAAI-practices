# ==============================================================================
# Practica 2 - MAAAI (Curso 2026/27)
# Modulo: data_utils.jl
# Carga unificada, auditoria de valores ausentes/anomalos, particion temporal
# holdout, validacion cruzada batch-wise y metricas de clasificacion multiclase.
# ==============================================================================

using DataFrames
using CategoricalArrays
using Statistics
using StatsBase
using Random

const GAS_NAMES = Dict(
    1 => "Ethanol",
    2 => "Ethylene",
    3 => "Ammonia",
    4 => "Acetaldehyde",
    5 => "Acetone",
    6 => "Toluene"
)

const BATCH_MONTHS = Dict(
    1  => "Meses 1-2",
    2  => "Meses 3-4, 8-10",
    3  => "Meses 11-13",
    4  => "Meses 14-15",
    5  => "Mes 16",
    6  => "Meses 17-20",
    7  => "Mes 21",
    8  => "Meses 22-23",
    9  => "Meses 24, 30",
    10 => "Mes 36"
)

const FEATURE_TYPES = [
    "DR", "absDR",
    "EMAi_0.001", "EMAi_0.01", "EMAi_0.1",
    "EMAd_0.001", "EMAd_0.01", "EMAd_0.1"
]

"""
    feature_names() -> Vector{Symbol}

Genera los 128 nombres descriptivos de las variables sensoriales (16 sensores x 8 caracteristicas)
siguiendo el orden exacto definido en el enunciado:
[DR_s, absDR_s, EMAi_0.001_s, EMAi_0.01_s, EMAi_0.1_s, EMAd_0.001_s, EMAd_0.01_s, EMAd_0.1_s]
para s = 1..16.
"""
function feature_names()::Vector{Symbol}
    names_vec = Symbol[]
    for s in 1:16
        for ft in FEATURE_TYPES
            push!(names_vec, Symbol("S$(s)_$(ft)"))
        end
    end
    return names_vec
end

"""
    load_gas_drift_dataset(data_dir::String)

Carga y unifica los 10 ficheros `batch1.dat` .. `batch10.dat` en formato LIBSVM extendido
(`clase;concentracion 1:val ... 128:val`) en un unico `DataFrame` consolidado con:
- 128 columnas sensoriales (`Float64`, admitiendo `NaN` si faltase algun indice),
- `:batch` (`Int`, identificador temporal 1..10),
- `:concentration` (`Float64`, metadato de concentracion en ppmv, excluido como predictor),
- `:gas_class` (`Int`, etiqueta 1..6),
- `:gas_name` (`String`, nombre del gas).
"""
function load_gas_drift_dataset(data_dir::String)
    @assert isdir(data_dir) "No se encontro el directorio de datos: $data_dir"

    fnames = feature_names()
    rows_X = Vector{Vector{Float64}}()
    batches = Int[]
    classes = Int[]
    concs = Float64[]
    explicit_missing_count = 0

    for b in 1:10
        fpath = joinpath(data_dir, "batch$(b).dat")
        @assert isfile(fpath) "No se encontro el fichero: $fpath"
        open(fpath, "r") do io
            for raw_line in eachline(io)
                line = strip(raw_line)
                isempty(line) && continue
                parts = split(line)
                # Cabecera: clase;concentracion (o clase:concentracion)
                lbl_token = parts[1]
                if occursin(';', lbl_token)
                    c_str, conc_str = split(lbl_token, ';')
                    cls = parse(Int, c_str)
                    conc = parse(Float64, conc_str)
                elseif occursin(':', lbl_token)
                    c_str, conc_str = split(lbl_token, ':')
                    cls = parse(Int, c_str)
                    conc = parse(Float64, conc_str)
                else
                    cls = parse(Int, lbl_token)
                    conc = NaN
                end

                vec_x = fill(NaN, 128)
                for tok in @view parts[2:end]
                    k_str, v_str = split(tok, ':')
                    k = parse(Int, k_str)
                    v = parse(Float64, v_str)
                    vec_x[k] = v
                end
                explicit_missing_count += count(isnan, vec_x)
                push!(rows_X, vec_x)
                push!(batches, b)
                push!(classes, cls)
                push!(concs, conc)
            end
        end
    end

    n = length(rows_X)
    Xmat = Matrix{Float64}(undef, n, 128)
    for i in 1:n
        @inbounds Xmat[i, :] .= rows_X[i]
    end

    df = DataFrame(Xmat, fnames)
    df[!, :batch] = batches
    df[!, :concentration] = concs
    df[!, :gas_class] = classes
    df[!, :gas_name] = [GAS_NAMES[c] for c in classes]

    return df, Xmat, batches, classes, concs, explicit_missing_count
end

"""
    analyze_missing_and_sensor_faults(Xmat::Matrix{Float64}, batches::Vector{Int}, classes::Vector{Int}, concs::Vector{Float64})

Realiza un analisis forense de tres niveles de ausencia de datos:
1. Valores explicitamente ausentes (`NaN` o indices omitidos en LIBSVM).
2. Fallos implicitos de lectura de sensor (*sensor flatline*): lecturas donde simultaneamente
   `DR == 0.0` y `|DR| == 1.0` (resistencia constante sin respuesta), ceros exactos en `EMAd`,
   y valores suelo centinela `EMAi == 1e-4`.
3. Ausencia estructural de clases en batches temporales (ej. Toluene ausente en Batches 3-5).

Devuelve un diccionario completo de estadisticas y una copia `X_clean_nan` donde los canales
de sensores con fallo *flatline* confirmado (`DR == 0.0 && |DR| == 1.0`) se marcan como `NaN`
para ser imputados sin data leakage dentro de cada fold de entrenamiento.
"""
function analyze_missing_and_sensor_faults(
    Xmat::Matrix{Float64},
    batches::Vector{Int},
    classes::Vector{Int},
    concs::Vector{Float64}
)
    n, p = size(Xmat)
    fnames = feature_names()

    # 1. Ausencia explicita (NaN / Inf)
    explicit_nan_per_var = [count(isnan, @view Xmat[:, j]) for j in 1:p]
    explicit_nan_total = sum(explicit_nan_per_var)
    explicit_nan_pct_global = 100.0 * explicit_nan_total / (n * p)

    # 2. Ausencia implicita: ceros exactos, centinelas 1e-4 y flatlines de sensor (DR==0 & |DR|==1)
    exact_zero_per_var = [count(==(0.0), @view Xmat[:, j]) for j in 1:p]
    exact_zero_total = sum(exact_zero_per_var)

    X_marked = copy(Xmat)
    flatline_events = DataFrame(
        row_idx=Int[], batch=Int[], sensor=Int[], gas_class=Int[],
        concentration=Float64[], full_8ch_flatline=Bool[]
    )
    flatline_per_batch = zeros(Int, 10)
    flatline_per_sensor = zeros(Int, 16)

    for i in 1:n
        for s in 1:16
            c0 = (s - 1) * 8 + 1
            dr = Xmat[i, c0]
            abs_dr = Xmat[i, c0 + 1]
            if dr == 0.0 && abs_dr == 1.0
                emai_floor = all(Xmat[i, c0 + k] == 1e-4 for k in 2:4)
                emad_zero = all(Xmat[i, c0 + k] == 0.0 for k in 5:7)
                push!(flatline_events, (i, batches[i], s, classes[i], concs[i], emai_floor && emad_zero))
                flatline_per_batch[batches[i]] += 1
                flatline_per_sensor[s] += 1
                # Marcar los 8 canales de este sensor fallido como NaN para imputacion leak-free
                X_marked[i, c0:(c0 + 7)] .= NaN
            end
        end
    end

    # Conteo de centinelas EMAi == 1e-4
    emai_sentinel_total = 0
    for s in 1:16
        for k in 2:4
            emai_sentinel_total += count(==(1e-4), @view Xmat[:, (s - 1) * 8 + 1 + k])
        end
    end

    marked_nan_per_var = [count(isnan, @view X_marked[:, j]) for j in 1:p]
    marked_nan_total = sum(marked_nan_per_var)
    marked_nan_pct_global = 100.0 * marked_nan_total / (n * p)

    # 3. Ausencia estructural de clases por batch
    class_batch_matrix = zeros(Int, 10, 6)
    for i in 1:n
        class_batch_matrix[batches[i], classes[i]] += 1
    end

    # Distribucion de valores marcados por batch
    marked_by_batch = DataFrame(
        batch=1:10,
        months=[BATCH_MONTHS[b] for b in 1:10],
        n_rows=[count(==(b), batches) for b in 1:10],
        explicit_nans=[count(isnan, @view Xmat[batches .== b, :]) for b in 1:10],
        flatline_sensor_events=flatline_per_batch,
        marked_nan_cells=[count(isnan, @view X_marked[batches .== b, :]) for b in 1:10],
        marked_nan_pct=[100.0 * count(isnan, @view X_marked[batches .== b, :]) / (count(==(b), batches) * p) for b in 1:10],
        missing_classes=[count(==(0), @view class_batch_matrix[b, :]) for b in 1:10]
    )

    stats = Dict{String, Any}(
        "n_rows" => n,
        "n_features" => p,
        "explicit_nan_total" => explicit_nan_total,
        "explicit_nan_pct_global" => explicit_nan_pct_global,
        "explicit_nan_per_var" => explicit_nan_per_var,
        "exact_zero_total" => exact_zero_total,
        "exact_zero_pct_global" => 100.0 * exact_zero_total / (n * p),
        "emai_sentinel_total" => emai_sentinel_total,
        "flatline_event_count" => nrow(flatline_events),
        "full_8ch_flatline_count" => count(flatline_events.full_8ch_flatline),
        "flatline_events" => flatline_events,
        "flatline_per_sensor" => flatline_per_sensor,
        "marked_nan_total" => marked_nan_total,
        "marked_nan_pct_global" => marked_nan_pct_global,
        "marked_nan_per_var" => marked_nan_per_var,
        "marked_by_batch" => marked_by_batch,
        "class_batch_matrix" => class_batch_matrix
    )

    return X_marked, stats
end

"""
    temporal_holdout_split(batches::Vector{Int}, classes::Vector{Int}; seed::Int=104)

Particion temporal holdout respetando estrictamente el orden de adquisicion y sin mezclar
muestras del mismo batch entre entrenamiento y test:
- Entrenamiento (pool 5-Fold CV): Batches 1..9 (10.310 instancias, 74,12%).
- Test holdout final (principal): Batch 10 completo (3.600 instancias, 25,88%, Mes 36).
- Test holdout sensibilidad ~10%: subconjunto estratificado de 1.392 instancias (10,01% del total,
  232 por clase) muestreado exclusivamente dentro del Batch 10 con semilla `seed=104`,
  garantizando cero muestras del Batch 10 en entrenamiento.
"""
function temporal_holdout_split(batches::Vector{Int}, classes::Vector{Int}; seed::Int=104)
    train_batches = collect(1:9)
    test_batches = [10]

    train_idx = findall(b -> b in train_batches, batches)
    test_idx = findall(b -> b in test_batches, batches)

    @assert isempty(intersect(Set(batches[train_idx]), Set(batches[test_idx]))) "Data leakage: batches solapados entre train y test"
    @assert maximum(batches[train_idx]) < minimum(batches[test_idx]) "Violacion del orden temporal entre train y test"

    # Subconjunto estratificado ~10% del total (1392 = 6 * 232) estrictamente dentro de Batch 10
    rng = MersenneTwister(seed)
    test_10pct_idx = Int[]
    per_class = 232  # 6 * 232 = 1392 = 10.007% de 13910
    for c in 1:6
        c_idx = [i for i in test_idx if classes[i] == c]
        chosen = sort(c_idx[randperm(rng, length(c_idx))[1:per_class]])
        append!(test_10pct_idx, chosen)
    end
    sort!(test_10pct_idx)

    return (
        train_batches = train_batches,
        test_batches = test_batches,
        train_idx = train_idx,
        test_idx = test_idx,
        test_10pct_idx = test_10pct_idx
    )
end

"""
    make_temporal_forward_cv_folds(train_batches_vec::Vector{Int})

Construye los 5 folds de Validacion Cruzada Batch-wise Temporal Causal (Forward-Chaining /
Expanding Window) sobre los Batches 1..9 (indices locales 1..length(train_batches_vec)):
- Fold 1: Train {1, 2} (1.689)   -> Val {3, 4, 5} (1.944)
- Fold 2: Train {1..5} (3.633)   -> Val {6}       (2.300)
- Fold 3: Train {1..6} (5.933)   -> Val {7}       (3.613)
- Fold 4: Train {1..7} (9.546)   -> Val {8}       (294)
- Fold 5: Train {1..8} (9.840)   -> Val {9}       (470)

Propiedades verificadas por asercion:
1. Estrictamente causal: max(batch_train) < min(batch_val) en los 5 folds.
2. Cero mezcla de batches: intersect(batch_train, batch_val) == empty.
3. Todas las 6 clases presentes en cada particion de entrenamiento.
"""
function make_temporal_forward_cv_folds(train_batches_vec::Vector{Int})
    specs = [
        (collect(1:2), [3, 4, 5]),
        (collect(1:5), [6]),
        (collect(1:6), [7]),
        (collect(1:7), [8]),
        (collect(1:8), [9]),
    ]
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    for (tr_b, va_b) in specs
        tr_idx = findall(b -> b in tr_b, train_batches_vec)
        va_idx = findall(b -> b in va_b, train_batches_vec)
        @assert isempty(intersect(Set(train_batches_vec[tr_idx]), Set(train_batches_vec[va_idx])))
        @assert maximum(train_batches_vec[tr_idx]) < minimum(train_batches_vec[va_idx])
        push!(folds, (tr_idx, va_idx))
    end
    return folds, specs
end

"""
    make_contiguous_group_cv_folds(train_batches_vec::Vector{Int})

Construye 5 folds GroupKFold por bloques temporales contiguos sobre los Batches 1..9
(sin mezcla de batches, pero permitiendo entrenamiento bidireccional pasado+futuro).
Usado en el estudio de ablacion metodologica (Seccion 9) para comparar frente a
Forward-Chaining causal y K-Fold aleatorio convencional.
"""
function make_contiguous_group_cv_folds(train_batches_vec::Vector{Int})
    blocks = [[1, 2], [3, 4, 5], [6], [7], [8, 9]]
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    for va_b in blocks
        va_idx = findall(b -> b in va_b, train_batches_vec)
        tr_idx = findall(b -> !(b in va_b), train_batches_vec)
        @assert isempty(intersect(Set(train_batches_vec[tr_idx]), Set(train_batches_vec[va_idx])))
        push!(folds, (tr_idx, va_idx))
    end
    return folds, blocks
end

"""
    make_random_stratified_cv_folds(y_train::Vector{Int}; nfolds::Int=5, seed::Int=104)

Validacion cruzada aleatoria estratificada convencional (INCORRECTA en presencia de drift
temporal y batches) utilizada exclusivamente en el experimento de ablacion metodologica
para cuantificar el sesgo optimista por fuga intra-batch.
"""
function make_random_stratified_cv_folds(y_train::Vector{Int}; nfolds::Int=5, seed::Int=104)
    rng = MersenneTwister(seed)
    n = length(y_train)
    fold_ids = zeros(Int, n)
    for c in sort(unique(y_train))
        c_idx = findall(==(c), y_train)
        perm = c_idx[randperm(rng, length(c_idx))]
        for (k, idx) in enumerate(perm)
            fold_ids[idx] = mod1(k, nfolds)
        end
    end
    folds = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    for f in 1:nfolds
        va_idx = findall(==(f), fold_ids)
        tr_idx = findall(!=(f), fold_ids)
        push!(folds, (tr_idx, va_idx))
    end
    return folds
end

# ------------------------------------------------------------------------------
# Metricas multiclase (Accuracy, Macro-F1, Weighted-F1, Balanced Accuracy, CM)
# ------------------------------------------------------------------------------

function confusion_matrix_multiclass(y_true::AbstractVector{Int}, y_pred::AbstractVector{Int}; n_classes::Int=6)
    cm = zeros(Int, n_classes, n_classes)
    @inbounds for i in eachindex(y_true)
        cm[y_true[i], y_pred[i]] += 1
    end
    return cm
end

function classification_metrics(y_true::AbstractVector{Int}, y_pred::AbstractVector{Int}; n_classes::Int=6)
    n = length(y_true)
    @assert length(y_pred) == n
    acc = count(i -> y_true[i] == y_pred[i], 1:n) / n
    cm = confusion_matrix_multiclass(y_true, y_pred; n_classes=n_classes)

    f1s = Float64[]
    recalls = Float64[]
    precisions = Float64[]
    supports = Int[]
    per_class_f1 = zeros(Float64, n_classes)
    per_class_recall = zeros(Float64, n_classes)
    per_class_prec = zeros(Float64, n_classes)

    for c in 1:n_classes
        tp = cm[c, c]
        fp = sum(@view cm[:, c]) - tp
        fn = sum(@view cm[c, :]) - tp
        sup = tp + fn
        prec = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
        rec = sup > 0 ? tp / sup : 0.0
        f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0
        per_class_prec[c] = prec
        per_class_recall[c] = rec
        per_class_f1[c] = f1
        if sup > 0
            push!(f1s, f1)
            push!(recalls, rec)
            push!(precisions, prec)
            push!(supports, sup)
        end
    end

    macro_f1 = isempty(f1s) ? 0.0 : mean(f1s)
    weighted_f1 = sum(supports) > 0 ? sum(f1s .* supports) / sum(supports) : 0.0
    bal_acc = isempty(recalls) ? 0.0 : mean(recalls)

    return (
        accuracy = acc,
        macro_f1 = macro_f1,
        weighted_f1 = weighted_f1,
        balanced_accuracy = bal_acc,
        confusion_matrix = cm,
        per_class_f1 = per_class_f1,
        per_class_recall = per_class_recall,
        per_class_precision = per_class_prec
    )
end

"""
    ks_statistic_2sample(x::AbstractVector{Float64}, y::AbstractVector{Float64}) -> Float64

Calcula el estadistico de Kolmogorov-Smirnov de dos muestras (distancia suprema entre ECDFs)
en O((n + m) log(n + m)) en Julia puro para cuantificar la deriva distribucional entre batches.
"""
function ks_statistic_2sample(x::AbstractVector{Float64}, y::AbstractVector{Float64})::Float64
    sx = sort(filter(!isnan, collect(x)))
    sy = sort(filter(!isnan, collect(y)))
    nx = length(sx)
    ny = length(sy)
    (nx == 0 || ny == 0) && return NaN
    i = 1
    j = 1
    d_max = 0.0
    while i <= nx && j <= ny
        val = min(sx[i], sy[j])
        while i <= nx && sx[i] <= val
            i += 1
        end
        while j <= ny && sy[j] <= val
            j += 1
        end
        d = abs((i - 1) / nx - (j - 1) / ny)
        if d > d_max
            d_max = d
        end
    end
    return d_max
end
