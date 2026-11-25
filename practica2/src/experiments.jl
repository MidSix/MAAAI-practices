# ==============================================================================
# Practica 2 - MAAAI (Curso 2026/27)
# Modulo: experiments.jl
# Ejecucion sistematica y reproducible (semilla 104) de:
# - Ablacion de escalado (Min-Max vs Standard vs Robust)
# - Modelos basicos y seleccion/proyeccion de atributos (48 configuraciones en 5-Fold CV)
# - Modelos de ensemble en 5-Fold CV (Bagging, AdaBoost SAMME, EvoTree, RF, XGB, LGBM, CatBoost)
# - Evaluacion final sobre Test Holdout Temporal (Batch 10 completo + sensibilidad 10%)
# - Contrastes estadisticos de hipotesis (Friedman, Nemenyi CD, Wilcoxon-Holm, ANOVA)
# - Ablacion metodologica de validacion (Aleatoria vs GroupKFold vs Causal vs Holdout)
# - Significacion de variables (RF Gini y XGBoost Gain) y estabilidad entre batches
# - Proyecciones 2D (PCA, LDA, ICA, t-SNE, Isomap, LLE) ajustadas en train -> test
# ==============================================================================

using DataFrames
using CSV
using JSON
using Statistics
using StatsBase
using Random
using HypothesisTests
using Distributions
using DecisionTree
using XGBoost
using Printf

# ------------------------------------------------------------------------------
# Evaluacion de un pipeline sobre un conjunto de folds (5-Fold CV)
# ------------------------------------------------------------------------------
function evaluate_pipeline_cv(
    pipe::GasClassificationPipeline,
    X::Matrix{Float64},
    y::Vector{Int},
    folds::Vector{Tuple{Vector{Int}, Vector{Int}}}
)
    nfolds = length(folds)
    accs = zeros(Float64, nfolds)
    mf1s = zeros(Float64, nfolds)
    wf1s = zeros(Float64, nfolds)
    bacc = zeros(Float64, nfolds)
    dims = zeros(Int, nfolds)

    t0 = time()
    for (f, (tr_idx, va_idx)) in enumerate(folds)
        st = fit_pipeline(pipe, X[tr_idx, :], y[tr_idx])
        preds, _ = predict_pipeline(st, X[va_idx, :])
        m = classification_metrics(y[va_idx], preds)
        accs[f] = m.accuracy
        mf1s[f] = m.macro_f1
        wf1s[f] = m.weighted_f1
        bacc[f] = m.balanced_accuracy
        dims[f] = st.n_features_final
    end
    elapsed = time() - t0

    return (
        name = pipe.name,
        scaler = string(pipe.scaler_method),
        filter = string(pipe.filter_spec),
        filter_k = pipe.filter_k,
        projection = string(pipe.proj_spec),
        classifier = string(pipe.clf_spec),
        clf_params = string(pipe.clf_params),
        avg_dim = mean(dims),
        acc_mean = mean(accs),
        acc_std = std(accs),
        macro_f1_mean = mean(mf1s),
        macro_f1_std = std(mf1s),
        weighted_f1_mean = mean(wf1s),
        bal_acc_mean = mean(bacc),
        fold_accs = accs,
        fold_macro_f1s = mf1s,
        elapsed_sec = elapsed,
        pipe = pipe
    )
end

# ------------------------------------------------------------------------------
# Ablacion de escaladores (Min-Max vs Z-score vs Robust) sobre 5-Fold CV
# ------------------------------------------------------------------------------
function run_scaling_comparison(
    Xtr::Matrix{Float64},
    ytr::Vector{Int},
    folds::Vector{Tuple{Vector{Int}, Vector{Int}}}
)
    rows = DataFrame(
        scaler=String[], classifier=String[],
        acc_mean=Float64[], acc_std=Float64[],
        macro_f1_mean=Float64[], elapsed_sec=Float64[]
    )
    for scl in (:minmax, :standard, :robust)
        for (cname, cspec, cpar) in [
            ("SVM(C=1.0)", :svm, Dict{Symbol,Any}(:C => 1.0)),
            ("KNN(k=10)",  :knn, Dict{Symbol,Any}(:k => 10)),
            ("MLP([100])", :mlp, Dict{Symbol,Any}(:hidden => [100], :epochs => 60, :seed => 104))
        ]
            p = GasClassificationPipeline("$(scl)_$(cname)", scl, :none, 128, :none, nothing, cspec, cpar)
            res = evaluate_pipeline_cv(p, Xtr, ytr, folds)
            push!(rows, (string(scl), cname, res.acc_mean, res.acc_std, res.macro_f1_mean, res.elapsed_sec))
        end
    end
    return rows
end

# ------------------------------------------------------------------------------
# Fase 3: Modelos basicos, 7 filtros y 4 proyecciones (48 configuraciones)
# ------------------------------------------------------------------------------
function build_base_experiment_suite(; scaler::Symbol=:minmax, filter_k::Int=32)
    pipes = GasClassificationPipeline[]

    # Bloque A: Barrido completo de las 9 configuraciones base sin reduccion
    for C in (0.1, 0.5, 1.0)
        push!(pipes, GasClassificationPipeline(
            "None | None | SVM(C=$C)", scaler, :none, 128, :none, nothing,
            :svm, Dict{Symbol,Any}(:C => C)
        ))
    end
    for k in (1, 10, 20)
        push!(pipes, GasClassificationPipeline(
            "None | None | KNN(k=$k)", scaler, :none, 128, :none, nothing,
            :knn, Dict{Symbol,Any}(:k => k)
        ))
    end
    for h in ([50], [100], [100, 50])
        hstr = "[" * join(h, ",") * "]"
        push!(pipes, GasClassificationPipeline(
            "None | None | MLP($hstr)", scaler, :none, 128, :none, nothing,
            :mlp, Dict{Symbol,Any}(:hidden => h, :epochs => 70, :seed => 104)
        ))
    end

    # Bloque B: Proyecciones (PCA 95%, LDA 5D, ICA 12D) sin filtrado previo x 3 familias
    for (proj_s, proj_p, plbl) in [(:pca, 0.95, "PCA(95%)"), (:lda, 5, "LDA(5D)"), (:ica, 12, "ICA(12D)")]
        push!(pipes, GasClassificationPipeline(
            "None | $plbl | SVM(C=1.0)", scaler, :none, 128, proj_s, proj_p,
            :svm, Dict{Symbol,Any}(:C => 1.0)
        ))
        push!(pipes, GasClassificationPipeline(
            "None | $plbl | KNN(k=10)", scaler, :none, 128, proj_s, proj_p,
            :knn, Dict{Symbol,Any}(:k => 10)
        ))
        push!(pipes, GasClassificationPipeline(
            "None | $plbl | MLP([100])", scaler, :none, 128, proj_s, proj_p,
            :mlp, Dict{Symbol,Any}(:hidden => [100], :epochs => 70, :seed => 104)
        ))
    end

    # Bloque C: Comparativa controlada de los 6 filtros de caracteristicas (k=32, sin proyeccion) x 3 familias
    filter_list = [
        (:anova,    "ANOVA($filter_k)"),
        (:pearson,  "Pearson($filter_k)"),
        (:spearman, "Spearman($filter_k)"),
        (:kendall,  "Kendall($filter_k)"),
        (:mi,       "MutualInfo($filter_k)"),
        (:rfe,      "RFE_LR($filter_k)")
    ]
    for (fspec, flbl) in filter_list
        push!(pipes, GasClassificationPipeline(
            "$flbl | None | SVM(C=1.0)", scaler, fspec, filter_k, :none, nothing,
            :svm, Dict{Symbol,Any}(:C => 1.0)
        ))
        push!(pipes, GasClassificationPipeline(
            "$flbl | None | KNN(k=10)", scaler, fspec, filter_k, :none, nothing,
            :knn, Dict{Symbol,Any}(:k => 10)
        ))
        push!(pipes, GasClassificationPipeline(
            "$flbl | None | MLP([100])", scaler, fspec, filter_k, :none, nothing,
            :mlp, Dict{Symbol,Any}(:hidden => [100], :epochs => 70, :seed => 104)
        ))
    end

    # Bloque D: Combinaciones Filtro x Proyeccion (garantizando proyecciones en TODOS los filtros
    # y cobertura rica de los hiperparametros de MLP, KNN y SVM)
    extra_combos = [
        ("ANOVA($filter_k) | PCA(95%) | MLP([50])",       :anova,    :pca, 0.95, :mlp, Dict{Symbol,Any}(:hidden => [50], :epochs => 70, :seed => 104)),
        ("ANOVA($filter_k) | LDA(5D) | SVM(C=0.5)",       :anova,    :lda, 5,    :svm, Dict{Symbol,Any}(:C => 0.5)),
        ("Pearson($filter_k) | PCA(95%) | KNN(k=20)",     :pearson,  :pca, 0.95, :knn, Dict{Symbol,Any}(:k => 20)),
        ("Pearson($filter_k) | LDA(5D) | MLP([100,50])",  :pearson,  :lda, 5,    :mlp, Dict{Symbol,Any}(:hidden => [100, 50], :epochs => 70, :seed => 104)),
        ("Spearman($filter_k) | PCA(95%) | SVM(C=1.0)",   :spearman, :pca, 0.95, :svm, Dict{Symbol,Any}(:C => 1.0)),
        ("Spearman($filter_k) | LDA(5D) | KNN(k=1)",      :spearman, :lda, 5,    :knn, Dict{Symbol,Any}(:k => 1)),
        ("Kendall($filter_k) | ICA(10D) | MLP([50])",     :kendall,  :ica, 10,   :mlp, Dict{Symbol,Any}(:hidden => [50], :epochs => 70, :seed => 104)),
        ("Kendall($filter_k) | PCA(95%) | SVM(C=0.1)",    :kendall,  :pca, 0.95, :svm, Dict{Symbol,Any}(:C => 0.1)),
        ("MutualInfo($filter_k) | PCA(95%) | KNN(k=10)",  :mi,       :pca, 0.95, :knn, Dict{Symbol,Any}(:k => 10)),
        ("MutualInfo($filter_k) | LDA(5D) | SVM(C=1.0)",  :mi,       :lda, 5,    :svm, Dict{Symbol,Any}(:C => 1.0)),
        ("RFE_LR($filter_k) | PCA(95%) | MLP([100,50])",  :rfe,      :pca, 0.95, :mlp, Dict{Symbol,Any}(:hidden => [100, 50], :epochs => 70, :seed => 104)),
        ("RFE_LR($filter_k) | LDA(5D) | SVM(C=1.0)",      :rfe,      :lda, 5,    :svm, Dict{Symbol,Any}(:C => 1.0)),
    ]
    for (nm, fs, ps, pp, cs, cp) in extra_combos
        push!(pipes, GasClassificationPipeline(nm, scaler, fs, filter_k, ps, pp, cs, cp))
    end

    return pipes
end

# ------------------------------------------------------------------------------
# Fase 4A: Ensembles en 5-Fold Temporal CV (Sin reduccion y con PCA 95%)
# ------------------------------------------------------------------------------
function build_ensemble_cv_suite(; scaler::Symbol=:minmax)
    pipes = GasClassificationPipeline[]

    for (ps, pp, plbl) in [(:none, nothing, "None"), (:pca, 0.95, "PCA(95%)")]
        # BaggingClassifier: KNN (k=5) con 10 y 50 estimadores
        for n_est in (10, 50)
            push!(pipes, GasClassificationPipeline(
                "None | $plbl | Bagging_KNN5(n=$n_est)", scaler, :none, 128, ps, pp,
                :bagging_knn, Dict{Symbol,Any}(:k => 5, :n_estimators => n_est, :bagging_fraction => 0.8, :seed => 104)
            ))
        end
        # AdaBoost: Stump Trees (SAMME multiclase) con 5 clasificadores base
        push!(pipes, GasClassificationPipeline(
            "None | $plbl | AdaBoost_Stump(n=5)", scaler, :none, 128, ps, pp,
            :adaboost, Dict{Symbol,Any}(:n_iter => 5)
        ))
        # EvoTree: 50 y 100 estimadores, learning_rate = 0.2
        for nr in (50, 100)
            push!(pipes, GasClassificationPipeline(
                "None | $plbl | EvoTree(n=$nr,lr=0.2)", scaler, :none, 128, ps, pp,
                :evotree, Dict{Symbol,Any}(:nrounds => nr, :eta => 0.2, :max_depth => 6, :seed => 104)
            ))
        end
    end

    # Adicionalmente evaluamos RF, XGBoost, LightGBM y CatBoost en el mismo 5-Fold CV
    # para poder compararlos con rigor estadistico (Test de Friedman / Nemenyi) y medir
    # su brecha de generalizacion CV -> Holdout Test (Batch 10)
    push!(pipes, GasClassificationPipeline(
        "None | None | RandomForest(500,d=10)", scaler, :none, 128, :none, nothing,
        :rf, Dict{Symbol,Any}(:n_trees => 500, :max_depth => 10, :seed => 104)
    ))
    push!(pipes, GasClassificationPipeline(
        "None | None | XGBoost(default)", scaler, :none, 128, :none, nothing,
        :xgboost, Dict{Symbol,Any}(:seed => 104)
    ))
    push!(pipes, GasClassificationPipeline(
        "None | None | LightGBM(default)", scaler, :none, 128, :none, nothing,
        :lightgbm, Dict{Symbol,Any}(:seed => 104)
    ))
    push!(pipes, GasClassificationPipeline(
        "None | None | CatBoost(default)", scaler, :none, 128, :none, nothing,
        :catboost, Dict{Symbol,Any}(:seed => 104)
    ))

    return pipes
end

# ------------------------------------------------------------------------------
# Fase 5: Contrastes Estadisticos sobre resultados de 5-Fold CV
# (Shapiro/Normalidad, Friedman, Post-hoc Nemenyi CD, Wilcoxon-Holm, One-Way ANOVA)
# ------------------------------------------------------------------------------

# Valores criticos exactos q_{alpha=0.05} / sqrt(2) para el test post-hoc de Nemenyi
# (Demsar 2006, JMLR, Table 5, alpha = 0.05, dos colas)
const NEMENYI_Q_005 = Dict(
    2  => 1.960,
    3  => 2.343,
    4  => 2.569,
    5  => 2.728,
    6  => 2.850,
    7  => 2.949,
    8  => 3.031,
    9  => 3.102,
    10 => 3.164,
    11 => 3.219,
    12 => 3.268
)

function holm_bonferroni_adjust(pvals::Vector{Float64})::Vector{Float64}
    m = length(pvals)
    order = sortperm(pvals)
    adj = zeros(Float64, m)
    running_max = 0.0
    for (rank_i, orig_idx) in enumerate(order)
        val = pvals[orig_idx] * (m - rank_i + 1)
        running_max = max(running_max, val)
        adj[orig_idx] = min(1.0, running_max)
    end
    return adj
end

function run_statistical_tests(model_names::Vector{String}, fold_scores_list::Vector{Vector{Float64}})
    k = length(model_names)
    N = length(fold_scores_list[1])
    @assert all(length(v) == N for v in fold_scores_list)

    # 1. Rangos promedio por fold (rango 1 = mejor accuracy, con promedio en empates)
    rank_mat = Matrix{Float64}(undef, N, k)
    tie_corr_sum = 0.0
    for d in 1:N
        vals = [-fold_scores_list[j][d] for j in 1:k]  # negativo: mayor accuracy -> rango 1
        rk = StatsBase.tiedrank(vals)
        rank_mat[d, :] .= rk
        # Correccion exacta por empates dentro del bloque d: sum(t^3 - t)
        for (_, cnt) in countmap(rk)
            if cnt > 1
                tie_corr_sum += cnt^3 - cnt
            end
        end
    end
    avg_ranks = vec(mean(rank_mat; dims=1))

    # 2. Test de Friedman (bloques emparejados: N folds x k modelos) + Iman-Davenport F_F
    #    Formula canonica (Demsar 2006, JMLR):
    #    chi2_F = (12 * N / (k * (k + 1))) * (sum(R_j^2) - k * (k + 1)^2 / 4)
    #    con correccion por empates si existen empates dentro de algun fold
    ss_ranks = sum(avg_ranks .^ 2) - k * (k + 1)^2 / 4.0
    denom_f = k * (k + 1) / 12.0 - tie_corr_sum / (12.0 * N * (k - 1))
    friedman_q = denom_f > 0 ? (N * ss_ranks / denom_f) : 0.0
    p_friedman = Distributions.ccdf(Distributions.Chisq(k - 1), max(friedman_q, 0.0))
    # Estadistico F_F de Iman-Davenport (mas potente para N pequeno)
    iman_denom = N * (k - 1) - friedman_q
    iman_F = iman_denom > 1e-12 ? ((N - 1) * friedman_q / iman_denom) : Inf
    iman_p = isfinite(iman_F) ? Distributions.ccdf(Distributions.FDist(k - 1, (k - 1) * (N - 1)), max(iman_F, 0.0)) : 0.0

    # 3. Post-hoc Nemenyi: CD = q_alpha * sqrt(k * (k + 1) / (6 * N))
    q_alpha = get(NEMENYI_Q_005, k, 3.164)
    cd = q_alpha * sqrt(k * (k + 1) / (6.0 * N))

    nemenyi_pairs = DataFrame(
        model_a=String[], model_b=String[],
        rank_a=Float64[], rank_b=Float64[],
        rank_diff=Float64[], cd=Float64[], significant_nemenyi=Bool[]
    )
    for i in 1:(k - 1)
        for j in (i + 1):k
            diff = abs(avg_ranks[i] - avg_ranks[j])
            push!(nemenyi_pairs, (
                model_names[i], model_names[j],
                avg_ranks[i], avg_ranks[j], diff, cd, diff > cd
            ))
        end
    end

    # 4. Pairwise Wilcoxon Signed-Rank Test + ajuste Holm-Bonferroni
    wilcoxon_rows = DataFrame(
        model_a=String[], model_b=String[],
        mean_diff=Float64[], p_raw=Float64[], p_holm=Float64[], significant_holm=Bool[]
    )
    raw_ps = Float64[]
    pair_meta = Tuple{String, String, Float64}[]
    for i in 1:(k - 1)
        for j in (i + 1):k
            a = fold_scores_list[i]
            b = fold_scores_list[j]
            md = mean(a) - mean(b)
            p_w = all(a .== b) ? 1.0 : HypothesisTests.pvalue(HypothesisTests.SignedRankTest(a, b))
            push!(raw_ps, p_w)
            push!(pair_meta, (model_names[i], model_names[j], md))
        end
    end
    adj_ps = holm_bonferroni_adjust(raw_ps)
    for idx in eachindex(raw_ps)
        ma, mb, md = pair_meta[idx]
        push!(wilcoxon_rows, (ma, mb, md, raw_ps[idx], adj_ps[idx], adj_ps[idx] < 0.05))
    end

    # 5. One-Way ANOVA (referencia parametrica)
    anova_test = HypothesisTests.OneWayANOVATest(fold_scores_list...)
    p_anova = HypothesisTests.pvalue(anova_test)
    anova_F = (sum(anova_test.SStᵢ) / anova_test.DFt) / max(sum(anova_test.SSeᵢ) / anova_test.DFe, 1e-15)

    # 6. Chequeo de normalidad por modelo (ExactOneSampleKSTest frente a Normal ajustada)
    norm_df = DataFrame(
        model=String[], mean_acc=Float64[], std_acc=Float64[],
        min_acc=Float64[], max_acc=Float64[], ks_normal_p=Float64[]
    )
    for j in 1:k
        v = fold_scores_list[j]
        μ, σ = mean(v), std(v)
        p_ks = σ > 1e-12 ? HypothesisTests.pvalue(HypothesisTests.ExactOneSampleKSTest(v, Normal(μ, σ))) : 1.0
        push!(norm_df, (model_names[j], μ, σ, minimum(v), maximum(v), p_ks))
    end

    return (
        k = k,
        N = N,
        friedman_stat = friedman_q,
        friedman_p = p_friedman,
        iman_davenport_F = iman_F,
        iman_davenport_p = iman_p,
        anova_F = anova_F,
        anova_p = p_anova,
        avg_ranks = avg_ranks,
        q_alpha = q_alpha,
        cd = cd,
        nemenyi_pairs = nemenyi_pairs,
        wilcoxon_pairs = wilcoxon_rows,
        summary_df = norm_df
    )
end

# ------------------------------------------------------------------------------
# Fase 6: Ablacion Metodologica (Validacion Incorrecta vs Correcta y Data Leakage)
# ------------------------------------------------------------------------------
function run_validation_methodology_ablation(
    Xtr::Matrix{Float64},
    ytr::Vector{Int},
    batches_tr::Vector{Int},
    Xte::Matrix{Float64},
    yte::Vector{Int},
    folds_causal::Vector{Tuple{Vector{Int}, Vector{Int}}};
    scaler::Symbol=:minmax
)
    folds_random = make_random_stratified_cv_folds(ytr; nfolds=5, seed=104)
    folds_group, _ = make_contiguous_group_cv_folds(batches_tr)

    rep_pipes = [
        GasClassificationPipeline("SVM(C=1.0)", scaler, :none, 128, :none, nothing, :svm, Dict{Symbol,Any}(:C => 1.0)),
        GasClassificationPipeline("KNN(k=10)",  scaler, :none, 128, :none, nothing, :knn, Dict{Symbol,Any}(:k => 10)),
        GasClassificationPipeline("MLP([100])", scaler, :none, 128, :none, nothing, :mlp, Dict{Symbol,Any}(:hidden => [100], :epochs => 70, :seed => 104)),
        GasClassificationPipeline("RF(500,d=10)", scaler, :none, 128, :none, nothing, :rf, Dict{Symbol,Any}(:n_trees => 500, :max_depth => 10, :seed => 104)),
    ]

    df = DataFrame(
        model=String[],
        random_cv_mean=Float64[], random_cv_std=Float64[],
        group_cv_mean=Float64[], group_cv_std=Float64[],
        causal_cv_mean=Float64[], causal_cv_std=Float64[],
        holdout_b10_acc=Float64[],
        optimism_gap_random_vs_test=Float64[],
        gap_causal_vs_test=Float64[]
    )

    for p in rep_pipes
        r_rand = evaluate_pipeline_cv(p, Xtr, ytr, folds_random)
        r_grp  = evaluate_pipeline_cv(p, Xtr, ytr, folds_group)
        r_caus = evaluate_pipeline_cv(p, Xtr, ytr, folds_causal)
        st_full = fit_pipeline(p, Xtr, ytr)
        preds_te, _ = predict_pipeline(st_full, Xte)
        m_te = classification_metrics(yte, preds_te)

        push!(df, (
            p.name,
            r_rand.acc_mean, r_rand.acc_std,
            r_grp.acc_mean, r_grp.acc_std,
            r_caus.acc_mean, r_caus.acc_std,
            m_te.accuracy,
            r_rand.acc_mean - m_te.accuracy,
            r_caus.acc_mean - m_te.accuracy
        ))
    end
    return df
end

# ------------------------------------------------------------------------------
# Fase 7: Significacion de Variables (Random Forest Gini + XGBoost Gain)
#         y Estabilidad entre Batches + Correlacion con Drift (KS)
# ------------------------------------------------------------------------------
function extract_xgb_gain_importance(bst, n_features::Int)::Vector{Float64}
    imp_dict = XGBoost.importance(bst, "total_gain")
    vals = zeros(Float64, n_features)
    for (k, v) in imp_dict
        val = Float64(v isa AbstractVector ? v[1] : v)
        if k isa Integer
            # XGBoost.jl v2.x devuelve claves enteras 1-based (1..n_features)
            if 1 <= k <= n_features
                vals[k] = val
            end
        else
            ks = string(k)
            if startswith(ks, "f")
                idx1 = parse(Int, ks[2:end]) + 1
                if 1 <= idx1 <= n_features
                    vals[idx1] = val
                end
            else
                idx1 = tryparse(Int, ks)
                if idx1 !== nothing && 1 <= idx1 <= n_features
                    vals[idx1] = val
                end
            end
        end
    end
    s = sum(vals)
    return s > 0 ? (vals ./ s) : vals
end

function run_feature_importance_and_stability(
    X_all::Matrix{Float64},
    y_all::Vector{Int},
    batches_all::Vector{Int},
    train_idx::Vector{Int},
    test_idx::Vector{Int};
    scaler::Symbol=:minmax,
    seed::Int=104
)
    fnames = feature_names()
    p = length(fnames)

    Xtr = X_all[train_idx, :]
    ytr = y_all[train_idx]
    Xte = X_all[test_idx, :]

    # Preprocesado leak-free en train completo
    imp = MedianImputer()
    imp_fr, _, _ = MMI.fit(imp, 0, Xtr)
    Ztr = MMI.transform(imp, imp_fr, Xtr)
    Zte = MMI.transform(imp, imp_fr, Xte)

    scl = MinMaxScaler(method=scaler)
    scl_fr, _, _ = MMI.fit(scl, 0, Ztr)
    Ztr_s = MMI.transform(scl, scl_fr, Ztr)
    Zte_s = MMI.transform(scl, scl_fr, Zte)

    # 1. Importancia global en entrenamiento (Batches 1..9): RF (500 arboles, d=10) y XGBoost
    rf_fit = fit_classifier(:rf, Dict{Symbol,Any}(:n_trees => 500, :max_depth => 10, :seed => seed), Ztr_s, ytr)
    rf_global = Float64.(rf_fit.importances)
    rf_global ./= max(sum(rf_global), 1e-12)

    xgb_fit = fit_classifier(:xgboost, Dict{Symbol,Any}(:seed => seed), Ztr_s, ytr)
    xgb_global = extract_xgb_gain_importance(xgb_fit.bst, p)

    # 2. Deriva distribucional (KS) por variable entre Train (B1-9) y Test (B10)
    ks_drift = [ks_statistic_2sample(@view(Ztr[:, j]), @view(Zte[:, j])) for j in 1:p]

    # Tabla por variable
    sensor_id = [div(j - 1, 8) + 1 for j in 1:p]
    feat_type = [FEATURE_TYPES[mod1(j, 8)] for j in 1:p]
    feat_df = DataFrame(
        feature_idx = 1:p,
        feature_name = string.(fnames),
        sensor = sensor_id,
        feature_type = feat_type,
        rf_importance = rf_global,
        xgb_importance = xgb_global,
        ks_drift_tr_vs_b10 = ks_drift
    )

    # Agregacion por sensor (1..16)
    sensor_df = combine(groupby(feat_df, :sensor),
        :rf_importance => sum => :rf_imp_sum,
        :xgb_importance => sum => :xgb_imp_sum,
        :ks_drift_tr_vs_b10 => mean => :ks_drift_mean
    )

    # Agregacion por tipo de caracteristica (8 familias)
    ftype_df = combine(groupby(feat_df, :feature_type),
        :rf_importance => sum => :rf_imp_sum,
        :xgb_importance => sum => :xgb_imp_sum,
        :ks_drift_tr_vs_b10 => mean => :ks_drift_mean
    )

    # 3. Estabilidad de importancia entre bloques temporales (B1-2, B3-5, B6, B7, B8-9, B10*)
    #    (*B10 incluido exclusivamente para diagnostico forense post-hoc de drift, nunca en training)
    block_defs = [
        ("B1-2", [1, 2]),
        ("B3-5", [3, 4, 5]),
        ("B6",   [6]),
        ("B7",   [7]),
        ("B8-9", [8, 9]),
        ("B10",  [10])
    ]
    nb = length(block_defs)
    rf_block_imp = Matrix{Float64}(undef, p, nb)
    xgb_block_imp = Matrix{Float64}(undef, p, nb)

    for (bi, (blbl, bvec)) in enumerate(block_defs)
        idx = findall(b -> b in bvec, batches_all)
        Xb = X_all[idx, :]
        yb = y_all[idx]
        # Imputar y escalar localmente en el bloque
        imp_b = MedianImputer(); fr_ib, _, _ = MMI.fit(imp_b, 0, Xb)
        Zb = MMI.transform(imp_b, fr_ib, Xb)
        scl_b = MinMaxScaler(method=scaler); fr_sb, _, _ = MMI.fit(scl_b, 0, Zb)
        Zb_s = MMI.transform(scl_b, fr_sb, Zb)

        rf_b = fit_classifier(:rf, Dict{Symbol,Any}(:n_trees => 200, :max_depth => 10, :seed => seed), Zb_s, yb)
        v_rf = Float64.(rf_b.importances)
        rf_block_imp[:, bi] .= v_rf ./ max(sum(v_rf), 1e-12)

        # Remapear clases presentes contiguamente para XGBoost (B3-5 solo tiene clases 1..5)
        u_cls = sort(unique(yb))
        cmap = Dict(c => (i - 1) for (i, c) in enumerate(u_cls))
        y0 = Float32.([cmap[c] for c in yb])
        dmat = XGBoost.DMatrix(Float32.(Zb_s); label=y0)
        bst_b = XGBoost.xgboost(dmat; num_round=100, objective="multi:softprob", num_class=length(u_cls), seed=seed, verbosity=0, watchlist=NamedTuple())
        xgb_block_imp[:, bi] .= extract_xgb_gain_importance(bst_b, p)
    end

    # Correlacion de rangos Spearman entre bloques (RF y XGB) + Jaccard Top-20
    rf_spearman = Matrix{Float64}(undef, nb, nb)
    xgb_spearman = Matrix{Float64}(undef, nb, nb)
    rf_jaccard20 = Matrix{Float64}(undef, nb, nb)
    xgb_jaccard20 = Matrix{Float64}(undef, nb, nb)
    for i in 1:nb, j in 1:nb
        rf_spearman[i, j] = corspearman(rf_block_imp[:, i], rf_block_imp[:, j])
        xgb_spearman[i, j] = corspearman(xgb_block_imp[:, i], xgb_block_imp[:, j])
        t_rf_i = Set(sortperm(rf_block_imp[:, i]; rev=true)[1:20])
        t_rf_j = Set(sortperm(rf_block_imp[:, j]; rev=true)[1:20])
        rf_jaccard20[i, j] = length(intersect(t_rf_i, t_rf_j)) / length(union(t_rf_i, t_rf_j))
        t_xg_i = Set(sortperm(xgb_block_imp[:, i]; rev=true)[1:20])
        t_xg_j = Set(sortperm(xgb_block_imp[:, j]; rev=true)[1:20])
        xgb_jaccard20[i, j] = length(intersect(t_xg_i, t_xg_j)) / length(union(t_xg_i, t_xg_j))
    end

    # Correlacion entre importancia global y magnitud de drift KS
    corr_rf_xgb = corspearman(rf_global, xgb_global)
    corr_rf_drift = corspearman(rf_global, ks_drift)
    corr_xgb_drift = corspearman(xgb_global, ks_drift)

    return (
        feat_df = feat_df,
        sensor_df = sensor_df,
        ftype_df = ftype_df,
        block_names = [b[1] for b in block_defs],
        rf_block_imp = rf_block_imp,
        xgb_block_imp = xgb_block_imp,
        rf_spearman = rf_spearman,
        xgb_spearman = xgb_spearman,
        rf_jaccard20 = rf_jaccard20,
        xgb_jaccard20 = xgb_jaccard20,
        corr_rf_xgb = corr_rf_xgb,
        corr_rf_drift = corr_rf_drift,
        corr_xgb_drift = corr_xgb_drift
    )
end

# ------------------------------------------------------------------------------
# Fase 8: Proyecciones Bidimensionales (PCA, LDA, ICA, t-SNE, Isomap, LLE)
#         Ajustadas ESTRICTAMENTE en entrenamiento y proyectadas a test.
# ------------------------------------------------------------------------------
function compute_2d_separability_knn(Ztr::Matrix{Float64}, ytr::Vector{Int},
                                     Zte::Matrix{Float64}, yte::Vector{Int}; k::Int=5)::Float64
    D = Distances.pairwise(Euclidean(), Ztr', Zte'; dims=2)
    n_te = size(Zte, 1)
    correct = 0
    for i in 1:n_te
        nn = partialsortperm(@view(D[:, i]), 1:k)
        counts = zeros(Int, 6)
        for idx in nn
            counts[ytr[idx]] += 1
        end
        if argmax(counts) == yte[i]
            correct += 1
        end
    end
    return correct / n_te
end

function run_2d_visualizations_fit_transform(
    Xtr::Matrix{Float64},
    ytr::Vector{Int},
    btr::Vector{Int},
    Xte::Matrix{Float64},
    yte::Vector{Int},
    bte::Vector{Int};
    scaler::Symbol=:minmax,
    seed::Int=104
)
    # 1. Imputacion y escalado ajustados UNICAMENTE en entrenamiento
    imp = MedianImputer(); fr_i, _, _ = MMI.fit(imp, 0, Xtr)
    Ztr = MMI.transform(imp, fr_i, Xtr)
    Zte = MMI.transform(imp, fr_i, Xte)

    scl = MinMaxScaler(method=scaler); fr_s, _, _ = MMI.fit(scl, 0, Ztr)
    Ztr_s = MMI.transform(scl, fr_s, Ztr)
    Zte_s = MMI.transform(scl, fr_s, Zte)

    # Subconjunto estratificado de entrenamiento (300/clase = 1800 pts) para ajuste de manifolds
    # y visualizacion limpia sin sobreimpresion masiva, mas subconjunto estratificado de test (200/clase = 1200)
    sub_tr = stratified_subsample_indices(ytr, 300; seed=seed)
    sub_te = stratified_subsample_indices(yte, 200; seed=seed)

    Ztr_sub = Ztr_s[sub_tr, :]
    ytr_sub = ytr[sub_tr]
    btr_sub = btr[sub_tr]
    Zte_sub = Zte_s[sub_te, :]
    yte_sub = yte[sub_te]
    bte_sub = bte[sub_te]

    embeddings = Dict{String, NamedTuple}()
    sep_df = DataFrame(technique=String[], test_2d_5nn_acc=Float64[], train_fit_mode=String[])

    # 1. PCA 2D (ajustado en TODO train, aplicado a train_sub y test_sub)
    all_sub_idx = collect(1:length(ytr_sub))
    pca2 = PCAProjector(pratio=1.0, maxoutdim=2)
    fr_pca, _, rep_pca = MMI.fit(pca2, 0, Ztr_s)
    emb_tr = MMI.transform(pca2, fr_pca, Ztr_sub)
    emb_te = MMI.transform(pca2, fr_pca, Zte_sub)
    acc2d = compute_2d_separability_knn(emb_tr, ytr_sub, emb_te, yte_sub)
    embeddings["PCA"] = (Z_tr=emb_tr, Z_te=emb_te, ref_idx=all_sub_idx, extra=rep_pca)
    push!(sep_df, ("PCA", acc2d, "Parametrico lineal (fit en 10.310 train)"))

    # 2. LDA 2D (supervisado, ajustado en TODO train)
    lda2 = LDAProjector(outdim=2, regcoef=1e-4)
    fr_lda, _, rep_lda = MMI.fit(lda2, 0, Ztr_s, ytr)
    emb_tr = MMI.transform(lda2, fr_lda, Ztr_sub)
    emb_te = MMI.transform(lda2, fr_lda, Zte_sub)
    acc2d = compute_2d_separability_knn(emb_tr, ytr_sub, emb_te, yte_sub)
    embeddings["LDA"] = (Z_tr=emb_tr, Z_te=emb_te, ref_idx=all_sub_idx, extra=rep_lda)
    push!(sep_df, ("LDA", acc2d, "Parametrico supervisado (fit en 10.310 train)"))

    # 3. ICA 2D (ajustado en TODO train)
    ica2 = ICAProjector(outdim=2, maxiter=500, tol=1e-3, seed=seed)
    fr_ica, _, rep_ica = MMI.fit(ica2, 0, Ztr_s)
    emb_tr = MMI.transform(ica2, fr_ica, Ztr_sub)
    emb_te = MMI.transform(ica2, fr_ica, Zte_sub)
    acc2d = compute_2d_separability_knn(emb_tr, ytr_sub, emb_te, yte_sub)
    embeddings["ICA"] = (Z_tr=emb_tr, Z_te=emb_te, ref_idx=all_sub_idx, extra=rep_ica)
    push!(sep_df, ("ICA", acc2d, "Parametrico lineal blanqueado (fit en 10.310 train)"))

    # 4. t-SNE 2D (wrapper OOS: fit en Ztr_sub, transform OOS sobre Zte_sub)
    tsne2 = OutOfSampleManifold2D(method=:tsne, k_neighbors=15, perplexity=30.0, max_iter=500, seed=seed)
    fr_tsne, _, _ = MMI.fit(tsne2, 0, Ztr_sub)
    emb_tr = fr_tsne.Y_ref
    emb_te = MMI.transform(tsne2, fr_tsne, Zte_sub)
    acc2d = compute_2d_separability_knn(emb_tr, ytr_sub[fr_tsne.ref_idx], emb_te, yte_sub)
    embeddings["t-SNE"] = (Z_tr=emb_tr, Z_te=emb_te, ref_idx=fr_tsne.ref_idx, extra=(n_ref=length(fr_tsne.ref_idx),))
    push!(sep_df, ("t-SNE", acc2d, "Wrapper OOS kernel k-NN (fit en $(length(fr_tsne.ref_idx)) train)"))

    # 5. Isomap 2D (wrapper OOS: fit en Ztr_sub, transform OOS sobre Zte_sub)
    iso2 = OutOfSampleManifold2D(method=:isomap, k_neighbors=15, seed=seed)
    fr_iso, _, _ = MMI.fit(iso2, 0, Ztr_sub)
    emb_tr = fr_iso.Y_ref
    emb_te = MMI.transform(iso2, fr_iso, Zte_sub)
    acc2d = compute_2d_separability_knn(emb_tr, ytr_sub[fr_iso.ref_idx], emb_te, yte_sub)
    embeddings["Isomap"] = (Z_tr=emb_tr, Z_te=emb_te, ref_idx=fr_iso.ref_idx, extra=(n_ref=length(fr_iso.ref_idx),))
    push!(sep_df, ("Isomap", acc2d, "Wrapper OOS kernel k-NN (fit en $(length(fr_iso.ref_idx)) train)"))

    # 6. LLE 2D (wrapper OOS: fit en Ztr_sub, transform OOS Saul-Roweis sobre Zte_sub)
    lle2 = OutOfSampleManifold2D(method=:lle, k_neighbors=15, seed=seed)
    fr_lle, _, _ = MMI.fit(lle2, 0, Ztr_sub)
    emb_tr = fr_lle.Y_ref
    emb_te = MMI.transform(lle2, fr_lle, Zte_sub)
    acc2d = compute_2d_separability_knn(emb_tr, ytr_sub[fr_lle.ref_idx], emb_te, yte_sub)
    embeddings["LLE"] = (Z_tr=emb_tr, Z_te=emb_te, ref_idx=fr_lle.ref_idx, extra=(n_ref=length(fr_lle.ref_idx),))
    push!(sep_df, ("LLE", acc2d, "Wrapper OOS pesos locales LLE (fit en $(length(fr_lle.ref_idx)) train)"))

    return (
        embeddings = embeddings,
        ytr_sub = ytr_sub,
        btr_sub = btr_sub,
        yte_sub = yte_sub,
        bte_sub = bte_sub,
        sep_df = sep_df
    )
end
