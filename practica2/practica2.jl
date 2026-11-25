# ==============================================================================
# PRACTICA 2 - MODELOS AVANZADOS DE APRENDIZAJE AUTOMATICO I (Curso 2026/27)
# Sistema Robusto de Clasificacion Multiclase frente a Deriva Temporal de Sensores
# Dataset: UCI Gas Sensor Array Drift Dataset at Different Concentrations
#
# Script autonomo y reproducible (semilla 104):
# Ejecutar sin modificaciones ni parametros adicionales:
#     julia practica2.jl
# Genera automaticamente todas las tablas (results/), metricas y figuras (figures/)
# incluidas en la memoria (Memoria_Practica2.pdf).
# ==============================================================================

using Pkg
Pkg.activate(@__DIR__)

using Random
using Statistics
using StatsBase
using DataFrames
using CSV
using JSON
using Printf

const SEED = 104
Random.seed!(SEED)

const BASE_DIR    = @__DIR__
const DATA_DIR    = joinpath(BASE_DIR, "gas+sensor+array+drift+dataset+at+different+concentrations")
const RESULTS_DIR = joinpath(BASE_DIR, "results")
const FIGURES_DIR = joinpath(BASE_DIR, "figures")
mkpath(RESULTS_DIR)
mkpath(FIGURES_DIR)

include(joinpath(BASE_DIR, "src", "data_utils.jl"))
include(joinpath(BASE_DIR, "src", "mlj_wrappers.jl"))
include(joinpath(BASE_DIR, "src", "experiments.jl"))
include(joinpath(BASE_DIR, "src", "visualizations.jl"))

function main()
    t_total_start = time()
    println("="^88)
    println("PRACTICA 2 - MAAAI | CLASIFICACION DE GASES CON DERIVA TEMPORAL DE SENSORES")
    println("Julia $(VERSION) | Semilla fija: $(SEED)")
    println("="^88)

    # ==========================================================================
    # FASE 1: CARGA, UNIFICACION Y ANALISIS EXPLORATORIO INICIAL
    # ==========================================================================
    println("\n[FASE 1] Carga y unificacion de datos (Batches 1..10)...")
    df, X_raw, batches, classes, concs, explicit_missing = load_gas_drift_dataset(DATA_DIR)
    n_rows, n_feat = size(X_raw)
    n_batches = length(unique(batches))
    n_classes = length(unique(classes))

    println("  - Instancias totales : $n_rows")
    println("  - Variables totales  : $(n_feat + 1) (128 caracteristicas sensoriales + 1 identificador de batch)")
    println("  - Metadato excluido  : concentration (ppmv) — retenido solo para EDA, excluido de X para evitar leakage")
    println("  - Numero de batches  : $n_batches (Meses 1 a 36)")
    println("  - Numero de clases   : $n_classes ($(join([GAS_NAMES[c] for c in 1:6], ", ")))")
    flush(stdout)

    # ==========================================================================
    # FASE 2: ANALISIS FORENSE DE VALORES AUSENTES Y FALLOS DE SENSORES
    # ==========================================================================
    println("\n[FASE 2] Analisis de valores ausentes (explicitos, implicitos y estructurales)...")
    X_marked, miss_stats = analyze_missing_and_sensor_faults(X_raw, batches, classes, concs)
    println(@sprintf("  - Valores nulos explicitos (NaN / omitidos LIBSVM): %d / %d (%.4f%%)",
                     miss_stats["explicit_nan_total"], n_rows * n_feat, miss_stats["explicit_nan_pct_global"]))
    println(@sprintf("  - Ceros exactos en lecturas de sensores           : %d celdas (%.4f%%)",
                     miss_stats["exact_zero_total"], miss_stats["exact_zero_pct_global"]))
    println(@sprintf("  - Centinelas suelo EMAi == 1e-4                   : %d celdas",
                     miss_stats["emai_sentinel_total"]))
    println(@sprintf("  - Fallos flatline confirmados (DR=0 & |DR|=1)     : %d eventos (%d de 8 canales completos, mayoritariamente Sensor 4 en Batch 2)",
                     miss_stats["flatline_event_count"], miss_stats["full_8ch_flatline_count"]))
    println(@sprintf("  - Celdas marcadas como NaN para imputacion segura : %d (%.4f%% global)",
                     miss_stats["marked_nan_total"], miss_stats["marked_nan_pct_global"]))
    println("  - Ausencia estructural de clase: Toluene (clase 6) = 0 muestras en Batches 3, 4 y 5.")
    flush(stdout)

    CSV.write(joinpath(RESULTS_DIR, "missing_by_batch.csv"), miss_stats["marked_by_batch"])
    CSV.write(joinpath(RESULTS_DIR, "flatline_events.csv"), miss_stats["flatline_events"])

    # Figuras de EDA y Drift
    plot_fig01_eda_batches_classes(batches, miss_stats["class_batch_matrix"],
                                   joinpath(FIGURES_DIR, "fig01_eda_batches_classes.png"))
    plot_fig02_missing_sensor_faults(X_raw, batches, miss_stats,
                                     joinpath(FIGURES_DIR, "fig02_missing_sensor_faults.png"))
    plot_fig03_temporal_drift_eda(X_raw, batches,
                                  joinpath(FIGURES_DIR, "fig03_temporal_drift_eda.png"))

    # ==========================================================================
    # FASE 3: PARTICION HOLDOUT TEMPORAL Y VALIDACION CRUZADA 5-FOLD BATCH-WISE
    # ==========================================================================
    println("\n[FASE 3] Particion Holdout Temporal y Validacion Cruzada 5-Fold Batch-wise...")
    holdout_split = temporal_holdout_split(batches, classes; seed=SEED)
    Xtr = X_marked[holdout_split.train_idx, :]
    ytr = classes[holdout_split.train_idx]
    btr = batches[holdout_split.train_idx]
    Xte = X_marked[holdout_split.test_idx, :]
    yte = classes[holdout_split.test_idx]
    bte = batches[holdout_split.test_idx]

    # Indice relativo dentro de Xte para la submuestra estratificada ~10% (1.392 instancias, seed 104)
    test_global_to_local = Dict(g => l for (l, g) in enumerate(holdout_split.test_idx))
    test_10pct_local = [test_global_to_local[g] for g in holdout_split.test_10pct_idx]
    Xte_10 = Xte[test_10pct_local, :]
    yte_10 = yte[test_10pct_local]

    folds_causal, fold_specs = make_temporal_forward_cv_folds(btr)

    println("  - Batches Entrenamiento (5-Fold CV): $(holdout_split.train_batches) -> $(length(holdout_split.train_idx)) muestras ($(round(100*length(holdout_split.train_idx)/n_rows, digits=2))%)")
    println("  - Batch Holdout Test Final         : $(holdout_split.test_batches) -> $(length(holdout_split.test_idx)) muestras ($(round(100*length(holdout_split.test_idx)/n_rows, digits=2))%, Mes 36)")
    println("  - Submuestra sensibilidad 10% (B10): $(length(test_10pct_local)) muestras ($(round(100*length(test_10pct_local)/n_rows, digits=2))%, seed=$SEED, 232/clase)")
    println("  - Folds 5-Fold Temporal Causal (Forward-Chaining):")
    for (fi, ((tr_b, va_b), (tr_i, va_i))) in enumerate(zip(fold_specs, folds_causal))
        println("      Fold $fi: Train Batches $tr_b (n=$(length(tr_i))) -> Val Batches $va_b (n=$(length(va_i)))")
    end

    # Verificacion unitaria de Kendall tau-b rapido vs StatsBase.corkendall
    sample_x = X_raw[1:300, 1]
    sample_yb = BitVector(classes[1:300] .== 1)
    @assert isapprox(fast_binary_kendall_tau_b(sample_x, sample_yb),
                     StatsBase.corkendall(sample_x, Float64.(sample_yb)); atol=1e-12)
    println("  [OK] Verificacion exacta de fast_binary_kendall_tau_b frente a StatsBase.corkendall (tol=1e-12).")

    # Comparativa de escaladores en 5-Fold CV (justificacion empirica de Min-Max)
    println("\n  Comparando escaladores (Min-Max vs Z-score vs Robust) en 5-Fold Causal CV...")
    scaling_df = run_scaling_comparison(Xtr, ytr, folds_causal)
    CSV.write(joinpath(RESULTS_DIR, "scaling_comparison.csv"), scaling_df)
    for r in eachrow(scaling_df)
        println(@sprintf("    %-8s | %-11s | Acc = %5.2f%% +/- %5.2f%% | Macro-F1 = %5.2f%%",
                         r.scaler, r.classifier, r.acc_mean*100, r.acc_std*100, r.macro_f1_mean*100))
    end
    flush(stdout)
    # Seleccionar Min-Max segun el enunciado (siempre ajustado en train)
    PRIMARY_SCALER = :minmax

    # ==========================================================================
    # FASE 4: MODELOS BASICOS Y SELECCION / PROYECCION DE ATRIBUTOS (30%)
    # ==========================================================================
    println("\n[FASE 4] Entrenamiento de modelos basicos, 7 filtros y 4 proyecciones (5-Fold Causal CV)...")
    base_pipes = build_base_experiment_suite(; scaler=PRIMARY_SCALER, filter_k=32)
    base_results = Any[]
    for (idx, pipe) in enumerate(base_pipes)
        res = evaluate_pipeline_cv(pipe, Xtr, ytr, folds_causal)
        push!(base_results, res)
        println(@sprintf("  [%02d/%02d] %-44s | d=%5.1f | Acc=%5.2f%% +/- %5.2f%% | MF1=%5.2f%% | %.1fs",
                         idx, length(base_pipes), res.name, res.avg_dim,
                         res.acc_mean*100, res.acc_std*100, res.macro_f1_mean*100, res.elapsed_sec))
        flush(stdout)
    end

    base_df = DataFrame(
        name = [r.name for r in base_results],
        scaler = [r.scaler for r in base_results],
        filter = [r.filter for r in base_results],
        filter_k = [r.filter_k for r in base_results],
        projection = [r.projection for r in base_results],
        classifier = [r.classifier for r in base_results],
        clf_params = [r.clf_params for r in base_results],
        avg_dim = [r.avg_dim for r in base_results],
        acc_mean = [r.acc_mean for r in base_results],
        acc_std = [r.acc_std for r in base_results],
        macro_f1_mean = [r.macro_f1_mean for r in base_results],
        macro_f1_std = [r.macro_f1_std for r in base_results],
        weighted_f1_mean = [r.weighted_f1_mean for r in base_results],
        bal_acc_mean = [r.bal_acc_mean for r in base_results],
        fold1_acc = [r.fold_accs[1] for r in base_results],
        fold2_acc = [r.fold_accs[2] for r in base_results],
        fold3_acc = [r.fold_accs[3] for r in base_results],
        fold4_acc = [r.fold_accs[4] for r in base_results],
        fold5_acc = [r.fold_accs[5] for r in base_results],
        elapsed_sec = [r.elapsed_sec for r in base_results]
    )
    CSV.write(joinpath(RESULTS_DIR, "base_models_cv.csv"), base_df)
    plot_fig04_base_models_cv(base_df, joinpath(FIGURES_DIR, "fig04_base_models_cv.png"))

    # ==========================================================================
    # FASE 5: MODELOS DE ENSEMBLE EN 5-FOLD CV Y EVALUACION FINAL HOLDOUT (20%)
    # ==========================================================================
    println("\n[FASE 5A] Ensembles en 5-Fold Causal CV (Sin reduccion y PCA 95%)...")
    ens_pipes = build_ensemble_cv_suite(; scaler=PRIMARY_SCALER)
    ens_results = Any[]
    for (idx, pipe) in enumerate(ens_pipes)
        res = evaluate_pipeline_cv(pipe, Xtr, ytr, folds_causal)
        push!(ens_results, res)
        println(@sprintf("  [%02d/%02d] %-44s | d=%5.1f | Acc=%5.2f%% +/- %5.2f%% | MF1=%5.2f%% | %.1fs",
                         idx, length(ens_pipes), res.name, res.avg_dim,
                         res.acc_mean*100, res.acc_std*100, res.macro_f1_mean*100, res.elapsed_sec))
        flush(stdout)
    end

    ens_df = DataFrame(
        name = [r.name for r in ens_results],
        filter = [r.filter for r in ens_results],
        projection = [r.projection for r in ens_results],
        classifier = [r.classifier for r in ens_results],
        clf_params = [r.clf_params for r in ens_results],
        avg_dim = [r.avg_dim for r in ens_results],
        acc_mean = [r.acc_mean for r in ens_results],
        acc_std = [r.acc_std for r in ens_results],
        macro_f1_mean = [r.macro_f1_mean for r in ens_results],
        macro_f1_std = [r.macro_f1_std for r in ens_results],
        bal_acc_mean = [r.bal_acc_mean for r in ens_results],
        fold1_acc = [r.fold_accs[1] for r in ens_results],
        fold2_acc = [r.fold_accs[2] for r in ens_results],
        fold3_acc = [r.fold_accs[3] for r in ens_results],
        fold4_acc = [r.fold_accs[4] for r in ens_results],
        fold5_acc = [r.fold_accs[5] for r in ens_results],
        elapsed_sec = [r.elapsed_sec for r in ens_results]
    )
    CSV.write(joinpath(RESULTS_DIR, "ensembles_cv.csv"), ens_df)

    # Seleccionar las 5 mejores combinaciones de CV (1 KNN, 1 SVM, 1 MLP, 1 Bagging, 1 AdaBoost) + mejor EvoTree
    pick_best(list, ctype) = sort(filter(r -> r.classifier == ctype, list); by=r -> r.acc_mean, rev=true)[1]
    best_knn_cv = pick_best(base_results, "knn")
    best_svm_cv = pick_best(base_results, "svm")
    best_mlp_cv = pick_best(base_results, "mlp")
    best_bag_cv = pick_best(ens_results, "bagging_knn")
    best_ada_cv = pick_best(ens_results, "adaboost")
    best_evo_cv = pick_best(ens_results, "evotree")
    rf_cv       = pick_best(ens_results, "rf")
    xgb_cv      = pick_best(ens_results, "xgboost")
    lgb_cv      = pick_best(ens_results, "lightgbm")
    cat_cv      = pick_best(ens_results, "catboost")

    println("\n  Mejores combinaciones seleccionadas en 5-Fold Temporal CV:")
    for (lbl, r) in [("Mejor KNN", best_knn_cv), ("Mejor SVM", best_svm_cv),
                     ("Mejor MLP", best_mlp_cv), ("Mejor Bagging", best_bag_cv),
                     ("Mejor AdaBoost", best_ada_cv), ("Mejor EvoTree", best_evo_cv)]
        println(@sprintf("    %-16s -> %-42s (CV Acc = %5.2f%% +/- %5.2f%%)",
                         lbl, r.name, r.acc_mean*100, r.acc_std*100))
    end

    # [FASE 5B] Evaluacion final sobre Holdout Test (Batch 10) reentrenando en TODO train (B1..B9)
    println("\n[FASE 5B] Evaluacion final sobre Holdout Test (Batch 10, n=3.600 + sensibilidad 10%, n=1.392)...")
    flush(stdout)
    holdout_df = DataFrame(
        model_label=String[], pipeline_detail=String[],
        cv_acc_mean=Float64[], cv_acc_std=Float64[],
        test_acc_b10=Float64[], test_macro_f1_b10=Float64[],
        test_bal_acc_b10=Float64[],
        test_acc_10pct=Float64[], test_macro_f1_10pct=Float64[], test_bal_acc_10pct=Float64[],
        gen_gap_cv_minus_test=Float64[]
    )
    holdout_preds_store = Dict{String, Vector{Int}}()
    holdout_metrics_store = Dict{String, Any}()

    final_single_eval = [
        ("Best KNN",       best_knn_cv),
        ("Best SVM",       best_svm_cv),
        ("Best MLP",       best_mlp_cv),
        ("Best Bagging",   best_bag_cv),
        ("Best AdaBoost",  best_ada_cv),
        ("Best EvoTree",   best_evo_cv),
        ("Random Forest",  rf_cv),
        ("XGBoost",        xgb_cv),
        ("LightGBM",       lgb_cv),
        ("CatBoost",       cat_cv),
    ]

    for (lbl, rcv) in final_single_eval
        st = fit_pipeline(rcv.pipe, Xtr, ytr)
        preds_b10, _ = predict_pipeline(st, Xte)
        m_b10 = classification_metrics(yte, preds_b10)
        m_10  = classification_metrics(yte_10, preds_b10[test_10pct_local])
        holdout_preds_store[lbl] = preds_b10
        holdout_metrics_store[lbl] = m_b10
        push!(holdout_df, (
            lbl, rcv.name, rcv.acc_mean, rcv.acc_std,
            m_b10.accuracy, m_b10.macro_f1, m_b10.balanced_accuracy,
            m_10.accuracy, m_10.macro_f1, m_10.balanced_accuracy,
            rcv.acc_mean - m_b10.accuracy
        ))
        println(@sprintf("  %-18s | CV=%5.2f%% | Test B10=%5.2f%% (MF1=%5.2f%%) | Test 10%%=%5.2f%% (MF1=%5.2f%%) | Gap=%+5.2f pp",
                         lbl, rcv.acc_mean*100, m_b10.accuracy*100, m_b10.macro_f1*100,
                         m_10.accuracy*100, m_10.macro_f1*100, (rcv.acc_mean - m_b10.accuracy)*100))
        flush(stdout)
    end

    # Hard Voting SVM: particion en 3 del conjunto de entrenamiento con semilla 104
    hv_preds, hv_info = fit_predict_hard_voting_svm(best_svm_cv.pipe, Xtr, ytr, Xte; seed=SEED)
    m_hv = classification_metrics(yte, hv_preds)
    m_hv10 = classification_metrics(yte_10, hv_preds[test_10pct_local])
    holdout_preds_store["Hard Voting SVM"] = hv_preds
    holdout_metrics_store["Hard Voting SVM"] = m_hv
    push!(holdout_df, (
        "Hard Voting SVM", "3x " * best_svm_cv.name * " (particion 3-way, seed=104)",
        NaN, NaN, m_hv.accuracy, m_hv.macro_f1, m_hv.balanced_accuracy,
        m_hv10.accuracy, m_hv10.macro_f1, m_hv10.balanced_accuracy, NaN
    ))
    println(@sprintf("  %-18s | CV=  N/A | Test B10=%5.2f%% (MF1=%5.2f%%) | Test 10%%=%5.2f%% (MF1=%5.2f%%) | particiones=%s, empates=%d",
                     "Hard Voting SVM", m_hv.accuracy*100, m_hv.macro_f1*100, m_hv10.accuracy*100, m_hv10.macro_f1*100,
                     string(hv_info.part_sizes), hv_info.ties))
    flush(stdout)

    # Stacking Ensemble: MLP meta-learner; base = Best SVM + Best KNN + Best MLP (OOF batch-wise)
    stk_preds, _ = fit_predict_batch_stacking(
        best_svm_cv.pipe, best_knn_cv.pipe, best_mlp_cv.pipe,
        Xtr, ytr, folds_causal, Xte; seed=SEED
    )
    m_stk = classification_metrics(yte, stk_preds)
    m_stk10 = classification_metrics(yte_10, stk_preds[test_10pct_local])
    holdout_preds_store["Stacking (MLP)"] = stk_preds
    holdout_metrics_store["Stacking (MLP)"] = m_stk
    push!(holdout_df, (
        "Stacking (MLP)", "Meta MLP([50]) sobre OOF(Best SVM + Best KNN + Best MLP)",
        NaN, NaN, m_stk.accuracy, m_stk.macro_f1, m_stk.balanced_accuracy,
        m_stk10.accuracy, m_stk10.macro_f1, m_stk10.balanced_accuracy, NaN
    ))
    println(@sprintf("  %-18s | CV=  N/A | Test B10=%5.2f%% (MF1=%5.2f%%) | Test 10%%=%5.2f%% (MF1=%5.2f%%)",
                     "Stacking (MLP)", m_stk.accuracy*100, m_stk.macro_f1*100, m_stk10.accuracy*100, m_stk10.macro_f1*100))
    flush(stdout)

    CSV.write(joinpath(RESULTS_DIR, "holdout_test_results.csv"), holdout_df)
    plot_fig05_ensembles_and_holdout(holdout_df, joinpath(FIGURES_DIR, "fig05_ensembles_and_holdout.png"))

    # Matriz de confusion del mejor modelo global en Batch 10 + F1 por clase de los referentes
    best_final_row = sort(holdout_df, :test_acc_b10; rev=true)[1, :]
    best_final_lbl = best_final_row.model_label
    best_final_m = holdout_metrics_store[best_final_lbl]
    per_class_f1_cmp = Dict(
        best_final_lbl => best_final_m.per_class_f1,
        "Best SVM" => holdout_metrics_store["Best SVM"].per_class_f1,
        "Best MLP" => holdout_metrics_store["Best MLP"].per_class_f1,
        "Random Forest" => holdout_metrics_store["Random Forest"].per_class_f1,
        "Stacking (MLP)" => holdout_metrics_store["Stacking (MLP)"].per_class_f1
    )
    plot_fig06_confusion_matrix_best(best_final_lbl, best_final_m.confusion_matrix,
                                     per_class_f1_cmp, joinpath(FIGURES_DIR, "fig06_confusion_matrix_best.png"))

    # Guardar tabla de F1 por clase de todos los modelos finales
    f1_cls_df = DataFrame(model = String[])
    for c in 1:6
        f1_cls_df[!, Symbol(GAS_NAMES[c])] = Float64[]
    end
    for lbl in holdout_df.model_label
        m = holdout_metrics_store[lbl]
        push!(f1_cls_df, (lbl, m.per_class_f1...))
    end
    CSV.write(joinpath(RESULTS_DIR, "holdout_per_class_f1.csv"), f1_cls_df)

    # ==========================================================================
    # FASE 6: CONTRASTES ESTADISTICOS DE HIPOTESIS SOBRE 5-FOLD CV
    # ==========================================================================
    println("\n[FASE 6] Contrastes estadisticos sobre los 5 folds emparejados de CV...")
    stat_models = [
        ("Best SVM",      best_svm_cv.fold_accs),
        ("Best KNN",      best_knn_cv.fold_accs),
        ("Best MLP",      best_mlp_cv.fold_accs),
        ("Best Bagging",  best_bag_cv.fold_accs),
        ("Best AdaBoost", best_ada_cv.fold_accs),
        ("Best EvoTree",  best_evo_cv.fold_accs),
        ("RandomForest",  rf_cv.fold_accs),
        ("XGBoost",       xgb_cv.fold_accs),
        ("LightGBM",      lgb_cv.fold_accs),
        ("CatBoost",      cat_cv.fold_accs),
    ]
    stat_names = [m[1] for m in stat_models]
    stat_vecs  = [m[2] for m in stat_models]
    stat_res = run_statistical_tests(stat_names, stat_vecs)

    println(@sprintf("  - Friedman Test (k=%d modelos, N=%d folds): Q = %.4f, p = %.6f | Iman-Davenport F = %.4f, p = %.6f",
                     stat_res.k, stat_res.N, stat_res.friedman_stat, stat_res.friedman_p,
                     stat_res.iman_davenport_F, stat_res.iman_davenport_p))
    println(@sprintf("  - One-Way ANOVA (referencia parametrica): F = %.4f, p = %.6f",
                     stat_res.anova_F, stat_res.anova_p))
    println(@sprintf("  - Post-hoc Nemenyi (alpha=0.05): q_alpha = %.3f, CD = %.3f",
                     stat_res.q_alpha, stat_res.cd))
    for (nm, rk, v) in zip(stat_names, stat_res.avg_ranks, stat_vecs)
        println(@sprintf("      %-15s | Rango medio = %4.2f | Acc = %5.2f%% +/- %5.2f%%",
                         nm, rk, mean(v)*100, std(v)*100))
    end
    CSV.write(joinpath(RESULTS_DIR, "stat_summary_cv.csv"), stat_res.summary_df)
    CSV.write(joinpath(RESULTS_DIR, "stat_nemenyi_pairs.csv"), stat_res.nemenyi_pairs)
    CSV.write(joinpath(RESULTS_DIR, "stat_wilcoxon_holm_pairs.csv"), stat_res.wilcoxon_pairs)

    # ==========================================================================
    # FASE 7: ABLACION METODOLOGICA (VALIDACION CORRECTA VS INCORRECTA)
    # ==========================================================================
    println("\n[FASE 7] Ablacion metodologica: Validacion Aleatoria vs GroupKFold vs Causal vs Holdout...")
    val_ablation_df = run_validation_methodology_ablation(
        Xtr, ytr, btr, Xte, yte, folds_causal; scaler=PRIMARY_SCALER
    )
    CSV.write(joinpath(RESULTS_DIR, "validation_ablation.csv"), val_ablation_df)
    for r in eachrow(val_ablation_df)
        println(@sprintf("  %-13s | RandCV=%5.2f%% | GroupCV=%5.2f%% | CausalCV=%5.2f%% | HoldoutB10=%5.2f%% | Sesgo Rand=%+5.2f pp",
                         r.model, r.random_cv_mean*100, r.group_cv_mean*100,
                         r.causal_cv_mean*100, r.holdout_b10_acc*100, r.optimism_gap_random_vs_test*100))
    end
    plot_fig10_statistical_cd_and_validation_ablation(
        stat_res, stat_names, val_ablation_df,
        joinpath(FIGURES_DIR, "fig10_statistical_cd_and_validation_ablation.png")
    )

    # ==========================================================================
    # FASE 8: VISUALIZACIONES 2D (PCA, LDA, ICA, t-SNE, Isomap, LLE)
    # ==========================================================================
    println("\n[FASE 8] Generando representaciones 2D (fit en Train -> transform sobre Test)...")
    vis_res = run_2d_visualizations_fit_transform(Xtr, ytr, btr, Xte, yte, bte; scaler=PRIMARY_SCALER, seed=SEED)
    CSV.write(joinpath(RESULTS_DIR, "projections_2d_separability.csv"), vis_res.sep_df)
    for r in eachrow(vis_res.sep_df)
        println(@sprintf("  %-8s | 2D Test 5-NN Acc = %5.2f%% | %s",
                         r.technique, r.test_2d_5nn_acc*100, r.train_fit_mode))
    end
    plot_fig07_projections_2d_class(vis_res, joinpath(FIGURES_DIR, "fig07_projections_2d_class.png"))
    plot_fig08_projections_2d_drift(vis_res, joinpath(FIGURES_DIR, "fig08_projections_2d_drift.png"))

    # ==========================================================================
    # FASE 9: SIGNIFICACION DE VARIABLES (RF + XGBOOST) Y ESTABILIDAD TEMPORAL
    # ==========================================================================
    println("\n[FASE 9] Analisis de importancia de variables (RF Gini y XGBoost Gain) y estabilidad...")
    imp_res = run_feature_importance_and_stability(
        X_marked, classes, batches, holdout_split.train_idx, holdout_split.test_idx;
        scaler=PRIMARY_SCALER, seed=SEED
    )
    CSV.write(joinpath(RESULTS_DIR, "feature_importance_global.csv"), imp_res.feat_df)
    CSV.write(joinpath(RESULTS_DIR, "sensor_importance_drift.csv"), imp_res.sensor_df)
    CSV.write(joinpath(RESULTS_DIR, "feature_type_importance.csv"), imp_res.ftype_df)
    CSV.write(joinpath(RESULTS_DIR, "importance_stability_rf_spearman.csv"),
              DataFrame(imp_res.rf_spearman, Symbol.(imp_res.block_names)))
    CSV.write(joinpath(RESULTS_DIR, "importance_stability_xgb_spearman.csv"),
              DataFrame(imp_res.xgb_spearman, Symbol.(imp_res.block_names)))

    println(@sprintf("  - Correlacion Spearman RF vs XGBoost (global)  : %.4f", imp_res.corr_rf_xgb))
    println(@sprintf("  - Correlacion Spearman Importancia RF vs Drift : %.4f", imp_res.corr_rf_drift))
    println(@sprintf("  - Correlacion Spearman Importancia XGB vs Drift: %.4f", imp_res.corr_xgb_drift))
    plot_fig09_feature_importance_stability(imp_res, joinpath(FIGURES_DIR, "fig09_feature_importance_stability.png"))

    # Guardar resumen consolidado JSON para verificacion automatica e integridad de la memoria
    summary_json = Dict(
        "julia_version" => string(VERSION),
        "seed" => SEED,
        "n_rows" => n_rows,
        "n_features" => n_feat,
        "n_batches" => n_batches,
        "n_classes" => n_classes,
        "explicit_nan_total" => miss_stats["explicit_nan_total"],
        "flatline_events" => miss_stats["flatline_event_count"],
        "full_8ch_flatline_events" => miss_stats["full_8ch_flatline_count"],
        "marked_nan_total" => miss_stats["marked_nan_total"],
        "train_n" => length(holdout_split.train_idx),
        "test_b10_n" => length(holdout_split.test_idx),
        "test_10pct_n" => length(test_10pct_local),
        "best_final_model" => best_final_lbl,
        "best_final_b10_acc" => best_final_row.test_acc_b10,
        "best_final_b10_macro_f1" => best_final_row.test_macro_f1_b10,
        "best_final_10pct_acc" => best_final_row.test_acc_10pct,
        "best_final_10pct_macro_f1" => best_final_row.test_macro_f1_10pct,
        "friedman_stat" => stat_res.friedman_stat,
        "friedman_p" => stat_res.friedman_p,
        "iman_davenport_F" => stat_res.iman_davenport_F,
        "iman_davenport_p" => stat_res.iman_davenport_p,
        "nemenyi_cd" => stat_res.cd,
        "anova_F" => stat_res.anova_F,
        "anova_p" => stat_res.anova_p,
        "corr_rf_xgb" => imp_res.corr_rf_xgb,
        "corr_rf_drift" => imp_res.corr_rf_drift,
        "corr_xgb_drift" => imp_res.corr_xgb_drift,
        "total_runtime_sec" => round(time() - t_total_start; digits=2)
    )
    open(joinpath(RESULTS_DIR, "summary_metrics.json"), "w") do io
        JSON.print(io, summary_json, 2)
    end

    println("\n" * "="^88)
    println(@sprintf("EJECUCION COMPLETA FINALIZADA CON EXITO en %.1f s.", time() - t_total_start))
    println("Resultados guardados en: $RESULTS_DIR")
    println("Figuras guardadas en   : $FIGURES_DIR")
    println("="^88)
end

main()
