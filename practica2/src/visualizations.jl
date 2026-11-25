# ==============================================================================
# Practica 2 - MAAAI (Curso 2026/27)
# Modulo: visualizations.jl
# Generacion automatica y reproducible de todas las figuras de la memoria (300 DPI)
# mediante Plots.jl (backend GR en modo headless).
# Nota: En Plots.jl, `text(str, sz::Int, ...)` exige `Int` para el tamano de fuente
# (un Float64 se interpreta como angulo de rotacion). Todas las anotaciones usan `Int`.
# ==============================================================================

ENV["GKSwstype"] = "100"
using Plots
using DataFrames
using Statistics
using Printf

function setup_plot_theme()
    default(
        fontfamily = "sans-serif",
        titlefontsize = 10,
        guidefontsize = 9,
        tickfontsize = 8,
        legendfontsize = 7,
        dpi = 300,
        framestyle = :box,
        grid = true,
        gridalpha = 0.25,
        background_color = :white,
        foreground_color_border = :gray40
    )
end

const CLASS_COLORS = [
    RGB(0.12, 0.47, 0.71),  # 1 Ethanol (azul)
    RGB(1.00, 0.50, 0.05),  # 2 Ethylene (naranja)
    RGB(0.17, 0.63, 0.17),  # 3 Ammonia (verde)
    RGB(0.84, 0.15, 0.16),  # 4 Acetaldehyde (rojo)
    RGB(0.58, 0.40, 0.74),  # 5 Acetone (morado)
    RGB(0.55, 0.34, 0.29),  # 6 Toluene (marron)
]

const CLASS_LABELS_SHORT = ["Ethanol", "Ethylene", "Ammonia", "Acetaldeh.", "Acetone", "Toluene"]

# ------------------------------------------------------------------------------
# Fig 01: Distribucion de muestras por batch y matriz de clases x batch
# ------------------------------------------------------------------------------
function plot_fig01_eda_batches_classes(batches::Vector{Int}, class_batch_mat::Matrix{Int}, out_path::String)
    setup_plot_theme()
    counts = [count(==(b), batches) for b in 1:10]
    counts_tr = [b <= 9 ? Float64(counts[b]) : NaN for b in 1:10]
    counts_te = [b == 10 ? Float64(counts[b]) : NaN for b in 1:10]

    p1 = bar(
        1:10, counts_tr;
        color = RGB(0.18, 0.49, 0.73), linecolor = :black, linewidth = 0.6,
        label = "Train / 5-Fold CV (B1-B9: 10.310; 74,1%)",
        xlabel = "Batch temporal (Meses 1 a 36)",
        ylabel = "Numero de instancias",
        title = "(A) Instancias por Batch: Train vs Holdout Test",
        xticks = (1:10, ["B$b" for b in 1:10]),
        ylims = (0, maximum(counts) * 1.24),
        legend = :topleft
    )
    bar!(p1, 1:10, counts_te;
        color = RGB(0.85, 0.28, 0.24), linecolor = :black, linewidth = 0.6,
        label = "Holdout Test Final (B10: 3.600; 25,9%)")
    for b in 1:10
        annotate!(p1, b, counts[b] + 130, text("$(counts[b])", 7, :center, :black))
    end

    p2 = heatmap(
        1:6, 1:10, Float64.(class_batch_mat);
        c = :YlGnBu,
        xlabel = "Clase de gas",
        ylabel = "Batch temporal",
        title = "(B) Clases por Batch (Toluene = 0 en B3-B5)",
        xticks = (1:6, CLASS_LABELS_SHORT),
        xrotation = 18,
        yticks = (1:10, ["B$b" for b in 1:10]),
        yflip = true,
        colorbar_title = "Muestras"
    )
    for b in 1:10, c in 1:6
        v = class_batch_mat[b, c]
        txt_col = v > 420 ? :white : (v == 0 ? :red : :black)
        annotate!(p2, c, b, text("$(v)", 8, :center, txt_col))
    end

    plt = plot(p1, p2; layout=(1, 2), size=(1240, 460),
               left_margin=8Plots.mm, bottom_margin=8Plots.mm, top_margin=6Plots.mm, right_margin=5Plots.mm)
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 02: Analisis forense de fallos implicitos de sensores y DR negativos
# ------------------------------------------------------------------------------
function plot_fig02_missing_sensor_faults(Xmat::Matrix{Float64}, batches::Vector{Int}, miss_stats::Dict{String, Any}, out_path::String)
    setup_plot_theme()
    fl_per_sensor = miss_stats["flatline_per_sensor"]
    emai_per_sensor = [sum(count(==(1e-4), @view Xmat[:, (s - 1) * 8 + 1 + k]) for k in 2:4) for s in 1:16]

    p1 = bar(
        (1:16) .- 0.18, fl_per_sensor;
        bar_width = 0.36, label = "Flatlines (DR=0 & |DR|=1)", color = RGB(0.84, 0.15, 0.16),
        xlabel = "Sensor MOX (S1-S16)", ylabel = "Numero de eventos / celdas",
        title = "(A) Fallos implicitos de lectura por sensor",
        xticks = (1:16, ["S$s" for s in 1:16]), ylims = (0, 44), legend = :topright
    )
    bar!(p1, (1:16) .+ 0.18, emai_per_sensor;
        bar_width = 0.36, label = "Centinela suelo EMAi=1e-4", color = RGB(0.95, 0.65, 0.15))
    for s in 1:16
        if fl_per_sensor[s] > 0
            annotate!(p1, s - 0.18, fl_per_sensor[s] + 1.2, text("$(fl_per_sensor[s])", 7, :center, :darkred))
        end
        if emai_per_sensor[s] > 0
            annotate!(p1, s + 0.18, emai_per_sensor[s] + 1.2, text("$(emai_per_sensor[s])", 7, :center, :black))
        end
    end

    mbb = miss_stats["marked_by_batch"]
    neg_dr_pct = [100.0 * count(<(0.0), @view Xmat[batches .== b, 1:8:128]) / (count(==(b), batches) * 16) for b in 1:10]

    p2 = bar(
        (1:10) .- 0.18, mbb.flatline_sensor_events;
        bar_width = 0.36, label = "Eventos flatline (conteo)", color = RGB(0.18, 0.49, 0.73),
        xlabel = "Batch temporal", ylabel = "Eventos flatline / % DR<0 (x10)",
        title = "(B) Concentracion temporal de anomalias por Batch",
        xticks = (1:10, ["B$b" for b in 1:10]), ylims = (0, 17.5), legend = :topright
    )
    bar!(p2, (1:10) .+ 0.18, neg_dr_pct .* 10.0;
        bar_width = 0.36, label = "% lecturas DR < 0 (escala x10)", color = RGB(0.58, 0.40, 0.74))
    for b in 1:10
        ev = mbb.flatline_sensor_events[b]
        if ev > 0
            annotate!(p2, b - 0.18, ev + 0.5, text("$ev", 7, :center, :navy))
        end
    end

    plt = plot(p1, p2; layout=(1, 2), size=(1240, 440),
               left_margin=8Plots.mm, bottom_margin=7Plots.mm, top_margin=6Plots.mm, right_margin=5Plots.mm)
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 03: Deriva temporal (DR medio por batch + mapa de calor KS por sensor)
# ------------------------------------------------------------------------------
function plot_fig03_temporal_drift_eda(Xmat::Matrix{Float64}, batches::Vector{Int}, out_path::String)
    setup_plot_theme()
    mean_dr = [mean(@view Xmat[batches .== b, 1:8:128]) / 1000.0 for b in 1:10]
    ks_vs_b1 = zeros(Float64, 10)
    ks_sensor_batch = zeros(Float64, 16, 9)

    idx_b1 = findall(==(1), batches)
    for b in 2:10
        idx_b = findall(==(b), batches)
        ks_all = [ks_statistic_2sample(@view(Xmat[idx_b1, j]), @view(Xmat[idx_b, j])) for j in 1:128]
        ks_vs_b1[b] = mean(ks_all)
        for s in 1:16
            ks_sensor_batch[s, b - 1] = mean(@view ks_all[(s - 1) * 8 + 1 : s * 8])
        end
    end

    p1 = plot(
        1:10, mean_dr;
        marker = :circle, linewidth = 2.2, color = RGB(0.12, 0.47, 0.71),
        label = "DR medio (kOhm)", xlabel = "Batch temporal (Mes 1 -> Mes 36)",
        ylabel = "DR medio (kOhm)",
        ylims = (6.0, 43.0),
        title = "(A) Degradacion de sensibilidad (DR) y deriva KS vs B1",
        xticks = (1:10, ["B$b" for b in 1:10]), legend = :topright
    )
    p1_twin = twinx(p1)
    plot!(p1_twin, 1:10, ks_vs_b1;
        marker = :diamond, linewidth = 2.0, linestyle = :dash, color = RGB(0.84, 0.15, 0.16),
        label = "KS medio vs B1", ylabel = "Estadistico KS medio (0-1)",
        ylims = (0.0, 0.65), legend = :bottomleft
    )

    p2 = heatmap(
        2:10, 1:16, ks_sensor_batch;
        c = :inferno,
        xlabel = "Batch temporal (comparado frente a Batch 1)",
        ylabel = "Sensor MOX",
        title = "(B) Mapa de deriva distribucional (KS medio por sensor)",
        xticks = (2:10, ["B$b" for b in 2:10]),
        yticks = (1:16, ["S$s" for s in 1:16]),
        clims = (0.0, 0.75),
        colorbar_title = "KS"
    )

    plt = plot(p1, p2; layout=(1, 2), size=(1260, 450),
               left_margin=10Plots.mm, right_margin=8Plots.mm, bottom_margin=7Plots.mm, top_margin=6Plots.mm)
    savefig(plt, out_path)
end

# Auxiliar geometrico para barras horizontales exactas (evita el bug de orientation=:h en Plots v1.41)
function add_hbars!(p, y_centers::AbstractVector{<:Real}, widths::AbstractVector{<:Real};
                    bar_height::Float64=0.25, color=:steelblue, label::String="",
                    xerr::Union{Nothing, AbstractVector{<:Real}}=nothing)
    first_drawn = true
    hh = bar_height / 2.0
    for idx in eachindex(y_centers)
        w = widths[idx]
        isnan(w) && continue
        yc = y_centers[idx]
        shp = Plots.Shape([0.0, w, w, 0.0], [yc - hh, yc - hh, yc + hh, yc + hh])
        plot!(p, shp; fillcolor=color, linecolor=:black, linewidth=0.45,
              label=(first_drawn ? label : ""))
        first_drawn = false
        if xerr !== nothing && !isnan(xerr[idx])
            s = xerr[idx]
            lo = max(0.0, w - s)
            hi = w + s
            plot!(p, [lo, hi], [yc, yc]; color=:gray20, linewidth=1.1, label="")
            plot!(p, [lo, lo], [yc - hh * 0.45, yc + hh * 0.45]; color=:gray20, linewidth=1.0, label="")
            plot!(p, [hi, hi], [yc - hh * 0.45, yc + hh * 0.45]; color=:gray20, linewidth=1.0, label="")
        end
    end
end

# ------------------------------------------------------------------------------
# Fig 04: Resultados de Modelos Basicos y Seleccion de Atributos (5-Fold CV)
# ------------------------------------------------------------------------------
function plot_fig04_base_models_cv(base_df::DataFrame, out_path::String)
    setup_plot_theme()

    filters_order = ["none", "anova", "pearson", "spearman", "kendall", "mi", "rfe"]
    filter_labels = ["Sin filtro", "ANOVA", "Pearson", "Spearman", "Kendall", "MutualInfo", "RFE-LR"]

    sub_noproj = filter(r -> r.projection == "none", base_df)
    svm_vals = Float64[]
    knn_vals = Float64[]
    mlp_vals = Float64[]
    for f in filters_order
        s_row = filter(r -> r.filter == f && r.classifier == "svm" && occursin("1.0", r.clf_params), sub_noproj)
        k_row = filter(r -> r.filter == f && r.classifier == "knn" && occursin("10", r.clf_params), sub_noproj)
        m_row = filter(r -> r.filter == f && r.classifier == "mlp" && occursin("[100]", r.clf_params), sub_noproj)
        push!(svm_vals, isempty(s_row) ? 0.0 : s_row.acc_mean[1] * 100)
        push!(knn_vals, isempty(k_row) ? 0.0 : k_row.acc_mean[1] * 100)
        push!(mlp_vals, isempty(m_row) ? 0.0 : m_row.acc_mean[1] * 100)
    end

    x = 1:length(filters_order)
    p1 = bar(x .- 0.25, svm_vals; bar_width=0.24, label="SVM (C=1.0)", color=RGB(0.12, 0.47, 0.71),
             xticks=(x, filter_labels), xrotation=22, ylims=(0, 100),
             ylabel="Accuracy 5-Fold CV (%)", xlabel="Tecnica de filtrado (k=32)",
             title="(A) Efecto del filtrado de caracteristicas (sin proyeccion)", legend=:bottomright)
    bar!(p1, x, knn_vals; bar_width=0.24, label="KNN (k=10)", color=RGB(1.00, 0.50, 0.05))
    bar!(p1, x .+ 0.25, mlp_vals; bar_width=0.24, label="MLP ([100])", color=RGB(0.17, 0.63, 0.17))

    sorted_df = sort(base_df, :acc_mean; rev=true)
    top_n = min(14, nrow(sorted_df))
    top_df = sorted_df[1:top_n, :]
    rev_names = reverse(top_df.name)
    rev_means = reverse(top_df.acc_mean .* 100)
    rev_stds  = reverse(top_df.acc_std .* 100)

    p2 = plot(
        yticks = (1:top_n, rev_names),
        ylims = (0.3, top_n + 0.7),
        xlims = (0, 108),
        xlabel = "Accuracy 5-Fold Temporal CV (% +/- SD)",
        title = "(B) Top 14 pipelines de modelos basicos (5-Fold Causal CV)",
        legend = false
    )
    add_hbars!(p2, collect(1:top_n), rev_means; bar_height=0.60, color=RGB(0.20, 0.55, 0.68), xerr=rev_stds)
    for i in 1:top_n
        hi = rev_means[i] + rev_stds[i]
        annotate!(p2, min(hi + 1.2, 99.5), i, text(@sprintf("%.1f%%", rev_means[i]), 7, :left, :black))
    end

    plt = plot(p1, p2; layout=(1, 2), size=(1460, 540),
               left_margin=12Plots.mm, bottom_margin=8Plots.mm, top_margin=6Plots.mm, right_margin=6Plots.mm)
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 05: Ensembles y Evaluacion Final: CV vs Holdout Test (Batch 10)
# ------------------------------------------------------------------------------
function plot_fig05_ensembles_and_holdout(holdout_df::DataFrame, out_path::String)
    setup_plot_theme()
    df_ord = sort(holdout_df, :test_acc_b10; rev=false)
    n = nrow(df_ord)
    y_pos = collect(1:n)

    cv_vals = [isnan(v) ? NaN : v * 100 for v in df_ord.cv_acc_mean]
    te_vals = df_ord.test_acc_b10 .* 100
    te10_vals = df_ord.test_acc_10pct .* 100

    p = plot(
        yticks = (y_pos, df_ord.model_label),
        ylims = (0.3, n + 0.7),
        xlims = (0, 108),
        xlabel = "Accuracy (%)",
        title = "Comparativa final: 5-Fold Temporal CV vs Holdout Test (Batch 10 y submuestra 10%)",
        legend = :bottomright
    )
    add_hbars!(p, y_pos .+ 0.25, cv_vals; bar_height=0.23,
               color=RGB(0.25, 0.55, 0.80), label="5-Fold Causal CV (B1-B9)")
    add_hbars!(p, y_pos, te_vals; bar_height=0.23,
               color=RGB(0.85, 0.28, 0.24), label="Holdout Test Batch 10 completo (n=3.600)")
    add_hbars!(p, y_pos .- 0.25, te10_vals; bar_height=0.23,
               color=RGB(0.95, 0.65, 0.15), label="Holdout 10% estratificado (n=1.392, seed=104)")

    for i in 1:n
        c_max = isnan(cv_vals[i]) ? max(te_vals[i], te10_vals[i]) : max(cv_vals[i], te_vals[i], te10_vals[i])
        annotate!(p, c_max + 1.0, y_pos[i], text(@sprintf("B10: %.1f%%", te_vals[i]), 7, :left, :darkred))
    end

    plt = plot(p; size=(1180, 600),
               left_margin=12Plots.mm, bottom_margin=7Plots.mm, top_margin=6Plots.mm, right_margin=6Plots.mm)
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 06: Matriz de Confusion del mejor modelo en Batch 10 + F1 por clase
# ------------------------------------------------------------------------------
function plot_fig06_confusion_matrix_best(
    best_model_name::String,
    cm::Matrix{Int},
    per_class_f1_dict::Dict{String, Vector{Float64}},
    out_path::String
)
    setup_plot_theme()
    row_sums = sum(cm; dims=2)
    cm_pct = 100.0 .* cm ./ max.(row_sums, 1)

    p1 = heatmap(
        1:6, 1:6, cm_pct;
        c = :Blues, clims = (0, 100), yflip = true,
        xticks = (1:6, CLASS_LABELS_SHORT),
        yticks = (1:6, CLASS_LABELS_SHORT),
        xrotation = 20,
        xlabel = "Clase predicha", ylabel = "Clase real",
        title = "(A) Matriz de confusion (%) en Batch 10: $(best_model_name)",
        colorbar_title = "% fila"
    )
    for r in 1:6, c in 1:6
        v = cm[r, c]
        pct = cm_pct[r, c]
        tc = pct > 55.0 ? :white : :black
        annotate!(p1, c, r, text(@sprintf("%d\n(%.1f%%)", v, pct), 7, :center, tc))
    end

    mnames = sort(collect(keys(per_class_f1_dict)))
    nm = length(mnames)
    x = collect(1:6)
    w = 0.75 / max(nm, 1)
    p2 = plot(
        xticks = (x, CLASS_LABELS_SHORT), xrotation = 20,
        ylims = (0, 118), ylabel = "F1-Score (%)", xlabel = "Gas objetivo",
        title = "(B) Rendimiento F1 por clase en Holdout Test (Batch 10)",
        legend = :top, legend_columns = 3
    )
    pal = [RGB(0.12, 0.47, 0.71), RGB(0.17, 0.63, 0.17), RGB(0.84, 0.15, 0.16), RGB(0.58, 0.40, 0.74), RGB(1.00, 0.50, 0.05)]
    for (idx, mn) in enumerate(mnames)
        offset = (idx - (nm + 1) / 2.0) * w
        bar!(p2, x .+ offset, per_class_f1_dict[mn] .* 100;
             bar_width = w * 0.90, label = mn, color = pal[mod1(idx, length(pal))])
    end

    plt = plot(p1, p2; layout=(1, 2), size=(1260, 490),
               left_margin=8Plots.mm, bottom_margin=8Plots.mm, top_margin=6Plots.mm, right_margin=5Plots.mm)
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 07 y Fig 08: Visualizaciones 2D (PCA, LDA, ICA, t-SNE, Isomap, LLE)
# ------------------------------------------------------------------------------
function plot_fig07_projections_2d_class(vis_res, out_path::String)
    setup_plot_theme()
    techs = ["PCA", "LDA", "ICA", "t-SNE", "Isomap", "LLE"]
    sep_map = Dict(row.technique => row.test_2d_5nn_acc for row in eachrow(vis_res.sep_df))
    panels = Any[]

    for (ti, tech) in enumerate(techs)
        emb = vis_res.embeddings[tech]
        Ztr = emb.Z_tr
        Zte = emb.Z_te
        ytr = vis_res.ytr_sub[emb.ref_idx]
        yte = vis_res.yte_sub
        acc2d = sep_map[tech] * 100

        p = plot(
            title = @sprintf("%s (Test 2D 5-NN: %.1f%%)", tech, acc2d),
            xlabel = "Dim 1", ylabel = "Dim 2",
            legend = (ti == 1 ? :topright : false)
        )
        for c in 1:6
            itr = findall(==(c), ytr)
            scatter!(p, Ztr[itr, 1], Ztr[itr, 2];
                color = CLASS_COLORS[c], alpha = 0.16, markersize = 2.0,
                markerstrokewidth = 0, label = "")
        end
        for c in 1:6
            ite = findall(==(c), yte)
            scatter!(p, Zte[ite, 1], Zte[ite, 2];
                color = CLASS_COLORS[c], alpha = 0.80, markersize = 2.8,
                markershape = :circle, markerstrokewidth = 0.2, markerstrokecolor = :white,
                label = (ti == 1 ? CLASS_LABELS_SHORT[c] : ""))
        end
        push!(panels, p)
    end

    plt = plot(panels...; layout=(2, 3), size=(1360, 820),
               left_margin=6Plots.mm, bottom_margin=6Plots.mm, top_margin=5Plots.mm,
               plot_title="Proyecciones 2D ajustadas en Train (fondo tenue) y aplicadas sobre Test Batch 10 (puntos)")
    savefig(plt, out_path)
end

function plot_fig08_projections_2d_drift(vis_res, out_path::String)
    setup_plot_theme()
    techs = ["PCA", "LDA", "ICA", "t-SNE", "Isomap", "LLE"]
    panels = Any[]

    for (ti, tech) in enumerate(techs)
        emb = vis_res.embeddings[tech]
        Ztr = emb.Z_tr
        Zte = emb.Z_te
        btr = vis_res.btr_sub[emb.ref_idx]
        bte = vis_res.bte_sub

        Zall = vcat(Ztr, Zte)
        ball = Float64.(vcat(btr, bte))

        p = scatter(
            Zall[:, 1], Zall[:, 2];
            marker_z = ball, c = :viridis, clims = (1, 10),
            alpha = 0.58, markersize = 2.2, markerstrokewidth = 0,
            title = "$tech: Trayectoria temporal (B1 -> B10)",
            xlabel = "Dim 1", ylabel = "Dim 2",
            colorbar = (ti in (3, 6)),
            colorbar_title = "Batch",
            legend = false
        )
        push!(panels, p)
    end

    plt = plot(panels...; layout=(2, 3), size=(1360, 780),
               left_margin=6Plots.mm, bottom_margin=6Plots.mm, top_margin=5Plots.mm,
               plot_title="Deriva temporal (Sensor Drift) en representaciones 2D coloreadas por Batch (1..10)")
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 09: Importancia de Variables (RF + XGBoost), Estabilidad y Drift
# ------------------------------------------------------------------------------
function plot_fig09_feature_importance_stability(imp_res, out_path::String)
    setup_plot_theme()

    fdf = copy(imp_res.feat_df)
    fdf[!, :comb] = 0.5 .* (fdf.rf_importance .+ fdf.xgb_importance)
    top15 = sort(fdf, :comb; rev=true)[1:15, :]
    y15 = collect(1:15)
    max_imp15 = max(maximum(top15.rf_importance), maximum(top15.xgb_importance)) * 100
    p1 = plot(
        yticks = (y15, reverse(top15.feature_name)),
        xlims = (0, max_imp15 * 1.18),
        ylims = (0.3, 15.7),
        xlabel = "Importancia relativa (%)",
        title = "(A) Top 15 variables mas relevantes",
        legend = :bottomright
    )
    add_hbars!(p1, y15 .+ 0.18, reverse(top15.rf_importance .* 100);
               bar_height=0.34, color=RGB(0.12, 0.47, 0.71), label="Random Forest (Gini)")
    add_hbars!(p1, y15 .- 0.18, reverse(top15.xgb_importance .* 100);
               bar_height=0.34, color=RGB(1.00, 0.50, 0.05), label="XGBoost (Total Gain)")

    sdf = sort(imp_res.sensor_df, :sensor)
    p2 = bar(
        sdf.sensor .- 0.18, sdf.rf_imp_sum .* 100;
        bar_width = 0.35, label = "RF (%)", color = RGB(0.12, 0.47, 0.71),
        xticks = (1:16, ["S$s" for s in 1:16]),
        ylims = (0.0, 26.0),
        xlabel = "Sensor MOX", ylabel = "Importancia acumulada (%)",
        title = "(B) Importancia por sensor vs Deriva KS (B1-9 -> B10)",
        legend = :top
    )
    bar!(p2, sdf.sensor .+ 0.18, sdf.xgb_imp_sum .* 100;
        bar_width = 0.35, label = "XGB (%)", color = RGB(1.00, 0.50, 0.05))
    p2_tw = twinx(p2)
    plot!(p2_tw, sdf.sensor, sdf.ks_drift_mean;
        color = :red, marker = :diamond, linewidth = 1.8, linestyle = :dash,
        ylabel = "KS medio (drift)", ylims = (0.0, 0.40), label = "KS drift", legend = :topright)

    bn = imp_res.block_names
    nb = length(bn)
    p3 = heatmap(
        1:nb, 1:nb, imp_res.rf_spearman;
        c = :RdYlBu, clims = (0.0, 1.0), yflip = true,
        xticks = (1:nb, bn), yticks = (1:nb, bn),
        xlabel = "Bloque temporal", ylabel = "Bloque temporal",
        title = "(C) Estabilidad Spearman de importancia RF entre batches",
        colorbar_title = "rho"
    )
    for i in 1:nb, j in 1:nb
        v = imp_res.rf_spearman[i, j]
        annotate!(p3, j, i, text(@sprintf("%.2f", v), 8, :center, v > 0.78 || v < 0.22 ? :white : :black))
    end

    max_rf_pct = maximum(fdf.rf_importance .* 100)
    p4 = scatter(
        fdf.ks_drift_tr_vs_b10, fdf.rf_importance .* 100;
        color = RGB(0.17, 0.55, 0.55), alpha = 0.75, markersize = 4.5,
        xlims = (minimum(fdf.ks_drift_tr_vs_b10) - 0.02, maximum(fdf.ks_drift_tr_vs_b10) + 0.08),
        ylims = (0.0, max_rf_pct * 1.18),
        xlabel = "Deriva KS por variable (Train B1-9 vs Test B10)",
        ylabel = "Importancia global RF (%)",
        title = @sprintf("(D) Importancia vs Deriva (Spearman rho_RF=%.2f, rho_XGB=%.2f)",
                         imp_res.corr_rf_drift, imp_res.corr_xgb_drift),
        legend = false
    )
    for r in eachrow(sort(fdf, :rf_importance; rev=true)[1:5, :])
        annotate!(p4, r.ks_drift_tr_vs_b10 + 0.005, r.rf_importance * 100,
                  text(r.feature_name, 7, :left, :black))
    end

    plt = plot(p1, p2, p3, p4; layout=(2, 2), size=(1380, 880),
               left_margin=10Plots.mm, right_margin=8Plots.mm, bottom_margin=7Plots.mm, top_margin=6Plots.mm)
    savefig(plt, out_path)
end

# ------------------------------------------------------------------------------
# Fig 10: Diagrama de Diferencia Critica (Nemenyi) + Ablacion de Validacion
# ------------------------------------------------------------------------------
function plot_fig10_statistical_cd_and_validation_ablation(
    stat_res,
    model_names::Vector{String},
    val_ablation_df::DataFrame,
    out_path::String
)
    setup_plot_theme()

    k = stat_res.k
    ranks = stat_res.avg_ranks
    cd = stat_res.cd
    ord = sortperm(ranks)
    s_names = model_names[ord]
    s_ranks = ranks[ord]

    p1 = plot(
        xlims = (0.2, k + 0.8),
        ylims = (0.0, k + 2.8),
        yticks = false,
        xticks = 1:k,
        xlabel = "Rango promedio en 5-Fold Temporal CV (menor = mejor)",
        title = @sprintf("(A) Diagrama CD de Nemenyi (Friedman p=%.4f, CD=%.2f)",
                         stat_res.friedman_p, cd),
        legend = false,
        framestyle = :axes,
        grid = :x
    )
    y_axis = k + 1.6
    plot!(p1, [1, k], [y_axis, y_axis]; color=:black, linewidth=1.5)
    for r in 1:k
        plot!(p1, [r, r], [y_axis - 0.15, y_axis + 0.15]; color=:black, linewidth=1.2)
    end
    plot!(p1, [1.0, 1.0 + cd], [y_axis + 0.65, y_axis + 0.65]; color=:crimson, linewidth=2.5)
    plot!(p1, [1.0, 1.0], [y_axis + 0.52, y_axis + 0.78]; color=:crimson, linewidth=2.0)
    plot!(p1, [1.0 + cd, 1.0 + cd], [y_axis + 0.52, y_axis + 0.78]; color=:crimson, linewidth=2.0)
    annotate!(p1, 1.0 + cd / 2.0, y_axis + 0.98, text(@sprintf("CD = %.2f", cd), 8, :center, :crimson))

    for (idx, (nm, rk)) in enumerate(zip(s_names, s_ranks))
        y_row = k - idx + 0.8
        scatter!(p1, [rk], [y_row]; color=RGB(0.12, 0.47, 0.71), markersize=5.5)
        plot!(p1, [rk, rk], [y_row, y_axis]; color=:gray65, linestyle=:dot, linewidth=0.9)
        if rk <= (k + 1) / 2.0
            annotate!(p1, rk + 0.16, y_row, text(@sprintf("%s (%.2f)", nm, rk), 8, :left, :black))
        else
            annotate!(p1, rk - 0.16, y_row, text(@sprintf("%s (%.2f)", nm, rk), 8, :right, :black))
        end
    end

    cliques = Tuple{Int, Int}[]
    for i in 1:k
        j_max = i
        for j in (i + 1):k
            if s_ranks[j] - s_ranks[i] <= cd
                j_max = j
            end
        end
        if j_max > i && (isempty(cliques) || j_max > cliques[end][2])
            push!(cliques, (i, j_max))
        end
    end
    for (ci, (i1, i2)) in enumerate(cliques)
        yc = y_axis - 0.35 - 0.25 * (ci - 1)
        plot!(p1, [s_ranks[i1], s_ranks[i2]], [yc, yc]; color=:black, linewidth=3.2)
    end

    m_lbls = val_ablation_df.model
    x = collect(1:nrow(val_ablation_df))
    p2 = bar(x .- 0.30, val_ablation_df.random_cv_mean .* 100;
             bar_width=0.19, label="5-Fold Aleatorio (INCORRECTO: fuga batch)", color=RGB(0.84, 0.15, 0.16),
             xticks=(x, m_lbls), ylims=(0, 132),
             ylabel="Accuracy (%)", xlabel="Clasificador representativo",
             title="(B) Impacto del esquema de validacion sobre el sesgo optimista",
             legend=:top, legend_columns=2)
    bar!(p2, x .- 0.10, val_ablation_df.group_cv_mean .* 100;
         bar_width=0.19, label="5-Fold GroupKFold (sin mezcla, bidireccional)", color=RGB(0.95, 0.65, 0.15))
    bar!(p2, x .+ 0.10, val_ablation_df.causal_cv_mean .* 100;
         bar_width=0.19, label="5-Fold Temporal Causal (nuestro protocolo)", color=RGB(0.12, 0.47, 0.71))
    bar!(p2, x .+ 0.30, val_ablation_df.holdout_b10_acc .* 100;
         bar_width=0.19, label="Holdout Test Final (Batch 10, Mes 36)", color=RGB(0.17, 0.63, 0.17))

    plt = plot(p1, p2; layout=(1, 2), size=(1380, 540),
               left_margin=8Plots.mm, bottom_margin=8Plots.mm, top_margin=6Plots.mm, right_margin=6Plots.mm)
    savefig(plt, out_path)
end
