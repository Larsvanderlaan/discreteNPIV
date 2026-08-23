#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

out_dir <- arg_value("--out-dir", "experiments/generated/figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

main_summary <- arg_value("--main-summary", "experiments/results/main/summary.csv")
projection_summary <- arg_value(
  "--projection-summary",
  "experiments/results/large_n/summary.csv"
)

num <- function(x) suppressWarnings(as.numeric(x))

regime_label <- function(x) {
  out <- ifelse(x == "covered", "Point identified",
    ifelse(x == "vanishing", "Asymptotically identified",
      ifelse(x == "persistent", "Persistent gap", x)
    )
  )
  factor(out, levels = c("Point identified", "Asymptotically identified", "Persistent gap"))
}

method_label <- function(x) {
  out <- x
  out[x == "2sls_dml"] <- "nonsplit 2SLS-DML"
  out[x == "loo_npjive"] <- "LOO DML-JIVE"
  out[x == "loo_npjive_star"] <- "LOO DML-JIVE*"
  out[x == "single_split_npjive"] <- "single-split DML-JIVE"
  out[x == "single_split_npjive_star"] <- "single-split DML-JIVE*"
  factor(
    out,
    levels = c(
      "nonsplit 2SLS-DML",
      "LOO DML-JIVE", "LOO DML-JIVE*",
      "single-split DML-JIVE", "single-split DML-JIVE*"
    )
  )
}

table_method_label <- function(x) {
  as.character(method_label(x))
}

save_plot <- function(plot, stem, width, height) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  ggsave(png_path, plot, width = width, height = height, dpi = 320, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, bg = "white")
  c(png_path, pdf_path)
}

save_stacked_plots <- function(top_plot, bottom_plot, stem, width, height) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  png(png_path, width = width, height = height, units = "in", res = 320, bg = "white")
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(2, 1, heights = unit(c(1, 1), "null"))))
  print(top_plot, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(bottom_plot, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  dev.off()
  pdf(pdf_path, width = width, height = height)
  grid.newpage()
  pushViewport(viewport(layout = grid.layout(2, 1, heights = unit(c(1, 1), "null"))))
  print(top_plot, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
  print(bottom_plot, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
  dev.off()
  c(png_path, pdf_path)
}

write_latex_table <- function(df, path, caption, label) {
  con <- file(path, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  cat("\\begin{table}[t]\n\\centering\n", file = con)
  cat("\\caption{", caption, "}\n", sep = "", file = con, append = TRUE)
  cat("\\label{", label, "}\n", sep = "", file = con, append = TRUE)
  cat("\\resizebox{\\textwidth}{!}{%\n", file = con, append = TRUE)
  cat("\\begin{tabular}{", paste(rep("l", ncol(df)), collapse = ""), "}\n", sep = "", file = con, append = TRUE)
  cat("\\toprule\n", file = con, append = TRUE)
  cat(paste(names(df), collapse = " & "), "\\\\\n", file = con, append = TRUE)
  cat("\\midrule\n", file = con, append = TRUE)
  for (i in seq_len(nrow(df))) {
    cat(paste(as.character(df[i, ]), collapse = " & "), "\\\\\n", file = con, append = TRUE)
  }
  cat("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n", file = con, append = TRUE)
}

main_methods <- c(
  "2sls_dml",
  "single_split_npjive",
  "single_split_npjive_star",
  "loo_npjive",
  "loo_npjive_star"
)

make_main_summary <- function(summary_path) {
  replications_path <- file.path(dirname(summary_path), "replications.csv")
  if (!file.exists(replications_path)) {
    main <- read.csv(summary_path, stringsAsFactors = FALSE)
    main <- subset(
      main,
      growth_path == "fixed_n30" &
        regime %in% c("covered", "vanishing", "persistent") &
        method %in% main_methods
    )
    main$K <- num(main$n_instruments)
    main$Coverage <- num(main$estimand_coverage)
    main$RMSE <- num(main$estimand_rmse)
    main$Bias <- num(main$estimand_bias)
    main$SESD <- num(main$se_over_estimand_empirical_sd)
    main$GapSE <- num(main$mean_normalized_identification_gap)
    main$Target <- ifelse(main$target_estimand == "projection", "\\(\\tau_K\\)", "\\(\\tau\\)")
    return(main)
  }

  reps <- read.csv(replications_path, stringsAsFactors = FALSE)
  reps <- subset(
    reps,
    row_type == "estimate" &
      growth_path == "fixed_n30" &
      regime %in% c("covered", "vanishing", "persistent") &
      method %in% main_methods
  )
  reps$estimand_error_value <- ifelse(
    is.na(num(reps$estimand_error)),
    ifelse(reps$target_estimand == "projection", num(reps$projection_error), num(reps$error)),
    num(reps$estimand_error)
  )
  reps$estimand_covered_value <- ifelse(
    reps$estimand_covered %in% c(TRUE, "TRUE", "True", "true", 1, "1"),
    1,
    0
  )
  reps$se_value <- num(reps$se)
  reps$gap_value <- num(reps$normalized_identification_gap)
  keys <- c("growth_path", "regime", "method", "target_estimand", "n_instruments")
  split_rows <- split(reps, interaction(reps[keys], drop = TRUE, lex.order = TRUE))
  out <- do.call(rbind, lapply(split_rows, function(df) {
    err <- df$estimand_error_value
    se <- df$se_value
    empirical_sd <- if (length(err) > 1) sd(err) else NA_real_
    data.frame(
      growth_path = df$growth_path[1],
      regime = df$regime[1],
      method = df$method[1],
      target_estimand = df$target_estimand[1],
      n_instruments = df$n_instruments[1],
      n_replications = nrow(df),
      K = num(df$n_instruments[1]),
      Coverage = mean(df$estimand_covered_value),
      RMSE = sqrt(mean(err^2)),
      Bias = mean(err),
      SESD = ifelse(is.na(empirical_sd) || empirical_sd == 0, NA_real_, mean(se) / empirical_sd),
      GapSE = mean(df$gap_value),
      Target = ifelse(df$target_estimand[1] == "projection", "\\(\\tau_K\\)", "\\(\\tau\\)"),
      stringsAsFactors = FALSE
    )
  }))
  rownames(out) <- NULL
  out
}

main <- make_main_summary(main_summary)
main$Regime <- regime_label(main$regime)
main$Method <- method_label(main$method)

coverage_ref <- data.frame(Regime = factor(levels(main$Regime), levels = levels(main$Regime)))
method_colors <- c(
  "nonsplit 2SLS-DML" = "#D95F02",
  "single-split DML-JIVE" = "#7570B3",
  "single-split DML-JIVE*" = "#66A61E",
  "LOO DML-JIVE" = "#1B9E77",
  "LOO DML-JIVE*" = "#E7298A"
)
method_shapes <- c(
  "nonsplit 2SLS-DML" = 17,
  "single-split DML-JIVE" = 15,
  "single-split DML-JIVE*" = 8,
  "LOO DML-JIVE" = 16,
  "LOO DML-JIVE*" = 18
)
method_breaks <- c(
  "nonsplit 2SLS-DML",
  "LOO DML-JIVE",
  "LOO DML-JIVE*",
  "single-split DML-JIVE",
  "single-split DML-JIVE*"
)
method_legend_labels <- expression(
  "nonsplit 2SLS-DML (target"~tau*")",
  "LOO DML-JIVE (target"~tau*")",
  "LOO DML-JIVE* (target"~tau[K]*")",
  "single-split DML-JIVE (target"~tau*")",
  "single-split DML-JIVE* (target"~tau[K]*")"
)

main_coverage_plot <- ggplot(main, aes(K, Coverage, color = Method, shape = Method)) +
  geom_hline(data = coverage_ref, aes(yintercept = 0.95), inherit.aes = FALSE,
    linetype = "dashed", linewidth = 0.35, color = "grey35") +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.2) +
  facet_wrap(~Regime, nrow = 1) +
  scale_x_continuous(breaks = c(50, 100, 200, 400)) +
  scale_y_continuous(limits = c(0, 1), breaks = c(0.25, 0.5, 0.75, 0.95)) +
  scale_color_manual(values = method_colors, breaks = method_breaks, labels = method_legend_labels) +
  scale_shape_manual(values = method_shapes, breaks = method_breaks, labels = method_legend_labels) +
  labs(
    x = NULL,
    y = "Coverage",
    color = NULL,
    shape = NULL
  ) +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey70")
  )

rmse_breaks <- c(0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 25)
main_rmse_plot <- ggplot(main, aes(K, RMSE, color = Method, shape = Method)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.2) +
  facet_wrap(~Regime, nrow = 1) +
  scale_x_continuous(breaks = c(50, 100, 200, 400)) +
  scale_y_log10(breaks = rmse_breaks, labels = rmse_breaks) +
  scale_color_manual(values = method_colors, breaks = method_breaks, labels = method_legend_labels) +
  scale_shape_manual(values = method_shapes, breaks = method_breaks, labels = method_legend_labels) +
  labs(
    x = "Number of historical experiments K",
    y = "RMSE (log scale)",
    color = NULL,
    shape = NULL
  ) +
  guides(
    color = guide_legend(nrow = 2, byrow = TRUE),
    shape = guide_legend(nrow = 2, byrow = TRUE)
  ) +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey70")
  )
main_paths <- save_stacked_plots(main_coverage_plot, main_rmse_plot, "main_results", width = 9.6, height = 6.5)

k400 <- subset(main, K == 400)
k400 <- k400[order(k400$Regime, k400$Method), ]
main_table <- data.frame(
  Regime = as.character(k400$Regime),
  Method = table_method_label(k400$method),
  Target = k400$Target,
  Bias = sprintf("%.3f", k400$Bias),
  RMSE = sprintf("%.3f", k400$RMSE),
  Coverage = sprintf("%.3f", k400$Coverage),
  `SE/SD` = sprintf("%.2f", k400$SESD),
  `Gap/SE` = sprintf("%.2f", k400$GapSE),
  check.names = FALSE
)
write.csv(main_table, file.path(out_dir, "main_k400_table.csv"), row.names = FALSE)
write_latex_table(
  main_table,
  file.path(out_dir, "main_k400_table.tex"),
  "Fixed-$n_k=30$ results at $K=400$; DML-JIVE* estimates the projection from the observed samples.",
  "tab:main-k400"
)

projection <- read.csv(projection_summary, stringsAsFactors = FALSE)
projection <- subset(
  projection,
  growth_path == "large_n400" &
    regime == "persistent" &
    num(n_instruments) == 400 &
    num(n_per_instrument) == 400 &
    method %in% c(
      "loo_npjive", "loo_npjive_star",
      "single_split_npjive", "single_split_npjive_star"
    )
)
projection$Method <- method_label(projection$method)
projection$Estimator <- ifelse(
  grepl("^loo_", projection$method),
  "LOO",
  "single split"
)
projection$Estimator <- factor(projection$Estimator, levels = c("LOO", "single split"))
projection$Correction <- ifelse(
  grepl("_star$", projection$method),
  "DML-JIVE* (projection corrected)",
  "DML-JIVE"
)
projection$Correction <- factor(
  projection$Correction,
  levels = c("DML-JIVE", "DML-JIVE* (projection corrected)")
)
projection$TrueCoverage <- num(projection$coverage)
projection$ProjectionCoverage <- num(projection$projection_coverage)
projection$ProjectionRMSE <- num(projection$projection_rmse)
projection$SESD <- num(projection$se_over_estimand_empirical_sd)
if (all(is.na(projection$SESD))) {
  projection$SESD <- num(projection$se_over_empirical_sd)
}

projection_long <- rbind(
  data.frame(projection[, c("Estimator", "Correction")], Metric = "Coverage: original target", Value = projection$TrueCoverage),
  data.frame(projection[, c("Estimator", "Correction")], Metric = "Coverage: approximating target", Value = projection$ProjectionCoverage),
  data.frame(projection[, c("Estimator", "Correction")], Metric = "RMSE: approximating target", Value = projection$ProjectionRMSE)
)
projection_long$Metric <- factor(
  projection_long$Metric,
  levels = c(
    "Coverage: original target",
    "Coverage: approximating target",
    "RMSE: approximating target"
  )
)
projection_ref <- data.frame(
  Metric = factor(
    c("Coverage: original target", "Coverage: approximating target"),
    levels = levels(projection_long$Metric)
  ),
  y = c(0.95, 0.95)
)

projection_plot <- ggplot(projection_long, aes(Estimator, Value, fill = Correction)) +
  geom_hline(
    data = projection_ref,
    aes(yintercept = y),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 0.35,
    color = "grey35"
  ) +
  geom_col(
    width = 0.65,
    position = position_dodge(width = 0.72),
    color = "grey20",
    linewidth = 0.2
  ) +
  facet_wrap(~Metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c(
    "DML-JIVE" = "#7570B3",
    "DML-JIVE* (projection corrected)" = "#E7298A"
  )) +
  labs(
    x = NULL,
    y = NULL,
    fill = NULL
  ) +
  theme_bw(base_size = 10) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey70")
  )
projection_paths <- save_plot(projection_plot, "projection_correction", width = 8.2, height = 3.8)

projection_table <- projection[order(projection$Method), ]
projection_table <- data.frame(
  Method = table_method_label(projection_table$method),
  `Coverage vs. \\(\\tau\\)` = sprintf("%.3f", projection_table$TrueCoverage),
  `Coverage vs. \\(\\tau_K\\)` = sprintf("%.3f", projection_table$ProjectionCoverage),
  `Approximating-target bias` = sprintf("%.3f", num(projection_table$projection_bias)),
  `Approximating-target RMSE` = sprintf("%.3f", projection_table$ProjectionRMSE),
  `SE/SD` = sprintf("%.2f", projection_table$SESD),
  check.names = FALSE
)
write.csv(projection_table, file.path(out_dir, "large_cell_table.csv"), row.names = FALSE)
write_latex_table(
  projection_table,
  file.path(out_dir, "large_cell_table.tex"),
  "Large-cell results under persistent nonidentification at $K=400,n_k=400$; DML-JIVE* estimates the projection from the observed samples.",
  "tab:projection-k400-n400"
)

manifest <- data.frame(
  artifact = c(
    basename(main_paths),
    "main_k400_table.csv",
    "main_k400_table.tex",
    basename(projection_paths),
    "large_cell_table.csv",
    "large_cell_table.tex"
  ),
  source = c(
    rep(main_summary, length(main_paths)),
    main_summary,
    main_summary,
    rep(projection_summary, length(projection_paths)),
    projection_summary,
    projection_summary
  )
)
write.csv(manifest, file.path(out_dir, "figure_manifest.csv"), row.names = FALSE)

cat("Wrote paper visuals to ", out_dir, "\n", sep = "")
cat("Main source: ", main_summary, "\n", sep = "")
cat("Projection source: ", projection_summary, "\n", sep = "")
