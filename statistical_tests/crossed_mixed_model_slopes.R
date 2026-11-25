# Crossed mixed model of the per-song word-level gain, sung against hummed versions, with random slopes.
# dr ~ condition + (condition | listener) + (condition | song) + (1 | listener:song), condition coded +0.5 (sung) and
# -0.5 (hummed), REML, Satterthwaite degrees of freedom (lmerTest). Singular fits fall back to (a) uncorrelated random
# effects, (b) dropping the smaller of the two random slopes, (c) no random slopes.
# Arguments: input CSV (columns listener, song, condition, dr), output CSV of the fixed effect, output CSV of the
# variance components.
suppressMessages({library(lmerTest)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1]); d$listener <- factor(d$listener); d$song <- factor(d$song)
fits <- list(
  full = dr ~ condition + (condition | listener) + (condition | song) + (1 | listener:song),
  a_uncorrelated = dr ~ condition + (condition || listener) + (condition || song) + (1 | listener:song))
res <- NULL
for (nm in names(fits)) {
  m <- lmer(fits[[nm]], data = d, REML = TRUE)
  if (!isSingular(m)) { res <- list(name = nm, m = m); break }
}
if (is.null(res)) {
  m <- lmer(dr ~ condition + (condition || listener) + (condition || song) + (1 | listener:song), data = d, REML = TRUE)
  vc <- as.data.frame(VarCorr(m)); sl <- vc[vc$var1 %in% "condition", ]
  smaller <- sl$grp[which.min(sl$vcov)]
  fb <- if (grepl("listener", smaller)) dr ~ condition + (1 | listener) + (condition || song) + (1 | listener:song) else dr ~ condition + (condition || listener) + (1 | song) + (1 | listener:song)
  m <- lmer(fb, data = d, REML = TRUE)
  if (!isSingular(m)) res <- list(name = paste0("b_drop_slope_", smaller), m = m)
  if (is.null(res)) { m <- lmer(dr ~ condition + (1 | listener) + (1 | song) + (1 | listener:song), data = d, REML = TRUE); res <- list(name = "c_no_slopes", m = m) }
}
co <- summary(res$m)$coefficients
vc <- as.data.frame(VarCorr(res$m))
out <- data.frame(model = res$name, estimate = co["condition", "Estimate"], se = co["condition", "Std. Error"], df = co["condition", "df"],
                  t = co["condition", "t value"], singular = isSingular(res$m))
write.csv(out, args[2], row.names = FALSE)
write.csv(vc[, c("grp", "var1", "var2", "vcov")], args[3], row.names = FALSE)
