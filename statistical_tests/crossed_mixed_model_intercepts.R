# Crossed mixed model of the per-song word-level gain without random slopes: dr ~ 0 + version + (1 | listener) +
# (1 | song), REML, with contrasts for each version and for sung - hummed (Satterthwaite, lmerTest); sensitivity model
# with (1 | listener:song) added.
# Arguments: input CSV (columns listener, version, song, dr), output CSV of the contrasts, output CSV of the sensitivity
# contrasts, output CSV of the variance components.
suppressMessages({library(lmerTest)})
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1]); d$version <- factor(d$version, levels = c("speech", "song", "melody"))
d$listener <- factor(d$listener); d$song <- factor(d$song)
m <- lmer(dr ~ 0 + version + (1 | listener) + (1 | song), data = d, REML = TRUE)
ct <- function(L) { r <- contest(m, L, joint = FALSE, ddf = "Satterthwaite"); c(r$Estimate, r$`Std. Error`, r$df, r$`t value`) }
out <- rbind(spoken = ct(c(1, 0, 0)), sung = ct(c(0, 1, 0)), hummed = ct(c(0, 0, 1)), sung_minus_hummed = ct(c(0, 1, -1)))
m2 <- lmer(dr ~ 0 + version + (1 | listener) + (1 | song) + (1 | listener:song), data = d, REML = TRUE)
m_keep <- m; m <- m2
out2 <- rbind(spoken = ct(c(1, 0, 0)), sung = ct(c(0, 1, 0)), hummed = ct(c(0, 0, 1)), sung_minus_hummed = ct(c(0, 1, -1)))
m <- m_keep
write.csv(data.frame(term = rownames(out2), estimate = out2[, 1], se = out2[, 2], df = out2[, 3], t = out2[, 4]), args[3], row.names = FALSE)
vc <- as.data.frame(VarCorr(m))
write.csv(data.frame(term = rownames(out), estimate = out[, 1], se = out[, 2], df = out[, 3], t = out[, 4]), args[2], row.names = FALSE)
write.csv(vc[, c("grp", "vcov")], args[4], row.names = FALSE)
