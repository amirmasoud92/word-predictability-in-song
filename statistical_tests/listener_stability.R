# Listener intraclass correlation of the single-segment word-level gain (Alice): dr ~ 1 + (1 | listener) + (1 | segment),
# with a 95 % interval from 1000 parametric bootstrap draws. Arguments: input CSV (listener, segment, dr), output CSV,
# and an optional integer seed of the bootstrap.
suppressMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2) set.seed(as.integer(args[3]))
icc <- function(m) { v <- as.data.frame(VarCorr(m)); lv <- v$vcov[v$grp == "listener"]; lv / sum(v$vcov) }
d <- read.csv(args[1]); d$segment <- factor(d$segment)
m <- lmer(dr ~ 1 + (1 | listener) + (1 | segment), data = d)
b <- bootMer(m, icc, nsim = 1000, type = "parametric")
r <- c(icc(m), quantile(b$t, c(.025, .975), na.rm = TRUE))
write.csv(data.frame(icc = r[1], lo = r[2], hi = r[3]), args[2], row.names = FALSE)
