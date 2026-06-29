library(ppcor)
args <- commandArgs(trailingOnly = T)
inFile <- args[1]
outFile <-  args[2]
maxRegulatorsPerTarget <- ifelse(length(args) >= 3, suppressWarnings(as.integer(args[3])), NA)
pValueCutoff <- ifelse(length(args) >= 4, suppressWarnings(as.numeric(args[4])), 1.0)
if (is.na(maxRegulatorsPerTarget) || maxRegulatorsPerTarget <= 0) {
  maxRegulatorsPerTarget <- Inf
}
if (is.na(pValueCutoff)) {
  pValueCutoff <- 1.0
}

# input expression data
inputExpr <- read.table(inFile, sep=",", header = 1, row.names = 1)
geneNames <- rownames(inputExpr)
rownames(inputExpr) <- c(geneNames)

# Run pcor using spearman's correlation as mentioned in the PNI paper 
# Link to paper: https://www.pnas.org/content/114/23/5822

pcorResults=  pcor(x= t(as.matrix(inputExpr)), method = "spearman")

write.table(
  data.frame(Gene1 = character(), Gene2 = character(), corVal = numeric(), pValue = numeric()),
  outFile,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

for (targetIndex in seq_along(geneNames)) {
  scores <- pcorResults$estimate[, targetIndex]
  pvalues <- pcorResults$p.value[, targetIndex]
  scores[targetIndex] <- NA
  pvalues[targetIndex] <- NA

  valid <- which(is.finite(scores) & is.finite(pvalues))
  if (length(valid) == 0) {
    next
  }

  significant <- valid[pvalues[valid] <= pValueCutoff]
  nonsignificant <- setdiff(valid, significant)
  significant <- significant[order(abs(scores[significant]), decreasing = TRUE)]
  nonsignificant <- nonsignificant[order(abs(scores[nonsignificant]), decreasing = TRUE)]
  selected <- c(significant, nonsignificant)
  if (!is.infinite(maxRegulatorsPerTarget)) {
    selected <- head(selected, maxRegulatorsPerTarget)
  }

  if (length(selected) == 0) {
    next
  }

  outDF <- data.frame(
    Gene1 = geneNames[selected],
    Gene2 = geneNames[targetIndex],
    corVal = scores[selected],
    pValue = pvalues[selected]
  )
  write.table(outDF, outFile, sep = "\t", quote = FALSE, row.names = FALSE,
              col.names = FALSE, append = TRUE)
}
