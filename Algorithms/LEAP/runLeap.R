library(LEAP)

args <- commandArgs(trailingOnly = T)
inFile <- args[1]
maxLag <- as.numeric(args[2])
outFile <-  args[3]
maxRegulatorsPerTarget <- ifelse(length(args) >= 4, suppressWarnings(as.integer(args[4])), NA)
if (is.na(maxRegulatorsPerTarget) || maxRegulatorsPerTarget <= 0) {
  maxRegulatorsPerTarget <- Inf
}

# input expression data
inputExpr <- read.table(inFile, sep=",", header = 1, row.names = 1)
geneNames <- rownames(inputExpr)
rownames(inputExpr) <- c()
# Run LEAP's compute Max. Absolute Correlation
# MAC_cutoff is set to zero to get a score for all TFs
# max_lag_prop is set to the max. recommended value from the paper's supplementary file
# Link to paper: https://academic.oup.com/bioinformatics/article/33/5/764/2557687

MAC_results = MAC_counter(data = inputExpr, max_lag_prop=maxLag, MAC_cutoff = 0, 
                          file_name = "temp", lag_matrix = FALSE, symmetric = FALSE)

write.table(
  data.frame(Gene1 = character(), Gene2 = character(), Score = numeric()),
  outFile,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

targetIndices <- unique(MAC_results[, 'Column gene index'])
for (targetIndex in targetIndices) {
  targetRows <- which(MAC_results[, 'Column gene index'] == targetIndex)
  if (length(targetRows) == 0) {
    next
  }
  orderedRows <- targetRows[order(abs(MAC_results[targetRows, 'Correlation']), decreasing = TRUE)]
  selectedRows <- orderedRows
  if (!is.infinite(maxRegulatorsPerTarget)) {
    selectedRows <- head(orderedRows, maxRegulatorsPerTarget)
  }
  outDF <- data.frame(
    Gene1 = geneNames[MAC_results[selectedRows, 'Row gene index']],
    Gene2 = geneNames[targetIndex],
    Score = MAC_results[selectedRows, 'Correlation']
  )
  write.table(outDF, outFile, sep = "\t", quote = FALSE, row.names = FALSE,
              col.names = FALSE, append = TRUE)
}
