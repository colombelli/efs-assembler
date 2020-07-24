select <- function(df) {

  df$class <- as.factor(df$class)
  cat("Reliefing... :)\n")

  sink("NUL")
  attScores <- attrEval(class ~ ., df, estimator="Relief") 
  sink()


  cat("Processing output...\n")
  rankDf <- as.data.frame(attScores)
  rankDf <- rankDf[order(-rankDf$attScores),,drop=FALSE]
  rankDf["rank"] <- c(1:length(rankDf$attScores))
  rankDf["attScores"] <- NULL

  return(rankDf)
}