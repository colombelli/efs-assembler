select <- function(df) {


  df$class <- as.factor(df$class)
  cat("Classifying with One Rule algorithm...\n")
  rankDf <- oneR(class ~ ., df) 


  cat("Processing output...\n")
  rankDf <- rankDf[order(-rankDf$attr_importance),,drop=FALSE]
  rankDf["rank"] <- c(1:length(rankDf$attr_importance))
  rankDf["attr_importance"] <- NULL

  return(rankDf)
}