select <- function(df) {

  df$class <- as.factor(df$class)
  dfx <- df[c(1:nrow(df)), c(1:length(df)-1)]
  dfy <- dfy <- df$class


  cat("Calculating SU importance...\n")
  rankDf <- information_gain(x=dfx, y=dfy, type = "symuncert")


  cat("Processing output...\n")
  rankDf <- rankDf[order(-rankDf$importance),,drop=FALSE]
  rownames(rankDf) <- rankDf$attributes
  rankDf$attributes <- NULL
  rankDf["rank"] <- c(1:length(rankDf$importance))
  rankDf$importance <- NULL

  return(rankDf)
}