# suppressPackageStartupMessages(library(ggstatsplot))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(readxl))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(ggpubr))
suppressPackageStartupMessages(library(broom))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(class))
suppressPackageStartupMessages(library(gmodels))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(e1071))
suppressPackageStartupMessages(library(ISLR))
suppressPackageStartupMessages(library(pROC))
suppressPackageStartupMessages(library(lattice))
suppressPackageStartupMessages(library(kknn))
suppressPackageStartupMessages(library(multiROC))
# suppressPackageStartupMessages(library(MLeval))
suppressPackageStartupMessages(library(AppliedPredictiveModeling))
suppressPackageStartupMessages(library(corrplot))
suppressPackageStartupMessages(library(Hmisc))
suppressPackageStartupMessages(library(rattle))
suppressPackageStartupMessages(library(Hmisc))
suppressPackageStartupMessages(library(broom)) # to add : AUGMENT
suppressPackageStartupMessages(library(rattle))
suppressPackageStartupMessages(library(quantmod)) 
suppressPackageStartupMessages(library(nnet))
suppressPackageStartupMessages(library(NeuralNetTools))
suppressPackageStartupMessages(library(neuralnet))
suppressPackageStartupMessages(library(klaR))
suppressPackageStartupMessages(library(kernlab))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(cluster))
suppressPackageStartupMessages(library(factoextra))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(fpc))
suppressPackageStartupMessages(library(gplots))
suppressPackageStartupMessages(library(pheatmap))
# suppressPackageStartupMessages(library(d3heatmap))
suppressPackageStartupMessages(library(clValid))
suppressPackageStartupMessages(library(clustertend))
suppressPackageStartupMessages(library(factoextra))
suppressPackageStartupMessages(library(ggfortify))
suppressPackageStartupMessages(library(splines))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(leaps))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(car))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(repr))
suppressPackageStartupMessages(library(splines))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(elasticnet))
suppressPackageStartupMessages(library(pcr))
suppressPackageStartupMessages(library(pls))
suppressPackageStartupMessages(library(earth))
suppressPackageStartupMessages(library("monomvn"))
suppressPackageStartupMessages(library("import"))
suppressPackageStartupMessages(library("mboost"))
suppressPackageStartupMessages(library("Cubist"))
suppressPackageStartupMessages(library("elasticnet"))
suppressPackageStartupMessages(library("fastICA"))
suppressPackageStartupMessages(library("pls"))
suppressPackageStartupMessages(library("caretEnsemble"))

######################################################
######################################################


FILE="Concrete_Data.csv"
######################################################
######################################################
# file = read.delim(FILE, sep = "\t", header=TRUE, stringsAsFactors=F)
file = read.csv(FILE)
######################################################
######################################################
# str(file)
class(file)
dim(file)
# summary(file) 
# all the FEATURES are NUMERICAL
######################################################
######################################################
# we choose shorter NAMES for the FEATURES

# colnames(file)
dim(file)

file2 = file[, c('Cement..component.1..kg.in.a.m.3.mixture.',
                 'Fly.Ash..component.3..kg.in.a.m.3.mixture.',
                 'Concrete.compressive.strength.MPa..megapascals..')]

file2 = as.data.frame(lapply(file2, as.numeric))
# head(file2, 2)  

file3 = na.omit(file2)
# names(file3) = c("LDN193189", "FINGOLIMOD", "AS")
names(file3) = c("L", "F", "AS")

dim(file3)
head(file3, 2)
df = file3

# names(df)[1]
# names(df)[2]
# names(df)[3]


head(file,2)
colnames(file)

# cor(file$wes.Aneuploidy.Score, file$wgs.Aneuploidy.Score)
# cor(file$wes.BPs.Lost, file$wgs.BPs.Lost)
# cor(file$wes.FGA, file$wgs.FGA)
# cor(file$wes.Aneu.EV, file$wgs.Aneu.EV)

# let's work with a generic drug name : L
DRUG = names(df)[1]
# DRUG = names(df)[2]
DRUG

# Beside GLM, LASSO, RIDGE, we add : 
# Polynomial regression. It adds polynomial terms or quadratic terms (square, cubes, etc) to a regression.
# Spline regression. It fits a smooth curve with a series of polynomial segments. 
# The values delimiting the spline segments are called Knots.
# Generalized additive models (GAM). Fits spline models with automated selection of knots.



# preparing for CARET : 
# setup cross validation and control parameters

metrics = "RMSE"
# metrics <- c("RMSE", "Rsquared", "error")
# metrics = "Rsquared" 
# metrics = "error"

# Create train control with repeated cross-validation
control <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3,
  # search = "grid",
  verbose = FALSE)





# visualization in ggplot2
options(repr.plot.width=5, repr.plot.height=5)

# displaying the 1st drug

ggplot(df, aes(x = L , y = AS)) +
  geom_point() + 
stat_smooth(aes(color = "lm"), method = "lm", se = FALSE) +
  stat_smooth(aes(color = "glm"), method = "glm", se = FALSE) +
  stat_smooth(aes(color = "gam"), method = "gam", se = FALSE) +
  stat_smooth(aes(color = "loess"), method = "loess", se = FALSE) +
  stat_smooth(aes(color = "rlm"), method = "rlm", se = FALSE) +
  stat_smooth(aes(color = "auto"), method = "auto", se = FALSE) +
  scale_color_manual(
    values = c(
      "lm" = "blue",
      "glm" = "red",
      "gam" = "green",
      "loess" = "purple",
      "rlm" = "orange",
      "auto" = "gray"
    ),
    name = "Method"
  ) + 
ggtitle(paste("drug :", names(df)[1], " metrics :", names(df)[3], sep=""))

# displaying the 2nd drug

ggplot(df, aes(x = F , y = AS)) +
  geom_point() + 
stat_smooth(aes(color = "lm"), method = "lm", se = FALSE) +
  stat_smooth(aes(color = "glm"), method = "glm", se = FALSE) +
  stat_smooth(aes(color = "gam"), method = "gam", se = FALSE) +
  stat_smooth(aes(color = "loess"), method = "loess", se = FALSE) +
  stat_smooth(aes(color = "rlm"), method = "rlm", se = FALSE) +
  stat_smooth(aes(color = "auto"), method = "auto", se = FALSE) +
  scale_color_manual(
    values = c(
      "lm" = "blue",
      "glm" = "red",
      "gam" = "green",
      "loess" = "purple",
      "rlm" = "orange",
      "auto" = "gray"
    ),
    name = "Method"
  ) +
ggtitle(paste("drug :", names(df)[2], " metrics :", names(df)[3], sep=""))



# working with the drug : L



trainIndex <- createDataPartition(df$AS, p = 0.8, list=FALSE, times=1)
subTrain <- df[trainIndex,]
subTest <- df[-trainIndex,]

dim(subTrain)
dim(subTest)



# LINEAR MODEL



reg_model <- lm(AS ~ L, data = subTrain)

# Linear Regression Model Summary
reg_model
summary(reg_model)
summary(reg_model)$coefficient

# Making Diagnostic Plots:
reg_model.diagnostics <- augment(reg_model)
# head(reg_model.diagnostics)
# tail(reg_model.diagnostics)

# Displaying:
ggplot(reg_model.diagnostics, aes(L, AS)) +
geom_point() +
stat_smooth(method = lm, se = FALSE) 

# Diagnostic Plots
# par(mfrow = c(2, 2))
plot(reg_model)

# Diagnostic Plots
# autoplot(reg_model)

# Cook's Distance is used to evaluate the Influential Points that will alter the Regression Analysis 
# or the coefficents values.
# By default Cook's Distance more than 4/( n - p - 1) defines an influential value. 

# Cook's distance
plot(reg_model, 4)
# Cook's distance vs Leverage
plot(reg_model, 6)

# Residuals :

residuals <- resid(reg_model)
predictedValues <- predict(reg_model)







# LOG TRANSFORMATION
# model <- lm(AS ~ log(L), data = subTrain)
# ggplot(df, aes(x = F , y = AS)) +
#  geom_point() + 
#  stat_smooth(aes(color = "log"), formula = y ~ log(x) )







## REGRESSION ANALYSIS : using GAM



gmmodel <- gam(AS ~ s(L), data = subTrain) 
gmmodel
summary(gmmodel)

gampredictions <- gmmodel %>% predict(subTest) 
 
# Model performance 
rmse = RMSE(gampredictions, subTest$AS)

cat("rmse :")
rmse

# R^2
cat("rsquared :")
rsquared <- caret::R2(gampredictions, subTest$AS)
rsquared 

# MAE
mae <- MAE(gampredictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse/mean(subTest$AS)
cat("error rate :")
error.rate



## REGRESSION ANALYSIS : using SPLINE REGRESSION



knots <- quantile(subTrain$AS, p = c( 0.25, 0.5, 0.75))
knots

splinemodel <- lm(AS ~ bs(L, knots = knots), data = subTrain)

splinemodel
summary(splinemodel)

predictions <- splinemodel %>% predict(subTest)

# Model performance 
cat("rmse :")
rmse = RMSE(predictions, subTest$AS)
rmse

#R^2
cat("rsquared :")
rsquared <- caret::R2(predictions, subTest$AS)
rsquared 

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse/mean(subTest$AS)
cat("error rate :")
error.rate



## REGRESSION ANALYSIS : using POLYNOMIAL REGRESSION



poly_reg <- lm(AS ~ poly(L, 2), data = subTrain)
poly_reg
summary(poly_reg)

predictions <- poly_reg %>% predict(subTest)

# RMSE
cat("rmse :")
rmse <- RMSE(predictions, subTest$AS)
rmse

# R^2
cat("rsquared :")
rsquared <- caret::R2(predictions, subTest$AS)
rsquared 

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse/mean(subTest$L)
cat("error rate :")
error.rate



## REGRESSION ANALYSIS : using MARS (Multivariate Adaptive Regression Splines)



earth_reg <- earth(AS ~ L, data = subTrain)
earth_reg
summary(earth_reg)

predictions <- poly_reg %>% predict(subTest)

# RMSE
cat("rmse :")
rmse <- RMSE(predictions, subTest$AS)
rmse

# R^2
cat("rsquared :")
rsquared <- caret::R2(predictions, subTest$AS)
rsquared 

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse/mean(subTest$L)
cat("error rate :")
error.rate

set.seed(123)
# TUNING the MARS MODEL : create a tuning grid

hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
  )
head(hyper_grid)

# cross validated model
tuned_mars <- train(
  AS ~ L, 
  data = df, 
  method = "earth",
  trControl = control, 
  tuneLength = 10,
  metric = metrics,
  na.action  = na.omit, 
  tuneGrid = hyper_grid
)

# best model
tuned_mars$bestTune

# plot results
# ggplot(tuned_mars, aes(x = nprune, y = RMSE)) + geom_line()

options(repr.plot.width = 4, repr.plot.height=4)
ggplot(tuned_mars)

# Variable Importance
# ggplot(varImp(tuned_mars))

# plot(tuned_mars)

# Print summary
summary(tuned_mars)

# Extract predictions on a new dataset (subTest)
predictions <- predict(tuned_mars, newdata = subTest)
# Extract residuals
residuals <- residuals(tuned_mars)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')







# REGRESSION ANALYSIS : using LASSO REGRESSION **
# REGRESSION ANALYSIS : using RIDGE REGRESSION **

# REGRESSION ANALYSIS : MODEL SELECTION based on AIC
# REGRESSION ANALYSIS : MODEL SELECTION based on BIC



# REGRESSION ANALYSIS : in CARET



# the TRAINING and the TESTING datasets **

# setup cross validation and control parameters

metrics = "RMSE"
# metrics <- c("RMSE", "Rsquared", "error")
# metrics = "Rsquared" 
# metrics = "error"

# Create train control with repeated cross-validation
control <- trainControl(
  method = "repeatedcv",
  number = 3,
  repeats = 3,
  # search = "grid",
  verbose = FALSE
)



# Train the linear regression model : LM



fit.LM <- caret::train(
  AS ~ L, 
  data = df, 
  method = "lm", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics,
  na.action  = na.omit, 
  VERBOSE = FALSE
)

# Print summary
summary(fit.LM)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.LM, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.LM)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



#  Train a RANDOM FOREST
#  method = "ranger" 
#  Train a XGBOOST
#  method = "xgbTree"
#  method = "pls", 
#  method = "ridge"
#  method = "lasso"
#  method = "pcr" 
# Train a RANDOM FOREST
#  method = "ranger" 
# Train a XGBOOST
#  method = "xgbTree" 
# Train a NEURALNET
#  method = "neuralnet" 
# method = "ridge"
# method = "lasso"

# other mtehods that we can use :
#! treebag # Bagged CART
#! earth # Multivariate Adaptive Regression Splines 
# rpart
#! cubist
#! bridge : bayesian ridge regression
#! blasso : bayesian lasso regression
# krlsPoly  Polynomial Kernel Regularized Least Squares
# ? ridge : not functional
# ? lasso : not functional
# blackboost : boosted trees
# bayesglm
# glm.nb
# glmboost
# gamboost
# bam : Generalized Additive Model using Splines
# lars : Least Angle Regression
# nnls : non-negative least squares regression
# gbm



# GAM



fit.GAM <- caret::train(
  AS ~ L, 
  data = df, 
  method = "gam", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics,
  na.action = na.omit, 
  VERBOSE = FALSE
)

# Print summary
summary(fit.GAM)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.GAM, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.GAM)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



# KNN

fit.KNN <- caret::train(
  AS ~ L, 
  data = df, 
  method = "knn", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics,
  na.action = na.omit, 
  VERBOSE = FALSE
)

# Print summary
summary(fit.KNN)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.KNN, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.KNN)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



# Decision Tree

fit.DT <- caret::train(
  AS ~ L,  
  data = df, 
  method = "rpart", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics, 
  na.action = na.omit
)

# Print summary
summary(fit.DT)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.DT, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.DT)

# RMSE
rmse <- caret::RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')

# to visualize the DECISION TREE
# library(rpart.plot)
# rpart.plot(fit.DT$finalModel)





fit.TB <- caret::train(
  AS ~ L,  
  data = df, 
  method = "treebag", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics, 
  na.action = na.omit, 
  VERBOSE = FALSE
)

# Print summary
summary(fit.TB)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.TB, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.TB)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



fit.MARS <- caret::train(
  AS ~ L,  
  data = df, 
  method = "earth", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics, 
  na.action = na.omit
  # VERBOSE = FALSE
)

# Print summary
summary(fit.MARS)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.MARS, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.MARS)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



fit.CUBIST <- caret::train(
  AS ~ L,  
  data = df, 
  method = "cubist", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics, 
  na.action = na.omit, 
  VERBOSE = FALSE
)

# Print summary
summary(fit.CUBIST)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.CUBIST, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.CUBIST)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



# RIDGE REGRESSION



fit.RIDGE <- caret::train(
  AS ~ L,  
  data = df, 
  method = "bridge", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics, 
  na.action = na.omit
)

# Print summary
summary(fit.RIDGE)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.RIDGE, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.RIDGE)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



fit.LASSO <- caret::train(
  AS ~ L,  
  data = df, 
  method = "blasso", 
  trControl = control, 
  tuneLength = 10,
  metric = metrics, 
  na.action = na.omit
)

# Print summary
summary(fit.LASSO)

# Extract predictions on a new dataset (subTest)
predictions <- predict(fit.LASSO, newdata = subTest)
# Extract residuals
residuals <- residuals(fit.LASSO)

# RMSE
rmse <- RMSE(predictions, subTest$AS)
cat('RMSE:', rmse, '\n')

# R-squared
rsquared <- caret::R2(predictions, subTest$AS)
cat('R-squared:', rsquared, '\n')

# MAE
mae <- MAE(predictions, subTest$AS)
cat('MAE:', mae, '\n')

# Error Rate
error.rate = rmse / mean(subTest$AS)
cat('Error Rate:', error.rate, '\n')



# Building ENSEMBLE MODELS

algorithmList <- c(
'glm',
'gam',
'knn',
'rpart',
'treebag',
'earth',
'cubist',
'bridge',
'blasso'
)

suppressMessages(
    
models <- caretEnsemble::caretList( AS ~ L, 
                     data = df, 
                     trControl = control, 
                     methodList = algorithmList, 
                     metric="RMSE",
                     # metric="ROC",
                     # preProcess = PREPROCESS, 
                     tuneLength = 20)
)

results <- resamples(models)
print("summary of all these models :")
summary(results)

# xyplot(resamples(models))
# models



# Check the correlation between the models (ideally the models should have low correlations) :
options(repr.plot.width = 12, repr.plot.height=12)

# models
modelCor(results)
# splom(results)
# results

options(repr.plot.width = 6, repr.plot.height = 6)
pheatmap(modelCor(results), 
        cluster_rows = FALSE, 
        cluster_cols = FALSE)



# ENSEMBLE model
set.seed(123)

# Create an ensemble model using the caretEnsemble package
ensemble_model <- caretEnsemble(models)
summary(ensemble_model)

# Make predictions on the test set
ensemble_predictions <- predict(ensemble_model, newdata = subTest)
# ensemble_predictions

# Extract residuals
# ensemble_residuals <- residuals(ensemble_model)
# ensemble_residuals

# RMSE
ensemble_rmse <- RMSE(ensemble_predictions, subTest$AS)
cat('RMSE:', ensemble_rmse, '\n')
# R-squared
ensemble_rsquared <- caret::R2(ensemble_predictions, subTest$AS)
cat('R-squared:', ensemble_rsquared, '\n')
# MAE
ensemble_mae <- MAE(ensemble_predictions, subTest$AS)
cat('MAE:', ensemble_mae, '\n')
# Error Rate
ensemble.error.rate = ensemble_rmse / mean(subTest$AS)
cat('Error Rate:', ensemble.error.rate, '\n')

# display :
options(repr.plot.width = 6, repr.plot.height = 6)
plot(ensemble_model)



options(digits = 3)
model_results_RMSE <- data.frame(
 GLM = min(models$glm$results$RMSE),
 GAM = min(models$gam$results$RMSE),
 KNN = min(models$knn$results$RMSE),
 RPART = min(models$rpart$results$RMSE),
 TREEBAG = min(models$treebag$results$RMSE), 
 MARS = min(models$earth$results$RMSE),
 CUBIST = min(models$cubist$results$RMSE), 
 BRIDGE = min(models$bridge$results$RMSE),
 BLASSO = min(models$blasso$results$RMSE), 
 ENSEMBLE = ensemble_rmse
 )
print(model_results_RMSE)

options(digits = 3)
model_results_Rsquared <- data.frame(
 GLM = min(models$glm$results$Rsquared),
 GAM = min(models$gam$results$Rsquared),
 KNN = min(models$knn$results$Rsquared),
 RPART = min(models$rpart$results$Rsquared),
 TREEBAG = min(models$treebag$results$Rsquared), 
 MARS = min(models$earth$results$Rsquared),
 CUBIST = min(models$cubist$results$Rsquared), 
 BRIDGE = min(models$bridge$results$Rsquared),
 BLASSO = min(models$blasso$results$Rsquared), 
 ENSEMBLE = ensemble_rsquared
 )
print(model_results_Rsquared)

options(digits = 3)
model_results_MAE <- data.frame(
 GLM = min(models$glm$results$MAE),
 GAM = min(models$gam$results$MAE),
 KNN = min(models$knn$results$MAE),
 RPART = min(models$rpart$results$MAE),
 TREEBAG = min(models$treebag$results$MAE), 
 MARS = min(models$earth$results$MAE),
 CUBIST = min(models$cubist$results$MAE), 
 BRIDGE = min(models$bridge$results$MAE),
 BLASSO = min(models$blasso$results$MAE), 
 ENSEMBLE = ensemble_mae 
 )
print(model_results_MAE)

# in order to print, we place all the metrics in a data frame

METRICS = rbind(model_results_RMSE,
                model_results_Rsquared,
                model_results_MAE)

# Assign row names to the data frame
rownames(METRICS) <- c("RMSE", "Rsquared", "MAE")
METRICS

write.table(METRICS, 
            file = paste("drug", DRUG, "RMSE.R2.MAE.txt", sep="."), 
            sep="\t", quote=FALSE, row.names = FALSE, col.names = TRUE)



options(repr.plot.width = 6, repr.plot.height=6)

barplot(as.matrix(model_results_RMSE), 
        main = paste("drug : ", DRUG, " : RMSE", sep="."),
        xlab = "models",
        ylab = "RMSE",
        col = "red",  # You can change the color as desired
        cex.names = 0.7  # Adjust the size of the names if needed
)

png(paste("drug", DRUG, "RMSE.png", sep="."), width = 800, height = 600)
barplot(as.matrix(model_results_RMSE), 
        main = paste("drug : ", DRUG, " : RMSE", sep="."),
        xlab = "models",
        ylab = "RMSE",
        col = "red",  # You can change the color as desired
        cex.names = 0.7  # Adjust the size of the names if needed
)
dev.off()

options(repr.plot.width = 6, repr.plot.height=6)

barplot(as.matrix(model_results_Rsquared), 
        main = paste("drug :", DRUG, " : Rsquared", sep=""),
        xlab = "Models",
        ylab = "Rsquared",
        col = "darkgreen",  # You can change the color as desired
        cex.names = 0.7  # Adjust the size of the names if needed
)

png(paste("drug", DRUG, "Rsquared.png", sep="."), width = 800, height = 600)
barplot(as.matrix(model_results_Rsquared), 
        main = paste("drug : ", DRUG, " : Rsquared", sep=""),
        xlab = "Models",
        ylab = "Rsquared",
        col = "darkgreen",  # You can change the color as desired
        cex.names = 0.7  # Adjust the size of the names if needed
)
dev.off()

options(repr.plot.width = 6, repr.plot.height = 6)

barplot(as.matrix(model_results_MAE), 
        main = paste("drug : ", DRUG, ": MAE", sep="."),
        xlab = "models",
        ylab = "MAE",
        col = "black",  # You can change the color as desired
        cex.names = 0.7  # Adjust the size of the names if needed
)

png(paste("drug", DRUG, "MAE.png", sep="."), width = 800, height = 600)
barplot(as.matrix(model_results_MAE), 
        main = paste("drug :", DRUG, ": MAE", sep=""),
        xlab = "models",
        ylab = "MAE",
        col = "black",  # You can change the color as desired
        cex.names = 0.7  # Adjust the size of the names if needed
)
dev.off()



# Evaluating the STACK MODELS
# too slow

# stackControl = trainControl(method='repeatedcv', 
#                            number=10, 
#                            repeats=10,
#                            savePredictions = TRUE,
#                            # classProbs = TRUE,
#                            # preProcess = PREPROCESS,
#                            verbose=TRUE)

# rf_Stack = caretStack(models, method='rf', metric='RMSE', trControl = stackControl)
# print(rf_Stack)
# rf_Stack_predict <- predict(rf_Stack, newdata = testing)

# gbm_Stack = caretStack(models, method='gbm', metric='RMSE', trControl = stackControl)
# print(gbm_Stack)
# gbm_Stack_predict <- predict(gbm_Stack, newdata = testing)





# VISUALIZATIONS :

# Polynomial regression

options(repr.plot.width = 6, repr.plot.height=6)

ggplot(subTrain, aes(L, AS) ) +
  geom_point() +
  stat_smooth(aes(color = "poly1"), method = "lm", formula = y ~ poly(x, 1, raw = TRUE ), se=FALSE) +
  stat_smooth(aes(color = "poly2"), method = "lm", formula = y ~ poly(x, 2, raw = TRUE ), se=FALSE) +
  stat_smooth(aes(color = "poly3"), method = "lm", formula = y ~ poly(x, 3, raw = TRUE ), se=FALSE) +
  stat_smooth(aes(color = "poly4"), method = "lm", formula = y ~ poly(x, 4, raw = TRUE ), se=FALSE) +
  stat_smooth(aes(color = "poly5"), method = "lm", formula = y ~ poly(x, 5, raw = TRUE ), se=FALSE) +
  stat_smooth(aes(color = "auto"), method = "auto", se = FALSE) +
  scale_color_manual(
    values = c(
      "poly1" = "blue",
      "poly2" = "red",
      "poly3" = "green",
      "poly4" = "purple",
      "poly5" = "orange",
      "auto" = "gray"
    ),
    name = "Method"
  ) +
ggtitle(paste("polynomical regression : drug :", names(df)[2], " metrics :", names(df)[3], sep=""))

# Spline regression

options(repr.plot.width = 6, repr.plot.height=6)

ggplot(subTrain, aes(L, AS) ) +
  geom_point() +
  stat_smooth(aes(color = "spline1"), method = "lm", formula = y ~ splines::bs(x, df = 1), se=FALSE) +
  stat_smooth(aes(color = "spline2"), method = "lm", formula = y ~ splines::bs(x, df = 2), se=FALSE) +
  stat_smooth(aes(color = "spline3"), method = "lm", formula = y ~ splines::bs(x, df = 3), se=FALSE) +
  stat_smooth(aes(color = "spline4"), method = "lm", formula = y ~ splines::bs(x, df = 4), se=FALSE) +
  stat_smooth(aes(color = "spline5"), method = "lm", formula = y ~ splines::bs(x, df = 5), se=FALSE) +
  stat_smooth(aes(color = "gam"), method = "gam", formula = y ~ s(x), se = FALSE) +
  scale_color_manual(
    values = c(
      "spline1" = "blue",
      "spline2" = "red",
      "spline3" = "green",
      "spline4" = "purple",
      "spline5" = "orange",
      "gam" = "gray"
    ),
    name = "Method"
  ) +
ggtitle(paste("spline regression : drug :", names(df)[2], " metrics :", names(df)[3], sep=""))



# GAM

options(repr.plot.width = 6, repr.plot.height=6)

ggplot(subTrain, aes(L, AS) ) +
  geom_point() +
  stat_smooth(aes(color = "spline1"), method = "lm", formula = y ~ splines::bs(x, df = 1), se=FALSE) +
  stat_smooth(aes(color = "spline2"), method = "lm", formula = y ~ splines::bs(x, df = 2), se=FALSE) +
  stat_smooth(aes(color = "spline3"), method = "lm", formula = y ~ splines::bs(x, df = 3), se=FALSE) +
  stat_smooth(aes(color = "spline4"), method = "lm", formula = y ~ splines::bs(x, df = 4), se=FALSE) +
  stat_smooth(aes(color = "spline5"), method = "lm", formula = y ~ splines::bs(x, df = 5), se=FALSE) +
  stat_smooth(aes(color = "gam"), method = "gam", formula = y ~ s(x), se = FALSE) +
  scale_color_manual(
    values = c(
      "spline1" = "blue",
      "spline2" = "red",
      "spline3" = "green",
      "spline4" = "purple",
      "spline5" = "orange",
      "gam" = "gray"
    ),
    name = "Method"
  ) +
ggtitle(paste("spline regression : drug :", names(df)[2], " metrics :", names(df)[3], sep=""))





# Another way to display the SUMMARY : 

summary(resamples(list(
    GLM = fit.LM,
 GAM = fit.GAM ,
 KNN = fit.KNN ,
 RPART = fit.DT ,
 TREEBAG = fit.TB, 
 MARS = fit.MARS,
 CUBIST = fit.CUBIST , 
 BRIDGE = fit.RIDGE ,
 BLASSO = fit.LASSO
 )))






