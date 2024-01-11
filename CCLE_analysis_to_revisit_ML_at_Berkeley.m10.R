suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(reshape2))
suppressPackageStartupMessages(library(readxl))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(NeuralNetTools))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(class))
suppressPackageStartupMessages(library(gmodels))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(e1071))
suppressPackageStartupMessages(library(klaR))
suppressPackageStartupMessages(library(kernlab))
suppressPackageStartupMessages(library(rattle))
suppressPackageStartupMessages(library(doParallel))   
suppressPackageStartupMessages(library(NeuralNetTools))
suppressPackageStartupMessages(library(neuralnet))
suppressPackageStartupMessages(library(GGally))
suppressPackageStartupMessages(library(klaR))
suppressPackageStartupMessages(library(vip))
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(pROC))
suppressPackageStartupMessages(library("gbm"))
suppressPackageStartupMessages(library("ada"))
suppressPackageStartupMessages(library("xgboost"))
suppressPackageStartupMessages(library("randomForest"))
suppressPackageStartupMessages(library("caretEnsemble"))
suppressPackageStartupMessages(library("pheatmap"))
suppressPackageStartupMessages(library("caTools"))
suppressPackageStartupMessages(library("cvms"))
suppressPackageStartupMessages(library("Boruta"))
suppressPackageStartupMessages(library("Amelia"))
suppressPackageStartupMessages(library("mice"))
suppressPackageStartupMessages(library("VIM"))
suppressPackageStartupMessages(library("missForest"))

# It is part of a class that I have taken at UC Berkeley
# for variable importance, to evaluate VIP
# https://koalaverse.github.io/vip/articles/vip.html
# https://topepo.github.io/caret/
# the dataset is decribed on : 
# https://archive.ics.uci.edu/dataset/320/student+performance

library(doParallel) 
rCluster <- makePSOCKcluster(6) 
registerDoParallel(rCluster)  
set.seed(123)

FILE1="student-mat.tsv"
student <- read.delim(FILE1, sep="\t", header=T, stringsAsFactors=F)
head(student, 2)

summary(student)
str(student)
class(student)

ggplot(data = student) + 
       geom_bar(mapping = aes(x=G3, fill=G3))

student1 <- subset(student, select = -c(G1, G2))

student2 <- subset(student1, 
                   select = -c(school, sex, address, famsize, Pstatus, 
                   Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, 
                   higher, internet, romantic))

str(student2)
student2$G3 = as.factor(student2$G3)

student3 = subset(student2, 
                  select= c(age, traveltime, studytime, failures, absences, G3))

student3$G3 = as.integer(student3$G3)
student4 = student3[student3$G3 > 2, ]
dim(student4) 

ggplot(data = student4) + 
       geom_bar(mapping = aes(x=G3, fill=G3))

## TRANSFORMING G3 into RANGES of PASS and NO-PASS :

student3$G3 = as.integer(student3$G3)

student3$RESULT[student3$G3 <= 10] = "NO_PASS"
student3$RESULT[student3$G3 >=10 ] = "PASS"

student3 <- subset(student3, select = -c(G3))
student3$RESULT = as.factor(student3$RESULT)

# TO IDENTIFY the CORRELATED PREDICTORS or LINEAR DEPENDENCIES

head(student3, 2)
student3_predictors = student3[,1:5]
head(student3_predictors, 2)

# NOTES from CARET package :

# https://topepo.github.io/caret/pre-processing.html#creating-dummy-variables

# CORRELATIONS

# The code chunk below shows the effect of removing descriptors with absolute correlations above 0.75.
# upper.tri : Returns a matrix of logicals the same size of a given matrix with entries TRUE 
# in the lower or upper triangle.

# descrCor <- cor(student3[,1:5])
# summary(descrCor[upper.tri(descrCor)])

# LINEAR DEPENDENCIES
# findLinearCombos(student3[,1:5])

# preProcess FUNCTION : CENTER, SCALE, IMPUTATION (imputation is based on KNN)

# pp = preProcess(student3[,1:5], method = c("center", "scale"), na.remove=FALSE)
# pp

# SIMPLE SPLITTING

# trainIndex <- createDataPartition(student3[,1:5], 
#                                  p = .8, 
#                                  list = FALSE, 
#                                  times = 1)

# trainIndex

# A series of test/training partitions are created using
# ‘createDataPartition’ while ‘createResample’ creates one or more
# bootstrap samples. ‘createFolds’ splits the data into ‘k’ groups

# student3Train = student3[ trainIndex,]
# student3Test = student3[-trainIndex ]

# head(student3Train,2)
# head(student3Test, 2)

# trainControl can be used to specifiy the type of resampling:
# fitControl =  trainControl(## 10-fold CV
#                           method = "repeatedcv",
#                           number = 10,
#                           ## repeated ten times
#                           repeats = 10)

# CONFUSION MATRIX :
# confusionMatrix(data = test_set$pred, reference = test_set$obs, mode = "prec_recall" )

# Feature Selection Methods
# recursive feature elimination, 
# genetic algorithms, 
# and simulated annealing.
# Feature Selection using Univariate Filters

# Recursive Feature Elimination

# rfeControl
# rfeIter
# summary
# fit function
# pred
# rank 
# selectSize
# selectVar

# Genetic Alghoritms
# gaf
# ctrl <- gafsControl(functions = caretGA)
# obj <- gafs(x = predictors, 
#            y = outcome,
#            iters = 100,
#            gafsControl = ctrl,
#            ## Now pass options to `train`
#            method =
# ROC function

# Simulated Annealing
# saf
# ctrl <- safsControl(functions = caretSA)
# obj <- safs(x = predictors, 
#            y = outcome,
#            iters = 100,
#            safsControl = ctrl,
#            ## Now pass options to `train`
#            method =
# ROC function

head(student3,2)

# Visualize the data 
# inspiration from : 
# https://setscholars.net/end-to-end-machine-learning-ionosphere-prediction-in-r/
# split input and output

x = student3[, 1:5]
y = student3[, 6]

# scatterplot matrix
pairs(RESULT~. , data = student3, col=student3$RESULT)

# box and whisker plots for each attribute
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="box", scales=scales)

# density plots for each attribute by class value
featurePlot(x=x, y=y, plot="density", scales=scales)



# to check the MISSINGNESS 

# HOW MANY MISSING VALUES ARE :
# head(student3, 3)

sum(is.na(student3))

# https://bookdown.org/mike/data_analysis/imputation-missing-data.html
# suggestions regarding IMPUTATIONS
# remove categorical variables

student3_predictors = dplyr::select(student3, -RESULT)
head(student3_predictors, 2)





# imputation with MICE

# IMPUTATION with MICE : shall we use the data in df student3_predictors ?

############################################
############################################ PATTERNS in the MISSING DATA
# Missing pattern :
md.pattern(student3)
md.pairs(student3)
############################################
############################################ VIM
# Using mice for looking at missing data pattern
# md.pattern(training)

aggr_plot <- aggr(student3, col=c('navyblue','red'), 
                   numbers=TRUE, 
                   sortVars=TRUE, 
                   labels=names(student3), 
                   cex.axis=.7, 
                   gap=3, 
                   ylab=c("Histogram of missing data","Pattern"))

# Margin plots : 
# marginplot(student3[c(1,2)])
# marginplot(student3[c(2,3)])

############################################
############################################

mice_student3 <- mice(student3, method = "rf")
# str(mice_student3, 2)

mice_student3_complete <- complete(mice_student3)
# str(mice_student3_complete)

############################################
############################################

dim(student3)
dim(mice_student3_complete)

# summary(student3)
# summary(mice_student3_complete)

############################################
############################################ multiple options
# methods(mice)
# mice_student3 <- mice(x_train, 
#                      method = "rf", 
#                      m=5, 
#                      maxit=50, 
#                      seed=500)
############################################
############################################
# Inspecting the density of the data after IMPUTATION : 
# mice_student3$predictorMatrix
# mice_student3$visitSequence

# Select numeric columns from the data frame
numeric_columns <- mice_student3_complete[sapply(mice_student3_complete, is.numeric)]
# numeric_columns

# Create density or strip plots for each numeric column
density_plots <- lapply(numeric_columns, densityplot)
strip_plots <- lapply(numeric_columns, stripplot)

############################################
############################################
# Display the density plots
print(density_plots)
# Display the strip plots
print(strip_plots)
############################################
############################################ considering a variable
# names(mice_student3_complete)
# stripplot(mice_student3_complete[, "age"])
# densityplot(mice_student3_complete[, "age"])



# IMPUTATION with AMELIA

# IMPUTATION with AMELIA
# we do NOT use categorical variables :)

amelia_student3_predictors <- amelia(student3_predictors)
str(amelia_student3_predictors)

# multiple options 
# amelia_student3 <- amelia(x_train, 
#                          m = 3, 
#                          parallel = "multicore" , 
#                          noms = c('age','traveltime','studytime','failures','absences','RESULT'))

# To access the imputed data frames 
amelia_student3_predictors
# amelia_student3_predictors$imputations

# another strategy
# student3.mis <- prodNA(student3, noNA = 0.1)



set.seed(123)

# SPLIT DATASET
indxTrain <- createDataPartition(student3$RESULT, 
                                 p = .75, 
                                 list = FALSE)
# indxTrain

training <- student3[indxTrain,]
# training
testing <- student3[-indxTrain,]
# testing

## PRE-PROCESSING : we describe possible options 

# trainX  <- training[, names(training) != "RESULT"]
# preProcValues <- preProcess(x = trainX, method = c("center", "scale"))
# preProcValues

# in order to see the TRANSFORMED data :

preProcValues <- preProcess(training, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, training)
glimpse(trainTransformed)





# we separate :

# independent variables for TRAIN
# dependent variable for TRAIN
# independent variables for TEST
# dependent variables for TEST

X_train = training[, c(1,2,3,4,5)]  # independent variables for train
X_test = testing[, c(1,2,3,4,5)]    # dependent variables for train
head(X_train, 2)
head(X_test, 2)
x_train = X_train
x_test = X_test
head(x_train, 2)
head(x_test, 2)

# using BINARY CODING of 0 and 1
# Y_train <- as.integer(training$RESULT) - 1
# Y_test <- as.integer(testing$RESULT) - 1

# if we do not need to perform the BINARY ENCODING
y_train <- training$RESULT  # independent variables for test
y_test <- testing$RESULT    # dependent variables for test
head(y_train, 2)
head(y_test, 2)

# we encode PASS as 1, and NOT_PASS as 0

Y_train <- as.integer(training$RESULT == "PASS") # independent variables for test
Y_test <- as.integer(testing$RESULT == "PASS")    # dependent variables for test
head(Y_train, 2)
head(Y_test, 2)



# some documentation on : 
# http://rismyhammer.com/ml/Pre-Processing.html#:~:text=0%2C%201%5D.-,Imputation,(e.g.%20using%20the%20mean)

# notes about DATA PRE-PROCESSING :
# createFolds splits the data into k groups
# groupKFold splits the data based on a grouping factor

# Splitting Based on Predictors
# maxDissim is used to create sub–samples using a maximum dissimilarity approach. 
# This is particularly useful for unsupervised learning where there are no response variables.

# Splitting with Important Groups
# To split the data base by groups, groupKFold can be used

# DummyVars

# Zero and Near-Zero-Variance Predictors
# In some situations, the data generating mechanism can create predictors that only have a single unique value 
# (i.e. a “zero-variance predictor”). 
# For many models (excluding tree-based models), this may cause the model to crash or the fit to be unstable.
# The concern here that these predictors may become zero-variance predictors when the data are split into cross-validation/bootstrap sub-samples or that a few samples may have an undue influence on the model. 
# These “near-zero-variance” predictors may need to be identified and eliminated prior to modeling.
# nearZeroVar(student3)





# How to handle NA values :
# na.action = na.pass
# na.action = na.exclude 
# na.action = na.omit

# na.exclude over na.omit is that the former will retain the original number of rows in the data. 
# This may be useful where you need to retain the original size of the dataset - 
# for example it is useful when you want to compare predicted values to original values. 
# With na.omit you will end up with fewer rows so you won't as easily be able to compare.

# IMPUTATIONS : 

# The K closest neighbors are found in the training set and the value for the predictor is imputed using 
# these values (e.g. using the mean).
# Using this approach will automatically trigger preProcess to center and scale the data.

# bagImpute 
# medianImpute

# For each predictor in the data, a bagged tree is created using all of the other predictors in the training set.
# When a new sample has a missing predictor value, the bagged model is used to predict the value.

# TRANSFORMATIONS :

# In some cases there is a need to use principal component analysis (PCA) to transform the data to a 
# smaller sub–space where the new variables are uncorrelated with one another. 
# The preProcess class can apply this transformation by including pca in the method argument. 
# Doing this will also force scaling of the predictors.

# Remember, if you have categorical variables you must convert them first to dummy variables 
# before you can apply your processing (center, scale, pca, etc.).

# Similarly, independent component analysis (ICA) can also be used to find new variables that are linear 
# combinations of the original set such that the components are independent (as opposed to uncorrelated in PCA)

# spatialSign TRANSFORMATION
# spatialSign(student3_predictors)

# transformed <- spatialSign(training)
# transformed <- as.data.frame(transformed)
# head(transformed, 2)

# BOXCOX TRANSFORMATION
# preProcess(training, method = "BoxCox")

# Class Distance Calculations
# To generate new predictors variables based on distances to class centroids 
# (similar to how linear discriminant analysis works). 
# For each level of a factor variable, the class centroid and covariance matrix is calculated.

# Inadequate data pre-processing is one of the common reasons on why some predictive models fail.

# When shall we perform FEATURE ENGINEERING ?
# https://www.linkedin.com/posts/soledad-galli_should-feature-engineering-be-done-before-activity-7102606015589699584-D4KG/

# We should treat the test set as unseen data. Therefore, we can't use it for feature engineering or selection purposes. 
# If we did, it would mean that the test set was seen during the creation of the machine learning pipeline.

# Instead of machine learning models, think of machine learning pipelines. Within this context, the test set should be unseen by the entire pipeline, 
# which involves feature engineering, feature selection, and model optimization.

# If any part of modelling "sees" test data you may have information leaking. 
# Eg, test data should not be used when centering or standardizing variableas, 
# when imputing mean / median / modes, etc
# So, split at the beginning, and do not use until the time comes to validate your model.

# In a real - world application, all you will have is a training set to build and deploy a model, 
# and that model will predict the test examples that the application needs. 
# In this setting, it is not realistic to assume that you will have the test data to generate features, 
# as you will already have your model delivered. 
# So, the mechanism to generate the features can only be fed with training information.

# TO CENTER and SCALE the DATA ?

# Whether you need to center or scale your data depends on the specific requirements of the machine 
# learning algorithm you plan to use and the characteristics of your dataset. Here are some considerations:

# Centering:

# If the algorithm you are using is sensitive to the mean of the features, 
# centering the data might be beneficial.
# Centering the data by subtracting the mean ensures that the mean of each variable is zero. 
# This can be useful for interpretability, especially in linear models.

# Scaling:

# If the scale of the features matters for the algorithm you are using, you might want to scale the data.
# Algorithms that rely on distances between data points, such as k-nearest neighbors or support vector machines, 
# are sensitive to the scale of features. Scaling helps ensure that all features contribute equally to distance computations.

# Considerations:

# If your data ranges from -8 to +8 and your chosen algorithm is not sensitive to the mean or scale, 
# you might not need to center or scale the data.
# Some algorithms, like decision trees or random forests, 
# are not inherently sensitive to the scale or mean of features, 
# and they may perform well without scaling or centering.





preProcValues <- preProcess(training, method = c("center", "scale", "knnImpute"))
trainTransformed <- predict(preProcValues, training)
glimpse(trainTransformed)

# to use a GENERAL VARIABLE to specify the PRE-PROCESSING :

# PREPROCESS = c("center","scale", "knnImpute")
PREPROCESS = c("center","scale", "bagImpute")
# PREPROCESS = c("center","scale", "BoxCox")

# knnImpute data is scaled and centered by default
# we can't avoid scaling and centering your data when using method = "knnImpute", 

# however, method = "bagImpute" or method = "medianImpute" will not scale and center 
# the data unless we ask it to. 



## TRAINING PARAMETERS : 

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats=10, 
                     allowParallel=TRUE, 
                     classProbs = TRUE)



## Correlations between FEATURES

library(GGally)
library(klaR)

ggpairs(training)
ggpairs(testing)



# Near Zero Variance Predictors

# nzv <- nearZeroVar(student3_predictors)
# filteredDescr <- student3_predictors[, -nzv]

# dim(filteredDescr)
# head(filteredDescr, 2)
# dim(student3_predictors)

filteredDescr = student3_predictors
head(filteredDescr, 2)

# Correlated Predictors

# findCorrelation uses the following algorithm to flag predictors for removal :
descrCor <-  cor(filteredDescr)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .75)
highCorr 
summary(descrCor[upper.tri(descrCor)])

# the effect of removing descriptors with absolute correlations above 0.75.

highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
highlyCorDescr

# if we can find such predictors :

# filteredDescr2 <- filteredDescr[,-highlyCorDescr]
# descrCor2 <- cor(filteredDescr2)
# summary(descrCor2[upper.tri(descrCor2)])

# Linear Predictors
# uses the QR decomposition of a matrix to enumerate sets of linear combinations (if they exist).

# findLinearCombos will return a list that enumerates these dependencies. 
# This is often not easy to find in larger data sets! 
# For each linear combination, it will incrementally remove columns from the matrix and 
# test to see if the dependencies have been resolved. 
# findLinearCombos returns a vector of column positions can be removed to eliminate the linear dependencies:

comboInfo = findLinearCombos(student3_predictors)
comboInfo 

# if comboInfo$remove is different than 0
# student3_predictors[, -comboInfo$remove]

# CENTERING
# SCALING
# predict.preProcess is used to apply them to specific data set



### KNN



################################################
################################################
## TRAINING 

knnFit <- train( RESULT~ ., 
                 data = training, 
                 method = "knn", 
                 trControl = ctrl, 
                 preProcess = PREPROCESS, 
                 tuneLength = 20)

## OUTPUT :
knnFit

## png("the.results.knn.FIT.png")
plot(knnFit)
## dev.off()

## PREDICTIONS :
knnPredict <- predict(knnFit, newdata = testing)

## CONFUSION MATRIX
confusionMatrix_knn = confusionMatrix(knnPredict, testing$RESULT, mode = "everything")
# confusionMatrix(knnPredict, testing$RESULT, mode = "prec_recall" )

## ACCURACY 
## print("the ACCURACY of KNN model is :")
## mean(knnPredict == testing$RESULT)

## VARIABLE IMPORTANCE :
knnFit.vip <- varImp(knnFit)
print(knnFit.vip)
plot(knnFit.vip)

## ROC :
## Obtaining predicted probabilites for Test data
knn.probs = predict(knnFit,
                   newdata = testing,
                   type="prob")

rocCurve.knn <- roc(testing$RESULT, knn.probs[,"PASS"])
plot(rocCurve.knn, col=c(4))
print("AUC is : ")
auc(rocCurve.knn)

rocCurve.knn.auc = auc(rocCurve.knn) 



# using ANN 



################################################
################################################
## TRAINING
## nnnet package by defualt uses the Logistic Activation function

fit.nn <- train( RESULT~ ., 
                   data = training, 
                   method = "nnet", 
                   trControl = ctrl, 
                   preProcess = PREPROCESS, 
                   trace=FALSE,
                   verbose=FALSE, 
                   # tuneLength = 20, 
                   na.action = na.omit)

## OUTPUT :

# Size: Number of Hidden Layers.
# Decay: Is the regularization factor that offsets overfitting.
# Kappa: Evaluates the match is significant or by chance.

head(fit.nn$results)
tail(fit.nn$results)

# PLOT :
print(fit.nn)
plot(fit.nn)

# PREDICTIONS : 
fit.nn.predict <- predict(fit.nn, newdata = testing)

# CONFUSION MATRIX : 
confusionMatrix_nn = confusionMatrix(fit.nn.predict, testing$RESULT, mode = "everything")
# confusionMatrix(fit.nn.predict, testing$RESULT, mode="prec_recall")

# ACCURACY :
## mean(fit.nn.predict == testing$RESULT)

# VARIABLE IMPORTANCE :
fit.nn.vip <- varImp(fit.nn)
print(fit.nn.vip)
plot(fit.nn.vip)

# Graphical representation of an ANN :

plotnet(fit.nn)
title("Graphical Representation of Neural Network")

# We  may estimate the VARIABLE IMPORTANCE with :
# vip(fit.nn) # in another R package

## ROC :
## Obtaining predicted probabilites for Test data
nn.probs = predict(fit.nn,
                   newdata = testing,
                   type="prob")
# head(nn.probs, 2)

rocCurve.nn <- roc(testing$RESULT, nn.probs[,"PASS"])

plot(rocCurve.nn, col=c(4))

print("AUC is : ")
auc(rocCurve.nn)

rocCurve.nn.auc = auc(rocCurve.nn) 



# TRAINING and PREDICTIONS with ANN
# we are using other libraries :
# suppressPackageStartupMessages(library(NeuralNetTools))
# suppressPackageStartupMessages(library(neuralnet))



############################################
############################################
# LOGISTIC ACTIVATION FUNCTION :

model.nn1 <- neuralnet(RESULT ~ age + traveltime + studytime + failures + absences,
                       data = training, 
                       hidden=2, 
                       act.fct = "logistic", 
                       linear.output = FALSE)

# plot(model.nn1)

# TANH ACTIVATION FUNCTION :

model.nn2 <- neuralnet(RESULT ~ age + traveltime + studytime + failures + absences,
                       data = training, 
                       hidden=2, 
                       act.fct = "tanh", 
                       linear.output = FALSE)

plot(model.nn2)

############################################
############################################

model.nn1.results <- neuralnet::compute(model.nn1, testing)
head(model.nn1.results$net.result)

model.nn2.results <- neuralnet::compute(model.nn2, testing)
head(model.nn2.results$net.result)

############################################
############################################

# model.nn1.results
# model.nn2.results

############################################
############################################



# Training and Predictions with SVM

# SVM_LINEAR



############################################
############################################

svm_Linear <- train( RESULT~ ., 
                     data = training, 
                     method = "svmLinear", 
                     trControl = ctrl, 
                     preProcess = PREPROCESS, 
                     trace=FALSE,
                     verbose=FALSE, 
                     # tuneGrid = grid,
                     # tuneLength = 20, 
                     na.action = na.omit)

## OUTPUT of SVM_LINEAR

head(svm_Linear$results)
tail(svm_Linear$results)
print(svm_Linear)

## PREDICTIONS :

svm_Linear_predict <- predict(svm_Linear, newdata = testing)

## CONFUSION MATRIX :

confusionMatrix_svmLinear = confusionMatrix(svm_Linear_predict, testing$RESULT, mode = "everything")

### ACCURACY :
### mean(svm_Linear_predict == testing$RESULT )

## VARIABLE IMPORTANCE
 
svm_Linear.vip <- varImp(svm_Linear)
print(svm_Linear.vip)
plot(svm_Linear.vip)

## ROC :
## Obtaining predicted probabilites for Test data
svm_Linear.probs = predict(svm_Linear,
                   newdata = testing,
                   type="prob")
rocCurve.svm_Linear <- roc(testing$RESULT, svm_Linear.probs[,"PASS"])

plot(rocCurve.svm_Linear, col=c(4))

print("AUC is : ")
auc(rocCurve.svm_Linear)

rocCurve.svm_Linear.auc = auc(rocCurve.svm_Linear)



# SVM_RADIAL



############################################
############################################
# grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5))

svm_Radial <- train( RESULT~ ., 
                   data = training, 
                   method = "svmRadial", 
                   trControl = ctrl, 
                   preProcess = PREPROCESS, 
                   trace=FALSE,
                   verbose=FALSE, 
                   # tuneGrid = grid,
                   # tuneLength = 20, 
                   na.action = na.omit)

## OUTPUT of SVM_RADIAL 
head(svm_Radial$results)
tail(svm_Radial$results)

print(svm_Radial)
plot(svm_Radial)

## PREDICTIONS
svm_Radial_predict <- predict(svm_Radial, newdata = testing)

## CONFUSION MATRIX :
confusionMatrix_svmRadial = confusionMatrix(svm_Radial_predict, testing$RESULT, mode = "everything")
# confusionMatrix(svm_Radial_predict, testing$RESULT, mode = "prec_recall")

## ACCURACY : 
## mean(svm_Radial_predict == testing$RESULT)

## VARIABLE IMPORTANCE :
svm_Radial.vip <- varImp(svm_Radial)
print(svm_Radial.vip)
plot(svm_Radial.vip)

## ROC :
## Obtaining predicted probabilites for Test data
svm_Radial.probs = predict(svm_Radial,
                   newdata = testing,
                   type="prob")
rocCurve.svm_Radial <- roc(testing$RESULT, svm_Radial.probs[,"PASS"])
plot(rocCurve.svm_Radial, col=c(4))

print("AUC is : ")
auc(rocCurve.svm_Radial)

rocCurve.svm_Radial.auc = auc(rocCurve.svm_Radial)  

## in other R LIBRARY :

## TRAINING

model.ksvm1 <- ksvm(RESULT ~ age + traveltime + studytime + failures + absences, 
                    data = training, 
                    kernel="rbfdot")
model.ksvm1

model.ksvm2 <- ksvm(RESULT ~ age + traveltime + studytime + failures + absences, 
                    data = training, 
                    kernel="tanhdot")
model.ksvm2

# PREDICTIONS

model.ksvm1.results <- predict(model.ksvm1, testing, type="response")
head(model.ksvm1.results)

table(model.ksvm1.results, testing$RESULT)
agreement1 <-  model.ksvm1.results == testing$RESULT
table(agreement1)
prop.table(table(agreement1))

model.ksvm2.results <- predict(model.ksvm2, testing, type="response")
head(model.ksvm2.results)

table(model.ksvm2.results, testing$RESULT)
agreement2 <-  model.ksvm2.results == testing$RESULT
table(agreement2)
prop.table(table(agreement2))



## LDA

############################################
############################################

ldaFit <- train( RESULT~ ., 
                   data = training, 
                   method = "lda", 
                   trControl = ctrl, 
                   preProcess = PREPROCESS, 
                   trace=FALSE,
                   verbose=FALSE, 
                   # tuneGrid = grid,
                   # tuneLength = 20, 
                   na.action = na.omit)

## OUTPUT of SVM_RADIAL 
head(ldaFit$results)
tail(ldaFit$results)

print(ldaFit)
# plot(ldaFit)

## PREDICTIONS
ldaPredict <- predict(ldaFit, newdata = testing)

## CONFUSION MATRIX :
confusionMatrix_lda = confusionMatrix(ldaPredict, testing$RESULT, mode = "everything") 
## confusionMatrix(ldaPredict, testing$RESULT, mode = "prec_recall") 

## ACCURACY : 
## mean(ldaPredict == testing$RESULT)

## VARIABLE IMPORTANCE :
ldaFit.vip <- varImp(ldaFit)
print(ldaFit.vip)
plot(ldaFit.vip)

## ROC :
## Obtaining predicted probabilites for Test data
ldaFit.probs = predict(ldaFit,
                   newdata = testing,
                   type="prob")

rocCurve.ldaFit <- roc(testing$RESULT, ldaFit.probs[,"PASS"])
plot(rocCurve.ldaFit, col=c(4))

print("AUC is : ")
auc(rocCurve.ldaFit)

rocCurve.ldaFit.auc = auc(rocCurve.ldaFit)



## DECISION TREES

## rpart : Recursive Partitioning and is used for constructing decision trees. 
## Decision trees are built by recursively partitioning the data based on the values of input features.
## rpart: Builds a single decision tree.

# VARIABLE IMPORTANCE with DECISION TREES
# The relative importance of predictor 𝑋 is the sum of the squared improvements over all internal nodes 
# of the tree for which 𝑋 was chosen as the partitioning variables.



rpartFit <- train( RESULT~ ., 
                 data = training, 
                 method = "rpart", 
                 trControl = ctrl, 
                 preProcess = PREPROCESS, 
                 tuneLength = 20)

## OUTPUT :
rpartFit

## summary(rpartFit$finalModel)
## it outputs a very long summary

## PLOT : 
plot(rpartFit)

## PREDICTIONS :
rpartPredict <- predict(rpartFit, newdata = testing)

## CONFUSION MATRIX and
confusionMatrix_rpart = confusionMatrix(rpartPredict, testing$RESULT, mode = "everything")
# confusionMatrix(rpartPredict, testing$RESULT, mode = "prec_recall")

## ACCURACY :
## mean(rpartPredict == testing$RESULT)

## VARIABLE IMPORTANCE
rpartFit.vip <- varImp(rpartFit)
plot(rpartFit.vip)
print(rpartFit.vip)

## DISPLAYING THE TREE

plot(rpartFit$finalModel, 
    uniform=TRUE,
    main="Classification Tree")
text(rpartFit$finalModel, use.n.=TRUE, all=TRUE, cex=.8)

fancyRpartPlot(rpartFit$finalModel)

## ROC :
## Obtaining predicted probabilites for Test data
rpartFit.probs = predict(rpartFit,
                   newdata = testing,
                   type="prob")

rocCurve.rpartFit <- roc(testing$RESULT, rpartFit.probs[,"PASS"])
plot(rocCurve.rpartFit, col=c(4))

print("AUC is : ")
auc(rocCurve.rpartFit)

rocCurve.rpartFit.auc = auc(rocCurve.rpartFit)



# LOGISTIC REGRESSION



## TRAINING
## In the caret package in R, the method to use for logistic regression is typically specified as "glm" 
## (Generalized Linear Model) with the family set to "binomial".

logisticFit = train( RESULT ~ .,
  data = training,
  trControl = ctrl,
  method = "glm",
  family = "binomial", 
  preProcess = PREPROCESS, 
  tuneLength = 20)

## OUTPUT : 
logisticFit

## PREDICTIONS 
logisticPredict <- predict(logisticFit, newdata = testing)

## CONFUSION MATRIX 
confusionMatrix_logistic = confusionMatrix(logisticPredict, testing$RESULT, mode = "everything")
## confusionMatrix(logisticPredict, testing$RESULT, mode = "prec_recall")

## ACCURACY :
## mean(logisticPredict == testing$RESULT)

## VARIABLE IMPORTANCE
logisticFit.vip <- varImp(logisticFit)
plot(logisticFit.vip)
print(logisticFit.vip)

## ROC :
## Obtaining predicted probabilites for Test data
logisticFit.probs = predict(logisticFit,
                   newdata = testing,
                   type="prob")

# head(bagg.probs, 2)

rocCurve.logisticFit <- roc(testing$RESULT, logisticFit.probs[,"PASS"])
plot(rocCurve.logisticFit, col=c(4))

print("AUC is : ")
auc(rocCurve.logisticFit)

rocCurve.logisticFit.auc = auc(rocCurve.logisticFit)



## NAIVE BAYES



### THE BALANCE of the DATA in TRAINING and TESTING SETS

prop.table(table(training$RESULT)) * 100
prop.table(table(testing$RESULT)) * 100

## TRAINING : 

nbFit = train( RESULT~ ., 
                 data = training, 
                 method = "nb", 
                 # preProcess = PREPROCESS, 
                 trControl = ctrl) 

# OUTPUT :
nbFit

# PREDICT :
nbPredict <- predict(nbFit, newdata = testing)

# CONFUSION MATRIX :
confusionMatrix_nb = confusionMatrix(nbPredict, testing$RESULT, mode = "everything")

## ACCURACY :
## mean(nbPredict == testing$RESULT)

## VARIABLE IMPORTANCE :
nbFit.vip <- varImp(nbFit)
plot(nbFit.vip)
print(nbFit.vip)

## ROC :
## Obtaining predicted probabilites for Test data
nbFit.probs = predict(nbFit,
                   newdata = testing,
                   type="prob")
rocCurve.nbFit <- roc(testing$RESULT, nbFit.probs[,"PASS"])
plot(rocCurve.nbFit, col=c(4))

print("AUC is : ")
auc(rocCurve.nbFit)

rocCurve.nbFit.auc = auc(rocCurve.nbFit)







# TREE BAG



treebagFit <- train( RESULT~ ., 
                 data = training, 
                 method = "treebag", 
                 trControl = ctrl, 
                 preProcess = PREPROCESS, 
                 tuneLength = 20)

treebagFit

## OUTPUT :
treebagFit

# summary(treebagFit$finalModel)
## it outputs a very long summary

## PLOT : 
# plot(treebagFit)

## PREDICTIONS :
treebagPredict <- predict(treebagFit, newdata = testing)

## CONFUSION MATRIX and
confusionMatrix_treebag = confusionMatrix(treebagPredict, testing$RESULT, mode = "everything")
# confusionMatrix(treebagPredict, testing$RESULT, mode = "prec_recall")

## ACCURACY :
## mean(treebagPredict == testing$RESULT)

## VARIABLE IMPORTANCE
treebagFit.vip <- varImp(treebagFit)
plot(treebagFit.vip)
print(treebagFit.vip)

## Obtaining predicted probabilites for Test data
bagg.probs=predict(treebagFit,
                   newdata = testing,
                   type="prob")

rocCurve.bagg <- roc(testing$RESULT, bagg.probs[,"PASS"])
rocCurve.bagg
plot(rocCurve.bagg, col=c(4))

print("AUC is : ")
auc(rocCurve.bagg)

rocCurve.bagg.auc = auc(rocCurve.bagg)



## RANDOM FOREST
## is an ensemble learning method that constructs a multitude of decision trees during 
## training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.



rfFit <- train( RESULT~ ., 
                 data = training, 
                 method = "rf", 
                 trControl = ctrl, 
                 preProcess = PREPROCESS, 
                 tuneLength = 20)

rfFit

## OUTPUT :
rfFit

## summary(rfFit$finalModel)
## it outputs a very long summary

## PLOT : 
## plot(rfFit)

## PREDICTIONS :
rfPredict <- predict(rfFit, newdata = testing)

## CONFUSION MATRIX and
confusionMatrix_rf = confusionMatrix(rfPredict, testing$RESULT, mode = "everything")
# confusionMatrix(rfPredict, testing$RESULT, mode = "prec_recall")

## ACCURACY :
## mean(rfPredict == testing$RESULT)

## VARIABLE IMPORTANCE
rfFit.vip <- varImp(rfFit)
plot(rfFit.vip)
print(rfFit.vip)

## Obtaining predicted probabilites for Test data
rf.probs = predict(rfFit,
                   newdata = testing,
                   type="prob")

rocCurve.rf <- roc(testing$RESULT, rf.probs[,"PASS"])
rocCurve.rf
plot(rocCurve.rf, col=c(4))

print("AUC is : ")
auc(rocCurve.rf)

rocCurve.rf.auc = auc(rocCurve.rf)  



## RANDOM FOREST with BOOSTING
# Idea: Boosting focuses on sequentially training multiple weak learners 
# (models that are slightly better than random guessing) to correct the errors of their predecessors

# modelLookup("ada")
# modelLookup("gbm")



# GBM : Stochastic Gradient Boosting

gbmFit <- train( RESULT~ ., 
                 data = training, 
                 method = "gbm", 
                 trControl = ctrl, 
                 preProcess = PREPROCESS, 
                 tuneLength = 20)

## OUTPUT :
## gbmFit

## summary(gbmFit$finalModel)
## it outputs a very long summary

## PLOT : 
## plot(gbmFit)

## PREDICTIONS :
gbmPredict <- predict(gbmFit, newdata = testing)

## CONFUSION MATRIX and
## confusionMatrix(gbmPredict, testing$RESULT, mode = "prec_recall")
confusionMatrix_gbm = confusionMatrix(gbmPredict, testing$RESULT, mode = "everything")

## ACCURACY :
## mean(gbmPredict == testing$RESULT)

## VARIABLE IMPORTANCE
gbmFit.vip <- varImp(gbmFit)
plot(gbmFit.vip)
print(gbmFit.vip)

## Obtaining predicted probabilites for Test data
gbm.probs = predict(gbmFit,
                   newdata = testing,
                   type="prob")
rocCurve.gbm <- roc(testing$RESULT, gbm.probs[,"PASS"])
plot(rocCurve.gbm, col=c(4))

print("AUC is : ")
auc(rocCurve.gbm)

rocCurve.gbm.auc = auc(rocCurve.gbm) 



# Summarize data from these models



# DIFFERENCE between these ALGORITHMS :

algo_results <- resamples(list( KNN = knnFit,
                                NNET = fit.nn, 
                                SVML = svm_Linear, 
                                SVMR = svm_Radial, 
                                DT = rpartFit, 
                                LR = logisticFit, 
                                NB = nbFit, 
                                LDA = ldaFit,
                                CART = treebagFit,
                                RF = rfFit, 
                                GBM = gbmFit ))   


summary(algo_results)
# dotplot(algo_results)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(algo_results, scales=scales)

# splom(algo_results)

diffs <- diff(algo_results)
diffs



## Combine F1 scores in a LIST : 

F1_scores <- list(
knn_F1 = confusionMatrix_knn$byClass["F1"],
nn_F1 = confusionMatrix_nn$byClass["F1"],
svmLinear_F1 = confusionMatrix_svmLinear$byClass["F1"],
svmRadial_F1 = confusionMatrix_svmRadial$byClass["F1"],
lda_F1 = confusionMatrix_lda$byClass["F1"],
rpart_F1 = confusionMatrix_rpart$byClass["F1"],
logistic_F1 = confusionMatrix_logistic$byClass["F1"],
nb_F1 = confusionMatrix_nb$byClass["F1"],
treebag_F1 = confusionMatrix_treebag$byClass["F1"],
rf_F1 = confusionMatrix_rf$byClass["F1"],
gbm_F1 = confusionMatrix_gbm$byClass["F1"]
)

# Create boxplot
boxplot(F1_scores, 
       main = "F1 scores",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:11), 
       lty = 1, 
       lwd = 2, 
       las = 2, 
       xlab = "Methods", 
       ylab = "F1 score")

## Printing the F1 scores
print("The F1 scores :")

print("KNN")
confusionMatrix_knn$byClass["F1"]
print("Neural Net")
confusionMatrix_nn$byClass["F1"]
print("SVM Linear")
confusionMatrix_svmLinear$byClass["F1"]
print("SVM Radial")
confusionMatrix_svmRadial$byClass["F1"]
print("LDA")
confusionMatrix_lda$byClass["F1"]
print("DT")
confusionMatrix_rpart$byClass["F1"]
print("Logistic Regression")
confusionMatrix_logistic$byClass["F1"]
print("Naive Bayes")
confusionMatrix_nb$byClass["F1"]
print("CART")
confusionMatrix_treebag$byClass["F1"]
print("Random Forest")
confusionMatrix_rf$byClass["F1"]
print("GBM")
confusionMatrix_gbm$byClass["F1"]







## # Combine ROC curves into a LIST
roc_curves <- list(
  KNN = rocCurve.knn.auc,
  NN = rocCurve.nn.auc,
  SVM_Linear = rocCurve.svm_Linear.auc,
  SVM_Radial = rocCurve.svm_Radial.auc,
  LDA = rocCurve.ldaFit.auc,
  DT = rocCurve.rpartFit.auc,
  Logistic = rocCurve.logisticFit.auc,
  NaiveBayes = rocCurve.nbFit.auc,
  Bagging = rocCurve.bagg.auc,
  RF = rocCurve.rf.auc,
  GBM = rocCurve.gbm.auc
)

# Create boxplot
boxplot(roc_curves, 
       main = "ROC Curves",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:11), 
       lty = 1, 
       lwd = 2, 
       las = 2,  
       xlab = "ML models", 
       ylab = "AUC")

# MODEL COMPARISONS :
# ROC CURVES : , main = "ROC Curves"

plot(rocCurve.knn, col = c(1), main = "ROC Curves")
plot(rocCurve.nn, add = TRUE, col = c(2))
plot(rocCurve.svm_Linear, add = TRUE, col = c(3))
plot(rocCurve.svm_Radial, add = TRUE, col = c(4))
plot(rocCurve.ldaFit, add = TRUE, col = c(5))
plot(rocCurve.rpartFit, add = TRUE, col = c(6))
plot(rocCurve.logisticFit, add = TRUE, col = c(7))
plot(rocCurve.nbFit, add = TRUE, col = c(8))
plot(rocCurve.bagg, add = TRUE, col = c(9)) 
plot(rocCurve.rf, add = TRUE, col = c(10)) 
plot(rocCurve.gbm, add = TRUE, col = c(11)) 

legend("bottomright", 
       legend = c("KNN",
                  "NN",
                  "SVM_Linear",
                  "SVM_radial",
                  "LDA",
                  "DT",
                  "Logistic Regression",
                  "Naive Bayes",
                  "Bagging",
                  "RF",
                  "GBM"),
col = c(1:11), 
lty = 1, lwd = 2)



# algo_results
# str(algo_results)



# STACKING

# suggestions from : 
# https://github.com/archowdhury/Bagging-Boosting-and-Stacking-using-R/blob/master/Ensembling%20using%20R.R



# Create the mmodels
# or is data the full MATRIX, not separated in training and testing ?
# https://github.com/archowdhury/Bagging-Boosting-and-Stacking-using-R/blob/master/Ensembling%20using%20R.R

set.seed(123)

algorithmList <- c(
'knn',
'nnet',
'svmLinear',
'svmRadial',
'lda',
'rpart',
'glm',
'nb',
'rf',
'gbm', 
'treebag')


models <- caretList( RESULT~., 
                     data=training, 
                     trControl=ctrl, 
                     methodList=algorithmList, 
                     # metric="prec_recall",
                     metric="ROC",
                     preProcess = PREPROCESS, 
                     tuneLength = 20)

results <- resamples(models)

print("summary of all these models :")
# summary(results)
# dotplot(results)

# Check the correlation between the models (ideally the models should have low correlations) :

# models
modelCor(results)
# splom(results)
# results

pheatmap(modelCor(results), 
        cluster_rows = FALSE, 
        cluster_cols = FALSE)

### PREDICTIONS <- as.data.frame(predict(models, newdata=testing))
### print(PREDICTIONS)
### some notes at : 
### https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html





# ENSEMBLE model

# Create an ensemble model using the caretEnsemble package
ensemble_model <- caretEnsemble(models)

# Make predictions on the test set
ensemble_predictions <- predict(ensemble_model, newdata = testing)

# Evaluate the performance of the ensemble model
confusionMatrix_ensemble = confusionMatrix(ensemble_predictions, testing$RESULT, mode = "everything")
print(confusionMatrix_ensemble)

# Variable Importance 
print("Variable Importance")
ensemble.vip <- varImp(ensemble_model, scale = FALSE)
# print(ensemble.vip)

# plot(ensemble.vip)
pheatmap(ensemble.vip, 
         cluster_rows = FALSE,  # Disable row clustering
         cluster_cols = FALSE  # Disable column clustering
)

# Extract F1 score
confusionMatrix_ensemble_F1 <- confusionMatrix_ensemble$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_ensemble_F1))

## Obtaining predicted probabilites for Test data
ensemble.probs = predict(ensemble_model,
                         newdata = testing,
                         type="prob")


rocCurve.ensemble <- roc(testing$RESULT, ensemble.probs)
# rocCurve.ensemble <- roc(testing$RESULT, ensemble.probs[, "PASS"])
plot(rocCurve.ensemble, col=c(4))

print("AUC is : ")
auc(rocCurve.ensemble)

rocCurve.ensemble.auc = auc(rocCurve.ensemble) 

# Extract F1 score
confusionMatrix_ensemble_F1 <- confusionMatrix_ensemble$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_ensemble_F1))





# STACKING MODELS

# Stack the models using Random Forest

stackControl = trainControl(method='repeatedcv', 
                            number=10, repeats=10,
                            savePredictions = TRUE,
                            classProbs = TRUE,
                            # preProcess = PREPROCESS,
                            verbose=TRUE)

rf_Stack = caretStack(models, method='rf', metric='F1', trControl = stackControl)
print(rf_Stack)

rf_Stack_predict <- predict(rf_Stack, newdata = testing)

## CONFUSION MATRIX :
confusionMatrix_rf_Stack_predict = confusionMatrix(rf_Stack_predict, testing$RESULT, mode="everything")
## confusionMatrix_rf_Stack_predict = confusionMatrix(rf_Stack_predict, testing$RESULT, mode="prec_recall")

## ACCURACY :
## mean(rf_Stack_predict == testing$RESULT)

## Obtaining predicted probabilites for Test data
rf_Stack.probs = predict(rf_Stack,
                   newdata = testing,
                   type="prob")

rocCurve.rf_Stack <- roc(testing$RESULT, rf_Stack.probs)
plot(rocCurve.rf_Stack, col=c(4))

print("AUC is : ")
auc(rocCurve.rf_Stack)

rocCurve.rf_Stack.auc = auc(rocCurve.rf_Stack) 

# Extract F1 score
confusionMatrix_rf_Stack_predict_F1 <- confusionMatrix_rf_Stack_predict$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_rf_Stack_predict_F1))

rocCurve.rf_Stack.auc



## VARIABLE IMPORTANCE

###  to extract the VARIABLE IMPORTANCE : rf_Stack$models
base_models <- rf_Stack$models
base_model_importances <- lapply(base_models, varImp)

### base_model_importances

rf_Stack_knn = data.frame(rf_Stack_knn_importance = base_model_importances$knn$importance[,"PASS"], 
                          Variable = rownames(base_model_importances$knn$importance))

rf_Stack_nnet = data.frame(rf_Stack_nnet_importance = base_model_importances$nnet$importance$Overall, 
                           Variable = rownames(base_model_importances$nnet$importance))

rf_Stack_svmLinear = data.frame(rf_Stack_svmLinear_importance = base_model_importances$svmLinear$importance[,"PASS"],
             Variable = rownames(base_model_importances$svmLinear$importance))

rf_Stack_svmRadial = data.frame(rf_Stack_svmRadial_importance = base_model_importances$svmRadial$importance[,"PASS"],
             Variable = rownames(base_model_importances$svmRadial$importance))

rf_Stack_lda = data.frame(rf_Stack_lda_importance = base_model_importances$lda$importance[,"PASS"],
             Variable = rownames(base_model_importances$lda$importance))

rf_Stack_rpart = data.frame(rf_Stack_rpart_importance = base_model_importances$rpart$importance$Overall,
             Variable = rownames(base_model_importances$rpart$importance))

rf_Stack_glm = data.frame(rf_Stack_glm_importance = base_model_importances$glm$importance$Overall,
             Variable = rownames(base_model_importances$glm$importance))

rf_Stack_nb = data.frame(rf_Stack_nb_importance = base_model_importances$nb$importance[,"PASS"],
             Variable = rownames(base_model_importances$nb$importance))

rf_Stack_treebag = data.frame(rf_Stack_treebag_importance = base_model_importances$treebag$importance$Overall,
             Variable = rownames(base_model_importances$treebag$importance))

rf_Stack_rf = data.frame(rf_Stack_rf_importance = base_model_importances$rf$importance$Overall,
             Variable = rownames(base_model_importances$rf$importance))

rf_Stack_gbm = data.frame(rf_Stack_gbm_importance = base_model_importances$gbm$importance$Overall,
             Variable = rownames(base_model_importances$gbm$importance))

## Including all these models into a Data Frame : 

rf_Stack_list <- list(rf_Stack_knn, 
                      rf_Stack_nnet, 
                      rf_Stack_svmLinear,
                      rf_Stack_svmRadial, 
                      rf_Stack_lda, 
                      rf_Stack_rpart,
                      rf_Stack_glm, 
                      rf_Stack_nb, 
                      rf_Stack_treebag,
                      rf_Stack_rf, 
                      rf_Stack_gbm)

# Merge data frames based on "Variable"
rf_Stack_list_importance <- Reduce(function(x, y) merge(x, y, by = "Variable", all = TRUE), rf_Stack_list)

# Make a data frame
rf_Stack_list_importance.df = as.data.frame(rf_Stack_list_importance)
rownames(rf_Stack_list_importance.df) = rf_Stack_list_importance.df$Variable

# Show a heatmap                                   
rf_Stack_list_importance.df <- rf_Stack_list_importance.df[, -1]
pheatmap(rf_Stack_list_importance.df, cluster_cols = FALSE, cluster_rows = FALSE)

rf_Stack_list_importance





# Stack the models using GBM

# Stack the models using GBM
stackControl = trainControl(method='repeatedcv', 
                            number=10, repeats=3,
                            savePredictions = TRUE,
                            classProbs = TRUE,
                            # preProcess = PROCESS,
                            verbose=TRUE)

gbm_Stack = caretStack(models, method='gbm', metric='Accuracy', trControl = stackControl)
print(gbm_Stack)
# str(gbm_Stack)

gbm_Stack_predict <- predict(gbm_Stack, newdata = testing)

## CONFUSION MATRIX and
confusionMatrix_gbm_Stack_predict  = confusionMatrix(gbm_Stack_predict, testing$RESULT, mode = "everything")

## ACCURACY :
# mean(gbm_Stack_predict == testing$RESULT)

## Obtaining predicted probabilites for Test data
gbm_Stack.probs = predict(gbm_Stack,
                    newdata = testing,
                    type="prob")

rocCurve.gbm_Stack <- roc(testing$RESULT, gbm_Stack.probs)
plot(rocCurve.gbm_Stack, col=c(4))

print("AUC is : ")
auc(rocCurve.gbm_Stack)

rocCurve.gbm_Stack.auc = auc(rocCurve.gbm_Stack) 

# Extract F1 score
confusionMatrix_gbm_Stack_predict_F1 <- confusionMatrix_gbm_Stack_predict$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_gbm_Stack_predict_F1))

## VARIABLE IMPORTANCE

###  to extract the VARIABLE IMPORTANCE : rf_Stack$models
base_models <- gbm_Stack$models
base_model_importances <- lapply(base_models, varImp)

### base_model_importances

gbm_Stack_knn = data.frame(gbm_Stack_knn_importance = base_model_importances$knn$importance[,"PASS"], 
                           Variable = rownames(base_model_importances$knn$importance))

gbm_Stack_nnet = data.frame(gbm_Stack_nnet_importance = base_model_importances$nnet$importance$Overall, 
                            Variable = rownames(base_model_importances$nnet$importance))

gbm_Stack_svmLinear = data.frame(gbm_Stack_svmLinear_importance = base_model_importances$svmLinear$importance[,"PASS"],
             Variable = rownames(base_model_importances$svmLinear$importance))

gbm_Stack_svmRadial = data.frame(gbm_Stack_svmRadial_importance = base_model_importances$svmRadial$importance[,"PASS"],
             Variable = rownames(base_model_importances$svmRadial$importance))

gbm_Stack_lda = data.frame(gbm_Stack_lda_importance = base_model_importances$lda$importance[,"PASS"],
             Variable = rownames(base_model_importances$lda$importance))

gbm_Stack_rpart = data.frame(gbm_Stack_rpart_importance = base_model_importances$rpart$importance$Overall,
             Variable = rownames(base_model_importances$rpart$importance))

gbm_Stack_glm = data.frame(gbm_Stack_glm_importance = base_model_importances$glm$importance$Overall,
             Variable = rownames(base_model_importances$glm$importance))

gbm_Stack_nb = data.frame(gbm_Stack_nb_importance = base_model_importances$nb$importance[,"PASS"],
             Variable = rownames(base_model_importances$nb$importance))

gbm_Stack_treebag = data.frame(gbm_Stack_treebag_importance = base_model_importances$treebag$importance$Overall,
             Variable = rownames(base_model_importances$treebag$importance))

gbm_Stack_rf = data.frame(gbm_Stack_rf_importance = base_model_importances$rf$importance$Overall,
             Variable = rownames(base_model_importances$rf$importance))

gbm_Stack_gbm = data.frame(gbm_Stack_gbm_importance = base_model_importances$gbm$importance$Overall,
             Variable = rownames(base_model_importances$gbm$importance))

## Including all these models into a Data Frame : 

gbm_Stack_list <- list(gbm_Stack_knn, 
                      gbm_Stack_nnet, 
                      gbm_Stack_svmLinear,
                      gbm_Stack_svmRadial, 
                      gbm_Stack_lda, 
                      gbm_Stack_rpart,
                      gbm_Stack_glm, 
                      gbm_Stack_nb, 
                      gbm_Stack_treebag,
                      gbm_Stack_rf, 
                      gbm_Stack_gbm)

# Merge data frames based on "Variable"
gbm_Stack_list_importance <- Reduce(function(x, y) merge(x, y, by = "Variable", all = TRUE), gbm_Stack_list)

# Make a data frame
gbm_Stack_list_importance.df = as.data.frame(gbm_Stack_list_importance)
rownames(gbm_Stack_list_importance.df) = gbm_Stack_list_importance.df$Variable

# Show a heatmap                                   
gbm_Stack_list_importance.df <- gbm_Stack_list_importance.df[, -1]
pheatmap(gbm_Stack_list_importance.df, cluster_cols = FALSE, cluster_rows = FALSE)                                

gbm_Stack_list_importance.df

head(training, 2)
head(testing, 2)



# Plot again F1 score

F1_scores_again <- list(
knn_F1 = confusionMatrix_knn$byClass["F1"],
nn_F1 = confusionMatrix_nn$byClass["F1"],
svmLinear_F1 = confusionMatrix_svmLinear$byClass["F1"],
svmRadial_F1 = confusionMatrix_svmRadial$byClass["F1"],
lda_F1 = confusionMatrix_lda$byClass["F1"],
rpart_F1 = confusionMatrix_rpart$byClass["F1"],
logistic_F1 = confusionMatrix_logistic$byClass["F1"],
nb_F1 = confusionMatrix_nb$byClass["F1"],
treebag_F1 = confusionMatrix_treebag$byClass["F1"],
rf_F1 = confusionMatrix_rf$byClass["F1"],
gbm_F1 = confusionMatrix_gbm$byClass["F1"],
ensemble_F1 = confusionMatrix_ensemble$byClass["F1"],
rf_Stack_F1 = confusionMatrix_rf_Stack_predict$byClass["F1"],
gbm_Stack_F1 = confusionMatrix_gbm_Stack_predict$byClass["F1"]
)

# Create boxplot
boxplot(F1_scores_again, 
       main = "F1 scores",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:14), 
       lty = 1, 
       lwd = 2, 
       las = 2, 
       xlab = "Methods", 
       ylab = "F1 score")





# Plot again ROC

## # Combine ROC curves into a LIST
roc_curves_again <- list(
  KNN = rocCurve.knn.auc,
  NN = rocCurve.nn.auc,
  SVM_Linear = rocCurve.svm_Linear.auc,
  SVM_Radial = rocCurve.svm_Radial.auc,
  LDA = rocCurve.ldaFit.auc,
  DT = rocCurve.rpartFit.auc,
  Logistic = rocCurve.logisticFit.auc,
  NaiveBayes = rocCurve.nbFit.auc,
  Bagging = rocCurve.bagg.auc,
  RF = rocCurve.rf.auc,
  GBM = rocCurve.gbm.auc, 
  ensemble = rocCurve.ensemble.auc,
  rf_Stack = rocCurve.rf_Stack.auc,
  gbm_Stack = rocCurve.gbm_Stack.auc
)

boxplot(roc_curves, 
       main = "ROC Curves",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:14), 
       lty = 1, 
       lwd = 2, 
       las = 2,  
       xlab = "ML models", 
       ylab = "AUC")

# MODEL COMPARISONS :
# ROC CURVES : , main = "ROC Curves"

plot(rocCurve.knn, col = c(1), main = "ROC Curves")
plot(rocCurve.nn, add = TRUE, col = c(2))
plot(rocCurve.svm_Linear, add = TRUE, col = c(3))
plot(rocCurve.svm_Radial, add = TRUE, col = c(4))
plot(rocCurve.ldaFit, add = TRUE, col = c(5))
plot(rocCurve.rpartFit, add = TRUE, col = c(6))
plot(rocCurve.logisticFit, add = TRUE, col = c(7))
plot(rocCurve.nbFit, add = TRUE, col = c(8))
plot(rocCurve.bagg, add = TRUE, col = c(9)) 
plot(rocCurve.rf, add = TRUE, col = c(10)) 
plot(rocCurve.gbm, add = TRUE, col = c(11)) 
plot(rocCurve.ensemble, add = TRUE, col = c(12)) 
plot(rocCurve.rf_Stack, add = TRUE, col = c(13)) 
plot(rocCurve.gbm_Stack, add = TRUE, col = c(14)) 

legend("bottomright", 
       legend = c("KNN",
                  "NN",
                  "SVM_Linear",
                  "SVM_radial",
                  "LDA",
                  "DT",
                  "Logistic Regression",
                  "Naive Bayes",
                  "Bagging",
                  "RF",
                  "GBM", 
                  "Ensemble",
                  "RF_Stack", 
                  "GBM_Stack" ),
col = c(1:14), 
lty = 1, lwd = 2)

# Associations and Feature Importance:

# The feature importance or coefficients to understand which variables contribute the most to predictions. 
# This analysis might reveal associations between certain features and the target variable.

# High feature importance in a predictive model can indicate that a particular feature is strongly 
# associated with the target variable.
# For example, if you have a predictive model for predicting customer churn, 
# and the "Number of Customer Service Calls" is identified as a highly important feature, 
# it suggests an association between the number of service calls and the likelihood of churn.



head(training, 2)
head(testing, 2)



library(randomForest)
set.seed(123)

# Step I : Data Preparation
# Step II : Run the random forest model

# Create a random forest model
rf_model <- randomForest(RESULT ~ ., data = training, ntree = 500)

# The number of variables selected at each split is denoted by mtry in randomforest function.

mtry <- tuneRF(training[, 1:5], 
               training$RESULT, 
               ntreeTry=500,
               stepFactor=2, 
               improve=0.01, 
               trace=TRUE, 
               plot=TRUE)

# Step III : Find the optimal mtry value

best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]

print("mtry")
print(mtry)

print("best.m")
print(best.m)

# Step IV : Build model again using best mtry value.


rf_model <- randomForest(RESULT ~ ., 
                         data = training, 
                         ntree = 500, 
                         mtry=best.m, 
                         importance=TRUE)

print(rf_model)
plot(rf_model)

predictions <- predict(rf_model, testing)

# Evaluate the model
conf_matrix <- table(predictions, testing$RESULT)
conf_matrix

# Compute precision (Positive Predictive Value)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])

# Compute recall (Sensitivity)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Compute F1 score
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Precision (Positive Predictive Value):", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score, "\n")

# Evaluate variable importance
importance(rf_model)
varImpPlot(rf_model)

# not advisable, it plots only : MeanDecreaseAccuracy
# vip::vip(rf_model)
# str(importance(rf_model))
rf_model_predictor_importance = as.data.frame(importance(rf_model))




# Higher the value of mean decrease gini score , 
# higher the importance of the variable in the model



head(training, 2)
head(testing, 2)



# coding the XGBOOST MODELS :
# https://cran.r-project.org/web/packages/xgboost/vignettes/xgboostPresentation.html
# https://www.projectpro.io/recipes/apply-xgboost-for-classification-r



# in the code below, we encode NO_PASS as 0, and PASS as 1

# y_train <- as.integer(training$RESULT) - 1
# y_test <- as.integer(testing$RESULT) - 1
# head(y_train,20)
# head(y_test,20)

X_train = training[, c(1,2,3,4,5)]  # independent variables for train
X_test = testing[, c(1,2,3,4,5)]    # dependent variables for train
head(X_train, 2)
head(X_test, 2)

# using BINARY CODING of 0 and 1
# Y_train <- as.integer(training$RESULT) - 1
# Y_test <- as.integer(testing$RESULT) - 1

# if we do not need to perform the BINARY ENCODING
# y_train <- training$RESULT  # independent variables for test
# y_test <- testing$RESULT    # dependent variables for test

# we encode PASS as 1, and NOT_PASS as 0

Y_train <- as.integer(training$RESULT == "PASS") # independent variables for test
Y_test <- as.integer(testing$RESULT == "PASS")    # dependent variables for test
head(Y_train, 2)
head(Y_test, 2)

xgb_train <- xgb.DMatrix(data = as.matrix(X_train), label = Y_train)
xgb_test <- xgb.DMatrix(data = as.matrix(X_test), label = Y_test)



xgb_params <- list(
  eta = 0.01,
  max_depth = 8,
  gamma = 4,
  subsample = 0.75,
  objective = "binary:logistic",
  eval_metric = "logloss"
  # booster = "gbtree",
  # booster = "gblinear",
  # it is an important choice as the Var Imp varies depending on the objective function
  # objective = "multi:softprob",  
  # num_class = length(levels(student3$RESULT))
)

# xgboost
# watchlist <- list(train = xgb_train, test = xgb_test)

xgb_model <- xgb.train(
    data = xgb_train,
    params = xgb_params,
    nrounds = 5000,
    # watchlist = watchlist, 
    verbose = 1
)

summary(xgb_model)

importance_matrix <- xgb.importance(
  feature_names = colnames(xgb_train), 
  model = xgb_model
)

importance_matrix

xgb.plot.importance(importance_matrix)





# in order to check the model and to visualize the tree
# xgb.dump(xgb_model, with_stats = TRUE)
# xgb.plot.tree(model = xgb_model)

# in order to compute F1 SCORE : 

# converting the LABELS "PASS" and "NO_PASS" into BINARY 0 or 1 for the TESTING DATASET
testing$RESULT_binary <- ifelse(testing$RESULT == 'PASS', 1, 0)
head(testing, 5)

# Make predictions on the test set
predictions <- predict(xgb_model, newdata = xgb_test)
head(predictions, 2)

# Convert predicted values to class labels :
# predicted_labels <- ifelse(predictions > 0.5, "PASS", "NOT_PASS") 
# predicted_labels = factor(predicted_labels, levels = levels(testing$RESULT))
# predicted_labels <- as.factor(predicted_labels)
# Evaluate the model :
# confusionMatrix_xgboost = confusionMatrix(predicted_labels, testing$RESULT,  mode = "everything")
# confusionMatrix_xgboost
# head(predicted_labels, 2)
# head(testing$RESULT, 2)

# Convert predicted values to binary 0 or 1
predicted_values <- ifelse(predictions > 0.5, 1, 0)
head(predicted_values, 2)

# Confusion Matrix :
confusionMatrix_xgboost = confusionMatrix(as.factor(predicted_values), 
                                          as.factor(testing$RESULT_binary),  mode = "everything")
confusionMatrix_xgboost

# Extract F1 score
confusionMatrix_xgboost_F1 <- confusionMatrix_xgboost$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_xgboost_F1))

# The issue of F1 being reported as NA typically occurs when either precision (Pos Pred Value) 
# or recall is zero. The F1 score is defined as the harmonic mean of precision and recall, 
# and it becomes undefined (NA) when either precision or recall is zero.

# Obtaining predicted probabilites for Test data

# xgboost.probs = predict(predicted_labels,
#                   newdata = testing,
#                   type="prob")

# rocCurve.xgboost <- roc(testing$RESULT, xgboost.probs[,"PASS"])
# plot(rocCurve.xgboost, col=c(4))#
#
# print("AUC is : ")
# auc(rocCurve.xgboost)#
# rocCurve.xgboost.auc = auc(rocCurve.xgboost) 


# Create a ROC curve
xgb_roc <- roc(testing$RESULT, predicted_values)
# xgb_roc
plot(xgb_roc) 

# Calculate AUC
auc_score <- auc(xgb_roc)
cat("AUC score:", auc_score, "\n")
auc_score





# FEATURE SELECTION

# BORUTA

# how BORUTA works : SHADOW FEATURES

# Create duplicate copies of all independent variables. 
# When the number of independent variables in the original data is less than 5, 
# create at least 5 copies using existing variables.
# Shuffle the values of added duplicate copies to remove their correlations with the target variable. 
# It is called shadow features or permuted copies.
# Combine the original ones with shuffled copies
# Run a random forest classifier on the combined dataset and performs a variable importance measure 
# (the default is Mean Decrease Accuracy) to evaluate the importance of each variable where higher means 
# more important.
# Then Z score is computed. It means mean of accuracy loss divided by standard deviation of accuracy loss.
# Find the maximum Z score among shadow attributes (MZSA)
# Tag the variables as ‘unimportant’  when they have importance significantly lower than MZSA. 
# Then we permanently remove them from the process.
# Tag the variables as ‘important’  when they have importance significantly higher than MZSA.

# Since the whole process is dependent on permuted copies, 
# we repeat random permutation procedure to get statistically robust results



# BORUTA
# It creates a set of random noise variables associated with each attribute by shuffling each value’s 
# row location. After training a Random Forest model it checks the variable importance of each feature 
# against the distribution of corresponding noise variables.



library("Boruta")

boruta.train  <- Boruta( RESULT~ ., 
                  data = training, 
                  doTrace = 0)

fd <- boruta.train$finalDecision

table(fd)
as.data.frame(fd)
# TentativeRoughFix(boruta.train)
# getSelectedAttributes(boruta.train)
    
confirmed_vars <- fd[which(fd == "Confirmed")] %>% names
cat("confirmed variables : ")
print(confirmed_vars)

rejected_vars <- fd[which(fd != "Confirmed")] %>% names
cat("rejected variables : ")
print(rejected_vars)




head(x_train,2)
head(x_test,2)
head(y_train,2)
head(y_test,2)



# performing RFE (recursive feature elimination)

# Define the control parameters : 

ctrl <- rfeControl(functions = rfFuncs, # Random forest is used as the feature selection method
                   method = "cv",       # Cross-validation is used for resampling
                   number = 10)         # Number of variables to select

# Run the feature selection : 

model_rfe <- rfe(x_train,         # Remove the response variable from the predictor set
              y_train,            # Response variable
              sizes = c(1:10),    # Range of variable sizes to consider
              rfeControl = ctrl)  # Use the control parameters defined above

model_rfe
plot(model_rfe, type = c("g", "o"))

# Get the selected features 
selected_features <- predictors(model_rfe)
print(selected_features)

# For subsequent analysis, to use the matrix of the relevant features
# x2_train = x_train[, selected_features] # training data: selected features
# x2_test = x_test[, selected_features] # test data: selected features
# head(x2_train, 2)
# head(x2_test, 2)

# Genetic Alghoritms
# There are a few built-in sets of functions to use with gafs: caretGA, rfGA, and treebagGA
# ctrl <- gafsControl(functions = caretGA)
# ctrl <- gafsControl(functions = rfGA)
# ctrl <- gafsControl(functions = treebagGA)
# obj <- gafs(x = predictors, 
#            y = outcome,
#            iters = 100,
#            gafsControl = ctrl,
#            ## Now pass options to `train`
#            method =

set.seed(123)

# Define the control parameters
ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)

# Set the levels
LEVELS <- c("PASS","NO_PASS")     
 
# Compute 
ga_compute <- gafs(x = x_train, 
                   y = y_train,
                   iters = 100, # 100 generations of algorithm
                   popSize = 20, # population size for each generation
                   levels = LEVELS,
                   gafsControl = ga_ctrl)


# The plot also shows the average internal accuracy estimates as well as the average external estimates 
# calculated from the 10 out of sample predictions.
plot(ga_compute) + theme_bw() # Plot mean fitness (AUC) by generation

# Get the selected features
final <- ga_compute$ga$final # Get features selected by GA
cat("the selected features by GA :")
print(final)

# For subsequent analysis, to use the matrix of the relevant features
# x2_train = x_train[, final] # training data: selected features
# x2_test = x_test[, final] # test data: selected features
# head(x2_train, 2)
# head(x2_test, 2)




# Simulated Annealing

# ctrl <- safsControl(functions = caretSA)
# ctrl <- safsControl(functions = rfSA)
# obj <- safs(x = predictors, 
#            y = outcome,
#            iters = 100,
#            safsControl = ctrl,
             ## Now pass options to `train`
#            method =

set.seed(123)

# Define the control parameters
sa_ctrl <- safsControl(functions = rfSA,           # Assess fitness with RF
                       method = "repeatedcv",      # 10 fold cross validation
                       repeats = 5,
                       improve = 50)


LEVELS <- c("PASS","NO_PASS")     # Set the levels

# Compute 
sa_compute <- safs(x = x_train, 
                   y = y_train,
                   iters = 100, # 100 generations of algorithm
                   levels = LEVELS,
                   safsControl = sa_ctrl)



# The plot also shows the average internal accuracy estimates as well as the average external estimates 
# calculated from the 10 out of sample predictions.
plot(sa_compute) + theme_bw() # Plot mean fitness (AUC) by generation

# Get the selected features
final <- sa_compute$sa$final # Get features selected by SA
cat("the selected features by SA :")
print(final)
