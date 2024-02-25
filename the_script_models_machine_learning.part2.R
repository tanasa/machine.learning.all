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
suppressPackageStartupMessages(library("corrplot"))
# suppressPackageStartupMessages(library("RSNNS"))
# suppressPackageStartupMessages(library("VIP"))

############################################
############################################

# We have decided not to use the parallelism as it may compromise the running of some ML models

# rCluster <- makePSOCKcluster(6) 
# registerDoParallel(rCluster)  

set.seed(123)

# Choose the file to process : 

############################################
############################################

FILE = "missingness.m40_COLUMN.txt"
student <- read.delim(FILE, sep="\t", header=T, stringsAsFactors=F)
head(student, 2)

# summary(student)
# str(student)
cat("the dimensions of the matrix: ")
dim(student)
colnames(student)

table(student$wgs.Aneuploidy.Score.Cut)
table(student$wgs.Aneu.EV.Cut)
table(student$wgs.FGA.EGS.Cut)
table(student$wgs.BPs.Lost.EGS.Cut)
table(student$wgs.BPs.Gained.EGS.Cut)

# An IMPORTANT NOTE : 
# We are working with # 'wgs.FGA.EGS.Cut'

VARIABLE = 'wgs.BPs.amplified.Cut'

# in this particular analysis, we have recompute the variable "wgs.BPs.Gained.EGS.Cut", 
# given the fact that in the previous file, it has 6 NA values.
# we remove the rows (cell lines) where NA is present

student_recomputed = student
dim(student_recomputed)

student_recomputed = student_recomputed[!is.na(student_recomputed$'wgs.BPs.amplified.Cut'), ]
dim(student_recomputed)

median_value_AMP <- median(student_recomputed$wgs.BPs.Gained.EGS)
student_recomputed$wgs.BPs.Gained.EGS.Cut_recomputed = 
        ifelse(student_recomputed$wgs.BPs.Gained.EGS > median_value_AMP, "high", "small")

sum(is.na(student_recomputed$wgs.BPs.Gained.EGS.Cut_recomputed))

write.table(student_recomputed,
file = "missingness.m40_COLUMN.re.txt",
quote=FALSE, sep="\t", col.names = TRUE, row.names = FALSE)

student = student_recomputed
dim(student)

student1 <- subset(student, select = -c(
wes.ProfileID,
wes.Aneuploidy.Score,
wes.BPs.Lost,
wes.FGA,
wes.Aneu.EV,
wes.Datatype,
wes.CellLineName,
wgs.ProfileID,
wgs.Aneuploidy.Score,
wgs.BPs.Lost,
wgs.FGA,
wgs.Aneu.EV,
wgs.Datatype,
wgs.CellLineName,
wes.BPs.Lost.ratio,
wgs.BPs.Lost.ratio,
wes.FGA.ratio,
wgs.FGA.ratio,
ModelID,
wgs.FGA.EGS,
wgs.BPs.Lost.EGS,
wgs.BPs.Gained,
wgs.BPs.Gained.EGS,
X.GDSC1))          ### it is the 1st column

############################################
############################################

head(student1, 2)

# beside the drug response values, we keep the following DEPENDENT VARIABLES :
# 'wgs.Aneuploidy.Score.Cut',
# 'wgs.Aneu.EV.Cut',
# 'wgs.FGA.EGS.Cut',
# 'wgs.BPs.Lost.EGS.Cut',
# 'wgs.BPs.Gained.EGS.Cut'

# in the data frame student1, we have kept multiple DEPENDENT VARIABLES ; 
# in the next analysis, we are working with only ONE variable at a time :

student2 = subset(student1, 
                  select = -c(   wgs.Aneuploidy.Score.Cut,
                                 wgs.Aneu.EV.Cut, 
                                 wgs.FGA.EGS.Cut,
                                 wgs.BPs.Lost.EGS.Cut,
                                 wgs.BPs.Gained.EGS.Cut
                                 # wgs.BPs.Gained.EGS.Cut_recomputed
                              ))

head(student2, 2)

# to make a BackUp Copy :
# to rename the dependent variable as RESULT

student3 = student2
student3$RESULT = student3$wgs.BPs.Gained.EGS.Cut_recomputed 
student3 = subset(student3, 
                  select = -c(wgs.BPs.Gained.EGS.Cut_recomputed))

table(student3$RESULT)
head(student3, 3)

student3$RESULT = as.factor(student3$RESULT)
class(student3$RESULT)

colnames(student3)
dim(student3)

## To have more BACKUP copies : 

student4 = student3
student5 = student3
student6 = student3

############################################
############################################

# TO SPECIFY the INDEPENDENT and DEPENDENT variable for TRAINING and TESTING dataset

# independent variables for TRAIN
# dependent variable for TRAIN
# independent variables for TEST
# dependent variables for TEST

dim(student3)
index_predictors = dim(student3)[2]-1
index_predictors 
index_predicted = dim(student3)[2]
index_predicted

options(repr.plot.width = 15, repr.plot.height = 15 , repr.plot.res = 100)

# Visualize the data 
x = student3[, c(1:index_predictors)]
y = student3[, index_predicted]

# scatterplot matrix
# pairs(RESULT~. , data = student3, col=student3$RESULT)

# box and whisker plots for each attribute
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="box", scales=scales)

# density plots for each attribute by class value
featurePlot(x=x, y=y, plot="density", scales=scales)

# to check the MISSINGNESS

# to check the MISSINGNESS 
# how many values are in the DATA FRAME : 
dim(student3)
dimensions = dim(student3)[1] * (dim(student3)[2] - 1)
dimensions

# HOW MANY MISSING VALUES ARE :

number_of_missing_values = sum(is.na(student3))
percent_of_missing_values = number_of_missing_values / dimensions

cat("the percent of missing values : ")
percent_of_missing_values

student3_predictors = dplyr::select(student3, -RESULT)
head(student3_predictors, 2)

############################################
############################################

# Imputation with MICE

# Imputation with MICE
# using one of the back up copies :
student3 = student4
set.seed(123)

# IMPUTATION with MICE : 
# we could also use the data in df student3_predictors 

############################################
############################################ PATTERNS in the MISSING DATA

# Missing pattern :
options(repr.plot.width = 30, repr.plot.height = 30 , repr.plot.res = 100)
md.pattern(student3)
# md.pairs(student3)

############################################
############################################ PATTERNS in the MISSING DATA
# Missing pattern :
# md.pattern(student3)
# md.pairs(student3)
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

mice_student3 <- mice(student3, method = "rf", printFlag = FALSE)
# str(mice_student3, 2)

mice_student3_complete <- mice::complete(mice_student3)
# str(mice_student3_complete)

head(mice_student3_complete, 2)

############################################
############################################

dim(student3)
dim(mice_student3_complete)

cat("number of NA in the original dataset :")
sum(is.na(student3))

cat("number of NA in the imputed dataset :")
sum(is.na(mice_student3_complete))

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
# numeric_columns <- mice_student3_complete[sapply(mice_student3_complete, is.numeric)]
# numeric_columns

# Create density or strip plots for each numeric column
# density_plots <- lapply(numeric_columns, densityplot)
# strip_plots <- lapply(numeric_columns, stripplot)

############################################
############################################
# Display the density plots
# options(repr.plot.width = 30, repr.plot.height = 30 , repr.plot.res = 100)
# print(density_plots)
# Display the strip plots
# print(strip_plots)
############################################
############################################ considering a variable
# names(mice_student3_complete)
# stripplot(mice_student3_complete[, "age"])
# densityplot(mice_student3_complete[, "age"])
############################################
############################################

# Working with student4
# using one of the back up copies 

student3 = student4
set.seed(123)

options(repr.plot.width = 6, repr.plot.height = 6, repr.plot.res = 100)

# student3' is our data frame
original_data <- student3

# Specify the variables of interest
variables_of_interest <- colnames(original_data)

# Create an empty list to store individual plots
plots_list <- list()

# Loop through variables of interest
for (variable_of_interest in variables_of_interest) {
  # Combine the original and imputed datasets
  combined_data <- rbind(
    data.frame(value = original_data[, variable_of_interest], status = "Before Imputation"),
    data.frame(value = mice_student3_complete[, variable_of_interest], status = "After Imputation")
  )
  
  # Create an overlayed density plot for the current variable
  current_plot <- ggplot(combined_data, aes(x = value, fill = status)) +
    geom_density(alpha = 0.7) +
    labs(title = paste(variable_of_interest, " Before and After Imputation")) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 20),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 14)
    )
  
  # Store the current plot in the list
  plots_list[[variable_of_interest]] <- current_plot
}

# Print or view the individual plots
for (i in seq_along(plots_list)) {
  print(plots_list[[i]])
}

############################################
############################################

# Based on MICE IMPUTED DATA, generate a TRAINING SET and a TEST SET :

cat("number of NA values in the original dataset : ")
sum(is.na(student3))
cat("number of NA values in the MICE imputed dataset : ")
sum(is.na(mice_student3_complete))

# TO SPLIT and NOT TO IMPUTE, SCALE, or CENTER : 

set.seed(123)
                         
# SPLIT DATASET
indxTrain_mice <- createDataPartition(mice_student3_complete$RESULT, 
                                 p = .75, 
                                 list = FALSE)
# indxTrain

training3_mice <- mice_student3_complete[indxTrain_mice,]
# training
testing3_mice <- mice_student3_complete[-indxTrain_mice,]
# testing

dim(student3)
dim(training3_mice)
dim(testing3_mice)

levels(training3_mice$RESULT)
levels(testing3_mice$RESULT)

cat("number of NA values in the training data : ")
sum(is.na(training3_mice))
cat("number of NA values in the test data : ")
sum(is.na(testing3_mice))

############################################
############################################

# IMPUTATION using MISSFOREST package
# Working with student5

# using one of the back up copies :
student3 = student5
set.seed(123)

missforest_student3 <- missForest(student3)            # Impute missing values using 5 iterations
missforest_student3_ximp = missforest_student3$ximp
head(missforest_student3_ximp, 2)

cat("number of NA values in the original dataset : ")
sum(is.na(student3))
cat("number of NA values in the missForest imputed dataset : ")
sum(is.na(missforest_student3_ximp))

# TO SPLIT and NOT TO IMPUTE, SCALE, or CENTER : 

set.seed(123)
                         
# SPLIT DATASET
indxTrain2_mf <- createDataPartition(missforest_student3_ximp$RESULT, 
                                 p = .75, 
                                 list = FALSE)
# indxTrain

training2_mf <- missforest_student3_ximp[indxTrain2_mf,]
# training
testing2_mf <- missforest_student3_ximp[-indxTrain2_mf,]
# testing

dim(student3)
dim(training2_mf)
dim(testing2_mf)

levels(training2_mf$RESULT)
levels(training2_mf$RESULT)

cat("number of NA values in the training data : ")
sum(is.na(training2_mf))
cat("number of NA values in the test data : ")
sum(is.na(testing2_mf))

############################################
############################################

# IMPUTATION with AMELIA
set.seed(123)

# IMPUTATION with AMELIA
# we do NOT use categorical variables, or the RESPONSE variable
# it is problematic though, as not all NA are replaced.

amelia_student3_predictors <- amelia(student3_predictors, 
                                      m = 5, 
                                      parallel = "multicore" )
                                      ## noms = c('drug1','drug2','drug3','','','RESULT'

# To access the imputed data frames and to count the number of "NA"

cat("number of NA in the original dataset :")
sum(is.na(student3_predictors))

cat("number of NA in the imputed dataset :")
sum(is.na(amelia_student3_predictors$imputations[[1]]))
sum(is.na(amelia_student3_predictors$imputations[[2]]))
sum(is.na(amelia_student3_predictors$imputations[[3]]))
sum(is.na(amelia_student3_predictors$imputations[[4]]))
sum(is.na(amelia_student3_predictors$imputations[[5]]))


# Shall we use MICE IMPUTED DATA 
# the IMPUTATIONS in CARET, 
# or with MISS FORREST : 
# we will EVALUATE the options

# ML groups suggest to do the splitting before we impute and train ; 
# in our case, as we do deal with a percent of missing data, 
# and we are interested in Variable Importance, 
# we perform the PRE_PROCESSING and the IMPUTATIONS in a first step.

# To perform the analysis in a more natural order :
# SPLITING
# PRE-PROCESSING of the TRAINING data ONLY 

############################################
############################################ 

# PRE-PROCESSING with CARET
# We split the DATASET into TRAINING and TESTING based on the pre-processing performed with CARET

set.seed(123)

# In order to avoid potential bugs : 
# Working with student6
# using one of the back up copies :

student3 = student6

# we impute the databased on "bagImpute" algorithm

sum(is.na(student3$RESULT))

preProcValues_all <- preProcess(student3, method = c("center", "scale", "bagImpute"))
student3Transformed_all <- predict(preProcValues_all, student3)
glimpse(student3Transformed_all)

sum(is.na(student3))
sum(is.na(student3Transformed_all))

student3 = student3Transformed_all

############################################
############################################

set.seed(123)
student3 = student3Transformed_all

# SPLIT DATASET
indxTrain <- createDataPartition(student3$RESULT, 
                                 p = .75, 
                                 list = FALSE)
# indxTrain

training <- student3[indxTrain,]
# training
testing <- student3[-indxTrain,]
# testing

cat("the entire dataset :")
dim(student3)
cat("the training dataset :")
dim(training)
cat("the testing dataset :")
dim(testing)

############################################
############################################

## PRE-PROCESSING : after SPLITTING the DATA

# trainX  <- training[, names(training) != "RESULT"]
# preProcValues <- preProcess(x = trainX, method = c("center", "scale, "bagImpute))
# preProcValues <- preProcess(training, method = c("center", "scale", "bagImpute"))
# trainTransformed <- predict(preProcValues, training)
# glimpse(trainTransformed)
# using the TRANSFORMED DATA for DOWNSTREAM ML methods
# training = trainTransformed 

############################################
############################################

# data imputed with CARET

X_train = training[, c(1:index_predictors)]  # independent variables for train
X_test = testing[, c(1:index_predictors)]    # dependent variables for train
# head(X_train, 2)
# head(X_test, 2)

# a copy of the same variable
x_train = X_train
x_test = X_test
# head(x_train, 2)
# head(x_test, 2)

# using BINARY CODING of 0 and 1
# Y_train <- as.integer(training$RESULT) - 1
# Y_test <- as.integer(testing$RESULT) - 1

# if we do not need to perform the BINARY ENCODING
y_train <- training$RESULT  # independent variables for test
y_test <- testing$RESULT    # dependent variables for test
# head(y_train, 2)
# head(y_test, 2)

# we encode HIGH as 1, and SMALL as 0

Y_train <- as.integer(training$RESULT == "high")  # independent variables for test
Y_test <- as.integer(testing$RESULT == "high")    # dependent variables for test
# head(Y_train, 2)
# head(Y_test, 2)

# dimensions of the dataframes :
dim(X_train)
dim(X_test)
dim(x_train)
dim(x_test)
length(Y_train)
length(Y_test)
length(y_train)
length(y_test) 

############################################
############################################

# TO SPECIFY the INDEPENDENT and DEPENDENT variable for TRAINING and TESTING dataset

# data imputed with MICE

X_train_mice = training3_mice[, c(1:index_predictors)]  # independent variables for train
X_test_mice = testing3_mice[, c(1:index_predictors)]    # dependent variables for train
# head(X_train_mice, 2)
# head(X_test_mice, 2)

# a copy of the same variable
x_train_mice = X_train_mice
x_test_mice = X_test_mice
# head(x_train_mice, 2)
# head(x_test_mice, 2)

# using BINARY CODING of 0 and 1
# Y_train_mice <- as.integer(training3_mice$RESULT) - 1
# Y_test_mice <- as.integer(testing3_mice$RESULT) - 1

# if we do not need to perform the BINARY ENCODING
y_train_mice <- training3_mice$RESULT  # independent variables for test
y_test_mice <- testing3_mice$RESULT    # dependent variables for test
# head(y_train_mice, 2)
# head(y_test_mice, 2)

# we encode HIGH as 1, and SMALL as 0

Y_train_mice <- as.integer(training3_mice$RESULT == "high")  # independent variables for test
Y_test_mice <- as.integer(testing3_mice$RESULT == "high")    # dependent variables for test
# head(Y_train_mice, 2)
# head(Y_test_mice, 2)

dim(X_train_mice)
dim(X_test_mice)
dim(x_train_mice)
dim(x_test_mice)
length(Y_train_mice)
length(Y_test_mice)
length(y_train_mice)
length(y_test_mice) 

############################################
############################################

# TO SPECIFY the INDEPENDENT and DEPENDENT variable for TRAINING and TESTING dataset

# data imputed with MissForest

X_train_mf = training2_mf[, c(1:index_predictors)]  # independent variables for train
X_test_mf = testing2_mf[, c(1:index_predictors)]    # dependent variables for train
# head(X_train_mf, 2)
# head(X_test_mf, 2)

# a copy of the same variable
x_train_mf = X_train_mf
x_test_mf = X_test_mf
# head(x_train_mf, 2)
# head(x_test_mf, 2)

# using BINARY CODING of 0 and 1
# Y_train_mf <- as.integer(training2_mf$RESULT) - 1
# Y_test_mf <- as.integer(testing2_mf$RESULT) - 1

# if we do not need to perform the BINARY ENCODING
y_train_mf <- training2_mf$RESULT  # independent variables for test
y_test_mf <- testing2_mf$RESULT    # dependent variables for test
# head(y_train_mf, 2)
# head(y_test_mf, 2)

# we encode HIGH as 1, and SMALL as 0

Y_train_mf <- as.integer(training2_mf$RESULT == "high")  # independent variables for test
Y_test_mf <- as.integer(testing2_mf$RESULT == "high")    # dependent variables for test
# head(Y_train_mf, 2)
# head(Y_test_mf, 2)

dim(X_train_mf)
dim(X_test_mf)
dim(x_train_mf)
dim(x_test_mf)
length(Y_train_mf)
length(Y_test_mf)
length(y_train_mf)
length(y_test_mf) 

############################################
############################################

# to use a GENERAL VARIABLE to specify the PRE-PROCESSING :

PREPROCESS = c("center","scale", "bagImpute")

# PREPROCESS = c("center","scale", "knnImpute")
# PREPROCESS = c("center","scale", "BoxCox")

# knnImpute data is scaled and centered by default
# we can't avoid scaling and centering your data when using method = "knnImpute", 

# however, method = "bagImpute" or method = "medianImpute" will not scale and center the data 
# unless we ask it to. 

# How to handle NA values :
# na.action = na.pass
# na.action = na.exclude 
# na.action = na.omit

# na.exclude over na.omit is that the former will retain the original number of rows in the data. 
# This may be useful where you need to retain the original size of the dataset - 
# for example it is useful when you want to compare predicted values to original values. 
# With na.omit you will end up with fewer rows so you won't as easily be able to compare.

############################################
############################################

set.seed(123)

## TRAINING PARAMETERS : 

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats=10, 
                     allowParallel=TRUE, 
                     classProbs = TRUE)

# Near Zero Variance Predictors

nzv <- nearZeroVar(student3_predictors)
cat("Near Zero Variance Predictors :")
nzv

# filteredDescr <- student3_predictors[, -nzv]
# dim(filteredDescr)
# head(filteredDescr, 2)
# dim(student3_predictors)

filteredDescr = student3_predictors
head(filteredDescr, 2)
dim(student3)
dim(filteredDescr)

## Correlations between FEATURES

options(repr.plot.width = 15, repr.plot.height = 15)

suppressMessages({
suppressWarnings({
    
# ggpairs(training, aes(colour = RESULT))
# ggpairs(testing, aes(colour = RESULT))
ggpairs(student3, aes(colour = RESULT))
    
     })
  })

# Correlated Predictors

# findCorrelation uses the following algorithm to flag predictors for removal :
descrCor <-  cor(filteredDescr, use = 'pairwise.complete.obs')

# number of the predictors that are highly correlated
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .75)

cat("number of the predictors that are highly correlated :")
highCorr 
summary(descrCor[upper.tri(descrCor)])

options(repr.plot.width = 15, repr.plot.height = 15)
cat("the correlation plot is :")
corrplot(descrCor, method = "color")

# the effect of removing descriptors with absolute correlations above 0.75.
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
highlyCorDescr

# if we can find such predictors :

# filteredDescr2 <- filteredDescr[,-highlyCorDescr]
# descrCor2 <- cor(filteredDescr2)
# summary(descrCor2[upper.tri(descrCor2)])

############################################
############################################

# Linear Predictors
# uses the QR decomposition of a matrix to enumerate sets of linear combinations (if they exist).

# findLinearCombos will return a list that enumerates these dependencies. 
# This is often not easy to find in larger data sets! 
# For each linear combination, it will incrementally remove columns from the matrix and 
# test to see if the dependencies have been resolved. 
# findLinearCombos returns a vector of column positions can be removed to eliminate the linear dependencies:

# The data needs to be IMPUTED
sum(is.na(student3))
sum(is.na(mice_student3_complete))
sum(is.na(student3Transformed_all))
# sum(is.na(student3_mf))

# comboInfo = findLinearCombos(student3_predictors)
# comboInfo 

# The findLinearCombos function from the caret package does not directly handle missing values in the data. 
# If we have missing values in our data and want to compute linear combinations while accounting for 
# missing values, we may consider imputing or handling missing values first.

cat("the Linear Predictors :")
comboInfo = findLinearCombos(mice_student3_complete[,c(1:index_predictors)])
comboInfo 

# if comboInfo$remove is different than 0
# student3_predictors[, -comboInfo$remove]

############################################
############################################

# CENTERING
# SCALING
# BAGIMPUTE
# predict.preProcess is used to apply them to specific data set

print("parameters to pre-process the data :")
PREPROCESS

# if we would like to redo the split shown below :

set.seed(123)
student3 = student3Transformed_all

# SPLIT DATASET
# indxTrain <- createDataPartition(student3$RESULT, 
#                                  p = .75, 
#                                  list = FALSE)
# indxTrain

# training <- student3[indxTrain,]
# training
# testing <- student3[-indxTrain,]
# testing

# cat("the entire dataset :")
# dim(student3)
# cat("the training dataset :")
# dim(training)
# cat("the testing dataset :")
# dim(testing)

### KNN
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

################################################
################################################
## TRAINING 

knnFit <- train( RESULT~ ., 
                 data = training, 
                 method = "knn", 
                 trControl = ctrl 
                 # preProcess = PREPROCESS, 
                 # tuneLength = 20, 
                 # na.action = na.omit
                 )

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

rocCurve.knn <- roc(testing$RESULT, knn.probs[,"high"])
plot(rocCurve.knn, col=c(4))
print("AUC is : ")
auc(rocCurve.knn)

rocCurve.knn.auc = auc(rocCurve.knn) 

# using ANN 
# it requires significant computational resources 

set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)
sum(is.na(training))

################################################
################################################
## TRAINING
## nnet package by default uses the Logistic Activation function
## library(nnet)
        
fit.nn <- caret::train( RESULT~ ., 
                   data = training, 
                   # method = "nnet", 
                   # method = "neuralnet",
                   method = "mlp",
                   trControl = ctrl, 
                   tuneLength = 20,
                   # preProcess = PREPROCESS, 
                   # na.action = na.omit, 
                   trace=FALSE,
                   verbose=FALSE)

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
confusionMatrix_nn = caret::confusionMatrix(fit.nn.predict, testing$RESULT, mode = "everything")
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

rocCurve.nn <- roc(testing$RESULT, nn.probs[,"high"])

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
# Calling the functions directly from NEURALNET package

model.nn1 <- neuralnet(RESULT ~ . ,
                       data = training, 
                       hidden=2, 
                       act.fct = "logistic", 
                       linear.output = FALSE)

# plot(model.nn1)

# TANH ACTIVATION FUNCTION :

model.nn2 <- neuralnet(RESULT ~ .,
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

# SVM
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

# SVM_LINEAR

############################################
############################################

svm_Linear <- train( RESULT~ ., 
                     data = training, 
                     method = "svmLinear", 
                     trControl = ctrl, 
                     # preProcess = PREPROCESS,
                     # tuneGrid = grid,
                     # tuneLength = 20,
                     na.action = na.omit,
                     trace=FALSE,
                     verbose=FALSE 
                   )

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
rocCurve.svm_Linear <- roc(testing$RESULT, svm_Linear.probs[,"high"])

plot(rocCurve.svm_Linear, col=c(4))

print("AUC is : ")
auc(rocCurve.svm_Linear)

rocCurve.svm_Linear.auc = auc(rocCurve.svm_Linear)

# SVM_RADIAL
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

############################################
############################################
# grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5))

svm_Radial <- train( RESULT~ ., 
                   data = training, 
                   method = "svmRadial", 
                   trControl = ctrl, 
                   # preProcess = PREPROCESS, 
                   na.action = na.omit,
                   # tuneGrid = grid,
                   # tuneLength = 20, 
                   trace=FALSE,
                   verbose=FALSE)

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
rocCurve.svm_Radial <- roc(testing$RESULT, svm_Radial.probs[,"high"])
plot(rocCurve.svm_Radial, col=c(4))

print("AUC is : ")
auc(rocCurve.svm_Radial)

rocCurve.svm_Radial.auc = auc(rocCurve.svm_Radial)  

## with other R LIBRARY :

## TRAINING

model.ksvm1 <- ksvm(RESULT ~ ., 
                    data = training, 
                    na.action = na.omit,
                    kernel="rbfdot")
model.ksvm1

model.ksvm2 <- ksvm(RESULT ~ ., 
                    data = training,
                    na.action = na.omit,
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

## VARIABLE IMPORTANCE : model.ksvm1
## VARIABLE IMPORTANCE : model.ksvm2

## LDA
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

############################################
############################################

ldaFit <- train( RESULT~ ., 
                   data = training, 
                   method = "lda", 
                   trControl = ctrl, 
                   # preProcess = PREPROCESS, 
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

rocCurve.ldaFit <- roc(testing$RESULT, ldaFit.probs[,"high"])
plot(rocCurve.ldaFit, col=c(4))

print("AUC is : ")
auc(rocCurve.ldaFit)

rocCurve.ldaFit.auc = auc(rocCurve.ldaFit)

############################################
############################################

## DECISION TREES
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

## rpart : Recursive Partitioning and is used for constructing decision trees. 
## Decision trees are built by recursively partitioning the data based on the values of input features.
## rpart: Builds a single decision tree.
## The split is based on CART algorithm, using rpart() function from the package.

# VARIABLE IMPORTANCE with DECISION TREES
# The relative importance of predictor 𝑋 is the sum of the squared improvements over all internal nodes 
# of the tree for which 𝑋 was chosen as the partitioning variables.

rpartFit <- train( RESULT~ ., 
                 data = training, 
                 method = "rpart", 
                 trControl = ctrl, 
                 # preProcess = PREPROCESS,
                 na.action = na.omit, 
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

rocCurve.rpartFit <- roc(testing$RESULT, rpartFit.probs[,"high"])
plot(rocCurve.rpartFit, col=c(4))

print("AUC is : ")
auc(rocCurve.rpartFit)

rocCurve.rpartFit.auc = auc(rocCurve.rpartFit)

############################################
############################################

# LOGISTIC REGRESSION
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

## TRAINING
## In the caret package in R, the method to use for logistic regression is typically specified as "glm" 
## (Generalized Linear Model) with the family set to "binomial".

logisticFit = train( RESULT ~ .,
  data = training,
  trControl = ctrl,
  method = "glm",
  family = "binomial", 
  # preProcess = PREPROCESS, 
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

rocCurve.logisticFit <- roc(testing$RESULT, logisticFit.probs[,"high"])
plot(rocCurve.logisticFit, col=c(4))

print("AUC is : ")
auc(rocCurve.logisticFit)

rocCurve.logisticFit.auc = auc(rocCurve.logisticFit)

############################################
############################################

## NAIVE BAYES
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)


### THE BALANCE of the DATA in TRAINING and TESTING SETS

prop.table(table(training$RESULT)) * 100
prop.table(table(testing$RESULT)) * 100

## TRAINING : 

suppressWarnings({
nbFit = train( RESULT~ ., 
                 data = training, 
                 method = "nb", 
                 # preProcess = PREPROCESS, 
                 trControl = ctrl) 
})

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
rocCurve.nbFit <- roc(testing$RESULT, nbFit.probs[,"high"])
plot(rocCurve.nbFit, col=c(4))

print("AUC is : ")
auc(rocCurve.nbFit)

rocCurve.nbFit.auc = auc(rocCurve.nbFit)

############################################
############################################

## TREE BAG
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

treebagFit <- train( RESULT~ ., 
                 data = training, 
                 method = "treebag", 
                 trControl = ctrl, 
                 # preProcess = PREPROCESS, 
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

rocCurve.bagg <- roc(testing$RESULT, bagg.probs[,"high"])
rocCurve.bagg
plot(rocCurve.bagg, col=c(4))

print("AUC is : ")
auc(rocCurve.bagg)

rocCurve.bagg.auc = auc(rocCurve.bagg)

############################################
############################################

## RANDOM FOREST
## is an ensemble learning method that constructs a multitude of decision trees during 
## training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees.
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

rfFit <- train(  RESULT~ ., 
                 data = training, 
                 method = "rf", 
                 trControl = ctrl, 
                 # preProcess = PREPROCESS, 
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

rocCurve.rf <- roc(testing$RESULT, rf.probs[,"high"])
rocCurve.rf
plot(rocCurve.rf, col=c(4))

print("AUC is : ")
auc(rocCurve.rf)

rocCurve.rf.auc = auc(rocCurve.rf)  

############################################
############################################

## RANDOM FOREST with BOOSTING
## modelLookup("ada")
## modelLookup("gbm")

# GBM : Stochastic Gradient Boosting
set.seed(123)
options(repr.plot.width = 10, repr.plot.height=10)

gbmFit <- train( RESULT~ ., 
                 data = training, 
                 method = "gbm", 
                 trControl = ctrl, 
                 # preProcess = PREPROCESS, 
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
rocCurve.gbm <- roc(testing$RESULT, gbm.probs[,"high"])
plot(rocCurve.gbm, col=c(4))

print("AUC is : ")
auc(rocCurve.gbm)

rocCurve.gbm.auc = auc(rocCurve.gbm) 

############################################
############################################

# Summarize data from these models

KNN = knnFit
NNET = fit.nn
SVML = svm_Linear 
SVMR = svm_Radial 
DT = rpartFit
LR = logisticFit 
NB = nbFit 
LDA = ldaFit
CART = treebagFit
RF = rfFit 
GBM = gbmFit

# DIFFERENCE between these ALGORITHMS :

algo_results <- resamples(list( KNN = knnFit,
                                # NNET = fit.nn, 
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
dotplot(algo_results)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(algo_results, scales=scales)

# splom(algo_results)

diffs <- diff(algo_results)
diffs

############################################
############################################

## Combine F1 scores in a LIST : 

F1_scores <- list(
knn_F1 = confusionMatrix_knn$byClass["F1"],
# nn_F1 = confusionMatrix_nn$byClass["F1"],
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

options(repr.plot.width = 20, repr.plot.height = 10 , repr.plot.res = 100)

boxplot(F1_scores, 
       main = "F1 scores",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:11), 
       lty = 1, 
       lwd = 2, 
       las = 2, 
       # xlab = "Methods", 
       # ylab = "F1 score", 
       horizontal = TRUE)

## Printing the F1 scores
print("The F1 scores :")

print("KNN")
confusionMatrix_knn$byClass["F1"]
# print("Neural Net")
# confusionMatrix_nn$byClass["F1"]
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

############################################
############################################

## Combine ROC curves into a LIST

roc_curves <- list(
  KNN = rocCurve.knn.auc,
  # NN = rocCurve.nn.auc,
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

options(repr.plot.width = 15, repr.plot.height = 10 , repr.plot.res = 100)

boxplot(roc_curves, 
       main = "ROC Curves",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:11), 
       lty = 1, 
       lwd = 2, 
       las = 2,  
       # xlab = "ML models", 
       # ylab = "AUC", 
       horizontal = TRUE)

# MODEL COMPARISONS :
# ROC CURVES : , main = "ROC Curves"

plot(rocCurve.knn, col = c(1), main = "ROC Curves")
# plot(rocCurve.nn, add = TRUE, col = c(2))
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
                  # "NN",
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

############################################
############################################ # STACKING

# STACKING
# algo_results
# str(algo_results)

set.seed(123)

algorithmList <- c(
'knn',
# 'nnet',
'svmLinear',
'svmRadial',
# 'lda',
'rpart',
'glm',
'nb',
'rf',
'gbm', 
'treebag')

suppressMessages(
    
models <- caretList( RESULT~., 
                     data = training, 
                     trControl=ctrl, 
                     methodList=algorithmList, 
                     metric="prec_recall",
                     # metric="ROC",
                     # preProcess = PREPROCESS, 
                     tuneLength = 20)
)

results <- resamples(models)

print("summary of all these models :")
summary(results)
# dotplot(results)

# Check the correlation between the models (ideally the models should have low correlations) :

options(repr.plot.width = 12, repr.plot.height=12)

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


############################################
############################################ # ENSEMBLE model

# ENSEMBLE model

set.seed(123)

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
# rocCurve.ensemble <- roc(testing$RESULT, ensemble.probs[, "high"])
plot(rocCurve.ensemble, col=c(4))

print("AUC is : ")
auc(rocCurve.ensemble)

rocCurve.ensemble.auc = auc(rocCurve.ensemble) 

# Extract F1 score
confusionMatrix_ensemble_F1 <- confusionMatrix_ensemble$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_ensemble_F1))


############################################
############################################

# STACKING MODELS

set.seed(123)

# Stack the models using Random Forest

stackControl = trainControl(method='repeatedcv', 
                            number=10, 
                            repeats=10,
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


############################################
############################################


## VARIABLE IMPORTANCE

###  to extract the VARIABLE IMPORTANCE : rf_Stack$models
base_models <- rf_Stack$models
base_model_importances <- lapply(base_models, varImp)

### base_model_importances

rf_Stack_knn = data.frame(rf_Stack_knn_importance = base_model_importances$knn$importance[,"high"], 
                          Variable = rownames(base_model_importances$knn$importance))

# rf_Stack_nnet = data.frame(rf_Stack_nnet_importance = base_model_importances$nnet$importance$Overall, 
#                           Variable = rownames(base_model_importances$nnet$importance))

rf_Stack_svmLinear = data.frame(rf_Stack_svmLinear_importance = base_model_importances$svmLinear$importance[,"high"],
             Variable = rownames(base_model_importances$svmLinear$importance))

rf_Stack_svmRadial = data.frame(rf_Stack_svmRadial_importance = base_model_importances$svmRadial$importance[,"high"],
             Variable = rownames(base_model_importances$svmRadial$importance))

rf_Stack_lda = data.frame(rf_Stack_lda_importance = base_model_importances$lda$importance[,"high"],
             Variable = rownames(base_model_importances$lda$importance))

rf_Stack_rpart = data.frame(rf_Stack_rpart_importance = base_model_importances$rpart$importance$Overall,
             Variable = rownames(base_model_importances$rpart$importance))

rf_Stack_glm = data.frame(rf_Stack_glm_importance = base_model_importances$glm$importance$Overall,
             Variable = rownames(base_model_importances$glm$importance))

rf_Stack_nb = data.frame(rf_Stack_nb_importance = base_model_importances$nb$importance[,"high"],
             Variable = rownames(base_model_importances$nb$importance))

rf_Stack_treebag = data.frame(rf_Stack_treebag_importance = base_model_importances$treebag$importance$Overall,
             Variable = rownames(base_model_importances$treebag$importance))

rf_Stack_rf = data.frame(rf_Stack_rf_importance = base_model_importances$rf$importance$Overall,
             Variable = rownames(base_model_importances$rf$importance))

rf_Stack_gbm = data.frame(rf_Stack_gbm_importance = base_model_importances$gbm$importance$Overall,
             Variable = rownames(base_model_importances$gbm$importance))

## Including all these models into a Data Frame : 

rf_Stack_list <- list(rf_Stack_knn, 
                      # rf_Stack_nnet, 
                      rf_Stack_svmLinear,
                      rf_Stack_svmRadial, 
                      # rf_Stack_lda, 
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
pheatmap(rf_Stack_list_importance.df, cluster_cols = FALSE, cluster_rows = FALSE, horizontal = TRUE)

rf_Stack_list_importance


############################################
############################################


# Stack the models using GBM
set.seed(123)

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

gbm_Stack_knn = data.frame(gbm_Stack_knn_importance = base_model_importances$knn$importance[,"high"], 
                           Variable = rownames(base_model_importances$knn$importance))

# gbm_Stack_nnet = data.frame(gbm_Stack_nnet_importance = base_model_importances$nnet$importance$Overall, 
#                             Variable = rownames(base_model_importances$nnet$importance))

gbm_Stack_svmLinear = data.frame(gbm_Stack_svmLinear_importance = base_model_importances$svmLinear$importance[,"high"],
             Variable = rownames(base_model_importances$svmLinear$importance))

gbm_Stack_svmRadial = data.frame(gbm_Stack_svmRadial_importance = base_model_importances$svmRadial$importance[,"high"],
             Variable = rownames(base_model_importances$svmRadial$importance))

# gbm_Stack_lda = data.frame(gbm_Stack_lda_importance = base_model_importances$lda$importance[,"high"],
#             Variable = rownames(base_model_importances$lda$importance))

gbm_Stack_rpart = data.frame(gbm_Stack_rpart_importance = base_model_importances$rpart$importance$Overall,
             Variable = rownames(base_model_importances$rpart$importance))

gbm_Stack_glm = data.frame(gbm_Stack_glm_importance = base_model_importances$glm$importance$Overall,
             Variable = rownames(base_model_importances$glm$importance))

gbm_Stack_nb = data.frame(gbm_Stack_nb_importance = base_model_importances$nb$importance[,"high"],
             Variable = rownames(base_model_importances$nb$importance))

gbm_Stack_treebag = data.frame(gbm_Stack_treebag_importance = base_model_importances$treebag$importance$Overall,
             Variable = rownames(base_model_importances$treebag$importance))

gbm_Stack_rf = data.frame(gbm_Stack_rf_importance = base_model_importances$rf$importance$Overall,
             Variable = rownames(base_model_importances$rf$importance))

gbm_Stack_gbm = data.frame(gbm_Stack_gbm_importance = base_model_importances$gbm$importance$Overall,
             Variable = rownames(base_model_importances$gbm$importance))

## Including all these models into a Data Frame : 

gbm_Stack_list <- list(gbm_Stack_knn, 
                      # gbm_Stack_nnet, 
                      gbm_Stack_svmLinear,
                      gbm_Stack_svmRadial, 
                      # gbm_Stack_lda, 
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


############################################
############################################


# Plot again F1 score


F1_scores_again <- list(
knn_F1 = confusionMatrix_knn$byClass["F1"],
# nn_F1 = confusionMatrix_nn$byClass["F1"],
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
options(repr.plot.width = 12, repr.plot.height=10)
boxplot(F1_scores_again, 
       main = "F1 scores",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:14), 
       lty = 1, 
       lwd = 2, 
       las = 2, 
       xlab = "Methods", 
       ylab = "F1 score", horizontal = TRUE)



############################################
############################################


# Plot again ROC

## Combine ROC curves into a LIST

roc_curves_again <- list(
  KNN = rocCurve.knn.auc,
  # NN = rocCurve.nn.auc,
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

options(repr.plot.width = 16, repr.plot.height = 10)
boxplot(roc_curves, 
       main = "ROC Curves",
       auto.key = list(columns = 3, space = "right"),
       col = c(1:14), 
       lty = 1, 
       lwd = 2, 
       las = 2,  
       xlab = "ML models", 
       ylab = "AUC", horizontal = TRUE)

# MODEL COMPARISONS :

# ROC CURVES : , main = "ROC Curves"

plot(rocCurve.knn, col = c(1), main = "ROC Curves")
# plot(rocCurve.nn, add = TRUE, col = c(2))
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
                  # "Naive Bayes",
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

############################################
############################################

# We run the algorithms through the following datasets :

# training
# testing

# dim(X_train)
# dim(X_test)
# dim(x_train)
# dim(x_test)
# length(Y_train)
# length(Y_test)
# length(y_train)
# length(y_test) 

# training2_mf
# testing2_mf

# dim(X_train_mf)
# dim(X_test_mf)
# dim(x_train_mf)
# dim(x_test_mf)
# length(Y_train_mf)
# length(Y_test_mf)
# length(y_train_mf)
# length(y_test_mf) 

# testing3_mice
# training3_mice

# dim(X_train_mice)
# dim(X_test_mice)
# dim(x_train_mice)
# dim(x_test_mice)
# length(Y_train_mice)
# length(Y_test_mice)
# length(y_train_mice)
# length(y_test_mice)

############################################
############################################

# RANDOM FORESTS :

library(randomForest)
library(missForest)
set.seed(123)

# Random Forests are an ensemble learning method that combines multiple decision trees to make predictions. 
# Each tree in the forest is trained on a different random subset of the training data and features, 
# resulting in diverse individual models. The final prediction is made by aggregating the predictions 
# of all the trees, either through majority voting for classification tasks or averaging for regression tasks.
 
# Comments about Random Forests
# ntree - ntree by default is 500 trees.
# mtry - variables randomly samples as candidates at each split.

# 1. Draw ntree bootstrap samples.
# 2. For each bootstrap, grow an un-pruned tree by choosing 
# the best split based on a random sample of mtry predictors at each node
# 3. Predict new data using majority votes for classification 
# and average for regression based on ntree trees.



# IMPUTATION has been done with missForrest
set.seed(123)

# We prefer to use TRAINING2
options(repr.plot.width = 10, repr.plot.height=8)

# Step I : Data Preparation : 
# it has been described above

# Step II : Run the random forest model
rf_model <- randomForest(RESULT ~ ., data = training2_mf, ntree = 500)

# The number of variables selected at each split is denoted by mtry in randomforest function.

mtry <- tuneRF(training2_mf[, c(1:index_predictors)], 
               training2_mf$RESULT, 
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
                         data = training2_mf, 
                         ntree = 500, 
                         na.action = na.pass,
                         mtry = best.m, 
                         importance=TRUE)

print(rf_model)

options(repr.plot.width = 10, repr.plot.height = 10)
plot(rf_model)

predictions <- predict(rf_model, testing2_mf)

# Evaluate the model
conf_matrix <- table(predictions, testing2_mf$RESULT)
conf_matrix

# Compute precision (Positive Predictive Value)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])

# Compute recall (Sensitivity)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Compute F1 score
f1_score_rf_mf <- 2 * (precision * recall) / (precision + recall)

cat("Precision (Positive Predictive Value):", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score_rf_mf, "\n")

# Evaluate variable importance
options(repr.plot.width = 10, repr.plot.height=8)
importance(rf_model)

options(repr.plot.width = 18, repr.plot.height=8)
varImpPlot(rf_model)

# perhaps not advisable : MeanDecreaseAccuracy
# vip::vip(rf_model)
# str(importance(rf_model))

# VARIABLE IMPORTANCE
rf_model_predictor_importance = as.data.frame(importance(rf_model))
rf_model_predictor_importance

# “The first measure is computed from permuting OOB data: 
# For each tree, the prediction error on the out-of-bag portion of the data is recorded 
# (error rate for classification, MSE for regression). 
# Then the same is done after permuting each predictor variable. 
# The difference between the two are then averaged over all trees, 
# and normalized by the standard deviation of the differences.”

# “The second measure is the total decrease in node impurities from splitting on the variable, 
# averaged over all trees. For classification, the node impurity is measured by the Gini index. 
# For regression, it is measured by residual sum of squares.”

# randomForest::importance(rf_model, type = 1)
# randomForest::importance(rf_model, type = 2)

# Higher the value of mean decreases the gini score , 
# higher the importance of the variable in the model
set.seed(123)

# We prefer to use TRAINING that generated with CARET
set.seed(123)

# We prefer to use TRAINING that generated with CARET
options(repr.plot.width = 10, repr.plot.height=8)

# Step I : Data Preparation
# it has been described above

# Step II : Run the random forest model
rf_model <- randomForest(RESULT ~ ., data = training, ntree = 500)

# The number of variables selected at each split is denoted by mtry in randomforest function.

mtry <- tuneRF(training[, c(1:index_predictors)], 
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
                         na.action = na.pass,
                         mtry = best.m, 
                         importance=TRUE)

print(rf_model)
options(repr.plot.width = 10, repr.plot.height=8)
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
f1_score_rf <- 2 * (precision * recall) / (precision + recall)

cat("Precision (Positive Predictive Value):", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score_rf, "\n")

# Evaluate variable importance
options(repr.plot.width = 2, repr.plot.height=2)
importance(rf_model)

options(repr.plot.width = 18, repr.plot.height=8)
varImpPlot(rf_model)

# perhaps not advisable : MeanDecreaseAccuracy
# vip::vip(rf_model)
# str(importance(rf_model))

# VARIABLE IMPORTANCE
rf_model_predictor_importance = as.data.frame(importance(rf_model))
rf_model_predictor_importance

# USING the data imputed with MICE
set.seed(123)

# We prefer to use TRAINING that generated with CARET
options(repr.plot.width = 10, repr.plot.height=8)

# Step I : Data Preparation
# it has been shown above

# Step II : Run the random forest model
rf_model <- randomForest(RESULT ~ ., data = training3_mice, ntree = 500)

# The number of variables selected at each split is denoted by mtry in randomforest function.

mtry <- tuneRF(training3_mice[, c(1:index_predictors)], 
               training3_mice$RESULT, 
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
                         data = training3_mice, 
                         ntree = 500, 
                         na.action = na.pass,
                         mtry = best.m, 
                         importance=TRUE)

print(rf_model)
options(repr.plot.width = 10, repr.plot.height=8)
plot(rf_model)

predictions <- predict(rf_model, testing3_mice)

# Evaluate the model
conf_matrix <- table(predictions, testing3_mice$RESULT)
conf_matrix

# Compute precision (Positive Predictive Value)
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])

# Compute recall (Sensitivity)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])

# Compute F1 score
f1_score_rf_mice <- 2 * (precision * recall) / (precision + recall)

cat("Precision (Positive Predictive Value):", precision, "\n")
cat("Recall (Sensitivity):", recall, "\n")
cat("F1 Score:", f1_score_rf_mice, "\n")

# Evaluate variable importance
options(repr.plot.width = 2, repr.plot.height=2)
importance(rf_model)

options(repr.plot.width = 18, repr.plot.height=8)
varImpPlot(rf_model)

# perhaps not advisable : MeanDecreaseAccuracy
# vip::vip(rf_model)
# str(importance(rf_model))

# VARIABLE IMPORTANCE
rf_model_predictor_importance = as.data.frame(importance(rf_model))
rf_model_predictor_importance

# printing the F1 scores :

f1_score_rf
f1_score_rf_mf
f1_score_rf_mice

############################################
############################################

# XGBOOST MODELS :

# in the code below, we encode "low" / "small" as 0, and "high" as 1

# y_train <- as.integer(training$RESULT) - 1
# y_test <- as.integer(testing$RESULT) - 1
# head(y_train,20)
# head(y_test,20)

# we can run XGBOOST on multiple training and testing datasets that have been obtained 
# with 3 distinct imputation methods :

# training
# testing

# dim(X_train)
# dim(X_test)
# dim(x_train)
# dim(x_test)
# length(Y_train)
# length(Y_test)
# length(y_train)
# length(y_test) 

# training2_mf
# testing2_mf

# dim(X_train_mf)
# dim(X_test_mf)
# dim(x_train_mf)
# dim(x_test_mf)
# length(Y_train_mf)
# length(Y_test_mf)
# length(y_train_mf)
# length(y_test_mf) 

# testing3_mice
# training3_mice

# dim(X_train_mice)
# dim(X_test_mice)
# dim(x_train_mice)
# dim(x_test_mice)
# length(Y_train_mice)
# length(Y_test_mice)
# length(y_train_mice)
# length(y_test_mice)

# the IMPUTATION was performed in CARET
set.seed(123)

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

options(repr.plot.width = 16, repr.plot.height = 12)
xgb.plot.importance(importance_matrix)


# in order to check the model and to visualize the tree
# xgb.dump(xgb_model, with_stats = TRUE)
# xgb.plot.tree(model = xgb_model)

# in order to compute F1 SCORE : 

# converting the LABELS "high" and "low" into BINARY 1 or 0 for the TESTING DATASET
testing$RESULT_binary <- ifelse(testing$RESULT == 'high', 1, 0)
head(testing, 5)

# Make predictions on the test set
predictions <- predict(xgb_model, newdata = xgb_test)
head(predictions, 2)

# Convert predicted values to class labels :
# predicted_labels <- ifelse(predictions > 0.5, "high", "low") 
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

# Create a ROC curve
xgb_roc <- roc(testing$RESULT, predicted_values)
# xgb_roc
options(repr.plot.width = 8, repr.plot.height=8)
plot(xgb_roc) 

# Calculate AUC
auc_score <- auc(xgb_roc)
cat("AUC score:", auc_score, "\n")
auc_score

############################################
############################################

# the IMPUTATION was carried in missForrest
set.seed(123)

# We prefer to use TRAINING that generated with missForrest
options(repr.plot.width = 10, repr.plot.height=8)

xgb_train_mf <- xgb.DMatrix(data = as.matrix(X_train_mf), label = Y_train_mf)
xgb_test_mf <- xgb.DMatrix(data = as.matrix(X_test_mf), label = Y_test_mf)

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

xgb_model_mf <- xgb.train(
    data = xgb_train_mf,
    params = xgb_params,
    nrounds = 5000,
    # watchlist = watchlist, 
    verbose = 1
)

summary(xgb_model_mf)

importance_matrix_mf <- xgb.importance(
  feature_names = colnames(xgb_train_mf), 
  model = xgb_model_mf
)

importance_matrix_mf

options(repr.plot.width = 18, repr.plot.height = 14)
xgb.plot.importance(importance_matrix_mf)

############################################
############################################

# in order to compute F1 SCORE : 

# converting the LABELS "high" and "low" into BINARY 1 or 0 for the TESTING DATASET
testing2_mf$RESULT_binary <- ifelse(testing2_mf$RESULT == 'high', 1, 0)
head(testing2_mf, 5)

# Make predictions on the test set
predictions_mf <- predict(xgb_model_mf, newdata = xgb_test_mf)
head(predictions_mf, 2)

# Convert predicted values to class labels :
# predicted_labels <- ifelse(predictions > 0.5, "high", "low") 
# predicted_labels = factor(predicted_labels, levels = levels(testing$RESULT))
# predicted_labels <- as.factor(predicted_labels)

# Evaluate the model :
# confusionMatrix_xgboost = confusionMatrix(predicted_labels, testing$RESULT,  mode = "everything")
# confusionMatrix_xgboost
# head(predicted_labels, 2)
# head(testing$RESULT, 2)

# Convert predicted values to binary 0 or 1
predicted_values_mf <- ifelse(predictions_mf > 0.5, 1, 0)
head(predicted_values_mf, 2)

# Confusion Matrix :
confusionMatrix_xgboost_mf = confusionMatrix(as.factor(predicted_values_mf), 
                                          as.factor(testing2_mf$RESULT_binary),  mode = "everything")
confusionMatrix_xgboost_mf

# Extract F1 score
confusionMatrix_xgboost_F1_mf <- confusionMatrix_xgboost_mf$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_xgboost_F1_mf))

# Create a ROC curve
xgb_roc_mf <- roc(testing2_mf$RESULT, predicted_values_mf)
# xgb_roc
options(repr.plot.width = 8, repr.plot.height=8)
plot(xgb_roc_mf) 

# Calculate AUC
auc_score_mf <- auc(xgb_roc_mf)
cat("AUC score:", auc_score_mf, "\n")
auc_score_mf

############################################
############################################

# the IMPUTATION was performed in MICE
set.seed(123)

# We prefer to use TRAINING that generated with MICE

options(repr.plot.width = 10, repr.plot.height=8)

xgb_train_mice <- xgb.DMatrix(data = as.matrix(X_train_mice), label = Y_train_mice)
xgb_test_mice <- xgb.DMatrix(data = as.matrix(X_test_mice), label = Y_test_mice)

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

xgb_model_mice <- xgb.train(
    data = xgb_train_mice,
    params = xgb_params,
    nrounds = 5000,
    # watchlist = watchlist, 
    verbose = 1
)

summary(xgb_model_mice)

importance_matrix_mice <- xgb.importance(
  feature_names = colnames(xgb_train_mice), 
  model = xgb_model_mice
)

importance_matrix_mice

options(repr.plot.width = 18, repr.plot.height = 14)
xgb.plot.importance(importance_matrix_mice)

############################################
############################################

# in order to compute F1 SCORE : 

# converting the LABELS "high" and "low" into BINARY 1 or 0 for the TESTING DATASET
testing3_mice$RESULT_binary <- ifelse(testing3_mice$RESULT == 'high', 1, 0)
head(testing3_mice, 5)

# Make predictions on the test set
predictions_mice <- predict(xgb_model_mice, newdata = xgb_test_mice)
head(predictions_mice, 2)

# Convert predicted values to class labels :
# predicted_labels <- ifelse(predictions > 0.5, "high", "low") 
# predicted_labels = factor(predicted_labels, levels = levels(testing$RESULT))
# predicted_labels <- as.factor(predicted_labels)
# Evaluate the model :
# confusionMatrix_xgboost = confusionMatrix(predicted_labels, testing$RESULT,  mode = "everything")
# confusionMatrix_xgboost
# head(predicted_labels, 2)
# head(testing$RESULT, 2)

# Convert predicted values to binary 0 or 1
predicted_values_mice <- ifelse(predictions_mice > 0.5, 1, 0)
head(predicted_values_mice, 2)

# Confusion Matrix :
confusionMatrix_xgboost_mice = confusionMatrix(as.factor(predicted_values_mice), 
                                          as.factor(testing3_mice$RESULT_binary),  mode = "everything")
confusionMatrix_xgboost_mice

# Extract F1 score
confusionMatrix_xgboost_F1_mice <- confusionMatrix_xgboost_mice$byClass["F1"]
print(paste("F1 Score:", confusionMatrix_xgboost_F1_mice))

# Create a ROC curve
xgb_roc_mice <- roc(testing3_mice$RESULT, predicted_values_mice)
# xgb_roc
options(repr.plot.width = 8, repr.plot.height=8)
plot(xgb_roc_mice) 

# Calculate AUC
auc_score_mice <- auc(xgb_roc_mice)
cat("AUC score:", auc_score_mice, "\n")
auc_score_mice

# a summary of the F1 scores :
confusionMatrix_xgboost_F1 
confusionMatrix_xgboost_F1_mf
confusionMatrix_xgboost_F1_mice

############################################
############################################

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

############################################
############################################

# BORUTA

library("Boruta")
set.seed(123)

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

tentative_vars <- fd[which(fd == "Tentative")] %>% names
cat("tentative variables : ")
print(tentative_vars)

rejected_vars <- fd[which(fd == "Rejected")] %>% names
cat("rejected variables : ")
print(rejected_vars)

############################################
############################################

# FEATURE ELIMINATION
set.seed(123)

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

############################################
############################################

# In this section, we base our analysis on the dataset imputed with CARET

# performing RFE (recursive feature elimination)

# Define the control parameters : 

set.seed(123)

ctrl <- rfeControl(functions = rfFuncs, # Random forest is used as the feature selection method
                   method = "cv",       # Cross-validation is used for resampling
                   number = 10)         # Number of variables to select

# Run the feature selection : 

model_rfe <- rfe(x_train,         # Remove the response variable from the predictor set
              y_train,            # Response variable
              sizes = c(1:10),    # Range of variable sizes to consider
              rfeControl = ctrl)  # Use the control parameters defined above

model_rfe
options(repr.plot.width = 8, repr.plot.height=8)
plot(model_rfe, type = c("g", "o"))

# Get the selected features 
selected_features <- predictors(model_rfe)
print(selected_features)

# For subsequent analysis, to use the matrix of the relevant features
# x2_train = x_train[, selected_features] # training data: selected features
# x2_test = x_test[, selected_features] # test data: selected features
# head(x2_train, 2)
# head(x2_test, 2)

############################################
############################################

# Simulated Annealing
set.seed(123)

# Simulated Annealing
set.seed(123)

# Define the control parameters
sa_ctrl <- safsControl(functions = rfSA,           # Assess fitness with RF
                       method = "repeatedcv",      # 10 fold cross validation
                       repeats = 5,
                       improve = 50)


LEVELS <- c("high","low")     # Set the levels

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

# For subsequent analysis, to use the matrix of the relevant features
# x2_train = x_train[, final] # training data: selected features
# x2_test = x_test[, final] # test data: selected features
# head(x2_train, 2)
# head(x2_test, 2)

############################################
############################################

# Genetic Alghoritms
set.seed(123)

# Genetic Alghoritms
set.seed(123)

# Define the control parameters
ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel=TRUE, # Use parallel programming
                       allowParallel = TRUE)

# Set the levels
LEVELS <- c("high","low")     
# LEVELS = levels(factor(student3$RESULT)) 
LEVELS 

# Compute 
ga_compute <- gafs(x = x_train, 
                   y = y_train,
                   iters = 20, # 100 generations of algorithm
                   popSize = 20, # population size for each generation
                   levels = LEVELS,
                   gafsControl = ga_ctrl)


# The plot also shows the average internal accuracy estimates as well as the average external estimates 
# calculated from the 10 out of sample predictions.

options(repr.plot.width = 10, repr.plot.height=8)
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

############################################
############################################

# using a SUPER-LEARNER

library(SuperLearner)
# using SUPER LEARNERS
set.seed(123)
# available models
# listWrappers()
# to choose between these models, among other models :
# "SL.knn"              
# "SL.ksvm" 
# "SL.svm"   
# "SL.lda"             
# "SL.logreg"                    
# "SL.nnls"                         
# "SL.randomForest"     
# "SL.ranger"                     
# "SL.rpart"  
# "SL.gbm"         
# "SL.xgboost" 
# "SL.bayesglm" ### an implementation of logistic regression.

# we work with the following training and testing datasets

# training
# testing

# dim(X_train)
# dim(X_test)
# dim(x_train)
# dim(x_test)
# length(Y_train)
# length(Y_test)
# length(y_train)
# length(y_test) 

# training2_mf
# testing2_mf

# dim(X_train_mf)
# dim(X_test_mf)
# dim(x_train_mf)
# dim(x_test_mf)
# length(Y_train_mf)
# length(Y_test_mf)
# length(y_train_mf)
# length(y_test_mf) 

# testing3_mice
# training3_mice

# dim(X_train_mice)
# dim(X_test_mice)
# dim(x_train_mice)
# dim(x_test_mice)
# length(Y_train_mice)
# length(Y_test_mice)
# length(y_train_mice)
# length(y_test_mice)

# table(y_train)
# table(Y_train) # it is binary 

# table(y_train_mice)
# table(Y_train_mice) # it is binary 

# table(y_train_mf)
# table(Y_train_mf) # it is binary 

names(y_train)= "RESULT"
names(y_test)= "RESULT"
names(Y_train)= "RESULT"
names(Y_test)= "RESULT"

names(y_train_mice) = "RESULT"
names(Y_train_mice) = "RESULT"
names(y_test_mice) = "RESULT"
names(Y_test_mice) = "RESULT"


LEARNER_LIBRARY = c(
"SL.knn",              
"SL.ksvm", 
"SL.svm",   
"SL.lda",             
# "SL.logreg",                    
"SL.nnls",                         
"SL.randomForest",     
"SL.ranger",  # a faster version of RF                   
"SL.rpart",  
"SL.gbm",         
"SL.xgboost", 
"SL.bayesglm")

# for optimization
# "method.NNLS"
# "method.AUC"
# family to describe the error distribution
# gaussian 
# binomial 

# SuperLearner is an algorithm that uses cross-validation to estimate the performance of 
# multiple machine learning models, or the same model with different settings. 
# It then creates an optimal weighted average of those models, 
# which is also called an “ensemble”, using the test data performance.

# 1) All models must be trained on the same training set.
# 2) All models must be trained with the same number of CV folds.

# CaretEnsemble (Deane-Mayer and Knowles 2016), also provides an approach for stacking, 
# but it implements a bootsrapped (rather than cross-validated) version of stacking. 
# The bootstrapped version will train faster since bootsrapping (with a train/test set) 
# requires a fraction of the work of k-fold CV; however, the the ensemble performance often 
# suffers as a result of this shortcut.

# It is important to remember, that the best ensembles are not composed of the best performing algorithms, 
# but rather the algorithms that best complement each other to classify a prediction.

library("arm")
library("LogicReg")
set.seed(123)

SL = SuperLearner(Y = Y_train_mf, 
                      X = X_train_mf, 
                      family = binomial(),
                      # family = gaussian(),    # it is the default
                      # method = "method.NNLS", # it is the default
                      verbose = TRUE, 
                      cvControl = list(V = 10),
                      SL.library = LEARNER_LIBRARY)

SL
# Some models have a coefficient of zero, which means that it is not weighted as part of the ensemble anymore. 
# SuperLearner is calculating this risk for you and deciding on the optimal model mix that will reduce the error.

# Get V-fold cross-validated risk estimate
SL.cv.model <- CV.SuperLearner(Y = Y_train_mf,
                               X = X_train_mf,
                               family = binomial(),
                               V = 10, # V: The number of folds for ‘CV.SuperLearner’.
                               SL.library = LEARNER_LIBRARY)

# Print out the summary statistics
# The table shows the average risk estimate (Ave), the standard error (se), 
# the minimum risk estimate (Min), and the maximum risk estimate (Max) for each algorithm.

summary(SL.cv.model)

plot(SL.cv.model)

# Gather predictions for the tuned model
# predictions.tune = predict.SuperLearner(SL, newdata = as.matrix(X_test_mf), onlySL = TRUE)
# predictions.tune = predict.SuperLearner(SL, newdata = X_test_mf)

# Recode predictions
# conv.preds.tune <- ifelse(predictions.tune$pred>=0.5,1,0)

# Return the confusion matrix
# confusionMatrix(conv.preds.tune, y_test_mf)

# a function : create.Learner()

############################################
############################################