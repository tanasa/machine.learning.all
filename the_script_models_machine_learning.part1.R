###############################################################################
###############################################################################
# To compute the missingness 

library("tidyverse")
library("mice")
library("ggmice")
library("VIM")
library("lattice")

FILE="matrix.genome.instability.and.drugs.txt"
z = read.delim(FILE, header=T, sep="\t", stringsAsFactors = FALSE)

# separate the predictors
# compute size of the matrix
# compute the missingness

dim(z)
head(z,2)

# Extract the indexes of the predictors in the DATA FRAME (before the column wes.ProfileID)

column_name <- "wes.ProfileID"
index_of_column <- which(colnames(z) == column_name)
print(paste("Index of column '", column_name, "': ", index_of_column))

# Selecting the MATRIX with the PREDICTORS
drugs = z[, c(3:index_of_column-1)]
head(drugs, 2)

cat("the dimensions of the DRUG MATRIX :")
# dimensions of the MATRIX
dim(drugs)
# LENGTH
cat("the dimensions of the DRUG MATRIX : LENGTH")
dim(drugs)[1]
# WIDTH
cat("the dimensions of the DRUG MATRIX : WIDTH")
dim(drugs)[2]

cat("")
cat("")
cat("COMPUTE the MISSINGNESS in the MATRIX of DRUGS :")
cat("")
cat("")
cat("the missingness in the entire DRUG MATRIX :")
# dimensions of the MATRIX
total_missing <- sum(is.na(drugs))
print(total_missing)

# COLUMN
cat("")
cat("")
cat("the misingness in the DRUG MATRIX : per COLUMN")
missing_per_column <- colSums(is.na(drugs))
print(missing_per_column)
cat("")
cat("")

# ROW
cat("")
cat("")
cat("the dimensions of the DRUG MATRIX : per ROW ")
missing_per_row <- rowSums(is.na(drugs))
print(missing_per_row)
cat("")
cat("")

# Compute the percentage of missing values in the entire matrix
cat("")
cat("")
percentage_missing <- mean(is.na(drugs)) * 100
cat("")
cat("")

# Print the result
cat("Percentage of missing values in the entire matrix:", percentage_missing, "%\n")
cat("Another way to compute the percentage of missing values in the entire matrix:", percentage_missing, "%\n")
total_missing / (dim(drugs)[1] * dim(drugs)[2])

# Displaying the MISSING DATA 
# mice::md.pattern(drugs)

options(repr.plot.width = 20, repr.plot.height=20)
plot_pattern(drugs 
  # square = TRUE,
  # rotate = TRUE
)

options(repr.plot.width = 40, repr.plot.height=20)
visdat::vis_dat(drugs, palette = "default") +
        theme(axis.text.x = element_blank(), axis.title.x = element_blank())

aggr(drugs, numbers=TRUE, sortVars=TRUE, labels=names(drugs), cex.axis=.7, gap=3, 
     ylab=c("Proportion of missingness", "Missingness Pattern"))

###############################################################################
###############################################################################

FILE = "missingness.m10.col.txt"

m10_COLUMN = read.delim(FILE, header=T, sep="\t", stringsAsFactors = FALSE)
dim(m10_COLUMN)
head(m10_COLUMN,2)

# Extract the indexes of the predictors in the DATA FRAME (before the column wes.ProfileID)
column_name <- "wes.ProfileID"
index_of_column <- which(colnames(m10_COLUMN) == column_name)
print(paste("Index of column '", column_name, "': ", index_of_column))

# Selecting the MATRIX with the PREDICTORS
drugs = m10_COLUMN[, c(3:index_of_column-1)]
head(drugs, 2)

cat("the dimensions of the DRUG MATRIX :")
# dimensions of the MATRIX
dim(drugs)
# LENGTH
cat("the dimensions of the DRUG MATRIX : LENGTH")
dim(drugs)[1]
# WIDTH
cat("the dimensions of the DRUG MATRIX : WIDTH")
dim(drugs)[2]

cat("the missingness in the entire DRUG MATRIX :")
# dimensions of the MATRIX
total_missing <- sum(is.na(drugs))
print(total_missing)

# COLUMN
cat("the misingness in the DRUG MATRIX : per COLUMN")
missing_per_column <- colSums(is.na(drugs))
print(missing_per_column)
# ROW
cat("the dimensions of the DRUG MATRIX : per ROW ")
missing_per_row <- rowSums(is.na(drugs))
print(missing_per_row)

# Compute the percentage of missing values in the entire matrix
percentage_missing <- mean(is.na(drugs)) * 100

# Print the result
cat("Percentage of missing values in the entire matrix:", percentage_missing, "%\n")
cat("Another way to compute the percentage of missing values in the entire matrix:", percentage_missing, "%\n")
total_missing / (dim(drugs)[1] * dim(drugs)[2])

options(repr.plot.width = 4, repr.plot.height=4)
aggr(drugs, numbers=TRUE, sortVars=TRUE, labels=names(drugs), cex.axis=.7, gap=3, 
     ylab=c("Proportion of missingness","Missingness Pattern"))

###############################################################################
###############################################################################

FILE="missingness.m10.row.txt"
m10_ROW = read.delim(FILE, header=T, sep="\t", stringsAsFactors = FALSE)
dim(m10_ROW)
head(m10_ROW,2)

# Extract the indexes of the predictors in the DATA FRAME (before the column wes.ProfileID)
column_name <- "wes.ProfileID"
index_of_column <- which(colnames(m10_ROW) == column_name)
print(paste("Index of column '", column_name, "': ", index_of_column))

# Selecting the MATRIX with the PREDICTORS
drugs = m10_ROW[, c(3:index_of_column-1)]
head(drugs, 2)

cat("the dimensions of the DRUG MATRIX :")
# dimensions of the MATRIX
dim(drugs)
# LENGTH
cat("the dimensions of the DRUG MATRIX : LENGTH")
dim(drugs)[1]
# WIDTH
cat("the dimensions of the DRUG MATRIX : WIDTH")
dim(drugs)[2]

cat("the missingness in the entire DRUG MATRIX :")
# dimensions of the MATRIX
total_missing <- sum(is.na(drugs))
print(total_missing)

# COLUMN
cat("the misingness in the DRUG MATRIX : per COLUMN")
missing_per_column <- colSums(is.na(drugs))
print(missing_per_column)
# ROW
cat("the dimensions of the DRUG MATRIX : per ROW ")
missing_per_row <- rowSums(is.na(drugs))
print(missing_per_row)

# Compute the percentage of missing values in the entire matrix
percentage_missing <- mean(is.na(drugs)) * 100

# Print the result
cat("Percentage of missing values in the entire matrix:", percentage_missing, "%\n")
cat("Another way to compute the percentage of missing values in the entire matrix:", percentage_missing, "%\n")
total_missing / (dim(drugs)[1] * dim(drugs)[2])

options(repr.plot.width = 12, repr.plot.height=12)
aggr(drugs, numbers=TRUE, sortVars=TRUE, labels=names(drugs), cex.axis=.7, gap=3, 
     ylab=c("Proportion of missingness","Missingness Pattern"))

###############################################################################
###############################################################################

# Looking into data frames with less number of cells and more drugs

FILE="wgs.wes.drug.info.txt"
z = read.delim(FILE, header=T, sep="\t", stringsAsFactors = FALSE)

dim(z)
head(z, 2)

# Extract the indexes of the predictors in the DATA FRAME (before the column wes.ProfileID)
column_name <- "wes.ProfileID"
index_of_column <- which(colnames(z) == column_name)
print(paste("Index of column '", column_name, "': ", index_of_column))

# Selecting the MATRIX with the PREDICTORS
DRUGS = z[, c(3:index_of_column-1)]
head(DRUGS, 2)

df = DRUGS
dim(df)

###############################################################################
###############################################################################

# COUNTING the number of ROWS that have a percent of MISSINGNESS less than ...
dim(df)

######################################################### 
######################################################### 10 %
# Set the threshold for missing values (%)
percent = 0.1
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 20 %
# Set the threshold for missing values (%)
percent = 0.2
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))

######################################################### 
######################################################### 30 %
# Set the threshold for missing values (%)
percent = 0.3
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 40 %
# Set the threshold for missing values (%)
percent = 0.4
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 50 %
# Set the threshold for missing values (%)
percent = 0.5
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 60 %
# Set the threshold for missing values (%)
percent = 0.6
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 70 %
# Set the threshold for missing values (%)
percent = 0.7
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 80 %
# Set the threshold for missing values (%)
percent = 0.8
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 90 %
# Set the threshold for missing values (%)
percent = 0.9
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))
######################################################### 
######################################################### 100 %
# Set the threshold for missing values (%)
percent = 1
threshold <- ncol(df) * percent
# Filter rows with less than Xmissing values
selected_rows <- df[rowSums(is.na(df)) <= threshold, ]
# Print the result
cat("number of rows with NA less than :", percent )
dim(selected_rows)
sum(is.na(selected_rows))


###############################################################################
###############################################################################

# COUNTING the number of COLUMNS that have a percent of MISSINGNESS less than ...

# REVISITING the MISSINGNESS : number of COLUMNS with less than %NA

##########################################################
########################################################## 0.1
# Set the threshold for missing values (%)
percent <- 0.1
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.2
# Set the threshold for missing values (%)
percent <- 0.2
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.3
# Set the threshold for missing values (%)
percent <- 0.3
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.4
# Set the threshold for missing values (%)
percent <- 0.4
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.5
# Set the threshold for missing values (%)
percent <- 0.5
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.6
# Set the threshold for missing values (%)
percent <- 0.6
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.7
# Set the threshold for missing values (%)
percent <- 0.7
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.8
# Set the threshold for missing values (%)
percent <- 0.8
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 0.9
# Set the threshold for missing values (%)
percent <- 0.9
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

##########################################################
########################################################## 1
# Set the threshold for missing values (%)
percent <- 1
threshold <- nrow(df) * percent

# Filter cols with less than X missing values
selected_cols <- df[, colSums(is.na(df)) <= threshold]

# Print the result
cat("Number of columns with NA less than:", percent)
cat("\n")
cat("number of cell lines :")
print(dim(selected_cols)[1])
cat("\n")
cat("number of drugs :")
print(dim(selected_cols)[2])
cat("\n")
cat("number of missing NA : ")
print(sum(is.na(selected_cols)))

###############################################################################
###############################################################################

# starting from the file with MAX 40 missingness on the COLUMNS 
# in order to obtain another file with MAX 40 missingness on the COLUMNS, 
# and MAX 20 missingness on the ROWS .

FILE = "missingness.m40_COLUMN.txt"

m40_COLUMN = read.delim(FILE, header=T, sep="\t", stringsAsFactors = FALSE)
dim(m40_COLUMN)
head(m40_COLUMN,2)

# Extract the indexes of the predictors in the DATA FRAME (before the column wes.ProfileID)
column_name <- "wes.ProfileID"
index_of_column <- which(colnames(m40_COLUMN) == column_name)
print(paste("Index of column '", column_name, "': ", index_of_column))

# Selecting the MATRIX with the PREDICTORS
drugs = m40_COLUMN[, c(3:index_of_column-1)]
head(drugs, 2)
dim(drugs)

# index of the column that separates drug information from metadata
index_of_column

my_matrix = m40_COLUMN
dim(m40_COLUMN)

col_X <- 2
col_Y <- index_of_column - 1

# Calculate missingness for each row
missingness_row <- rowSums(is.na(my_matrix[, col_X : col_Y])) / dim(my_matrix[, col_X : col_Y])[2] 

# Set a threshold for missingness (e.g., 0.2) for each ROW : 0.2
threshold <- 0.2

# Select rows with missingness less than the threshold
selected_rows_m40_COLUMN_m20_ROW <- my_matrix[missingness_row < threshold, ]

# Print the selected rows
head(selected_rows_m40_COLUMN_m20_ROW, 3)
dim(selected_rows_m40_COLUMN_m20_ROW[, col_X : col_Y])

# missingness_row
# dim(my_matrix[, col_X : col_Y])[1]
# dim(my_matrix[, col_X : col_Y])[2]

write.table(selected_rows_m40_COLUMN_m20_ROW, 
file = "missingness.m40_COLUMN.m20_ROW.txt",
sep = "\t", quote=FALSE, col.names=TRUE, row.names = FALSE)

# to verify the MATRIX that we have just created

m40_COLUMN_m20_ROW = selected_rows_m40_COLUMN_m20_ROW

# Extract the indexes of the predictors in the DATA FRAME (before the column wes.ProfileID)
column_name <- "wes.ProfileID"
index_of_column <- which(colnames(m40_COLUMN_m20_ROW) == column_name)
print(paste("Index of column '", column_name, "': ", index_of_column))

# Selecting the MATRIX with the PREDICTORS
drugs = m40_COLUMN_m20_ROW[, c(3:index_of_column-1)]
head(drugs, 2)

cat("the dimensions of the DRUG MATRIX :")
# dimensions of the MATRIX
dim(drugs)
# LENGTH
cat("the dimensions of the DRUG MATRIX : LENGTH")
dim(drugs)[1]
# WIDTH
cat("the dimensions of the DRUG MATRIX : WIDTH")
dim(drugs)[2]

cat("the missingness in the entire DRUG MATRIX :")
# dimensions of the MATRIX
total_missing <- sum(is.na(drugs))
print(total_missing)

# COLUMN
cat("the misingness in the DRUG MATRIX : per COLUMN")
missing_per_column <- colSums(is.na(drugs))
print(missing_per_column)
# ROW
cat("the dimensions of the DRUG MATRIX : per ROW ")
missing_per_row <- rowSums(is.na(drugs))
print(missing_per_row)

# Compute the percentage of missing values in the entire matrix
percentage_missing <- mean(is.na(drugs)) * 100

# Print the result
cat("Percentage of missing values in the entire matrix:", percentage_missing, "%\n")
cat("Another way to compute the percentage of missing values in the entire matrix:", percentage_missing, "%\n")
total_missing / (dim(drugs)[1] * dim(drugs)[2])

options(repr.plot.width = 14, repr.plot.height = 6)
aggr(drugs, numbers=TRUE, sortVars=TRUE, labels=names(drugs), cex.axis=.7, gap=3, 
     ylab=c("Proportion of missingness","Missingness Pattern"))

##############################################################################
##############################################################################