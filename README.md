# This project is performed by Bana Zeleke
# Required packages: AppliedPredictiveModeling,caret,corrplot,FactoMineR,pls
# Library the required packages
lapply(c('AppliedPredictiveModeling','caret','corrplot','earth'),require,character.only = TRUE)

#Convert CarEconomy.xlsx to data frame
CarEconomy <- data.frame(CarEconomy)
# check the internal structure of the dataset
str(CarEconomy) 

# 1 : Examine the CarEconomy dataset for any missing values
# use colSums() to check if there are any missing values
sum(colSums(is.na(CarEconomy)))

# 2 create the predictors and outcome subset respectively
med <- CarEconomy[,1]
predictors <- CarEconomy[,-1]

# 3 Data Splitting: Split the dataset into training (75%) and test set (25%).
set.seed(0) 
training <- createDataPartition(med,p=0.75,list=FALSE)
predictors_train <- predictors[training,]
med_train <- med[training]
predictors_test <- predictors[-training,]
med_test <- med[-training]

# view the summary statistics of the predictors in the training set
summary(predictors_train)

# 4 Examine the training data for near-zero variance predictors and highly correlated predictors 
# visualize the training data to check the relationship between the outcome and randomly selected predictors- linear or not?
par(mfrow = c(2,2)) 
plot(predictors_train[,1],med_train,col="blue")
plot(predictors_train[,3],med_train,col="green")
plot(predictors_train[,5],med_train,col="yellow")
plot(predictors_train[,10],med_train,col="red")

# check if there's any near-zero variance predictors 
nzv <- nearZeroVar(predictors_train)

# use findCorrelation() to check predictors that have high correlations (using threshold value 0.9). If there are any found, then remove them from both the training and test set.
tooHigh <- findCorrelation(cor(predictors_train),.9)
predictors_trainfiltered <- predictors_train[,-tooHigh]


# 5 : Build and evaluate three regression modelssimple linear model (LM), MARS model, and KNN model, respectively.
#  creating a control function using 10 fold cross validation resampling technique
set.seed(0)
ctrl <- trainControl(method = "cv", index = createFolds(med_train, returnTrain = TRUE))

# Build a linear regression model (lm)
set.seed(0)
lmFit <- train(x = predictors_trainfiltered, y = med_train,
               method = "lm",
               trControl = ctrl)

lmFit # cross-validated RMSE = 6.47

# check and interpret the coefficients for the fitted linear regression model 
summary(lmFit) 

# Build Multivariate Adaptive Regression Splines (MARS) regression model, where "degree" and "nprune" are the tuning parameters- "degree" determines degrees of variable interactions, "nprune" determine the number of terms to retain.
set.seed(0)
marsTune <- train(x = predictors_train, y = med_train,
                  method = "earth",
                  tuneGrid = expand.grid(degree = 1:2, nprune = 2:38),
                  trControl = ctrl)
marsTune 

# visualize the model performance resulting from different parameter values
plot(marsTune)

# varImp() is used to estimate the predictors importance in the fitted model
marsImp <- varImp(marsTune)
plot(marsImp, top = 6) 

# Build K-Nearest Neighbors (KNN) regression model, where "k" is the turning parameter.
set.seed(0)
knnTune <- train(x = predictors_train, y = med_train,
                 method = "knn",
                 preProc = c("center", "scale"),
                 tuneGrid = data.frame(k = 1:30),
                 trControl = ctrl)
knnTune # cross-validated RMSE = 36.14 (k=2)
# plot knnTune model, and see how RMSE change with different number of neighbors (k)
plot(knnTune)

# 6  Which regression model would you recommend? Use your chosen model to predict the test data and report the model performance. 
# start with making a list
modellist <- list(lm=lmFit,mars=marsTune,knn=knnTune)
# collect resamples from the cv folds
resamps <- resamples(modellist)
dotplot(resamps,metric = "RMSE")
bwplot(resamps, metric = "RMSE")


# Based on above evaluation and comparison,I would Choose marsTune.
# Use the chosen model to predict new data 
pred <- predict(marsTune,predictors_test)
# Use postResample() to report the model performance on the test data. 
postResample(pred,med_test) # test RMSE = 36.29













