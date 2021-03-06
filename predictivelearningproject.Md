
#PredictiveLearningProject
author: "Beau Norgeot"
date: "July 23, 2015"


## Quick Summary:
The goal here is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict, for unlabeled set of observations, which of type of 5 different exercises was performed. (The exercises were all a combination of dumbell lifts, some done correctly, some done intentionally incorrectly). 

The training data can be found here https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
More info on the project can be found here:http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset)

### Load the data and get a quick feel for what's going on
```
library(caret); library(dplyr)
inTrain <- read.table("pml-training.csv", sep = ",", header = TRUE)
View(inTrain)
names(inTrain)
summary(inTrain)
```

## Gross Data Munging
Need to clean up the initial set by removing variables with nearly zero variance, variables that have very amounts of NA, and variables that cannot possibly be useful for prediction and/or won't generalize well (names, id-numbers, anything specific to the exact data set that was collected that wouldn't be shared by other data sets of the same kind/type). 

```
# remove variables w/very high proportions of NA
highNA <- sapply(inTrain, function(x) mean(is.na(x))) > 0.90
inTrain1 <- inTrain[,highNA == FALSE] #down to 93 vars from 160
#how do this with dplyr ?
#dpNA <- inTrain %>% summarise_each(funs(mean(is.na(.)))) #this returns the NA means for each column. I'd like to pipe this into a filter function and just remove all columns above a certain value, but can't figure out how
#dpTrain1 <- inTrain[,dpNA == FALSE] #This also has 93 vars. Not sure why this works w/o specifying a threshold

#remove varialbes w/nearly zero variance
nzv <- nearZeroVar(inTrain1)
inTrain2 <- inTrain1[,-nzv] #down to 59 vars

#remove vars that aren't relevant for prediction
inTrain3 <- inTrain2 %>% select(-(1:5)) #54 vars now. Looks like a decent starting df
summary(inTrain3) #There's no NA's in the rest of the set. na.fail() could have been used
```

## Split Data into Training, Testing, and Validation sets
Separate into a trainSet to build and explore, testSet to estimate out-of-sample error, and ValidationSet that will only be touched once ever. In this case, the data that is provided and actually titled 'test' will be used as a validation set. I want to use as much of the training data as possible to build the best model.

```
inTrain4 <- createDataPartition(inTrain3$classe, p=0.8, list = FALSE) #classe is what I want to predict
trainSet <- inTrain3[inTrain4,]
testSet <- inTrain3[-inTrain4,]
```

## Feature Selection

There's a lot of features in this data set, and since I don't have much domain knowledge here, and none of the features jump out at me as being the of obvious importance, and since trying to compare conditional boxplots (which is what I would normally do w/all of these continous variables) would be overwhelming; I'm going to use a decision tree to help identify important features, then maybe I'll dig deeper.
In RF's and decision trees in general **variable importance is biased in favor of continuous variables, variables with many categories, and highly correlated variables** Highly correlated vars could be removed prior to varImp but, for now, I'm not going to bother.

```
featureMod <- train(classe ~., method = "rpart", data = trainSet)
impFeatures <- varImp(featureMod) #Seems like there's only about 15 features that are contributing here
#get top vars
topVars <- impFeatures$importance
topVars$Vars <- row.names(topVars)
trainFeatures <- topVars[order(-topVars$Overall),][1:15,] #-topVars$Overall to sort in descending order
trainFeatures <- trainFeatures$Vars

#Does an rf produce different vars? Must add importance = T to get variable importance in an RF
featureMod1 <- train(classe ~., method = "rf", data = trainSet, importance = TRUE,trControl=trainControl(method = "cv", number = 5))
```

## Model Training

Random Forests are often a great place to start, espetially since the goal here is just a blackbox prediction. There aren't any NA's left, but if there were NAs, I'd impute with knn. Note no importance = T here. 

```
model <- train(classe ~  pitch_forearm + roll_forearm + magnet_dumbbell_y + roll_belt + num_window + yaw_belt + accel_belt_z +     magnet_belt_y + total_accel_belt+magnet_arm_x + accel_arm_x + magnet_dumbbell_z + roll_dumbbell + magnet_dumbbell_x + accel_dumbbell_y,
               method = "rf", data = trainSet,trControl=trainControl(method = "cv", number = 5))
model$finalModel #500 trees, 8 vars at each split
```

## Model Evaluation

Generally, I would have tried some other models, however the accucracy of this first model was crazy high on the test set. I'm just going to move forward and evaluate it to see if the accuracy persists. 

```
preds <- predict(model, newdata=testSet)
# confusion matrix to estimate of out-of-sample error
confusionMatrix(testSet$classe, preds)
#accurcay = 99.62, so out-of-sample error estimate is .38%
```

The model accuracy is rediculously high and predicted out-of-sample error is incredibly low. Out-of-Sample error = 1 - accuracy. This is our best estimate for the error rate we will see when we apply our predictions to a data set that we haven't seen or trained on. **The confusionMatrix shows an accurcay = 99.62%, so that means that my out-of-sample error estimate is only 0.38%.** There's no need to build and compare other models. Since a realatively small number of features were given to the model, and it selected an even smaller subset, I have reasonably high hopes that the model will generalize well. 

### Retrain the model on the full data set. 
The selected model only got to train on the 'trainSet' data, which was only 80% of the available training data. Now I'll retrain the model by including all of the available training data (trainSet + testSet) and run the final predictions on the validation set. 


```
finalMod <- train(classe ~  pitch_forearm + roll_forearm + magnet_dumbbell_y + roll_belt + num_window + yaw_belt + accel_belt_z +     magnet_belt_y + total_accel_belt+magnet_arm_x + accel_arm_x + magnet_dumbbell_z + roll_dumbbell + magnet_dumbbell_x + accel_dumbbell_y,
               method = "rf", data = inTrain3, trControl=trainControl(method = "cv", number = 5))
```

## Make predictions on the provided test set, my validation set

```
pmlTest <- read.table("pml-testing.csv", sep = ",", header = TRUE)
finalPreds <- predict(finalMod, newdata = pmlTest)

# convert predictions to character vector
finalPreds <- as.character(finalPreds)

# build function to write each prediction to a seperate file
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

# use function to write the files
pml_write_files(finalPreds)
```

### Results
This model made predictions on the validation set that were 100% accurate (judged by coursera profs), perfect score. 