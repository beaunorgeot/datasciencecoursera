
setwd("~/datasciencecoursera/titanic/")
train <- read.csv("train.csv")
test <- read.csv("test.csv")


library(plyr)
library(dplyr)
library(rpart)

test$Survived <- NA
combi <- rbind(train, test)

# Engineering of Title 
combi$Name = as.character(combi$Name)
combi$Title<-sapply(combi$Name, FUN=function(x){strsplit(x, split='[,.]')[[1]][2]})
combi$Title<-sub(' ', '', combi$Title)
combi$Title[combi$Title=='Mlle']<-'Miss'
combi$Title[combi$Title %in% c('Mme', 'Ms')]<-'Mrs'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')]<-'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')]<-'Lady'
combi$Title<-factor(combi$Title)

# Engineering of FamilySize
combi$FamilySize <-combi$SibSp+combi$Parch

# Engineering of FamilyID
combi$Surname<-sapply(combi$Name, FUN=function(x){strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID<-paste(combi$Surname,as.character(combi$FamilySize), sep="_")

# Bagging of Alone people
famIDs<-data.frame(table(combi$FamilyID))
famIDsFreq1<-famIDs[famIDs$Freq==1,]
combi$FamilyID[combi$FamilyID %in% famIDsFreq1$Var1] <-'Alone'
# Bagging of Small Families (of 2) 
famIDsFreq2<-famIDs[famIDs$Freq==2,]
combi$FamilyID[combi$FamilyID %in% famIDsFreq2$Var1] <-'Small'

combi$FamilyID<-factor(combi$FamilyID)

# Completion of NA fields in Age
Agefit <- rpart(Age ~ Pclass + Sex +SibSp + Parch + Fare + Title + FamilySize,
                data=combi[!is.na(combi$Age),],
                method="anova")
# added FamilySize (people alone -> 20-30 ?)
combi$Age[is.na(combi$Age)]<-predict(Agefit, combi[is.na(combi$Age),])

combi$Embarked[which(combi$Embarked == '')]<-"S"
combi$Embarked<-factor(combi$Embarked)

combi$Fare[which(is.na(combi$Fare))]<- median(combi$Fare, na.rm=TRUE)

combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <=1] <- "Alone"
combi$FamilyID2[combi$FamilySize >1 & combi$FamilySize <= 2] <- "Small"
combi$FamilyID2 <- factor(combi$FamilyID2)


trainSet<-combi[1:891,]
testSet<-combi[892:1309,]

#trainSet
trainSet$Survived<-factor(trainSet$Survived, levels=c(0,1), labels=c("Died", "Survived"))
trainSet$Pclass<-factor(trainSet$Pclass, levels=c(1,2,3), labels=c("First class", "Second class", "Third class"))

#Now repeat on test set making sure same names are used as input
testSet$Pclass<-factor(testSet$Pclass, levels=c(1,2,3), labels=c("First class", "Second class", "Third class"))
testSet$Fare <- ifelse(is.na(testSet$Fare), mean(testSet$Fare, na.rm= TRUE), testSet$Fare)

# The variables to include in the models
#Sex,Age,Pclass,SibSp,Fare,Embarked, Title, FamilySize, FamilyID2, and SURVIVED OF COURSE
trainSet <- trainSet %>% select(Sex,Age,Pclass,SibSp,Fare,Embarked,Title,FamilySize,FamilyID2, Survived)
testSet <- testSet %>% select(-Survived)
