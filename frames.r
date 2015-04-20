
library(data.table)
library(reshape2)

# downloaded and unzipped data to r wd()
# 2 folders of interest 'test'/'train'
# 3 files in each folder 'subject_..', 'y_..', 'X_..'

#The final goal is a table with a column for the subjects(identified by number), activities(by name),
# a column for the mean & std for each measurement of each action (eg tBodyAcc,). 
# There are ~14 measurement types with about 3 mean and 3 std measurements each

#read paired files from each folder and concatenate
#subjects
subjectTrainIn <- read.table("UCI HAR Dataset/train/subject_train.txt")
subjectTestIn <- read.table("UCI HAR Dataset/test/subject_test.txt")
subjects <- rbind(subjectTrainIn,subjectTestIn)
names(subjects) <- "subjNum"

#activities
actionTrainIn <- read.table("UCI HAR Dataset/train/y_train.txt")
actionTestIn <- read.table("UCI HAR Dataset/test/y_test.txt")
action <- rbind(actionTrainIn,actionTestIn)
names(action) <- "action"

#action measurement data
measurementsTrainIn <- read.table("UCI HAR Dataset/train/X_train.txt")
measurementsTestIn <- read.table("UCI HAR Dataset/test/X_test.txt")
measurements <- rbind(measurementsTrainIn,measurementsTestIn)

#measurement names are in the features.txt file
measurementNames <- read.table("UCI HAR Dataset/features.txt")
#set measurement column names to be the name of the associated feature instead of generic V1-V561
names(measurements) <- measurementNames$V2

# merge on columns to bring all datasets together
complete <- cbind(subjects,action,measurements)

#EXTRACT ONLY COLS WITH MEAN OR STD MEASUREMENTS
# Find columns with means or stds. 
#getMeanStd is a logical that returns true if a column name contains 'mean' or 'std'
getMeanStd <- grepl("mean\\(\\)", names(complete)) |
  grepl("std\\(\\)", names(complete)) |
  grepl("subjNum", names(complete)) |
  grepl("action", names(complete))

#I kept loosing subjNum and Actitivity during merge so, I'm specifying that they stay in the getMeanStd
#There's probably a better way to do this

#Subset based on getMeanStd,get only cols with mean/std
complete <- complete[,getMeanStd]

#The 'action' column is still using numbers to identify activities, switch to the action names
# Treats as nominal variables. Change 1-->Walking, 2-->Walking Upstairs, etc
complete$action <- factor(complete$action, labels=c("Walking",
                                                        "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"))

# Final Step, Create new tidy set with only avgs
# melt() defaults all columns with numeric values to being called "variables" with "values". Make var-name descriptive
molten <- melt(complete, id.vars=c("subjNum","action"),
               variable.name = "feature")

tidy <- dcast(molten, subjNum + action ~ feature, mean)

#write to Long form tidy data
write.csv(molten, "longTidy.csv", row.names=FALSE)
#write to Wide form tidy data
write.csv(tidy, "wideTidy.csv", row.names=FALSE)
