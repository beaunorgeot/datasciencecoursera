
# downloaded and unzipped data to r wd()
# 2 folders of interest 'test'/'train'
# 3 files in each folder 'subject_..', 'y_..', 'X_..'

#The final goal is a table with a column for the subjects(identified by number), activities(by name),
# a column for the mean & std for each measurement of each activity (eg tBodyAcc,). 
# There are ~14 measurement types with about 3 mean and 3 std measurements each

#read paired files from each folder and concatenate
#subjects
library(data.table)
subjectTrainIn <- read.table("UCI HAR Dataset/train/subject_train.txt")
subjectTestIn <- read.table("UCI HAR Dataset/test/subject_test.txt")
subjects <- rbind(subjectTrainIn,subjectTestIn)
names(subjects) <- "SubjNum"

#activities
activityTrainIn <- read.table("UCI HAR Dataset/train/y_train.txt")
activityTestIn <- read.table("UCI HAR Dataset/test/y_test.txt")
activity <- rbind(activityTrainIn,activityTestIn)
names(activity) <- "Activity"

#activity measurement data
measurementsTrainIn <- read.table("UCI HAR Dataset/train/X_train.txt")
measurementsTestIn <- read.table("UCI HAR Dataset/test/X_test.txt")
measurements <- rbind(measurementsTrainIn,measurementsTestIn)

#measurement names are in the features.txt file
measurementNames <- read.table("UCI HAR Dataset/features.txt")
#I couldn't figure out how to use setnames on a dt with wildcards
names(measurements) <- measurementNames$V2

# merge on columns to bring all datasets together
complete <- cbind(subjects,activity,measurements)

#EXTRACT ONLY COLS WITH MEAN OR STD MEASUREMENTS
# Find columns with means or stds. 
#getMeanStd is a logical that returns true if a column name contains 'mean' or 'std'
getMeanStd <- grepl("mean\\(\\)", names(complete)) |
  grepl("std\\(\\)", names(complete)) |
  grepl("SubjNum", names(complete)) |
  grepl("Activity", names(complete))

#I kept loosing subjNum and Actitivity during merge so, I'm specifying that they stay in the getMeanStd
#| grepl("SubjNumber",names(complete)) | grepl("Activity",names(complete))
#There's probably a better way to do this

#Subset based on getMeanStd,get only cols with mean/std
complete <- complete[,getMeanStd]

#The 'Activity' column is still using numbers to identify activities, switch to the activity names
# Treats as nominal variables. Change 1-->Walking, 2-->Walking Upstairs, etc
complete$Activity <- factor(complete$Activity, labels=c("Walking",
                                                        "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"))
#MAYBE IT MAKES SENSE TO DO THIS EARLIER IN SCRIPT?

# Final Step, Create new tidy set with only avgs


