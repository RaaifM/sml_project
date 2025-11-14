
# 2095: Statistical and Machine Learning - Project 

# I) PRE-PROCESSING

# 1) Libraries 

library("tibble")
library("dplyr")

# 2) Reading the data

# Getting location of the data file 

path <- file.path(here::here(""))
data_path <- file.path(path,"02_data","flight_data_2024.csv")
data <- read.csv(data_path)

# 3) Handling big data 
# Since the data is huge 7 million rows (1+ GB of data)
# We reduce the number of observations by removing some carriers

carrier_count <- table(data$op_unique_carrier)
print(carrier_count)

# In total we have 15 unique carriers 
# In this project we will only focus on the top 4 carriers by number of flights

# Sorting the above table 
sort_carrier_count <- sort(carrier_count, decreasing = TRUE)
top1_carriers <- head(sort_carrier_count, 1)
top1_codes <- names(top1_carriers)

# Here are the top 4 carriers: 
# WN = "Southwest Airlines", DL = "Delta Air Lines", 
# AA = "American Airlines", UA = "United Airlines"

# Filtering the data to only include top 1 carriers 

data_filtered <- data %>%
  filter(op_unique_carrier %in% top1_codes)

# 4) Removing redundant columns and after-even observed information columns 

drop_column_names <- c("fl_date","op_carrier_fl_num",
                       "origin_city_name","origin_state_nm",
                       "dest_city_name","dest_state_nm",
                       "crs_dep_time","dep_time","taxi_out",
                       "wheels_off","wheels_on","taxi_in",
                       "cancellation_code","actual_elapsed_time",
                       "air_time", "carrier_delay","weather_delay",
                       "nas_delay","security_delay","late_aircraft_delay",
                       "crs_arr_time","arr_time","year")

data_filtered <- data_filtered %>%
  select(-all_of(drop_column_names))

# 5) Constructing a new binary is_delayed column 

# Any flights with delay > 15, cancelled, or diverted is considered as delayed

data_filtered <- data_filtered %>%
  mutate(is_delayed = ifelse(cancelled ==1 | diverted == 1| coalesce(arr_delay > 15, FALSE), 1, 0))

# Let's get a first impression of how frequent delays are
table(data_filtered$is_delayed)

# 6) Dropping unnecessary columns (no longer needed)

data_filtered <- data_filtered %>%
  select(-c("cancelled","diverted","arr_delay"))

# 7) Make sure the data reads as factor where relevant 

# Converting the month, day_of_month, day_of_week, op_unique_carrier, origin, dest into factors

data_filtered$month <- factor(data_filtered$month)
data_filtered$day_of_month <- factor(data_filtered$day_of_month)
data_filtered$day_of_week <- factor(data_filtered$day_of_week)
data_filtered$op_unique_carrier <- factor(data_filtered$op_unique_carrier)
data_filtered$origin <- factor(data_filtered$origin)
data_filtered$dest <- factor(data_filtered$dest)

# 8) If working with only 1-carrier data 

data_filtered <- data_filtered %>%
  select(-op_unique_carrier)


# II) SUITABLE PREDICTIVE MODELS

# 1) Logistic regression model (link = logit)

# Splitting the data into training and testing sets 

train_ind <- sample(nrow(data_filtered), size = 0.7*nrow(data_filtered))
train_l_data <- data_filtered[train_ind,]
test_l_data <- data_filtered[-train_ind,]

# Fitting the model 

logit_model <- glm(is_delayed ~ ., family = binomial(), data = data_filtered)

test_l_data$pred <- predict(logit_model, test_l_data, type = "response")
  

  
  