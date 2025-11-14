
# 2095: Statistical and Machine Learning - Project 

# I) PRE-PROCESSING

set.seed("123")

# 1) Libraries 

library("tibble")
library("dplyr")
library("caret")
library("glmnet")

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
# data_filtered$day_of_month <- factor(data_filtered$day_of_month)
data_filtered$day_of_week <- factor(data_filtered$day_of_week)
data_filtered$op_unique_carrier <- factor(data_filtered$op_unique_carrier)
data_filtered$origin <- factor(data_filtered$origin)
data_filtered$dest <- factor(data_filtered$dest)

# 8) If working with only 1-carrier data 

data_filtered <- data_filtered %>%
  select(-op_unique_carrier)

# 9) The data is way too big (we don't want to work with 1M + rows)

reduced_rows <- sample(nrow(data_filtered),20000)
data_filtered_reduced <- data_filtered[reduced_rows,]

# 10) drop all rows consisting of NA

data_filtered_reduced <- na.omit(data_filtered_reduced)
sum(is.na(data_filtered_reduced))

# 11) We also have to drop departure delay because it is highly correlated to is_delay
# We get warning in R for Quasi-complete separation 
data_filtered_reduced <- data_filtered_reduced %>% 
  select(-dep_delay)

# 12) Since factors like month, and day give rise to a lot of categorical variables we group them logically
# Handling the high-cardinality variables in the following way 

data_filtered_reduced$season <- factor(
  dplyr::case_when(
    data_filtered_reduced$month %in% c(12,1,2)  ~ "Winter",
    data_filtered_reduced$month %in% 3:5        ~ "Spring",
    data_filtered_reduced$month %in% 6:8        ~ "Summer",
    data_filtered_reduced$month %in% 9:11       ~ "Fall"
  )
)

data_filtered_reduced$day_group <- factor(
  dplyr::case_when(
    data_filtered_reduced$day_of_month <= 10 ~ "Early",
    data_filtered_reduced$day_of_month <= 20 ~ "Mid",
    TRUE                        ~ "Late"
  )
)

# Remove month, and day of month 
data_filtered_reduced <- data_filtered_reduced %>%
  select(-c("month","day_of_month","day_of_week"))

# II) SUITABLE PREDICTIVE MODELS

# 1) Logistic regression model (link = logit)

# Splitting the data into training and testing sets 

train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_l_data <- data_filtered_reduced[train_ind,]
test_l_data <- data_filtered_reduced[-train_ind,]

# Fitting the model 

logit_model <- glm(is_delayed ~ ., family = binomial(), data = data_filtered_reduced)

test_l_data$pred <- predict(logit_model, test_l_data, type = "response")
  
# Logistic Model Summary 

summary(logit_model)

# Prediction classes

test_l_data$pred_class <- ifelse(test_l_data$pred >=0.5,1,0)
  
# Confusion matrix 

confusionMatrix(factor(test_l_data$pred_class), 
                factor(test_l_data$is_delayed))
table(train_l_data$is_delay)  

# AIC

AIC(logit_model)

# Significant factors 

# select rows with p-value < 0.05
coef_table <- summary(logit_model)$coefficients
signif_vars <- rownames(coef_table)[coef_table[,4] < 0.05]
signif_vars

# 2) lasso Logistic Regression 

# Uses k-folds validation internally so we don't need to split into test and training data sets
X <- model.matrix(is_delayed ~ ., data = data_filtered_reduced)[, -1]  
y <- data_filtered_reduced$is_delayed

# Hyper-parameter optimization 
cv_lasso <- cv.glmnet(X, y, family = "binomial", alpha = 1, nfolds = 5)

# Use lambda.1se for a sparse model
best_lambda <- cv_lasso$lambda.min
lasso_model <- glmnet(X, y, family = "binomial", alpha = 1, lambda = best_lambda)

# Now we can split the data into training and testing sets 

train_ind <- sample(nrow(data_filtered_reduced), size = 0.7*nrow(data_filtered_reduced))
train_lasso_data <- data_filtered_reduced[train_ind,]
test_lasso_data <- data_filtered_reduced[-train_ind,]

X_train <- model.matrix(is_delayed ~ ., train_lasso_data)[, -1]
y_train <- train_lasso_data$is_delayed

X_test  <- model.matrix(is_delayed ~ ., test_lasso_data)[, -1]
y_test  <- test_lasso_data$is_delayed

# Fitting the lasso model 

y_pred_prob <- predict(lasso_model, newx = X_test, type = "response")
y_pred_class <- ifelse(y_pred_prob >= 0.5, 1, 0)

# Confusion Matrix
y_test_factor <- factor(y_test, levels = c(0,1))
y_pred_factor <- factor(y_pred_class, levels = c(0,1))

confusionMatrix(y_pred_factor, y_test_factor)
