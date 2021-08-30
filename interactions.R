library(tidymodels)
library(rules)
library(lubridate)
library(timeDate)
library(stringr)
library(doMC)
registerDoMC(cores = 20)
tidymodels_prefer()
theme_set(theme_bw())

# ------------------------------------------------------------------------------

load("RData/chicago.RData")

# ------------------------------------------------------------------------------

stations <- names(chicago)
stations <- stations[!(stations %in% c("ridership", "date"))]

chicago <- 
  chicago %>% 
  mutate(
    change_1 = ifelse(date > ymd("2020-12-18"), 1, 0),  #  vac start
    change_2 = ifelse(date > ymd("2021-01-08"), 1, 0),  #  2nd doses
    change_3 = ifelse(date > ymd("2021-01-13"), 1, 0),  #  Hospitals and outpatient sites, 75 +
    change_4 = ifelse(date > ymd("2021-01-25"), 1, 0),  #  phase 1b
    change_5 = ifelse(date > ymd("2021-02-25"), 1, 0),  #  J&J available, phase 1B+
    change_6 = ifelse(date > ymd("2021-04-10"), 1, 0)   #  approved for kids
  )


# ------------------------------------------------------------------------------

test_days <- ymd("2021-05-18") + days(0:13)

chi_train <- chicago %>% filter(date < min(test_days))
chi_test  <- chicago %>% filter(date >= min(test_days))

min_rides <- min(chi_train$ridership)

chi_rs <-
  chi_train %>%
  sliding_period(
    index = "date",  
    period = "week",
    lookback = 52 * 19.25,
    assess_stop = 2,
    step = 2
  )

rs_dates <- 
  chi_rs %>% 
  mutate(date = map(splits, ~ min(assessment(.x)$date))) %>% 
  select(id, date) %>% 
  unnest(cols = date)

# ------------------------------------------------------------------------------

# define a few holidays
us_hol <- 
  listHolidays() %>% 
  str_subset("(^US)|(Easter)")

cubist_recipe <- 
  recipe(ridership ~ ., data = chi_train) %>% 
  step_holiday(date, holidays = us_hol) %>% 
  step_mutate(
    day_after_thx   = ifelse(month(date) == 11 & day(date) == 26, 1, 0),
    day_before_xmas = ifelse(month(date) == 12 & day(date) == 24, 1, 0),
    day_after_xmas  = ifelse(month(date) == 12 & day(date) == 26, 1, 0),
    new_years_eve    = ifelse(month(date) == 12 & day(date) == 30, 1, 0)
  ) %>% 
  step_date(date) %>% 
  step_rm(date) %>% 
  step_interact(~ all_of(stations):starts_with("change"))

glmnet_recipe <- 
  cubist_recipe %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# ------------------------------------------------------------------------------

cubist_spec <- 
  cubist_rules(committees = tune(), neighbors = tune()) %>% 
  set_engine("Cubist") 

cubist_workflow <- 
  workflow() %>% 
  add_recipe(cubist_recipe) %>% 
  add_model(cubist_spec) 

cubist_grid <-
  tidyr::crossing(committees = c(1:9, (1:5) * 10),
                  neighbors = c(0, 3, 6, 9)) 
cubist_tune <- 
  tune_grid(cubist_workflow, resamples = chi_rs, grid = cubist_grid) 

cubist_best <- select_best(cubist_tune, metric = "rmse")

cubist_interactions_test <- 
  cubist_workflow %>% 
  finalize_workflow(cubist_best) %>% 
  fit(chi_train) %>% 
  predict(chi_test) %>% 
  bind_cols(chi_test %>% select(date, ridership)) %>% 
  mutate(model = "cubist", method = "interactions", day = wday(date, label = TRUE))

cubist_interactions <- 
  cubist_tune %>% 
  collect_metrics(summarize = FALSE) %>% 
  filter(.metric == "rmse") %>% 
  inner_join(cubist_best, by = c("committees", "neighbors", ".config")) %>% 
  inner_join(rs_dates, by = "id") %>% 
  select(date, rmse = .estimate) %>% 
  mutate(model = "cubist", method = "interactions")


# ------------------------------------------------------------------------------

glmnet_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet") 

glmnet_workflow <- 
  workflow() %>% 
  add_recipe(glmnet_recipe) %>% 
  add_model(glmnet_spec) 

glmnet_grid <-
  tidyr::crossing(
    penalty = 10 ^ seq(-6, -1, length.out = 20),
    mixture = c(0.05, 0.2, 0.4, 0.6, 0.8, 1)
  )

glmnet_tune <- 
  tune_grid(glmnet_workflow, resamples = chi_rs, grid = glmnet_grid) 


glmnet_best <- select_best(glmnet_tune, metric = "rmse")

glmnet_interactions_test <-
  glmnet_workflow %>%
  finalize_workflow(glmnet_best) %>%
  fit(chi_train) %>%
  predict(chi_test) %>%
  bind_cols(chi_test %>% select(date, ridership)) %>%
  mutate(
    model = "glmnet",
    method = "interactions",
    day = wday(date, label = TRUE),
    .pred = ifelse(.pred <= min_rides, min_rides, .pred)
  )

glmnet_interactions <- 
  glmnet_tune %>% 
  collect_metrics(summarize = FALSE) %>% 
  filter(.metric == "rmse") %>% 
  inner_join(glmnet_best, by = c("penalty", "mixture", ".config")) %>% 
  inner_join(rs_dates, by = "id") %>% 
  select(date, rmse = .estimate) %>% 
  mutate(model = "glmnet", method = "interactions")

# ------------------------------------------------------------------------------

interactions <- 
  bind_rows(glmnet_interactions, cubist_interactions) %>% 
  filter(date > ymd("2020-04-01"))
interactions_test <- bind_rows(glmnet_interactions_test, cubist_interactions_test)

save(interactions, interactions_test, file = "RData/interactions.RData")

q("no")
