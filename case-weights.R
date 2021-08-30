library(tidymodels)
library(rules)
library(lubridate)
library(timeDate)
library(stringr)
library(furrr)
library(Cubist)
library(glmnet)

options(future.rng.onMisuse="ignore")
plan(multicore)

tidymodels_prefer()
theme_set(theme_bw())

# ------------------------------------------------------------------------------

load("RData/chicago.RData")

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
    new_years_eve    = ifelse(month(date) == 12 & day(date) == 30, 1, 0),
    case_weights     = ifelse(date >= ymd("2020-04-01"), 1, .1)
  ) %>% 
  step_date(date) %>% 
  step_rm(date) 

glmnet_recipe <- 
  cubist_recipe %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors(), -case_weights) %>% 
  step_normalize(all_numeric_predictors(), -case_weights)

# ------------------------------------------------------------------------------

fit_cubist <- function(df) {
  library(Cubist)
  cubist(x = df[, !(names(df) %in% c("ridership", "case_weight"))],
         y = df$ridership,
         weights = df$case_weights,
         committees = 50)
}

assess_cubist <- function(dat, model) {
  library(tidymodels, quietly = TRUE)
  x <- dat %>% select(-ridership)
  preds <- predict(model, x)
  tibble(.pred = preds, ridership = dat$ridership) %>% 
    rmse(ridership, .pred)
}

cubist_weighted <- 
  chi_rs %>% 
  mutate(
    recipe = map(splits, ~ prep(cubist_recipe, analysis(.x))),
    analysis  = map(recipe, ~ juice(.x)),
    assessment = map2(recipe, splits, ~ bake(.x, assessment(.y))),
    model = future_map(analysis, ~ fit_cubist(.x)),
    rmse = future_map2(assessment, model, assess_cubist)
  ) %>% 
  select(id, rmse) %>% 
  unnest(cols = rmse) %>% 
  inner_join(rs_dates, by = "id") %>% 
  select(date, rmse = .estimate) %>% 
  mutate(model = "cubist", method = "case weights")


cubist_recipe_prepped <- prep(cubist_recipe)
test_data <- cubist_recipe_prepped %>% bake(chi_test) %>%  select(-ridership)
final_cubist <- fit_cubist(juice(cubist_recipe_prepped))
cubist_weighted_test <-
  tibble(
    .pred = predict(final_cubist, test_data),
    ridership = chi_test$ridership,
    date = chi_test$date
  ) %>% 
  mutate(model = "cubist", method = "case weights", day = wday(date, label = TRUE))

# ------------------------------------------------------------------------------


fit_glmnet <- function(df) {
  library(glmnet)
  x <- df[, !(names(df) %in% c("ridership", "case_weights"))]
  x <- as.matrix(x)
  mod <- cv.glmnet(x, df$ridership, weights = df$case_weights)
  
  mod
}

assess_glmnet <- function(dat, model) {
  library(tidymodels, quietly = TRUE)
  library(glmnet)
  x <- dat %>% select(-ridership, -case_weights)
  x <- as.matrix(x)
  preds <- predict(model, x)
  tibble(.pred = preds[,1], ridership = dat$ridership) %>%
    rmse(ridership, .pred)
}

glmnet_weighted <- 
  chi_rs %>% 
  mutate(
    recipe = map(splits, ~ prep(glmnet_recipe, analysis(.x))),
    analysis  = map(recipe, ~ juice(.x)),
    assessment = map2(recipe, splits, ~ bake(.x, assessment(.y))),
    model = future_map(analysis, ~ fit_glmnet(.x)),
    rmse = future_map2(assessment, model, assess_glmnet)
  ) %>% 
  select(id, rmse) %>% 
  unnest(cols = rmse) %>% 
  inner_join(rs_dates, by = "id") %>% 
  select(date, rmse = .estimate) %>% 
  mutate(model = "glmnet", method = "case weights")


glmnet_recipe_prepped <- prep(glmnet_recipe)
test_data <- glmnet_recipe_prepped %>% bake(chi_test) %>%  select(-ridership, -case_weights)
final_glmnet <- fit_glmnet(juice(glmnet_recipe_prepped))
glmnet_weighted_test <-
  tibble(
    .pred = predict(final_glmnet, as.matrix(test_data))[,1],
    ridership = chi_test$ridership,
    date = chi_test$date
  ) %>% 
  mutate(model = "glmnet", method = "case weights", day = wday(date, label = TRUE))


# ------------------------------------------------------------------------------

case_weights <- 
  bind_rows(glmnet_weighted, cubist_weighted) %>% 
  filter(date > ymd("2020-04-01"))
case_weights_test <- bind_rows(glmnet_weighted_test, cubist_weighted_test)

save(case_weights, case_weights_test, file = "RData/case_weights.RData")

q("no")
