library(keras)
library(dplyr)
source("02-generator.R")

df <- readRDS("data/df.rds") %>% sample_frac(1)
ds <- data_generator(df, 32)

input <- layer_input(shape = c(74, 257, 1))

output <- input %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 29, activation = 'softmax')

model <- keras_model(input, output)

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
model %>% fit_generator(
  generator = ds,
  steps_per_epoch = 10,
  epochs = 1
)



