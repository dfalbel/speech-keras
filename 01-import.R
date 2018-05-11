library(stringr)
library(dplyr)

files <- fs::dir_ls(
  path = "data/speech_commands_v0.01/", 
  recursive = TRUE, 
  glob = "*.wav"
)

files <- files[!str_detect(files, "background_noise")]

df <- data_frame(
  fname = files, 
  class = fname %>% str_extract("1/.*/") %>% 
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)

saveRDS(df, "data/df.rds")
