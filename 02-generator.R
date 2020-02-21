# Creates a generator from a dataset.

library(tfdatasets)

data_generator <- function(df, batch_size, shuffle = TRUE, 
                           window_size_ms = 30, window_stride_ms = 10) {
  
  window_size <- as.integer(16000*window_size_ms/1000)
  stride <- as.integer(16000*window_stride_ms/1000)
  fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
  n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))
  
  ds <- tensor_slices_dataset(df)
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 100)  
  
  ds <- ds %>%
    dataset_map(function(obs) {
      
      # decoding wav files
      audio_binary <- tf$io$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- tf$audio$decode_wav(audio_binary, desired_channels = 1)
      samples <- wav$audio
      samples <- samples %>% tf$transpose(perm = c(1L, 0L))
      
      # create the spectrogram
      spectrogram <- tf$signal$stft(samples,
                                   frame_length = as.integer(window_size),
                                   frame_step = as.integer(stride))
      
      spectrogram <- tf$math$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 30L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(n_chunks, fft_size, NULL), shape(NULL)))
  
  ds
}

