# Creates a generator from a dataset.

library(tfdatasets)
audio_ops <- tf$contrib$framework$python$ops$audio_ops

data_generator <- function(df, batch_size, shuffle = TRUE) {
  
  ds <- tensor_slices_dataset(df) %>%
    dataset_map(function(obs) {
      
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
      
      # create the spectrogram
      spectrogram <- audio_ops$audio_spectrogram(
        wav$audio,
        window_size = wav$sample_rate/50,
        stride = wav$sample_rate/75,
        magnitude_squared = TRUE
      )
      
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      
      # create the  Mel-Frequency Cepstral Coefficients
      # x <- audio_ops$mfcc(
      #   spectrogram,
      #   wav$sample_rate,
      #   dct_coefficient_count = 40L
      # )
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(obs$class_id, 29L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 20000)  
  
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(NULL), shape(NULL, 74, 257)))
  
  ds
}



