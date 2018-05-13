# Download and uncompress data

download.file(
  url = "https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz", 
  destfile = "data/speech_commands_v0.01.tar.gz"
)

untar("speech_commands_v0.01.tar.gz")