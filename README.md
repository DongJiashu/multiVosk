# MultiVosk Speech Recognition Tool

This repo can realize: 
1. transcribe parrellel with multiple ASR vosk models using CPU.
2. compare transcription with utterance_file, which contains all references, and match sentence level result then save to pruned folder. punctuations, capitalizations, multi whitespace are ignored, similarity threshold can be adjusted.
3. generate report base on each models wer/cer.
4. combine multiple vosk modules' result using sentence level voting to calculate the most commen result. If no commen result, fallback to model priority approach (models with low WER published in VOSK website gets the higher priority)

used model in this example: 
1. vosk-small-en-us-0.15, 40MB, 9.85 (wer on librispeech)
2. vosk-en-us-0.22, 1.8GB, 5.69 (wer on librispeech)
3. vosk-model-en-us-0.22-lgraph,127MB, 7.82 (wer on librispeech)

# Environment set up 
'''
python3.11 -m venv venvs/vosk

module load FFmpeg

pip install --upgrade pip

pip install requests vosk jiwer numpy

source venvs/vosk/bin/activate
'''
# How to use 
'''
python transcribe.py data/task1/Nexdata_demo
'''
# What can be improved 
1. more models can be integrated, e.g. asr models from NEMO toolkit also have promising WER. I've tried in my experiment, the problem is to load Nemo models takes longer time than I expected, anyone have an idea how this problem can be tackeled is welcomed to contact. 
2. For the combined result, currently I only use sentence level voting. I've also tried word level voting and tried the word level confidence score from vosk, but the result is not so good, it causes either word eaten or adding, anyone have an idea how word level voting can be implemented, feel free to raise your idea. 

# Documentation
For other models visit [Vosk
Website](https://alphacephei.com/vosk/models).

For original repo visit [Vosk
repo](https://github.com/alphacep/vosk-api).

