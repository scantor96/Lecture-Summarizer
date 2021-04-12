import speech_recognition as sr
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

r = sr.Recognizer()

def get_transcript(path):
    sound = AudioSegment.from_wav(path)
    chunks = split_on_silence(sound,
                              min_silence_len = 500,
                              silence_thresh = sound.dBFS-14,
                              keep_silence=500
                              )
    folder = FOLDER_NAME
    
    if not os.path.isdir(folder):
        os.mkdir(folder)
    whole_text = open(WHOLE_TEXT_PATH, 'a')
    for i, audio_chunk in enumerate(chunks,start=1):
        chunk_name = os.path.join(folder,f"chunk{i}.wav")
        #audio_chunk.export(chunk_name,format="wav")
        with sr.AudioFile(chunk_name) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}."
                print(chunk_name, ":", text)
                whole_text.write(text)
                whole_text.write("\n")
                #whole_text += text

get_transcript(WAV_FILE)
