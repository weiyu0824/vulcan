import os
import time
import torch
import json
import librosa
import random
import tqdm
import pandas as pd
import numpy as np
from op import AudioSampler, NoiseReduction, WaveToText


knobs = [
    ('audio_sample_rate', [8000, 10000, 12000, 14000, 16000]),
    ('frequency_mask_width', [500, 1000, 2000, 3000, 4000]),
    ('model', ['wav2vec2-base', 'wav2vec2-large-10m', 'wav2vec2-large-960h', 'hubert-large', 'hubert-xlarge'])
]

def get_pipeline_args():
    return {
       'audio_sample_rate': os.environ.get('audio_sample_rate', '8k'),
       'frequency_mask_width': int(os.environ.get('frequency_mask_width', '500')),
       'model': os.environ.get('model', 'wav2vec2-base')
    }


ref_df = pd.read_csv('/data/wylin/VOiCES_devkit/references/filename_transcripts')
ref_df.set_index('file_name', inplace=True)
def random_load_speech_data():
    # randomly choose audio file for room, noise, specs
    room = random.choice(['rm1', 'rm2', 'rm3', 'rm4'])
    noise = random.choice(['babb', 'musi', 'none', 'tele'])
    path = '/data/wylin/VOiCES_devkit/distant-16k/speech/test/' + room + '/' + noise + '/'
    sp = random.choice([f for f in os.listdir(path) if f.startswith('sp')])
    path = path + sp + '/'
    filename = random.choice(os.listdir(path))

    # Audio
    audio, sr = librosa.load(path+filename)
    # Transcription
    transcript = ref_df.loc[filename.split('.')[0], 'transcript'] # split to remove .wav

    return audio, sr, transcript


def profile_pipeline():
    pipe_args = get_pipeline_args()
    
    profile_result = {}

    audio_sampler = AudioSampler(pipe_args)
    noise_reduction = NoiseReduction(pipe_args)
    wave_to_text = WaveToText(pipe_args)


    print('start profile this pipeline ...', pipe_args)
    start_time = time.time()

    print('profile latency & input size')
    # Profile args:
    num_profile_batch_latency  = 1 #Cab't be small because of warm-up

    for _ in tqdm.tqdm(range(num_profile_batch_latency)):
        audio, sr, transcript = random_load_speech_data()
        batch_data = {
            'audio': torch.tensor(np.expand_dims(audio, axis=0)),
            'sr': sr,
            'transcript': transcript
        }

        batch_data = audio_sampler.profile(batch_data, profile_compute_latency=True, profile_input_size=True)
        batch_data = noise_reduction.profile(batch_data, profile_compute_latency=True, profile_input_size=True)
        batch_data = wave_to_text.profile(batch_data, profile_compute_latency=True, profile_input_size=True)

    profile_result['audio_sampler_input_size'] = audio_sampler.get_input_size()
    profile_result['noise_reduction_input_size'] = noise_reduction.get_input_size()
    profile_result['wave_to_text_input_size'] = wave_to_text.get_input_size()
    profile_result['audio_sampler_compute_latency'] = audio_sampler.get_compute_latency()
    profile_result['noise_reduction_compute_latency'] = noise_reduction.get_compute_latency()
    profile_result['wave_to_text_compute_latency'] = wave_to_text.get_compute_latency()
     
    print(profile_result)

    print('profile accuracy')

    profile_result['total_profile_time'] = time.time() - start_time
    # Profile args:

    return profile_result


if __name__ == "__main__":
    # addr, port = 'localhost', 12343
    # start_connect(addr, port)

    records = []

    for audio_sr in knobs[0][1]:
        for freq_mask in knobs[1][1]:
            for model in knobs[2][1]:
                os.environ["audio_sample_rate"] = str(audio_sr)
                os.environ["frequency_mask_width"] = str(freq_mask)
                os.environ["model"] = str(model)

            result = profile_pipeline() 
            print(result)
            exit(0)
    
    with open ('profile_result.json', 'w') as fp:
        json.dump(records, fp)