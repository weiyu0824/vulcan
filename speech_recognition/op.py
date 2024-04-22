import numpy as np
import torch
import jiwer
import pickle
import torchaudio
import time

from base_op import ProcessOp, SourceOp

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
        emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
        Returns:
        str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1) # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        # indices = [i for i in indices if i != self.blank]
        transcript = "".join([self.labels[i] for i in indices])
        transcript = transcript.replace("|", " ")
        transcript = transcript.replace("-", "")
        return transcript

class AudioSampler(ProcessOp):
    def __init__(self, args):
        super().__init__()
        

    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))

        # compute:
        start_compute_time = time.time()
        if profile_compute_latency:
           self.compute_latencies.append(time.time() - start_compute_time)
        return batch_data 

class NoiseReduction(ProcessOp):
    def __init__(self, args):
        super().__init__()
        
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))
        
        # compute 
        start_compute_time = time.time()    
        if profile_compute_latency:
           self.compute_latencies.append(time.time() - start_compute_time)
        return batch_data

class WaveToText(ProcessOp):
    def __init__(self, args):
        super().__init__()

        model_name = args['model']
        if model_name == "wav2vec2-base":
            bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
        elif model_name == "wav2vec2-large-10m":
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_10M
        elif model_name == "wav2vec2-large-960h":
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        elif model_name == "hubert-large":
            bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        elif model_name == "hubert-xlarge":
            bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
        else:
            raise TypeError('model is not provided')

        self.model = bundle.get_model()
        self.decoder = None

        self.decoder = GreedyCTCDecoder(labels=bundle.get_labels())

        self.accuracy = []
         
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))
        # data
        audio = batch_data['audio']
        ground_transcript = batch_data['transcript']
        # compute
        start_compute_time = time.time()
        emission, _ = self.model(audio)
        pred_transcript = self.decoder(emission[0])
        if profile_compute_latency:
            self.compute_latencies.append(time.time()-start_compute_time) 
        # accuracy
        wer = jiwer.wer(ground_transcript, pred_transcript)
        self.accuracy.append(wer)
        return 

    def get_endpoint_accuracy(self):
        if len(self.accuracy) == 0:
            return 0
        return sum(self.accuracy) / len(self.accuracy)

class Decoder(ProcessOp):
    def __init__(self):
        pass
    def profile(self, batch_data): 
        pass
    def get_endpoint_accuracy(self):
        pass

# /data/wylin/VOiCES_devkit/distant-16k/speech/test



# import random
# import os
# import librosa
# def speech_file(room, noise, filename = ''):
#     if len(filename) == 0:
#         # randomly choose audio file for room, noise, specs
#         path = '/data/wylin/VOiCES_devkit/distant-16k/speech/test/'+room+'/'+noise+'/'
#         sp = random.choice([f for f in os.listdir(path) if f.startswith('sp')])
#         path = path+sp+'/'
#         filename = random.choice(os.listdir(path))
#     else:
#         room = filename[17:20] 
#         noise = filename[21:25]
#         path = 'distant-16k/speech/test'+room+'/'+noise+'/'+filename[26:filename.find('-ch')]+'/'
#     x, sr = librosa.load(path+filename)
#     duration = librosa.get_duration(path=path+filename)
#     # print(len(x))
#     # print(sr)
#     # print(duration)
#     return x, sr, path+filename, duration #[filename.find('sp'):filename.find('-mc')]
# # audio, sr, file_path, duration = speech_file('rm1', 'musi')

# import pandas as pd
# reference = pd.read_csv('/data/wylin/VOiCES_devkit/references/filename_transcripts')
# print(len(reference))
# print(reference.columns)
# filtered_df = reference[reference['file_name'] == 'Lab41-SRI-VOiCES-rm1-musi-sp5622-ch041172-sg0001-mc01-stu-clo-dg010']

# # Get the 'transcript' column from the filtered DataFrame
# transcript_for_file_a = filtered_df['transcript']

# # Print the transcript
# answer = [transcript_for_file_a.iloc[0].upper()]



# audio, sr = librosa.load('/data/wylin/VOiCES_devkit/distant-16k/speech/test/rm1/musi/sp5622/Lab41-SRI-VOiCES-rm1-musi-sp5622-ch041172-sg0001-mc01-stu-clo-dg010.wav')
# print(sr)
# audio = torch.tensor(np.expand_dims(audio, axis=0))



# import torchaudio
# """ 
# WAV2VEC2_ASR_BASE_960H
# WAV2VEC2_ASR_LARGE_10M
# WAV2VEC2_ASR_LARGE_960H
# HUBERT_ASR_LARGE
# HUBERT_ASR_XLARGE
# """
# bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
# # Build the model and load pretrained weight.
# model = bundle.get_model()
# # Resample audio to the expected sampling rate
# # waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
# # Extract acoustic features
# import time 
# st = time.time()
# emission, _ = model(audio)
# print(time.time() - st)
# # print(features)

# decoder = GreedyCTCDecoder(labels=bundle.get_labels())
# transcript: str = decoder(emission[0])


# print(transcript)

# import jiwer
# wer = jiwer.wer(answer, [transcript])
# print(wer)

# exit()



# # resample
# print('resample')
# audio = librosa.resample(audio, orig_sr=sr, target_sr=14000)

# audio = torch.tensor(np.expand_dims(audio, axis=0))

# # denoise
# # print('denoise')
# # import torch
# from torchgating import TorchGating as TG
# tg = TG(sr=int(sr), nonstationary=True, freq_mask_smooth_hz=4000).to(device)
# audio = tg(audio)




