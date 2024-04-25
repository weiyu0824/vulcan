import numpy as np
import torch
import jiwer
import pickle
import torchaudio
import time
from torchgating import TorchGating as TG

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
        transcript = transcript.lower()
        return transcript

class AudioSampler(ProcessOp):
    def __init__(self, args):
        super().__init__()
        self.target_sr = args['audio_sample_rate']
        

    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))
        
        sr = batch_data['sr']
        audio = batch_data['audio']

        # compute:
        start_compute_time = time.time()
        
        audio = torchaudio.functional.resample(audio, sr, self.target_sr)    

        batch_data['sr'] = self.target_sr
        batch_data['audio'] = audio

        if profile_compute_latency:
           self.compute_latencies.append(time.time() - start_compute_time)
        return batch_data 

class NoiseReduction(ProcessOp):
    def __init__(self, args):
        super().__init__()
        freq_mask = args['frequency_mask_width']
        sr = args['audio_sample_rate']
        self.tg = TG(sr=int(sr), nonstationary=True, freq_mask_smooth_hz=freq_mask).to(device)
        
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))
        
        audio = batch_data['audio'] 
        # compute 
        start_compute_time = time.time()    

        audio = self.tg(audio)
        batch_data['audio'] = audio

        if profile_compute_latency:
           self.compute_latencies.append(time.time() - start_compute_time)
        return batch_data

class WaveToText(ProcessOp):
    def __init__(self, args):
        super().__init__()

        model_name = args['model']
        if model_name == "wav2vec2-base":
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
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

        self.model = bundle.get_model().to(device)
         
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))
        # data
        audio = batch_data['audio'].to(device)
        
        # compute
        start_compute_time = time.time()
        with torch.no_grad():
            emission, _ = self.model(audio)
        
        batch_data['audio'] = None
        batch_data['emission'] = emission[0]
        
        if profile_compute_latency:
            self.compute_latencies.append(time.time()-start_compute_time) 

        return batch_data
        
       

class Decoder(ProcessOp):
    def __init__(self, args):
        super().__init__()

        model_name = args['model']
        if model_name == "wav2vec2-base":
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
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

        self.decoder = GreedyCTCDecoder(labels=bundle.get_labels())

        self.accuracy = []
    
    def profile(self, batch_data, profile_input_size=False, profile_compute_latency=False):
        if self.input_size == None and profile_input_size:
            self.input_size = len(pickle.dumps(batch_data))

        ground_transcript = batch_data['transcript']
        emission = batch_data['emission']

        # computer
        start_compute_time = time.time()
        pred_transcript = self.decoder(emission)
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
