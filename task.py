from dataclasses import dataclass

@dataclass
class Operator:
    name: str
    tpy: str

@dataclass
class Knob: 
    name: str
    choice: list 


video_analytics_knobs = [
    Knob("models", ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']),
    Knob("resize_factor", [0.9, 0.8, 0.7, 0.6, 0.5])
]
video_analytics_ops = [
    Operator("resize", 'basic'), 
    Operator("detector", 'basic')
]

speech_recongnition_knobs = [
    Knob("audio_sr", [8000, 10000, 12000, 14000, 16000]),
    Knob("freq_mask", [500, 1000, 2000, 3000, 4000]),
    Knob("model", ["wav2vec2-base", "wav2vec2-large-10m", "wav2vec2-large-960h", "hubert-large", "hubert-xlarge"])
]

speech_recognition_ops = [
    Operator("audio_sampler", "basic"), 
    Operator("noise_reduction", "basic"), 
    Operator("wave_to_text", "dnn"), 
    Operator("decoder", "basic")
]

speech_recognition_dnn_usg = {
    "wav2vec2-base": 1000,
    "wav2vec2-large-10m": 2000,
    "wav2vec2-large-960h": 2000, 
    "hubert-large": 2000,
    "hubert-xlarge": 5000
}