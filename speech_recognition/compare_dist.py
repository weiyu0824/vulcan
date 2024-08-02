import os
import json
import numpy as np
import matplotlib.pyplot as plt

knobs = [
    ('audio_sample_rate', [12000, 14000, 16000]),
    ('frequency_mask_width', [2000]),
    ('model', ['wav2vec2-large-10m', 'wav2vec2-large-960h', 'hubert-large', 'hubert-xlarge'])
]


# knobs = [
#     ('audio_sample_rate', [12000, 14000, 16000]),
#     ('frequency_mask_width', [1000, 2000, 4000]),
#     ('model', ['wav2vec2-large-10m'])
# ]


def get_pipeline_args():
    return {
       'audio_sample_rate': int(os.environ.get('audio_sample_rate', 8000)),
       'frequency_mask_width': int(os.environ.get('frequency_mask_width', '500')),
       'model': os.environ.get('model', 'wav2vec2-base')
    }
    
def load_cache(pipe_args):
    dump_filename = '_'.join([str(i) for i in pipe_args.values()])
    with open(f"./cache/{dump_filename}.json", 'r') as f:
        dump = json.load(f)
    return dump

def plot(res, title):
    plt.cla()
    colors = ['r', 'g', 'b']
    plt.ylim(0, 1)
    for r, c in zip(res, colors):
        # print(r[:5])
        plt.scatter(range(len(r)), r, s=1, c=c)
    plt.title(title)
    plt.savefig(f'./result/figures/compare_{title}.png')

def compare(u, v, w, title):
    u = u["results"].copy()
    v = v["results"].copy()
    w = w["results"].copy()
    keys = list(u.keys()).copy()
    keys = sorted(keys)
    res = [[], [], []]
    for k in keys:
        u1 = u[k]["metric"]
        v1 = v[k]["metric"]
        w1 = w[k]["metric"]
        res[0].append(u1)
        res[1].append(v1)
        res[2].append(w1)
        
    plot(res, title)
    
def exp():
    for audio_sr in knobs[0][1]:
        for freq_mask in knobs[1][1]:
            for model in knobs[2][1]:
                args = [{
                    "audio_sample_rate" : str(audio_sr),
                    "frequency_mask_width" : str(freq_mask),
                    "model" : str(model),
                } for freq_mask in [1000, 2000, 4000]]
                # args = [{
                #     "audio_sample_rate" : str(asr),
                #     "frequency_mask_width" : str(freq_mask),
                #     "model" : str(model),
                # } for asr in [12000, 14000, 16000]]
                print("Comparing", args)
                u = load_cache(args[0])
                v = load_cache(args[1])
                w = load_cache(args[2])
                compare(u, v, w, title=f"{audio_sr} {model}")
                

if __name__ == "__main__":
    exp()