import matplotlib.pyplot as plt
import json
import sys
import os
import numpy as np
import time


color_map = {
    "random": "red",
    "stratified": "blue",
    "bootstrap": "orange",
    "stratified_label_4": "blue",
    "stratified_label_8": "green",
    "guided_label_4_variance": "orange",
    "guided_label_4_minmax": "blue",
    "guided_label_4_sumL1": "green",
    "guided_label_4_avgL1": "orange",
    "guided_hidden_4_variance": "green",
    "guided_hidden_8_variance": "blue",
}

knobs = {
    'audio_sr': [14000, 16000],
    'freq_mask': [2000],
    'model': ['wav2vec2-large-960h', 'hubert-large']
}

methods = ["random", "guided_label_4_variance", "guided_hidden_4_variance", "guided_hidden_8_variance"]

# knobs = [
#     ('audio_sample_rate', [12000, 14000, 16000]),
#     ('frequency_mask_width', [2000]),
#     ('model', ['wav2vec2-base', 'wav2vec2-large-10m', 'hubert-large', 'hubert-xlarge'])
# ]

def draw_new(records, prefix=""):
    plt.cla()
    figures_dict = {}
    num_fig = 0
    for r in records:
        audio_sr = r['audio_sr']
        freq_mask = r['freq_mask']
        model = r['model']
        method = r['method']
        if model in knobs["model"] and audio_sr in knobs["audio_sr"] and freq_mask in knobs["freq_mask"]:
            key = f"{audio_sr}_{freq_mask}_{model}"
            if key not in figures_dict:
                num_fig += 1
                figures_dict[key] = []
            figures_dict[key].append(r)
            # print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, method: {method}")
    
    print("Num figure " ,num_fig)
    fig, axs = plt.subplots(num_fig, figsize=(15,15))
    fig.tight_layout(h_pad=3)    
    for axid, k in enumerate(figures_dict.keys()):
        ax = axs[axid]
        # ax = fig.add_subplot()
        for idx, r in enumerate(figures_dict[k]):
            record = r
            sample_idx = record['idx']
            method = record['method']
            audio_sr = record['audio_sr']
            freq_mask = record['freq_mask']
            model = record['model']
            acc = record['result']['accuracy']
            cul_acc = record['result']['cummulative_accuracy']

            # assert(np.average(acc) == cul_acc[-1])

            skip = 100
            xid = range(len(cul_acc[skip:]))
            
            if method != "random":
                ax.plot(xid, cul_acc[skip:], label=method, color=color_map[method], linewidth=0.1)
                print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, method: {method}")
            if sample_idx == 0 and method == "random":
                print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}")
                ax.set_xlabel('# sample')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{model} sr:{audio_sr} fm:{freq_mask}')
                ax.set_xticks([i for i in xid if i % 500 == 0], [str(i + skip) for i in xid if i % 500 == 0])
                ax.set_xlim(0, len(xid) - 1)
                # plt.axhline(y=ra_acc, color='g', linestyle='dashed', label='acc')
                ground_truth = cul_acc[-1]
                ax.axhline(y=ground_truth + 0.01, color='g', linestyle='dashed', label='acc+1%')
                ax.axhline(y=ground_truth - 0.01, color='g', linestyle='dashed', label='acc-1%')
                ax.set_ylim(ground_truth - 0.05, ground_truth + 0.05)
                
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    plt.savefig(f"./result/figures/{prefix}.png")
    print(f"Save to ./result/figures/{prefix}.png")


def draw_average(records, per, prefix=""):
    plt.cla()
    figures_dict = {}
    num_fig = 0
    for r in records:
        audio_sr = int(r['audio_sr'])
        freq_mask = int(r['freq_mask'])
        model = str(r['model'])
        method = str(r['method'])

        if model in knobs["model"] and audio_sr in knobs["audio_sr"] and freq_mask in knobs["freq_mask"]:
            key = f"{audio_sr}_{freq_mask}_{model}"
            if key not in figures_dict:
                num_fig += 1
                figures_dict[key] = []
                print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, method: {method}")
            figures_dict[key].append(r)

    print("Num figure", num_fig)
    fig, axs = plt.subplots(num_fig, figsize=(15,15))
    fig.tight_layout(h_pad=3)    

    if num_fig == 1:
        axs = [axs]
        
    for axid, k in enumerate(figures_dict.keys()):
        ax = axs[axid]
        cul_accs = {
            "random": [[] for i in range(6400)],
            "stratified_natural": [[] for i in range(6400)],
            "stratified_hidden": [[] for i in range(6400)],
            "stratified_label": [[] for i in range(6400)],
            "bootstrap": [[] for i in range(6400)],
            "stratified_label_4": [[] for i in range(6400)],
            "stratified_label_8": [[] for i in range(6400)],
            "guided_label_4_variance": [[] for i in range(6400)],
            "guided_label_4_minmax": [[] for i in range(6400)],
            "guided_label_4_sumL1": [[] for i in range(6400)],
            "guided_label_4_avgL1": [[] for i in range(6400)],
            "guided_hidden_4_variance": [[] for i in range(6400)],
            "guided_hidden_8_variance": [[] for i in range(6400)],
        }
        # ax = fig.add_subplot()
        for idx, r in enumerate(figures_dict[k]):
            record = r
            sample_idx = record['idx']
            method = record['method']
            audio_sr = record['audio_sr']
            freq_mask = record['freq_mask']
            model = record['model']
            acc = record['result']['accuracy']
            cul_acc = record['result']['cummulative_accuracy']

            # assert(np.average(acc) == cul_acc[-1])

            skip = 100
            xid = range(len(cul_acc[skip:]))
            
            for i in range(len(cul_acc)):
                cul_accs[method][i].append(cul_acc[i])
                # ax.plot(xid, cul_acc[skip:], label=method, color=color_map[method], linewidth=0.1)
                # print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, method: {method}")
            if sample_idx == 0 and method == "random":
                print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}")
                ax.set_xlabel('# sample')
                ax.set_ylabel('Accuracy')
                ax.set_title(f'{model} sr:{audio_sr} fm:{freq_mask}')
                ax.set_xticks([i for i in xid if i % 500 == 0], [str(i + skip) for i in xid if i % 500 == 0])
                ax.set_xlim(0, len(xid) - 1)
                # plt.axhline(y=ra_acc, color='g', linestyle='dashed', label='acc')
                ground_truth = cul_acc[-1]
                ax.axhline(y=ground_truth + 0.01, color='grey', linestyle='dashed', label='acc+1%')
                ax.axhline(y=ground_truth - 0.01, color='grey', linestyle='dashed', label='acc-1%')
                ax.set_ylim(ground_truth - 0.05, ground_truth + 0.05)
                
        def average(lst):
            return sum(lst) / len(lst)
        def variance(lst):
            return sum([(x - average(lst))**2 for x in lst]) / len(lst)
        def percentile(lst, p):
            return lst[int(float(len(lst)) * (p / 100.0))]
        
        for method in methods:
            upper_bound = []
            lower_bound = []
            for i in range(len(cul_accs[method])):
                acc = sorted(cul_accs[method][i])
                upper_bound.append(percentile(acc, per))
                lower_bound.append(percentile(acc, 100 - per))
            # ax.fill_between(xid, lower_bound[skip:], upper_bound[skip:], color=color_map[method], alpha=0.25)
            ax.plot(xid, lower_bound[skip:], color=color_map[method], label=method,linewidth=0.7)
            ax.plot(xid, upper_bound[skip:], color=color_map[method],linewidth=0.7)
            # ax.plot([], [], color=color_map[method], label=method)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="right")
    plt.savefig(f"./result/figures/{prefix}.png")
    print(f"Save to ./result/figures/{prefix}.png")

    
if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    date_expect = sys.argv[1]
    assert(len(date_expect) > 0)
    # method = sys.argv[1]
    # idx = sys.argv[1]

    # for record in records:
    #     plot(record, f"{method}:{idx}")
    
    # with open(f'./result/profile_random_0.json', 'r') as fp:
    #     records_random = json.load(fp)
    #     # ras.append(records_random[j])
    # with open(f'./result/weighted_feedback.json', 'r') as fp:
    #     records_weighted = json.load(fp)
    #         # sts.append(records_stratified[j])
    
    # plot2_short(records_random[0], records_weighted[0], f"")
    # plot2_short(records_random[1], records_weighted[1], f"")

    records = []
    for (dirpath, dirname, filenames) in os.walk("./result/"):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            method = filename.split("_")[0]
            date = filename.split("_")[1].split(".")[0]
            if date != date_expect:
                continue
            print("Load file: ", filename)
            # idx = filename.split("_")[1]
            with open(f'./result/{filename}', 'r') as fp:
                record = json.load(fp)
                records += record
                
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    # draw_new(records, f"all_{date_time}")
    for audio_sr in [14000, 16000]:
        knobs["audio_sr"] = [audio_sr]
        draw_average(records, 95, f"avg_p95_{audio_sr}_{date_time}")
        draw_average(records, 99, f"avg_p99_{audio_sr}_{date_time}")

    # plot_new(records_random[0], f"random_0")