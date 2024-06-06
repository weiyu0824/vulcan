import matplotlib.pyplot as plt
import json
import sys
import os
import numpy as np
import time


color_map = {
    "random": "r",
    "stratified": "b",
    "bootstrap": "g",
}


def draw_new(records, prefix=""):
    plt.cla()
    figures_dict = {}
    num_fig = 0
    for r in records:
        audio_sr = r['audio_sr']
        freq_mask = r['freq_mask']
        model = r['model']
        key = f"{audio_sr}_{freq_mask}_{model}"
        if key not in figures_dict:
            num_fig += 1
            figures_dict[key] = []
        figures_dict[key].append(r)
    
    fig, axs = plt.subplots(num_fig, 1, figsize=(10, 10))
        
    for axid, k in enumerate(figures_dict.keys()):
        ax = axs[axid]
        for idx, r in enumerate(figures_dict[k]):
            record = r
            method = record['method']
            audio_sr = record['audio_sr']
            freq_mask = record['freq_mask']
            model = record['model']
            acc = record['result']['accuracy']
            cul_acc = record['result']['cummulative_accuracy']

            # assert(np.average(acc) == cul_acc[-1])

            skip = 100
            xid = range(len(cul_acc[skip:]))
            
            
            ax.plot(xid, cul_acc[skip:], label=method, color=color_map[method], linewidth=0.25)
            
            if idx == 0:
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

    
if __name__ == "__main__":
    # assert(len(sys.argv) == 2)
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
            # idx = filename.split("_")[1]
            with open(f'./result/{filename}', 'r') as fp:
                record = json.load(fp)
                records += record
                
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    draw_new(records, f"all_{date_time}")
    # plot_new(records_random[0], f"random_0")