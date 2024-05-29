import matplotlib.pyplot as plt
import json
import sys


def plot(record, prefix=""):
    audio_sr = record['audio_sr']
    freq_mask = record['freq_mask']
    model = record['model']
    acc = record['result']['accuracy']
    cul_acc = record['result']['cummulative_accuracy']
    print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, acc: {acc}")
    
    x = range(len(cul_acc[100:]))

    plt.cla()
    plt.plot(x, cul_acc[100:], label='Cummulative Accuracy')
    plt.xticks([i for i in x if i % 500 == 0], [str(i + 100) for i in x if i % 500 == 0])
    plt.axhline(y=acc, color='r', linestyle='-', label='Accuracy')

    plt.xlabel('Index')
    plt.ylabel('Accuracy')
    plt.title(f'{prefix} plot')
    plt.legend()
    plt.savefig(f"./result/profile_{prefix}_{audio_sr}_{freq_mask}_{model}.png")

def plot2(ra, st, prefix=""):
    audio_sr = ra['audio_sr']
    freq_mask = ra['freq_mask']
    model = ra['model']
    ra_acc = ra['result']['accuracy']
    ra_cul_acc = ra['result']['cummulative_accuracy']
    st_acc = st['result']['accuracy']
    st_cul_acc =  st['result']['cummulative_accuracy']
    print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, acc: {ra_acc:2f}^{st_acc:2f}")
    assert(audio_sr == st['audio_sr'])
    assert(freq_mask == st['freq_mask'])
    assert(model == st['model'])
    assert(len(ra_cul_acc) == len(st_cul_acc))

    skip = 100
    x = range(len(ra_cul_acc[skip:]))
    plt.cla()
    plt.plot(x, ra_cul_acc[skip:], label='random', color='r')
    plt.plot(x, st_cul_acc[skip:], label='stratified', color='b')
    plt.xticks([i for i in x if i % 500 == 0], [str(i + skip) for i in x if i % 500 == 0])
    plt.xlim(0, len(x) - 1)
    # plt.axhline(y=ra_acc, color='g', linestyle='dashed', label='acc')
    plt.axhline(y=ra_acc + 0.01, color='g', linestyle='dashed', label='acc+1%')
    plt.axhline(y=ra_acc - 0.01, color='g', linestyle='dashed', label='acc-1%')

    plt.ylim(ra_acc - 0.05, ra_acc + 0.05)
    
    plt.xlabel('# sample')
    plt.ylabel('Accuracy')
    
    plt.title(f'{model} sr:{audio_sr} fm:{freq_mask}')
    plt.legend()
    plt.savefig(f"./result/profile_{prefix}_{audio_sr}_{freq_mask}_{model}.png")

def plot_all(ras, sts, prefix=""):
    assert(len(ras) == len(sts))
    for i in range(len(ras)):
        ra = ras[i]
        st = sts[i]
        audio_sr = ra['audio_sr']
        freq_mask = ra['freq_mask']
        model = ra['model']
        ra_acc = ra['result']['accuracy']
        ra_cul_acc = ra['result']['cummulative_accuracy']
        st_acc = st['result']['accuracy']
        st_cul_acc =  st['result']['cummulative_accuracy']
        print(f"audio_sr: {audio_sr}, freq_mask: {freq_mask}, model: {model}, acc: {ra_acc:2f}^{st_acc:2f}")
        assert(audio_sr == st['audio_sr'])
        assert(freq_mask == st['freq_mask'])
        assert(model == st['model'])
        assert(len(ra_cul_acc) == len(st_cul_acc))

        skip = 100
        x = range(len(ra_cul_acc[skip:]))
        plt.plot(x, ra_cul_acc[skip:], label='random', color='r', linewidth=0.25)
        plt.plot(x, st_cul_acc[skip:], label='stratified', color='b', linewidth=0.25)
        plt.xticks([i for i in x if i % 500 == 0], [str(i + skip) for i in x if i % 500 == 0])
        plt.xlim(0, len(x) - 1)
        # plt.axhline(y=ra_acc, color='g', linestyle='dashed', label='acc')
        if i == 0:
            plt.axhline(y=ra_acc + 0.01, color='g', linestyle='dashed', label='acc+1%')
            plt.axhline(y=ra_acc - 0.01, color='g', linestyle='dashed', label='acc-1%')
            plt.ylim(ra_acc - 0.05, ra_acc + 0.05)
        
            plt.xlabel('# sample')
            plt.ylabel('Accuracy')
        
            plt.title(f'{model} sr:{audio_sr} fm:{freq_mask}')
            plt.legend()
    plt.savefig(f"./result/profile_{prefix}_{audio_sr}_{freq_mask}_{model}.png")
    print(f"Save to ./result/profile_{prefix}_{audio_sr}_{freq_mask}_{model}.png")

    
if __name__ == "__main__":
    # assert(len(sys.argv) == 2)
    # method = sys.argv[1]
    # idx = sys.argv[1]
    ras = []
    sts = []
    for idx in range(10):
        with open(f'./result/profile_random_{idx}.json', 'r') as fp:
            records_random = json.load(fp)
            ras.append(records_random[1])
        with open(f'./result/profile_stratified_{idx}.json', 'r') as fp:
            records_stratified = json.load(fp)
            sts.append(records_stratified[1])

    # for record in records:
    #     plot(record, f"{method}:{idx}")
    
    # plot2(records_random[0], records_stratified[0], f"{idx}")
    # plot2(records_random[1], records_stratified[1], f"{idx}")

    plot_all(ras, sts, "all")