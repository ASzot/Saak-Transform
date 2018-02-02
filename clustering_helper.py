
# [cluster index] -> [Frequencies of real labels associated with that cluster]
def bin_labels(labels, pred_labels):
    freqs = {}
    for label, pred_label in zip(labels, pred_labels):
        if pred_label not in freqs:
            freqs[pred_label] = {}
        if label not in freqs[pred_label]:
            freqs[pred_label][label] = 0
        freqs[pred_label][label] += 1

    return freqs

def convert_bins_to_pcts(bins):
    total_vals = {}
    for b in bins:
        total_val = sum(bins[b].values())
        total_vals[b] = int(total_val)
        for i in bins[b]:
            bins[b][i] = (bins[b][i] / total_val * 100.)

        sorted_bin = sorted(bins[b].items(), key=lambda x: x[1], reverse=True)
        total_str = ['%i: %.2f%%' % (k, v) for k, v in sorted_bin]

        bins[b] = total_str

    return bins, total_vals



def bin_samples(samples, classes):
    bins = {}
    for sample, c in zip(samples, classes):
        if c not in bins:
            bins[c] = []
        bins[c].append(sample)

    return bins
