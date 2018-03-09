import numpy as np
from scipy import stats


def bin_entropies(binned_samples):
    entropies = []
    for k in binned_samples:
        flat = binned_samples[k].flatten()
        entropies.append(stats.entropy(flat))

    return entropies

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


# [cluster index] -> {label: prob 0.0 -1.0, ...}
def convert_bins_to_probs(bins):
    probs = {}
    totals = []
    for b in bins:
        total_val = sum(bins[b].values())
        totals.append(total_val)
        for i in bins[b]:
            bins[b][i] = (bins[b][i] / total_val)

        sorted_bin = sorted(bins[b].items(), key=lambda x: x[1], reverse=True)
        bins[b] = sorted_bin

    return bins, totals


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


# [class] -> [sample0, sample1, ...]
def bin_samples(samples, classes):
    bins = {}
    for sample, c in zip(samples, classes):
        if c not in bins:
            bins[c] = []
        bins[c].append(sample)

    for k in bins:
        bins[k] = np.array(bins[k])

    return bins

def bin_samples_labels(samples, classes, real_labels):
    bins = {}
    for sample, c, real_label in zip(samples, classes, real_labels):
        if c not in bins:
            bins[c] = []
        bins[c].append((sample, real_label))

    for k in bins:
        samples = [b[0] for b in bins[k]]
        labels = [b[1] for b in bins[k]]
        bins[k] = (np.array(samples), np.array(labels))

    return bins




def kl_div(a, b):
    return np.sum(a * np.log(a / b))

def entropy(a):
    return stats.entropy(a)
