import saak
from investigate_coeffs import cifar_labels
from scipy import stats

import numpy as np
import os

from classify import load_toy_dataset, f_test

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

plt.switch_backend('agg')


def create_data():
    data, labels = load_toy_dataset()

    data = data.reshape(-1, 3, 32, 32)

    data_dict = {}

    for label_str in cifar_labels:
        class_label = cifar_labels[label_str]

        class_data = [sample for sample, label in zip(data, labels) if label == class_label]
        class_data = np.array(class_data)

        data_dict[label_str] = class_data

    return data_dict


def plot_energy_dist(energy_dist):
    if len(energy_dist.shape) > 1:
        for ed in energy_dist:
            ed_max = np.amax(ed)
            print(ed_max)
            use_ed = ed / ed_max
            plt.plot(range(len(use_ed)), use_ed)
    else:
        ed_max = np.amax(energy_dist)
        use_ed = energy_dist / ed_max
        plt.plot(range(len(use_ed)), use_ed)

    plt.ylim([0, 0.4])
    plt.savefig('data/results/example_energy.png')


def compute_class_stats(data, class_name):
    print(data.shape)
    final_output = compute_energy(data)
    return final_output
    #descs = stats.describe(final_output, axis=1)
    #descs = descs[:10]

    #distribution = stats.gengamma(desc.mean, desc.variance, loc=desc.jk, scale=10)


    #print(final_output.shape)

    #mu = np.mean(final_output, axis=0)
    #std = np.std(final_output, axis=0)

    #print('Class: %s, Mu: %.4f, Std: %.4f' % (class_name, mu, std))

    #fft_fo = np.fft.fft(final_output, axis=1)
    #fft_fo_sq = fft_fo.T.dot(conj(fft_fo))
    #fft_fo_sq = np.square(fft_fo)
    ##sample_energy = np.sum(final_output, axis=1)
    ##print(sample_energy.shape)

def compute_energy(samples):
    #samples = np.fft.fft(samples, axis=1)
    samples = np.absolute(samples)
    samples = np.square(samples)

    if len(samples.shape) == 2:
        use_axis = 1
    else:
        use_axis = 0
    #samples = np.sum(samples, axis=use_axis)


    #return np.fft.fft(samples)
    return samples


def compute_energy_for_sample(sample):
    sample = sample.flatten()

    energy = compute_energy(sample)

    return energy


def compute_individual(class_data, use_i = None):
    if use_i is None:
        for i in range(len(class_data)):
            sample = class_data[i]
            compute_energy_for_sample(sample)
    else:
        energy = compute_energy(class_data[use_i])
        plot_energy_dist(energy)


def get_last_saak(data):
    filters, means, outputs = saak.multi_stage_saak_trans(data,
            energy_thresh=0.97)

    final_output = outputs[-1]
    final_output = final_output.reshape(-1, final_output.shape[1])

    return final_output, means, filters

def get_last_test_saak(data, means, filters):
    test_outputs = saak.test_multi_stage_saak_trans(data, means, filters)
    final_output = test_outputs[-1]
    final_output = final_output.reshape(-1, final_output.shape[1])

    return final_output

def normalize_energy(energy):
    return energy / np.amax(energy)


def add_other_class(data_dict, class_name, color, means, filters):
    TAKE_COUNT = 1
    other_class_data = data_dict[class_name]
    other_class = get_last_test_saak(other_class_data, means, filters)
    other_energies = compute_energy(other_class)
    #other_energies = other_energies.flatten()
    return other_energies
    #for other_energy in other_energies[:TAKE_COUNT]:
    #    plt.plot(range(len(other_energy)), normalize_energy(other_energy), color)


def compare_classes(use_class, data_dict):
    all_energies = []

    class_data = data_dict[use_class]
    class_data, means, filters = get_last_saak(class_data)
    energies = compute_energy(class_data)
    #energies = energies.flatten()

    all_energies.append(energies.T)
    for other_class in os.listdir('data/mcl'):
        if other_class == use_class:
            continue
        other_energies = add_other_class(data_dict, other_class, 'r', means, filters)
        all_energies.append(other_energies.T)

    all_band_mus = []
    all_band_sigmas = []
    for energy in all_energies:
        is_first = True
        band_mus = []
        band_sigmas = []
        for energy_band in energy:
            print(stats.shapiro(energy_band))
            band_mus.append(np.mean(energy_band))
            band_sigmas.append(np.std(energy_band))
            raise ValueError()

        all_band_mus.append(band_mus)
        all_band_sigmas.append(band_sigmas)

    all_band_mus = np.array(all_band_mus).T
    all_band_sigmas = np.array(all_band_sigmas).T

    print(all_band_mus.shape)
    print(all_band_sigmas.shape)
    #all_band_mus = all_band_mus[:10]
    #all_band_sigmas = all_band_sigmas[:10]

    use_path = 'data/results/bands/' + use_class + '/'

    if not os.path.exists(use_path):
        os.makedirs(use_path)

    index = 0
    for band_mus, band_sigmas in zip(all_band_mus, all_band_sigmas):
        is_first = True
        for mu, sigma in zip(band_mus, band_sigmas):
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            if is_first:
                use_color = 'r'
            else:
                use_color = 'k'
            plt.plot(x, mlab.normpdf(x, mu, sigma), use_color)
            is_first = False

        plt.savefig(use_path + str(index) + '.png')
        plt.clf()

        index += 1


if __name__ == '__main__':
    data_dict = create_data()

    for other_class in os.listdir('data/mcl'):
        compare_classes(other_class, data_dict)


    #for energy in energies[:1]:
    #    plt.plot(range(len(energy)), normalize_energy(energy), 'k')

    #add_other_class(data_dict, 'dog', 'r', means, filters)
    #add_other_class(data_dict, 'bird', 'y', means, filters)
    #add_other_class(data_dict, 'truck', 'm', means, filters)

    #plt.ylim([0, 0.4])


    #for class_name in data_dict:
    #    #compute_class_stats(class_data, class_name)
    #    raise ValueError()

    #for class_name in os.listdir('data/mcl/'):
    #    compute_class_stats('airplane')
    #    compute_individual(0)
    #    print('')
    #    raise ValueError()







