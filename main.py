import PCA
import animation
import head_model
import methods
import load
import plot
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tvbrain

""" 

Supplementary Module II: Computational Analysis of Data associated with Depression
Master of Clinical and Experimental Neuroscience 
University of Cologne

Thorben Greve
21.01.2025


"""

save = False
animate = False
load_preprocessed = False

''' IMPORT '''

dataset = 'TVB'
sampling_rates = {'Depresjon': 1 / 60,
                  'MODMA_3_rs': 250,
                  'MODMA_128_rs': 250,
                  'MODMA_128_task': 250,
                  'OpenNeuro_fMRI': False,
                  'OpenNeuro_MEG': False,
                  'OpenNeuro_66_rs': 500,
                  'MathTest': 500,
                  'TVB': 250}
n_channels = {'Depresjon': 1,
              'MODMA_3_rs': 3,
              'MODMA_128_rs': 128,
              'MODMA_128_task': 128,
              'OpenNeuro_fMRI': False,
              'OpenNeuro_MEG': False,
              'OpenNeuro_66_rs': 66,
              'MathTest': 1,
              'TVB': 76}

sampling_rate = sampling_rates[dataset]
n_channel = n_channels[dataset]

if not load_preprocessed:
    if dataset == 'Depresjon':
        data_list_MDD, data_list_HC = load.Depresjon(normalize=True)
    if dataset == 'MathTest':
        data_list_MDD, data_list_HC = load.MathTest()
    if dataset == 'OpenNeuro_66_rs':
        data_list_MDD, data_list_HC = load.OpenNeuro_66_rs()
    elif dataset == 'MODMA_3_rs':
        data_list_MDD, data_list_HC = load.MODMA_3_rs(preprocessing=True)
    elif dataset == 'MODMA_128_rs':
        data_list_MDD, data_list_HC = load.MODMA_128_rs(preprocessing=True, normalize=True)
else:
    if dataset == 'MODMA_128_rs':
        with open(f"preprocessed_data/{dataset}_MDD.pkl", "rb") as f:
            data_list_MDD = pickle.load(f)
        with open(f"preprocessed_data/{dataset}_HC.pkl", "rb") as f:
            data_list_HC = pickle.load(f)

''' PCA '''


def get_PCA_comp(data_list):
    comp_list = []
    for sub in data_list:
        comp = PCA.PCA_channel(sub)
        comp_list.append(comp)
    return comp_list


def get_ICA_comp(data_list):
    comp_list = []
    for sub in data_list:
        comp = PCA.ICA_channel(sub)
        comp_list.append(comp)
    return comp_list


# data_list_MDD = get_PCA_comp(data_list_MDD)
# data_list_HC = get_PCA_comp(data_list_HC)

''' THE VIRTUAL BRAIN '''

if dataset == "TVB":
    np.random.seed(42)
    tvb_sim_MDD = tvbrain.create_tvb_model(0.03, 0.05, 0.1, False, dt=0.1)
    results_MDD = tvb_sim_MDD.run()
    data_list_MDD = tvbrain.export_array(results_MDD, True)

    np.random.seed(1335)
    tvb_sim_HC = tvbrain.create_tvb_model(0.03, 0.3, 0.4, True, dt=0.1)
    results_HC = tvb_sim_HC.run()
    # tvbrain.plot_simulation(results_HC)
    data_list_HC = tvbrain.export_array(results_HC, True)

print('n MDD: ', len(data_list_MDD), ' ', data_list_MDD[0].shape, ' = [timepoints, channel]?')
print('n HC:  ', len(data_list_HC), ' ', data_list_HC[0].shape, ' = [timepoints, channel]?')

''' GET EXPONENT AND METRICS '''

if load_preprocessed:
    with open(f"preprocessed_data/{dataset}_exp_MDD.pkl", "rb") as f:
        exp_list_MDD = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_exp_HC.pkl", "rb") as f:
        exp_list_HC = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_hurst_MDD.pkl", "rb") as f:
        hurst_list_MDD = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_hurst_HC.pkl", "rb") as f:
        hurst_list_HC = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_sh_entr_MDD.pkl", "rb") as f:
        sh_entr_list_MDD = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_sh_entr_HC.pkl", "rb") as f:
        sh_entr_list_HC = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_spec_entr_MDD.pkl", "rb") as f:
        spec_entr_list_MDD = pickle.load(f)
    with open(f"preprocessed_data/{dataset}_spec_entr_HC.pkl", "rb") as f:
        spec_entr_list_HC = pickle.load(f)
else:
    exp_list_MDD = methods.get_exponent(data_list_MDD, sampling_rate)
    exp_list_HC = methods.get_exponent(data_list_HC, sampling_rate)
    hurst_list_MDD = methods.get_hurst(data_list_MDD)
    hurst_list_HC = methods.get_hurst(data_list_HC)
    print(data_list_MDD)
    sh_entr_list_MDD = methods.get_shannon_entropy(data_list_MDD)
    sh_entr_list_HC = methods.get_shannon_entropy(data_list_HC)
    spec_entr_list_MDD = methods.get_spectral_entropy(data_list_MDD, sampling_rate)
    spec_entr_list_HC = methods.get_spectral_entropy(data_list_HC, sampling_rate)

exp_list_all = methods.combine_MDD_HC(exp_list_MDD, exp_list_HC)
hurst_list_all = methods.combine_MDD_HC(hurst_list_MDD, hurst_list_HC)
sh_entr_list_all = methods.combine_MDD_HC(sh_entr_list_MDD, sh_entr_list_HC)
spec_entr_list_all = methods.combine_MDD_HC(spec_entr_list_MDD, spec_entr_list_HC)

sum_list_MDD = methods.get_summed_signal(data_list_MDD)
sum_list_HC = methods.get_summed_signal(data_list_HC)
sum_list_all = methods.combine_MDD_HC(sum_list_MDD, sum_list_HC)

# FD_list_MDD = methods.get_fractaldim(data_list_MDD)
# FD_list_HC = methods.get_fractaldim(data_list_HC)
# FD_list_all = methods.combine_MDD_HC(FD_list_MDD, FD_list_HC)


# metric = load.get_depression_metric(dataset)

# plot.plot_exp_against_metric(metric, exp_list_all, 'Scaling Exponent')
# plot.plot_exp_against_metric(metric, hurst_list_all, 'Hurst Exponent')
# plot.plot_exp_against_metric(metric, sh_entr_list_all, 'Shannon Entropy')
# plot.plot_exp_against_metric(metric, spec_entr_list_all, 'Spectral Entropy')

# plot.plot_exp_against_metric(metric, sum_list_all, 'Summed Signal')
# plot.plot_exp_against_metric(metric, FD_list_all, 'Fractal Dimension')

''' PLOT '''


plot.plot_exponent_hist(exp_list_MDD, exp_list_HC, n_channel, 'Scaling Exponent', density=True)
plot.plot_exponent_hist(hurst_list_MDD, hurst_list_HC, n_channel, 'Hurst Exponent', density=True)
plot.plot_exponent_hist(sh_entr_list_MDD, sh_entr_list_HC, n_channel, 'Shannon Entropy', density=True)
plot.plot_exponent_hist(spec_entr_list_MDD, spec_entr_list_HC, n_channel, 'Spectral Entropy', density=True)

# plot.plot_signal_and_FFT(data_list_MDD, sampling_rate, channel_wise=True, only_one=True)
plot.plot_signal_and_FFT(data_list_HC, sampling_rate, channel_wise=True, only_one=True)

# plot.plot_exponent_hist(sum_list_MDD, sum_list_HC, n_channel, 'Summed Signal', density=True)
# plot.plot_exponent_hist(FD_list_MDD, FD_list_HC, n_channel, 'Fractal Dimension', density=True)


# head_model.compute_mean_plot(exp_list_MDD,  exp_list_HC, 'Scaling Exponent MDD', n_channel)
# head_model.compute_mean_plot(exp_list_HC,  exp_list_MDD, 'Scaling Exponent HC', n_channel)
# head_model.compute_mean_plot(hurst_list_MDD,  hurst_list_HC, 'Hurst Exponent MDD', n_channel)
# head_model.compute_mean_plot(hurst_list_HC,  hurst_list_MDD, 'Hurst Exponent HC', n_channel)
# head_model.compute_mean_plot(sh_entr_list_MDD,  sh_entr_list_HC, 'Shannon Entropy MDD', n_channel)
# head_model.compute_mean_plot(sh_entr_list_HC,  sh_entr_list_MDD, 'Shannon Entropy HC', n_channel)
# head_model.compute_mean_plot(spec_entr_list_MDD,  spec_entr_list_HC, 'Spectral Entropy MDD', n_channel)
# head_model.compute_mean_plot(spec_entr_list_HC,  spec_entr_list_MDD, 'Spectral Entropy HC', n_channel)
#
#
# head_model.plot_electrodes_p_value(exp_list_MDD, exp_list_HC, n_channel)
# head_model.plot_electrodes_p_value(hurst_list_MDD, hurst_list_HC, n_channel)
# head_model.plot_electrodes_p_value(sh_entr_list_MDD, sh_entr_list_HC, n_channel)
# head_model.plot_electrodes_p_value(spec_entr_list_MDD, sh_entr_list_HC, n_channel)

''' SAVE AND EXPORT '''

if save:
    with open(f"preprocessed_data/{dataset}_MDD.pkl", "wb") as f:
        pickle.dump(data_list_MDD, f)
    with open(f"preprocessed_data/{dataset}_HC.pkl", "wb") as f:
        pickle.dump(data_list_HC, f)

    with open(f"preprocessed_data/{dataset}_exp_MDD.pkl", "wb") as f:
        pickle.dump(exp_list_MDD, f)
    with open(f"preprocessed_data/{dataset}_exp_HC.pkl", "wb") as f:
        pickle.dump(exp_list_HC, f)

    with open(f"preprocessed_data/{dataset}_hurst_MDD.pkl", "wb") as f:
        pickle.dump(hurst_list_MDD, f)
    with open(f"preprocessed_data/{dataset}_hurst_HC.pkl", "wb") as f:
        pickle.dump(hurst_list_HC, f)

    with open(f"preprocessed_data/{dataset}_sh_entr_MDD.pkl", "wb") as f:
        pickle.dump(sh_entr_list_MDD, f)
    with open(f"preprocessed_data/{dataset}_sh_entr_HC.pkl", "wb") as f:
        pickle.dump(sh_entr_list_HC, f)

    with open(f"preprocessed_data/{dataset}_spec_entr_MDD.pkl", "wb") as f:
        pickle.dump(spec_entr_list_MDD, f)
    with open(f"preprocessed_data/{dataset}_spec_entr_HC.pkl", "wb") as f:
        pickle.dump(spec_entr_list_HC, f)
    print('SAVED RESULTS')
