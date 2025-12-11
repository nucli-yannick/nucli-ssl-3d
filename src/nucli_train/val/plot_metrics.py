from __future__ import annotations

import os
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nucli_train.val.metrics import PSNR_builder, RMSE, SSIM_Builder, SUVMax


def get_standard_metrics():
    PSNR = PSNR_builder(data_range=20)
    SSIM = SSIM_Builder(data_range=20)
    return {"RMSE": RMSE, "SSIM":SSIM, "PSNR": PSNR, "SUVMax_difference": SUVMax}


def plot_metric(high_dose_images, low_dose_images, denoised_images, dataset_name, save_dir, metric, relative=True):
    metric_builder = get_standard_metrics()[metric]



    # First let's calculate all the data we need!

    num_slices = len(high_dose_images)
    models = denoised_images.keys()

    


    model_metrics = {model: metric_builder() for model in models}
    ld_metric = metric_builder()

    ld_values = []
    denoised_values = {model: [] for model in models}
    denoised_relative_values = {model: [] for model in models}

    for slice_ind, (high_dose_img, low_dose_img) in enumerate(zip(high_dose_images, low_dose_images)):
        ld_value = float(ld_metric.forward(high_dose_img, low_dose_img))
        ld_values.append(ld_value)
        for model in models:
            denoised_value = float(model_metrics[model].forward(high_dose_img, denoised_images[model][slice_ind]))
            denoised_values[model].append(denoised_value)
            if relative:
                relative_value = ((denoised_value/ld_value) - 1) * 100
                denoised_relative_values[model].append(relative_value)


    # Putting the data into formats we can use with pandas/pyplot

    avgs = {model: model_metrics[model].compute() for model in denoised_images.keys()}
    avgs["Low Dose"] = ld_metric.compute()
    if relative:
        avgs_relative = {model:sum(denoised_relative_values[model])/num_slices for model in models}
        


    data = {"Model": [], "Low Dose": [], "Denoised": [], "Relative Differences": []}
    for model, model_denoised_vals in denoised_values.items():
        data["Model"] += [model] * num_slices
        data["Low Dose"] += ld_values
        data["Denoised"] += model_denoised_vals
        if relative:
            data["Relative Differences"] += denoised_relative_values[model]

    if not relative:
        data.pop("Relative Differences", None)

    ''''tensor_ld, tensor_denoised = torch.tensor(data["Low Dose"]), torch.tensor(data["Denoised"])

    minima = {"Low Dose": tensor_ld.min(), "Denoised": tensor_denoised.min()}
    maxima = {"Low Dose": tensor_ld.max(), "Denoised": tensor_denoised.max()}'''





    # Now we visualize the data in several ways to gain more insight on performance differences between models :)



    # Plot the single number comparisons

    plt.figure(figsize=(10, 6))
    plt.bar(avgs.keys(), avgs.values(), color='skyblue')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(metric + " with high-dose on " + dataset_name, fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.ylim(torch.tensor(avgs.values()).min(), torch.tensor(avgs.values()).max())
    plt.savefig(os.path.join(save_dir, metric + '_absolute_' + dataset_name + '.png'))

    plt.cla() # Close axes
    plt.clf() # Close figure
    plt.close('all') # Close figure windows

    if relative:
        plt.figure(figsize=(10, 6))
        plt.bar(avgs_relative.keys(), avgs_relative.values(), color='skyblue')
        plt.xlabel('Model', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.title(metric + " difference (%) to low-dose on " + dataset_name, fontsize=16)
        plt.xticks(rotation=45, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        #plt.ylim(0.1, 0.18)
        plt.savefig(os.path.join(save_dir, metric + '_relative_' + dataset_name + '.png'))


    plt.cla() 
    plt.clf() 
    plt.close('all')


    # Absolute scatter


    df = pd.DataFrame(data, copy=True)


    bins = np.linspace(df["Low Dose"].min(), df["Low Dose"].max(), 20)
    df['Binned Low Dose'] = pd.cut(df["Low Dose"], bins=bins, labels=bins[:-1])


    binned_stats = df.groupby(['Model', 'Binned Low Dose']).agg(
        Mean_Low_Dose=('Low Dose', 'mean'),
        Mean_Denoised=('Denoised', 'mean'),
        Std_Denoised=('Denoised', 'std')
    ).reset_index()


    g = sns.JointGrid(data=df, x="Low Dose", y="Denoised", hue="Model", height=8)
    g.plot_marginals(sns.kdeplot, common_norm=False, fill=True, alpha=0.2, linewidth=0)


    for model in binned_stats['Model'].unique():
        subset = binned_stats[binned_stats['Model'] == model]
        print(subset.keys())
        g.ax_joint.errorbar(
            subset['Mean_Low_Dose'], 
            subset['Mean_Denoised'], 
            yerr=subset['Std_Denoised'], 
            fmt='o', 
            label=model, 
            capsize=3, 
            alpha=0.8
        )


    max_val_ld, max_val_denoised = binned_stats["Mean_Low_Dose"].max(), binned_stats["Mean_Denoised"].max()
    min_val_ld, min_val_denoised = binned_stats["Mean_Low_Dose"].min(), binned_stats["Mean_Denoised"].min()
    g.ax_joint.plot([min_val_ld, max_val_ld], [min_val_denoised, max_val_denoised], '--', color='gray')
    g.ax_joint.set_xlim(min_val_ld - 0.02*min_val_ld, max_val_ld + 0.02*min_val_ld)
    g.ax_joint.set_ylim(min_val_denoised - 0.02*min_val_denoised, max_val_denoised + 0.02*min_val_denoised)


    g.set_axis_labels("Low Dose " + metric, "Denoised " + metric, fontsize=12)
    g.ax_joint.legend(title="Model")
    plt.subplots_adjust(top=0.9) 
    g.fig.suptitle("Absolute" + metric)

    plt.savefig(os.path.join(save_dir, metric + '_scatter_absolute_' + dataset_name +  '.png'))

    plt.cla() 
    plt.clf() 
    plt.close('all')




    # Relative scatter

    if relative:
        df = pd.DataFrame(data, copy=True)


        bins = np.linspace(df["Low Dose"].min(), df["Low Dose"].max(), 20)
        df['Binned Low Dose'] = pd.cut(df["Low Dose"], bins=bins, labels=bins[:-1])


        binned_stats = df.groupby(['Model', 'Binned Low Dose']).agg(
            Mean_Low_Dose=('Low Dose', 'mean'),
            Mean_Relative_Differences=('Relative Differences', 'mean'),
            Std_Relative_Differences=('Relative Differences', 'std')
        ).reset_index()


        g = sns.JointGrid(data=df, x="Low Dose", y="Relative Differences", hue="Model", height=8)
        g.plot_marginals(sns.kdeplot, common_norm=False, fill=True, alpha=0.2, linewidth=0)


        for model in binned_stats['Model'].unique():
            subset = binned_stats[binned_stats['Model'] == model]
            g.ax_joint.errorbar(
                subset['Mean_Low_Dose'], 
                subset['Mean_Relative_Differences'], 
                yerr=subset['Std_Relative_Differences'], 
                fmt='o', 
                label=model, 
                capsize=3, 
                alpha=0.8
            )


        #max_val = max(binned_stats["Mean_Low_Dose"].max(), binned_stats["Mean_Relative_Differences"].max())
        #g.ax_joint.plot([0.0, max_val], [0.0, max_val], '--', color='gray')

        #g.ax_joint.set_ylim(binned_stats["Mean_Relative_Differences"].min(), binned_stats["Mean_Relative_Differences"].max())


        g.set_axis_labels("Low Dose " + metric, "Denoising impact (%) on " + metric, fontsize=12)
        g.ax_joint.legend(title="Model")
        plt.subplots_adjust(top=0.9) 
        g.fig.suptitle("Relative (%) " + metric)

        plt.savefig(os.path.join(save_dir, metric + '_scatter_relative_' + dataset_name +  '.png'))

        plt.cla() 
        plt.clf() 
        plt.close('all')


