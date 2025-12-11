from __future__ import annotations

from nucli_train.utils.registry import Registry

import mlflow

import numpy as np

import matplotlib.pyplot as plt

import skimage.metrics as skim

import pandas as pd

import os

DATA_RANGE = 20

def NRMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Normalized Root Mean Square Error between two image arrays.

    Args:
        y_true (np.ndarray): The ground truth image array.
        y_pred (np.ndarray): The predicted image array.

    Returns:
        float: The NRMSE value.
    """
    error = y_true - y_pred
    mse = np.mean(np.square(error))
    rmse = np.sqrt(mse)
    nrmse = rmse / np.sqrt(np.mean(np.square(y_true)) + 1e-8)  # Adding a small value to avoid division by zero
    return nrmse


def PSNR(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two image arrays.

    Args:
        y_true (np.ndarray): The ground truth image array.
        y_pred (np.ndarray): The predicted image array.

    Returns:
        float: The PSNR value.
    """
    mse = np.mean(np.square(y_true - y_pred))
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(DATA_RANGE / np.sqrt(mse))
    return psnr


def SSIM(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two image arrays.

    Args:
        y_true (np.ndarray): The ground truth image array.
        y_pred (np.ndarray): The predicted image array.

    Returns:
        float: The SSIM value.
    """
    return skim.structural_similarity(y_true, y_pred, data_range=DATA_RANGE)

EVALUATORS_REGISTRY = Registry('evaluators')

@EVALUATORS_REGISTRY.register('regression-evaluator')
class RegressionEvaluator:
    def __init__(self, save_dir):
        self.items_pred = {'center': [], 'dataset' : [], 'tracer': [], 'drf': [], 'subject_id': [], 'SUVstd_pred': [], 'SUVmean_pred': [], 'SUVmax_pred': [], 'SUVstd_gt': [], 'SUVmean_gt': [], 'SUVmax_gt': [],  'SSIM': [], 'PSNR': [], 'NRMSE': []}
        self.items_lc = {'dataset' : [], 'center': [], 'tracer': [], 'drf': [], 'subject_id': [], 'SUVstd_pred': [], 'SUVmean_pred': [], 'SUVmax_pred': [], 'SUVstd_gt': [], 'SUVmean_gt': [], 'SUVmax_gt': [],  'SSIM': [], 'PSNR': [], 'NRMSE': []}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

    def evaluate_batch(self, model_output, batch):
        lc = batch['lc'].squeeze().numpy().astype(np.float32)
        sc = batch['sc'].squeeze().numpy().astype(np.float32)

        prediction = model_output['predictions'].squeeze().numpy().astype(np.float32)

        sample_ids, centers, datasets, tracers, drfs = batch['sample_id'], batch['center'], batch['dataset'], batch['tracer'], batch['drf']

        suv_max_gt, suv_max_pred, suv_max_lc = np.max(sc, axis=(1, 2, 3)), np.max(prediction, axis=(1, 2, 3)), np.max(lc, axis=(1, 2, 3))
        suv_std_gt, suv_std_pred, suv_std_lc = np.std(sc, axis=(1, 2, 3)), np.std(prediction, axis=(1, 2, 3)), np.std(lc, axis=(1, 2, 3))
        suv_mean_gt, suv_mean_pred, suv_mean_lc = np.mean(sc, axis=(1, 2, 3)), np.mean(prediction, axis=(1, 2, 3)), np.mean(lc, axis=(1, 2, 3))

        for i in range(len(sample_ids)):
            self.items_pred['center'].append(centers[i])
            self.items_pred['dataset'].append(datasets[i])
            self.items_pred['tracer'].append(tracers[i])
            self.items_pred['drf'].append(drfs[i])
            self.items_pred['subject_id'].append(sample_ids[i])

            self.items_pred['SUVstd_pred'].append(suv_std_pred[i])
            self.items_pred['SUVmean_pred'].append(suv_mean_pred[i])
            self.items_pred['SUVmax_pred'].append(suv_max_pred[i])

            self.items_pred['SUVstd_gt'].append(suv_std_gt[i])
            self.items_pred['SUVmean_gt'].append(suv_mean_gt[i])
            self.items_pred['SUVmax_gt'].append(suv_max_gt[i])


            self.items_pred['SSIM'].append(SSIM(sc[i], prediction[i]))
            self.items_pred['PSNR'].append(PSNR(sc[i], prediction[i]))
            self.items_pred['NRMSE'].append(NRMSE(sc[i], prediction[i]))

            if self.items_lc is not None:
                self.items_lc['center'].append(centers[i])
                self.items_lc['dataset'].append(datasets[i])
                self.items_lc['tracer'].append(tracers[i])
                self.items_lc['drf'].append(drfs[i])
                self.items_lc['subject_id'].append(sample_ids[i])

                self.items_lc['SUVstd_pred'].append(suv_std_lc[i])
                self.items_lc['SUVmean_pred'].append(suv_mean_lc[i])
                self.items_lc['SUVmax_pred'].append(suv_max_lc[i])

                self.items_lc['SUVstd_gt'].append(suv_std_gt[i])
                self.items_lc['SUVmean_gt'].append(suv_mean_gt[i])
                self.items_lc['SUVmax_gt'].append(suv_max_gt[i])

                self.items_lc['SSIM'].append(SSIM(sc[i],lc[i]))
                self.items_lc['PSNR'].append(PSNR(sc[i], lc[i]))
                self.items_lc['NRMSE'].append(NRMSE(sc[i], lc[i]))





    def log_epoch(self, epoch):
        pred_df = pd.DataFrame(self.items_pred)
        lc_df = pd.DataFrame(self.items_lc)

        pred_df.to_csv(os.path.join(self.save_dir, f'predictions_epoch_{epoch}.csv'), index=False)
        lc_df.to_csv(os.path.join(self.save_dir, f'lc.csv'), index=False)


        # Reset the items for the next epoch
        self.items_pred = {key: [] for key in self.items_pred}
        self.items_lc = None

        


@EVALUATORS_REGISTRY.register('first-order-quantification-evaluator') 
class FirstOrderQuantificationEvaluator:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


        self.pred_values = {"SUVMax_difference" : [], "SUVMean_difference" : [], "SUVStd_difference" : []}

    def evaluate_batch(self, model_output, batch):
        target = batch['target'].squeeze().numpy()
        pred = model_output['predictions'].squeeze().numpy()

        suv_max_dif = np.max(pred, axis=(1, 2, 3)) - np.max(target, axis=(1, 2, 3))
        suv_std_dif = np.std(pred, axis=(1, 2, 3)) - np.std(target, axis=(1, 2, 3))
        suv_mean_dif = np.mean(pred, axis=(1, 2, 3)) - np.mean(target, axis=(1, 2, 3))

        self.pred_values["SUVMax_difference"].extend(suv_max_dif.tolist())
        self.pred_values["SUVStd_difference"].extend(suv_std_dif.tolist())
        self.pred_values["SUVMean_difference"].extend(suv_mean_dif.tolist())

    def log_epoch(self, epoch):
        mean_suv_max_dif = np.mean(self.pred_values["SUVMax_difference"])
        mean_suv_std_dif = np.mean(self.pred_values["SUVStd_difference"])
        mean_suv_mean_dif = np.mean(self.pred_values["SUVMean_difference"])

        mlflow.log_metric(f"SUVmax/{self.dataset_name}/Bias", mean_suv_max_dif, step=epoch)
        mlflow.log_metric(f"SUVstd/{self.dataset_name}/Bias", mean_suv_std_dif, step=epoch)
        mlflow.log_metric(f"SUVmean/{self.dataset_name}/Bias", mean_suv_mean_dif, step=epoch)

        std_suv_max_dif = np.std(self.pred_values["SUVMax_difference"])
        std_suv_std_dif = np.std(self.pred_values["SUVStd_difference"])
        std_suv_mean_dif = np.std(self.pred_values["SUVMean_difference"])

        upper_suv_max_dif = mean_suv_max_dif + 1.96*std_suv_max_dif 
        upper_suv_std_dif = mean_suv_std_dif + 1.96*std_suv_std_dif
        upper_suv_mean_dif = mean_suv_mean_dif + 1.96*std_suv_mean_dif

        lower_suv_max_dif = mean_suv_max_dif - 1.96*std_suv_max_dif
        lower_suv_std_dif = mean_suv_std_dif - 1.96*std_suv_std_dif
        lower_suv_mean_dif = mean_suv_mean_dif - 1.96*std_suv_mean_dif

        mlflow.log_metric(f"SUVmax/{self.dataset_name}/Upper", upper_suv_max_dif, step=epoch)
        mlflow.log_metric(f"SUVstd/{self.dataset_name}/Upper", upper_suv_std_dif, step=epoch)
        mlflow.log_metric(f"SUVmean/{self.dataset_name}/Upper", upper_suv_mean_dif, step=epoch)

        mlflow.log_metric(f"SUVmax/{self.dataset_name}/Lower", lower_suv_max_dif, step=epoch)
        mlflow.log_metric(f"SUVstd/{self.dataset_name}/Lower", lower_suv_std_dif, step=epoch)
        mlflow.log_metric(f"SUVmean/{self.dataset_name}/Lower", lower_suv_mean_dif, step=epoch)



        # log_epoch is only called when eval is done, so reset for next time
        self.pred_values = {"SUVMax_difference" : [], "SUVMean_difference" : [], "SUVStd_difference" : []}
        
@EVALUATORS_REGISTRY.register('save-preds')
class SavePredictionEvaluation:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def evaluate_batch(self, model_output, batch):
        self.prediction, self.batch = model_output['predictions'], batch
    def log_epoch(self, epoch):
        fig, axs = plt.subplots(min(self.batch['input'].shape[0], 3), 3)
        for i in range(min(self.batch['input'].shape[0], 3)):
            axs[i, 0].imshow(self.batch['input'][i, 0, :, 32, :].cpu().numpy(), cmap='gray', vmin=0.0, vmax=2.0)
            axs[i, 0].set_axis_off()
            axs[i, 1].imshow(self.prediction[i, 0, :, 32, :].cpu().numpy(), cmap='gray', vmin=0.0, vmax=2.0)
            axs[i, 1].set_axis_off()
            axs[i, 2].imshow(self.batch['target'][i, 0, :, 32, :].cpu().numpy(), cmap='gray', vmin=0.0, vmax=2.0)
            axs[i, 2].set_axis_off()
        plt.tight_layout()
        mlflow.log_figure(fig, artifact_file=f"{self.dataset_name}/predictions/epoch_{epoch}.png")
        plt.close(fig)




@EVALUATORS_REGISTRY.register('save-preds-VOCO')
class SavePredictionVOCO:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        print("SavePredictionVOCO initialized")

    def max_intensity_projection(self, images, axis):
        return np.max(images, axis=axis)

    def evaluate_batch(self, model_output, batch):
        self.bases, self.targets, self.pred_losses, self.reg_losses, self.gt_overlaps, self.num_target_crops, self.num_base_crops, self.batch_size, self.data = model_output['bases'], model_output['targets'], model_output['pred_loss'], model_output['reg_loss'], model_output['overlaps'], model_output["num_target"], model_output["num_base"], model_output["batch_size"], model_output["data"]
        self.selected_batch = np.random.randint(0, self.batch_size)
        print(f"bases shape: {self.bases.shape}, targets shape: {self.targets.shape}, selected batch: {self.selected_batch}")
        self.bases = self.bases[self.selected_batch * self.num_base_crops : self.selected_batch * self.num_base_crops + self.num_base_crops].detach().cpu().numpy()
        self.targets = self.targets[self.selected_batch * self.num_target_crops:self.selected_batch * self.num_target_crops + self.num_target_crops].detach().cpu().numpy()
        print(f"bases shape: {self.bases.shape}, targets shape: {self.targets.shape}, selected batch: {self.selected_batch}")
        self.data = self.data[self.selected_batch][0].detach().cpu().numpy()



    def log_matrix_artifact(self, cosine_sim_bt_base_target, overlaps_bt_base_target, cosine_sim_bt_base): 
        plt.rcParams.update(plt.rcParamsDefault)
    
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        
        im0 = axs[0].imshow(cosine_sim_bt_base_target, cmap='plasma', aspect='auto', vmin = 0, vmax = 1)
        axs[0].set_title("Cosine similarity\nbetween base crops and target crops")
        axs[0].set_xlabel("Base crops")
        axs[0].set_ylabel("Target crops")
        fig.colorbar(im0, ax=axs[0])

        
        im1 = axs[1].imshow(overlaps_bt_base_target, cmap='plasma', aspect='auto', vmin = 0, vmax = 1)
        axs[1].set_title("Overlaps\nbetween base crops and target crops")
        axs[1].set_xlabel("Base crops")
        axs[1].set_ylabel("Target crops")
        fig.colorbar(im1, ax=axs[1])

        
        im2 = axs[2].imshow(cosine_sim_bt_base, cmap='coolwarm', vmin = 0, vmax = 1, aspect='auto')
        axs[2].set_title("Cosine similarity\nbetween base crops")
        axs[2].set_xlabel("Base crops")
        axs[2].set_ylabel("Base crops")
        fig.colorbar(im2, ax=axs[2])

        plt.tight_layout()
        mlflow.log_figure(fig, "validation_epoch_" + str(self.current_epoch) + ".png") 
        plt.close(fig)

    def log_image_grid(self, images, n_images_axis1, n_images_axis2, title, cmap='gray'): 
        rows = n_images_axis1
        cols = n_images_axis2

        fig, axes = plt.subplots(n_images_axis1, n_images_axis2, figsize=(n_images_axis2 * 3, n_images_axis1 * 3))
        fig.suptitle(title, fontsize=16)

        for i in range(n_images_axis1):
            for j in range(n_images_axis2):
                print(f"Image index: {i * n_images_axis2 + j}")
                print(f"n_images_axis1: {n_images_axis1}, n_images_axis2: {n_images_axis2}")
                print(f"Total images: {len(images)}")
                idx = i * n_images_axis2 + j


                axes[i, j].imshow(images[idx], cmap=cmap)
                axes[i, j].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        mlflow.log_figure(fig, "validation_epoch_" + str(self.current_epoch) + "_" + title +".png")

    def log_target_crops_artifact(self, target_crops):

        target_crops_dim = target_crops[0].shape

    
        # Max intensity projections
        max_intensity_projection_x = self.max_intensity_projection(target_crops, axis=1)
        max_intensity_projection_y = self.max_intensity_projection(target_crops, axis=2)
        max_intensity_projection_z = self.max_intensity_projection(target_crops, axis=3)

        # -------- Création de la figure avec 3 lignes --------
        fig, axes = plt.subplots(3, self.num_target_crops, figsize=(self.num_target_crops * 2.5, 3 * 2.5))
        fig.suptitle("Target crops: max X / max Y / max Z slices", fontsize=18)
        
        # Affichage des slices avec marges
        for i in range(self.num_target_crops):
            axes[0, i].imshow(max_intensity_projection_x[i], cmap='gray', vmin = 0, vmax = 5)
            axes[0, i].axis('off')
            axes[1, i].imshow(max_intensity_projection_y[i], cmap='gray', vmin = 0, vmax = 5)
            axes[1, i].axis('off')
            axes[2, i].imshow(max_intensity_projection_z[i], cmap='gray', vmin = 0, vmax = 5)
            axes[2, i].axis('off')

        # Titres pour les lignes
        axes[0, 0].set_ylabel("Max-X", fontsize=14)
        axes[1, 0].set_ylabel("Max-Y", fontsize=14)
        axes[2, 0].set_ylabel("Max-Z", fontsize=14)

        # Espacement clair
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        mlflow.log_figure(fig, f"validation_epoch_{self.current_epoch}_target_crops_grid.png")
        plt.close(fig)


    def log_base_crops_artifact(self, base_crops): 
        

        base_crops_dim = base_crops[0].shape

        # Max intensity projections
        max_intensity_projection_x = self.max_intensity_projection(base_crops, axis=1)
        max_intensity_projection_y = self.max_intensity_projection(base_crops, axis=2)
        max_intensity_projection_z = self.max_intensity_projection(base_crops, axis=3)

        # -------- Création de la figure avec 3 lignes --------
        fig, axes = plt.subplots(3, self.num_base_crops, figsize=(self.num_base_crops * 2.5, 3 * 2.5))  # Taille ajustable
        fig.suptitle("Base crops: max X / max Y / max Z", fontsize=18)

        # Affichage des slices avec marges
        for i in range(self.num_base_crops):
            axes[0, i].imshow(max_intensity_projection_x[i], cmap='gray', vmin = 0, vmax = 5)
            axes[0, i].axis('off')
            axes[1, i].imshow(max_intensity_projection_y[i], cmap='gray', vmin = 0, vmax = 5)
            axes[1, i].axis('off')
            axes[2, i].imshow(max_intensity_projection_z[i], cmap='gray', vmin = 0, vmax = 5)
            axes[2, i].axis('off')

        # Titres pour les lignes
        axes[0, 0].set_ylabel("Max-X", fontsize=14)
        axes[1, 0].set_ylabel("Max-Y", fontsize=14)
        axes[2, 0].set_ylabel("Max-Z", fontsize=14)

        # Espacement clair
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        mlflow.log_figure(fig, f"validation_epoch_{self.current_epoch}_base_crops_grid.png")
        plt.close(fig)

    def log_data_artifact(self, data):

        data_dim = data[0].shape

        # Max intensity projections
        max_intensity_projection_x = self.max_intensity_projection(data, axis=0)
        max_intensity_projection_y = self.max_intensity_projection(data, axis=1)
        max_intensity_projection_z = self.max_intensity_projection(data, axis=2)

        # -------- Création de la figure avec 3 lignes --------
        fig, axes = plt.subplots(1, 3, figsize=(3 * 8, 8))  # Taille ajustable
        fig.suptitle("Original patch", fontsize=18)

        # Affichage des slices avec marges
        
        axes[0].imshow(max_intensity_projection_x, cmap='gray', vmin = 0, vmax = 5)
        axes[0].axis('off')
        axes[1].imshow(max_intensity_projection_y, cmap='gray', vmin = 0, vmax = 5)
        axes[1].axis('off')
        axes[2].imshow(max_intensity_projection_z, cmap='gray', vmin = 0, vmax = 5)
        axes[2].axis('off')

        # Titres pour les lignes
        axes[0].set_ylabel("Mid-X", fontsize=14)
        axes[1].set_ylabel("Mid-Y", fontsize=14)
        axes[2].set_ylabel("Mid-Z", fontsize=14)

        # Espacement clair
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        mlflow.log_figure(fig, f"validation_epoch_{self.current_epoch}_patch.png")
        plt.close(fig)



    def log_epoch(self, epoch):
        print("Logging epoch", epoch)
        self.current_epoch = epoch
        self.log_matrix_artifact(
            cosine_sim_bt_base_target=self.pred_losses["matrix"][self.selected_batch].detach().cpu().numpy(),
            overlaps_bt_base_target=self.gt_overlaps[self.selected_batch][0].detach().cpu().numpy(),
            cosine_sim_bt_base=self.reg_losses["matrix"][self.selected_batch].detach().cpu().numpy()
        )
        
        self.log_base_crops_artifact(self.bases)
        self.log_target_crops_artifact(self.targets)
        self.log_data_artifact(self.data)



@EVALUATORS_REGISTRY.register('save-preds-MAE')
class SavePredictionMAE:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        print("SavePredictionMAE initialized")

    def get_slices(self, volume, num_slices=5): 
        depth = volume.shape[2]
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        return [volume[:, :, idx] for idx in indices]

    def evaluate_batch(self, model_output, batch):
        self.batch_size = model_output['batch_size']
        self.selected_batch = np.random.randint(0, self.batch_size)

        self.losses = model_output['losses']
        self.prediction = model_output['output'][self.selected_batch][0].detach().cpu().numpy()
        self.masked_data = model_output['masked_data'][self.selected_batch][0].detach().cpu().numpy()
        self.data = model_output['data'][self.selected_batch][0].detach().cpu().numpy()


    def log_epoch(self, epoch):
        data_slices = self.get_slices(self.data, num_slices=5)
        masked_data_slices = self.get_slices(self.masked_data, num_slices=5)
        prediction_slices = self.get_slices(self.prediction, num_slices=5)

        fig, axes = plt.subplots(3, 5, figsize=(15, 9))

        for row, slices in enumerate([data_slices, masked_data_slices, prediction_slices]):
            for col, slice_img in enumerate(slices):
                ax = axes[row, col]
                ax.imshow(slice_img, cmap='gray', vmin=0.0, vmax=3.0)
                ax.axis('off')

        
        axes[0, 0].set_ylabel("Data", fontsize=12)
        axes[1, 0].set_ylabel("Masked", fontsize=12)
        axes[2, 0].set_ylabel("Output", fontsize=12)

        plt.tight_layout()

        mlflow.log_figure(fig, "validation_predictions_epoch_" + str(epoch) + ".png")  
        plt.close()





@EVALUATORS_REGISTRY.register('save-preds-MAE-convnext')
class SavePredictionMAEConvnext:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        print("SavePredictionMAEConvnext initialized")

    def get_slices(self, volume, num_slices=5): 
        depth = volume.shape[2]
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        return [volume[:, :, idx] for idx in indices]

    def evaluate_batch(self, model_output, batch):
        self.batch_size = model_output['batch_size']
        self.selected_batch = np.random.randint(0, self.batch_size)

        self.losses = model_output['losses']
        self.prediction = model_output['output'][self.selected_batch][0].detach().cpu().numpy()
        self.data = model_output['data'][self.selected_batch][0].detach().cpu().numpy()
        self.masked_data = model_output['masked_data'][self.selected_batch][0].detach().cpu().numpy()


    def log_epoch(self, epoch):
        data_slices = self.get_slices(self.data, num_slices=5)
        prediction_slices = self.get_slices(self.prediction, num_slices=5)
        masked_data_slices = self.get_slices(self.masked_data, num_slices=5)

        fig, axes = plt.subplots(3, 5, figsize=(15, 9))

        for row, slices in enumerate([data_slices, prediction_slices, masked_data_slices]):
            for col, slice_img in enumerate(slices):
                ax = axes[row, col]
                ax.imshow(slice_img, cmap='gray', vmin=0.0, vmax=3.0)
                ax.axis('off')

        
        axes[0, 0].set_ylabel("Data", fontsize=12)
        axes[1, 0].set_ylabel("Output", fontsize=12)
        axes[2, 0].set_ylabel("Masked", fontsize=12)

        plt.tight_layout()

        mlflow.log_figure(fig, "validation_predictions_epoch_" + str(epoch) + ".png")  
        plt.close()

    