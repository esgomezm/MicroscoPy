from pytorch_lightning.callbacks import Callback
from matplotlib import pyplot as plt
from . import utils, datasets

class PerformancePlotCallback_Pytorch(Callback):
    def __init__(self, x_test, y_test, img_saving_path, frequency=1):
        self.x_test = x_test
        self.y_test = y_test
        self.img_saving_path = img_saving_path
        self.frequency = frequency


    def on_train_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.frequency == 0:
            y_pred = pl_module.forward(self.x_test)

            ssim = utils.ssim_loss(self.y_test[0], y_pred[0])

            plt.switch_backend("agg")
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Input LR image")
            plt.imshow(self.x_test[0], "gray")
            plt.subplot(1, 4, 2)
            plt.title("Ground truth")
            plt.imshow(self.y_test[0], "gray")
            plt.subplot(1, 4, 3)
            plt.title("Prediction")
            plt.imshow(y_pred[0], "gray")
            plt.subplot(1, 4, 4)
            plt.title(f"SSIM: {ssim.numpy():.3f}")
            plt.imshow(1 - datasets.normalization(self.y_test[0] - y_pred[0]), "inferno")

            plt.tight_layout()
            plt.savefig(f"{self.img_saving_path}/{pl_module.current_epoch}.png")
            plt.close()