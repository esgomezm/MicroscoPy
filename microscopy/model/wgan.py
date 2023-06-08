import numpy as np
import torchvision

import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule

from ..datasets import PytorchDataset, ToTensor
from ..datasets import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate
from ..optimizer_scheduler_utils import select_optimizer, select_lr_schedule


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, p=1):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p),
        )

    def forward(self, x):
        return x + self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scaleFactor, k=3, p=1):
        super(UpsampleBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels * (scaleFactor**2), kernel_size=k, padding=p
            ),
            nn.PixelShuffle(scaleFactor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GeneratorUpsample(nn.Module):
    def __init__(self, n_residual=8, scale_factor=4):
        super(GeneratorUpsample, self).__init__()
        self.n_residual = n_residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2), nn.PReLU()
        )

        for i in range(n_residual):
            self.add_module("residual" + str(i + 1), ResidualBlock(64, 64))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.PReLU()
        )

        if scale_factor > 1:
            upsamples = [
                UpsampleBlock(64, 2) for x in range(int(np.log2(scale_factor)))
            ]
            self.upsample = nn.Sequential(
                *upsamples, nn.Conv2d(64, 1, kernel_size=5, padding=2)
            )
        else:
            self.upsample = nn.Sequential(nn.Conv2d(64, 1, kernel_size=5, padding=2))

    def forward(self, lr):
        # x = torch.cat((lr, noise), dim=1)
        y = self.conv1(lr)
        cache = y.clone()

        for i in range(self.n_residual):
            y = self.__getattr__("residual" + str(i + 1))(y)

        y = self.conv2(y)
        y = self.upsample(y + cache)
        # print ('G output size :' + str(y.size()))
        return torch.tanh(y)


class GeneratorModule(LightningModule):
    def __init__(self, n_residual=8, scale_factor=4, lr=0.001):
        super(GeneratorModule, self).__init__()

        self.save_hyperparameters()

        self.generator = GeneratorUpsample(
            n_residual=n_residual, scale_factor=scale_factor
        )

        self.l1loss = nn.L1Loss()

    def forward(self, x):
        y = self.generator(x)
        return y

    def training_step(self, batch, batch_idx):
        lr, hr = batch["lr"], batch["hr"]

        fake = self(lr)

        loss = self.l1loss(fake, hr)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        sched = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt, 0.0001, epochs=5, steps_per_epoch=712
            ),
            "interval": "step",
        }

        return [opt], [sched]


###


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            # nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, img):
        score = self.model(img)
        return torch.mean(score, dim=(-1, -2, -3))


###


class Discriminator_Complex(nn.Module):
    def __init__(self):
        super(Discriminator_Complex, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
        )

    def forward(self, img):
        score = self.model(img)
        return score


###

from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class WGANGP(LightningModule):
    def __init__(
        self,
        g_layers: int = 5,
        recloss: float = 10.0,
        lambda_gp: float = 10.0,
        n_critic_steps: int = 5,
        lr_patch_size_x: int = 128,
        lr_patch_size_y: int = 128,
        batchsize: int = 8,
        scale_factor: int = 2,
        datagen_sampling_pdf: int = 1,
        epochs: int = 151,
        learning_rate_d: float = 0.0001,
        learning_rate_g: float = 0.0001,
        g_optimizer: str = "Adam",
        d_optimizer: str = "Adam",
        g_scheduler: str = "OneCycle",
        d_scheduler: str = "OneCycle",
        rotation: bool = True,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        train_hr_path: str = "",
        train_lr_path: str = "",
        train_filenames: list = [],
        val_hr_path: str = "",
        val_lr_path: str = "",
        val_filenames: list = [],
        crappifier_method: str = "downsampleonly",
        gen_checkpoint: str = None,
        save_basedir: str = None,
        additonal_configuration: dict = {},
    ):
        super(WGANGP, self).__init__()

        self.save_hyperparameters()

        if gen_checkpoint is not None:
            checkpoint = torch.load(gen_checkpoint)
            self.generator = GeneratorModule(
                n_residual=checkpoint["n_residuals"],
                scale_factor=checkpoint["scale_factor"],
            )
            self.generator.load_state_dict(checkpoint["model_state_dict"])
            self.best_valid_loss = checkpoint["best_valid_loss"]
        else:
            self.generator = GeneratorModule(
                n_residual=g_layers, scale_factor=scale_factor
            )
            self.best_valid_loss = float("inf")

        self.discriminator = Discriminator()

        print(
            "Generators parameters: {}".format(
                sum(p.numel() for p in self.generator.parameters())
            )
        )
        print(
            "Discriminators parameters: {}".format(
                sum(p.numel() for p in self.discriminator.parameters())
            )
        )

        self.mae = nn.L1Loss()

        self.opt_g = None
        self.opt_d = None

        if train_hr_path or train_lr_path:
            self.len_data = self.train_dataloader().__len__()

    def save_model(self, filename):
        if self.hparams.save_basedir is not None:
            torch.save(
                {
                    "model_state_dict": self.generator.state_dict(),
                    "optimizer_state_dict": self.opt_g.state_dict(),
                    "n_residuals": self.hparams.g_layers,
                    "scale_factor": self.hparams.scale_factor,
                    "best_valid_loss": self.best_valid_loss,
                },
                self.hparams.save_basedir + "/" + filename,
            )
        else:
            raise Exception(
                "No save_basedir was specified in the construction of the WGAN object."
            )

    def forward(self, x):
        if isinstance(x, dict):
            return self.generator(x["lr"])
        else:
            return self.generator(x)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP

        Source: https://github.com/nocotan/pytorch-lightning-gans"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
            self.device
        )
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(d_interpolates.shape).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        # print("Interpolates", interpolates.shape)
        # print("d_interpolates", d_interpolates.shape)
        # print("gradients", len(gradients))
        gradients = gradients[0]
        # print("gradients", gradients.shape)

        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr, hr = batch["lr"], batch["hr"]

        # Optimize generator
        if optimizer_idx == 0:
            generated = self(lr)

            adv_loss = -1 * self.discriminator(generated).mean()
            error = self.mae(generated, hr)

            g_loss = adv_loss + error * self.hparams.recloss

            self.log("g_loss", g_loss, prog_bar=True, on_epoch=True)
            self.log("g_adv_loss", g_loss, prog_bar=True, on_epoch=True)
            self.log("g_l1", error, prog_bar=True, on_epoch=True)

            return g_loss

        # Optimize discriminator
        elif optimizer_idx == 1:
            generated = self(lr)

            real_logits = self.discriminator(hr).mean()
            fake_logits = self.discriminator(generated).mean()

            gradient_penalty = self.compute_gradient_penalty(hr.data, generated.data)

            wasserstein = real_logits - fake_logits

            d_loss = -wasserstein + self.hparams.lambda_gp * gradient_penalty

            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)
            self.log("d_wasserstein", wasserstein, prog_bar=False, on_epoch=True)
            self.log("d_real", real_logits, prog_bar=False, on_epoch=True)
            self.log("d_fake", fake_logits, prog_bar=False, on_epoch=True)
            self.log("d_gp", gradient_penalty, prog_bar=False, on_epoch=True)

            return d_loss

    def configure_optimizers(self):
        self.opt_g = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.g_optimizer,
            learning_rate=self.hparams.learning_rate_g,
            check_point=self.hparams.gen_checkpoint,
            parameters=self.generator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
        )

        self.opt_d = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.d_optimizer,
            learning_rate=self.hparams.learning_rate_d,
            check_point=None,
            parameters=self.discriminator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
        )

        sched_g = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.g_scheduler,
            data_len=self.len_data,
            number_of_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_g,
            monitor_loss="val_g_loss",
            name="g_lr",
            optimizer=self.opt_g,
            frequency=1,
            additional_configuration=self.hparams.additonal_configuration,
        )

        sched_d = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.d_scheduler,
            data_len=self.len_data,
            number_of_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_d,
            monitor_loss="val_g_loss",
            name="d_lr",
            optimizer=self.opt_d,
            frequency=self.hparams.n_critic_steps,
            additional_configuration=self.hparams.additonal_configuration,
        )

        return [self.opt_g, self.opt_d], [sched_g, sched_d]

    def validation_step(self, batch, batch_idx):
        # Right now used for just plotting, might want to change it later
        lr, hr = batch["lr"], batch["hr"]

        generated = self(lr)

        true = hr.cpu().numpy()
        fake = generated.cpu().numpy()
        xlr = lr.cpu().numpy()

        for i in range(lr.size(0)):
            ssim = structural_similarity(
                true[i, 0, ...], fake[i, 0, ...], data_range=1.0
            )
            self.log("val_ssim", ssim)

            psnr = peak_signal_noise_ratio(
                true[i, 0, ...], fake[i, 0, ...], data_range=1.0
            )
            self.log("val_psnr", psnr)

        return lr, hr, generated

    def validation_step_end(self, val_step_outputs):
        # Right now used for just plotting, might want to change it later
        lr, hr, generated = val_step_outputs

        adv_loss = -1 * self.discriminator(generated).mean()
        error = self.mae(generated, hr)

        g_loss = adv_loss + error * self.hparams.recloss

        self.log("val_g_loss", g_loss)
        self.log("val_g_l1", error)

        real_logits = self.discriminator(hr).mean()
        fake_logits = self.discriminator(generated).mean()

        wasserstein = real_logits - fake_logits

        self.log("val_d_wasserstein", wasserstein)

        if g_loss < self.best_valid_loss:
            self.best_valid_loss = g_loss
            self.save_model("best_checkpoint.pth")

    def on_train_end(self):
        self.save_model("last_checkpoint.pth")

    def train_dataloader(self):
        transformations = []

        if self.hparams.horizontal_flip:
            transformations.append(RandomHorizontalFlip())
        if self.hparams.vertical_flip:
            transformations.append(RandomVerticalFlip())
        if self.hparams.rotation:
            transformations.append(RandomRotate())

        transformations.append(ToTensor())

        transf = torchvision.transforms.Compose(transformations)

        if self.hparams.val_hr_path is None:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                val_split=0.1,
                validation=False,
            )

        else:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
            )

        return DataLoader(
            dataset, batch_size=self.hparams.batchsize, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        transf = ToTensor()

        if self.hparams.val_hr_path is None:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                val_split=0.1,
                validation=True,
            )
        else:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.val_hr_path,
                lr_data_path=self.hparams.val_lr_path,
                filenames=self.hparams.val_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
            )

        return DataLoader(
            dataset, batch_size=self.hparams.batchsize, shuffle=False, num_workers=0
        )

    def train_dataloader(self):
        transformations = []

        if self.hparams.horizontal_flip:
            transformations.append(RandomHorizontalFlip())
        if self.hparams.vertical_flip:
            transformations.append(RandomVerticalFlip())
        if self.hparams.rotation:
            transformations.append(RandomRotate())

        transformations.append(ToTensor())

        transf = torchvision.transforms.Compose(transformations)

        if self.hparams.val_hr_path is None:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                val_split=0.1,
                validation=False,
            )

        else:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
            )

        return DataLoader(
            dataset, batch_size=self.hparams.batchsize, shuffle=True, num_workers=1
        )

    def val_dataloader(self):
        transf = ToTensor()

        if self.hparams.val_hr_path is None:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.train_hr_path,
                lr_data_path=self.hparams.train_lr_path,
                filenames=self.hparams.train_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
                val_split=0.1,
                validation=True,
            )
        else:
            dataset = PytorchDataset(
                hr_data_path=self.hparams.val_hr_path,
                lr_data_path=self.hparams.val_lr_path,
                filenames=self.hparams.val_filenames,
                scale_factor=self.hparams.scale_factor,
                crappifier_name=self.hparams.crappifier_method,
                lr_patch_shape=(
                    self.hparams.lr_patch_size_x,
                    self.hparams.lr_patch_size_y,
                ),
                transformations=transf,
                datagen_sampling_pdf=self.hparams.datagen_sampling_pdf,
            )

        return DataLoader(
            dataset, batch_size=self.hparams.batchsize, shuffle=False, num_workers=1
        )
