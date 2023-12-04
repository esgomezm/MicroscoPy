# Model based on: https://github.com/lizhuoq/SRGAN

import numpy as np
import os
import math
import torch
from torch import nn
from torchvision.models.vgg import vgg16
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

import lightning as L

from ..optimizer_scheduler_utils import select_optimizer, select_lr_schedule

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, scale_factor) -> None:
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4), 
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, in_channels, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, in_channels) -> None:
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2), 

            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(512, 1024, kernel_size=1), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        
        # Adversarial Loss
        # we want real_out to be close 1, and fake_out to be close 0
        adversarial_loss = torch.mean(1 - out_labels)
        
        # Perception Loss
        out_images_3c = torch.cat([out_images, out_images, out_images], dim=1) # VGG16 needs a 3 channel input
        target_images_3c = torch.cat([target_images, target_images, target_images], dim=1) # VGG16 needs a 3 channel input
        with torch.no_grad():
            perception_loss = self.mse_loss(self.loss_network(out_images_3c), self.loss_network(target_images_3c))
        
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss 
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        return self.tv_loss_weight * 0.5 * (
            torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean() + 
            torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        )
    
class SRGAN(L.LightningModule):
    def __init__(
        self,
        datagen_sampling_pdf: int = 1,
        n_critic_steps: int = 5,
        data_len: int = 8,
        epochs: int = 151,
        scale_factor: int = 2,
        learning_rate_d: float = 0.0001,
        learning_rate_g: float = 0.0001,
        g_optimizer: str = None,
        d_optimizer: str = None,
        g_scheduler: str = None,
        d_scheduler: str = None,
        save_basedir: str = None,
        gen_checkpoint: str = None,
        additonal_configuration: dict = {},
        verbose: int = 0,
    ):
        super(SRGAN, self).__init__()
        self.save_hyperparameters()

        self.verbose = verbose
        print('self.verbose: {}'.format(self.verbose))

        if self.verbose > 0:
            print('\nVerbose: Model initialized (begining)\n')

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Free cuda memory
        torch.cuda.empty_cache()

        # Initialize generator and load the checkpoint in case is given
        if gen_checkpoint is None:
            self.generator = Generator(in_channels=1, scale_factor=scale_factor)
            self.best_valid_loss = float("inf")
        else:
            checkpoint = torch.load(gen_checkpoint)
            self.generator = Generator(in_channels=1, scale_factor=checkpoint["scale_factor"])
            self.generator.load_state_dict(checkpoint["model_state_dict"])
            self.best_valid_loss = checkpoint["best_valid_loss"]

        # self.discriminator = define_D(base_nf=64)
        self.discriminator = Discriminator(in_channels=1)

        # GD gan loss
        self.cri_gan = GeneratorLoss()

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = PeakSignalNoiseRatio()

        self.data_len = self.hparams.data_len

        # Metric lists for validation
        self.val_g_loss = []
        self.val_ssim = []
        

        if self.verbose > 0:
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
            print(
                "self.netF parameters: {}".format(
                    sum(p.numel() for p in self.netF.parameters())
                )
            )
        
        
        if self.verbose > 0:
            os.makedirs(f"{self.hparams.save_basedir}/training_images", exist_ok=True)

        self.step_schedulers = ['CosineDecay', 'OneCycle', 'MultiStepScheduler']
        self.epoch_schedulers = ['ReduceOnPlateau']

        if self.verbose > 0:
            print('\nVerbose: Model initialized (end)\n')

    def forward(self, x):
        if isinstance(x, dict):
            return self.generator(x["lr"])
        else:
            return self.generator(x)

    def training_step(self, batch, batch_idx):

        if self.verbose > 1:
            print('\nVerbose: Training step (begining)\n')

        lr, hr = batch["lr"], batch["hr"]

        # Extract the optimizers
        g_opt, d_opt = self.optimizers()

        # Extract the schedulers
        if self.hparams.g_scheduler == "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_d = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler == "Fixed":
            sched_g = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_g, sched_d = self.lr_schedulers()
        else:
            # There are no schedulers
            pass

        # The generator is updated every self.hparams.n_critic_steps
        if (batch_idx + 1) % self.hparams.n_critic_steps == 0:
            if self.verbose > 0:
                print(f'Generator updated on step {batch_idx + 1}')

            # Optimize generator
            # toggle_optimizer(): Makes sure only the gradients of the current optimizer's parameters are calculated
            #                     in the training step to prevent dangling gradients in multiple-optimizer setup.
            self.toggle_optimizer(g_opt)

            fake_hr = self.generator(lr)
            fake_out = self.discriminator(fake_hr).mean()

            l_g_total = self.cri_gan(fake_out, fake_hr, hr)

            self.log("g_loss", l_g_total, prog_bar=True, on_epoch=True)

            # Optimize generator
            self.manual_backward(l_g_total)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)
            
            if self.hparams.g_scheduler in self.step_schedulers:
                sched_g.step()

        # The discriminator is updated every step
        if (batch_idx + 1) % 1 == 0:
            if self.verbose > 0:
                print(f'Discriminator  updated on step {batch_idx + 1}')
            # Optimize discriminator
            self.toggle_optimizer(d_opt)

            fake_hr = self.generator(lr)

            real_out = self.discriminator(hr).mean()
            fake_out = self.discriminator(fake_hr).mean()

            l_d_total = 1 - real_out + fake_out

            self.log("d_loss", l_d_total, prog_bar=True, on_epoch=True)
            self.log("real_out", real_out, prog_bar=False, on_epoch=True)
            self.log("fake_out", fake_out, prog_bar=False, on_epoch=True)
 
            # Optimize discriminator/critic
            self.manual_backward(l_d_total)
            d_opt.step()
            d_opt.zero_grad()
            self.untoggle_optimizer(d_opt)

            if self.hparams.d_scheduler in self.step_schedulers:
                sched_d.step()

        if self.verbose > 0:
            print('\nVerbose: Training step (end)\n')

    def validation_step(self, batch, batch_idx):
        
        if self.verbose > 0:
            print('\nVerbose: validation_step (begining)\n')
            
        # Right now used for just plotting, might want to change it later
        lr, hr = batch["lr"], batch["hr"]
        fake_hr = self.generator(lr)
        fake_out = self.discriminator(fake_hr).mean()

        if self.verbose > 0:
            print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()}')
            print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()}')
            print(f'generated.shape: {fake_out.shape} generated.min: {fake_out.min()} generated.max: {fake_out.max()}')

        val_g_loss = self.cri_gan(fake_out, fake_hr, hr)
        val_ssim = self.ssim(fake_hr, hr)
        val_psnr = self.psnr(fake_hr, hr)

        self.val_g_loss.append(val_g_loss.cpu().numpy())
        self.val_ssim.append(val_ssim.cpu().numpy())

        if self.verbose > 0:
            print('\nVerbose: validation_step (end)\n')
    
        self.log("val_g_loss", val_g_loss, prog_bar=True, on_epoch=True)
        self.log("val_ssim", val_ssim, prog_bar=True, on_epoch=True)
        self.log("val_psnr", val_psnr, prog_bar=False, on_epoch=True)

    def on_validation_epoch_end(self):

        if self.verbose > 0:
            print('\nVerbose: on_validation_epoch_end (begining)\n')

        mean_val_g_loss = np.array(self.val_g_loss).mean()

        if self.verbose > 0:
            print(f'g_loss: {mean_val_g_loss}')
            print(f'self.best_valid_loss: {self.best_valid_loss}')

        if mean_val_g_loss < self.best_valid_loss:
            self.best_valid_loss = mean_val_g_loss
            self.save_model("best_checkpoint.pth")

        self.val_g_loss.clear() # free memory

        # Extract the schedulers
        if self.hparams.g_scheduler == "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_d = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler == "Fixed":
            sched_g = self.lr_schedulers()
        elif self.hparams.g_scheduler != "Fixed" and self.hparams.d_scheduler != "Fixed":
            sched_g, sched_d = self.lr_schedulers()
        else:
            # There are no schedulers
            pass
        
        mean_val_ssim = np.array(self.val_ssim).mean()

        # Note that step should be called after validate()
        if self.hparams.d_scheduler in self.epoch_schedulers:
            sched_d.step(mean_val_ssim)
        if self.hparams.g_scheduler in self.epoch_schedulers:
            sched_g.step(mean_val_ssim)

        self.val_ssim.clear() # free memory

        if self.verbose > 0:
            print('\nVerbose: on_validation_epoch_end (end)\n')

    def on_train_end(self):
        self.save_model("last_checkpoint.pth")

    def configure_optimizers(self):
        
        if self.verbose > 0:
            print('\nVerbose: configure_optimizers (begining)\n')
            print(f'Generator optimizer: {self.hparams.g_optimizer}')
            print(f'Discriminator optimizer: {self.hparams.d_optimizer}')
            print(f'Generator scheduler: {self.hparams.g_scheduler}')
            print(f'Discriminator scheduler: {self.hparams.d_scheduler}')

        self.opt_g = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.g_optimizer,
            learning_rate=self.hparams.learning_rate_g,
            check_point=self.hparams.gen_checkpoint,
            parameters=self.generator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        self.opt_d = select_optimizer(
            library_name="pytorch",
            optimizer_name=self.hparams.d_optimizer,
            learning_rate=self.hparams.learning_rate_d,
            check_point=None,
            parameters=self.discriminator.parameters(),
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_g = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.g_scheduler,
            data_len=self.data_len,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_g,
            monitor_loss="val_g_loss",
            name="g_lr",
            optimizer=self.opt_g,
            frequency=self.hparams.n_critic_steps,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        sched_d = select_lr_schedule(
            library_name="pytorch",
            lr_scheduler_name=self.hparams.d_scheduler,
            data_len=self.data_len,
            num_epochs=self.hparams.epochs,
            learning_rate=self.hparams.learning_rate_d,
            monitor_loss="val_g_loss",
            name="d_lr",
            optimizer=self.opt_d,
            frequency=1,
            additional_configuration=self.hparams.additonal_configuration,
            verbose=self.verbose
        )

        if sched_g is None and sched_d is None:
            scheduler_list = []
        else:
            scheduler_list = [sched_g, sched_d]

        return [self.opt_g, self.opt_d], scheduler_list
    
    
    def save_model(self, filename):
        if self.hparams.save_basedir is not None:
            torch.save(
                {
                    "model_state_dict": self.generator.state_dict(),
                    "optimizer_state_dict": self.opt_g.state_dict(),
                    "scale_factor": self.hparams.scale_factor,
                    "best_valid_loss": self.best_valid_loss,
                },
                self.hparams.save_basedir + "/" + filename,
            )
        else:
            raise Exception(
                "No save_basedir was specified in the construction of the WGAN object."
            )