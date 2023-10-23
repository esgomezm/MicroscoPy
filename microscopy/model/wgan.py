import numpy as np

import torch
from torch import nn

import lightning as L

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from ..optimizer_scheduler_utils import select_optimizer, select_lr_schedule

import os

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

class WGANGP(L.LightningModule):
    def __init__(
        self,
        g_layers: int = 5,
        recloss: float = 10.0,
        lambda_gp: float = 10.0,
        n_critic_steps: int = 5,
        data_len: int = 8,
        scale_factor: int = 2,
        epochs: int = 151,
        learning_rate_d: float = 0.0001,
        learning_rate_g: float = 0.0001,
        g_optimizer: str = "Adam",
        d_optimizer: str = "Adam",
        g_scheduler: str = "OneCycle",
        d_scheduler: str = "OneCycle",
        gen_checkpoint: str = None,
        save_basedir: str = None,
        additonal_configuration: dict = {},
        verbose: int = 0,
    ):
        super(WGANGP, self).__init__()
        self.save_hyperparameters()

        self.verbose = verbose
        print('self.verbose: {}'.format(self.verbose))

        if self.verbose > 1:
            print('\nVerbose: Model initialized (begining)\n')

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        # Free cuda memory
        torch.cuda.empty_cache()

        # Initialize generator and load the checkpoint in case is given
        if gen_checkpoint is None:
            self.generator = GeneratorUpsample(
                n_residual=g_layers, scale_factor=scale_factor
            )
            self.best_valid_loss = float("inf")
        else:
            checkpoint = torch.load(gen_checkpoint)
            self.generator = GeneratorUpsample(
                n_residual=checkpoint["n_residuals"],
                scale_factor=checkpoint["scale_factor"],
            )
            self.generator.load_state_dict(checkpoint["model_state_dict"])
            self.best_valid_loss = checkpoint["best_valid_loss"]

        self.discriminator = Discriminator()

        if self.verbose > 1:
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

        self.data_len = self.hparams.data_len

        self.validation_step_lr = []
        self.validation_step_hr = []
        self.validation_step_pred = []
        
        self.val_ssim = []
        
        if self.verbose > 1:
            os.makedirs(f"{self.hparams.save_basedir}/training_images", exist_ok=True)


        self.step_schedulers = ['CosineDecay', 'OneCycle', 'MultiStepScheduler']
        self.epoch_schedulers = ['ReduceOnPlateau']

        if self.verbose > 1:
            print('\nVerbose: Model initialized (end)\n')


    def forward(self, x):
        if isinstance(x, dict):
            return self.generator(x["lr"])
        else:
            return self.generator(x)
        
    def training_step(self, batch, batch_idx):
        
        if self.verbose > 1:
            print('\nVerbose: Training step (begining)\n')

        # Take a batch of data
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

            # Predict the HR image
            generated = self(lr)
            # Evaluate the real and the fake HR images
            real_logits = self.discriminator(hr).mean()
            fake_logits = self.discriminator(generated).mean()

            if self.verbose > 1:
                print('Generator step:')
                print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()}')
                print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()}')
                print(f'generated.shape: {generated.shape} generated.min: {generated.min()} generated.max: {generated.max()}')

            # Calculate the generator's loss
            adv_loss = -1 * fake_logits
            error = self.mae(generated, hr)
            g_loss = adv_loss + error * self.hparams.recloss

            # Log the losses
            self.log("g_loss", adv_loss, prog_bar=True, on_epoch=True)
            self.log("g_adv_loss", g_loss, prog_bar=True, on_epoch=True)
            self.log("g_l1", error, prog_bar=True, on_epoch=True)

            self.log("g_real", real_logits, prog_bar=False, on_epoch=True)
            self.log("g_fake", fake_logits, prog_bar=False, on_epoch=True)

            # Optimize generator
            self.manual_backward(g_loss)
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

            # Predict the HR image
            generated = self(lr)
            # Evaluate the real and the fake HR images
            real_logits = self.discriminator(hr).mean()
            fake_logits = self.discriminator(generated).mean()

            if self.verbose > 1:
                print('Discriminator step:')
                print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()}')
                print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()}')
                print(f'generated.shape: {generated.shape} generated.min: {generated.min()} generated.max: {generated.max()}')

            gradient_penalty = self.compute_gradient_penalty(hr.data, generated.data)

            wasserstein = real_logits - fake_logits
            d_loss = - wasserstein + self.hparams.lambda_gp * gradient_penalty

            # Log the losses
            self.log("d_loss", d_loss, prog_bar=True, on_epoch=True)
            self.log("d_wasserstein", wasserstein, prog_bar=False, on_epoch=True)
            self.log("d_gp", gradient_penalty, prog_bar=False, on_epoch=True)
            
            self.log("d_real", real_logits, prog_bar=False, on_epoch=True)
            self.log("d_fake", fake_logits, prog_bar=False, on_epoch=True)

            # Optimize discriminator/critic
            self.manual_backward(d_loss)
            d_opt.step()
            d_opt.zero_grad()
            self.untoggle_optimizer(d_opt)

            if self.hparams.d_scheduler in self.step_schedulers:
                sched_d.step()

        if self.verbose > 1:
            print('\nVerbose: Training step (end)\n')

    def validation_step(self, batch, batch_idx):
        
        if self.verbose > 1:
            print('\nVerbose: validation_step (begining)\n')
            
        # Right now used for just plotting, might want to change it later
        lr, hr = batch["lr"], batch["hr"]
        generated = self(lr)

        if self.verbose > 1:
            print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()}')
            print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()}')
            print(f'generated.shape: {generated.shape} generated.min: {generated.min()} generated.max: {generated.max()}')

        true = hr.cpu().numpy()
        fake = generated.cpu().numpy()
        # xlr = lr.cpu().numpy()

        for i in range(lr.size(0)):
            ssim = structural_similarity(
                true[i, 0, ...], fake[i, 0, ...], data_range=1.0
            )
            self.log("val_ssim", ssim)
            self.val_ssim.append(ssim)

            psnr = peak_signal_noise_ratio(
                true[i, 0, ...], fake[i, 0, ...], data_range=1.0
            )
            self.log("val_psnr", psnr)
            
        self.validation_step_lr.append(lr)
        self.validation_step_hr.append(hr)
        self.validation_step_pred.append(generated)

        if self.verbose > 1:
            print('\nVerbose: validation_step (end)\n')

        del lr, hr, generated

    def on_validation_epoch_end(self):
       
        if self.verbose > 1:
            print('\nVerbose: on_validation_epoch_end (begining)\n')

        # Right now used for just plotting, might want to change it later
        # lr, hr, generated = torch.stack(self.validation_step_outputs)
        lr = torch.cat(self.validation_step_lr, 0)
        hr = torch.cat(self.validation_step_hr, 0)
        generated = torch.cat(self.validation_step_pred, 0)
        
        if self.verbose > 1:
            print(f'lr.shape: {lr.shape} lr.min: {lr.min()} lr.max: {lr.max()} values: {lr[0,0,0,:10]}')
            print(f'hr.shape: {hr.shape} hr.min: {hr.min()} hr.max: {hr.max()} values: {hr[0,0,0,:10]}')
            print(f'generated.shape: {generated.shape} generated.min: {generated.min()} generated.max: {generated.max()} values: {generated[0,0,0,:10]}')

        adv_loss = -1 * self.discriminator(generated).mean()
        error = self.mae(generated, hr)

        g_loss = adv_loss + error * self.hparams.recloss

        self.log("val_g_loss", g_loss)
        self.log("val_g_l1", error)

        real_logits = self.discriminator(hr).mean()
        fake_logits = self.discriminator(generated).mean()

        wasserstein = real_logits - fake_logits

        self.log("val_d_wasserstein", wasserstein)

        if self.verbose > 1:
            print(f'g_loss: {g_loss}')
            print(f'self.best_valid_loss: {self.best_valid_loss}')

        if g_loss < self.best_valid_loss:
            self.best_valid_loss = g_loss
            self.save_model("best_checkpoint.pth")

        self.validation_step_lr.clear()  # free memory
        self.validation_step_hr.clear()  # free memory
        self.validation_step_pred.clear()  # free memory


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

        if self.verbose > 1:
            print('\nVerbose: on_validation_epoch_end (end)\n')

    def on_train_end(self):
        
        if self.verbose > 1:
            print('\nVerbose: on_train_end (begining)\n')

        self.save_model("last_checkpoint.pth")
        
        if self.verbose > 1:
            print('\nVerbose: on_train_end (end)\n')
            
    def configure_optimizers(self):
        
        if self.verbose > 1:
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

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP

        Source: https://github.com/nocotan/pytorch-lightning-gans"""
        # Random weight term for interpolation between real and fake samples
        # alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
        #     self.device
        # )
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)

        # Get random interpolation between real and fake samples
        # interpolates = (
        #     alpha * real_samples + ((1 - alpha) * fake_samples)
        # ).requires_grad_(True)
        # interpolates = interpolates.to(self.device)

        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates.requires_grad = True

        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates, device=self.device) 

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

        # gradients = gradients.view(gradients.size(0), -1).to(self.device)
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def save_model(self, filename):
        print(f'\nVerbose: Saving the model: {self.hparams.save_basedir + "/" + filename}\n')
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