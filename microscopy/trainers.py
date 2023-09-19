import numpy as np
import time
import csv
import os

from skimage import io

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint as tf_ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from . import datasets
from . import utils
from . import metrics
from . import model_utils
from . import optimizer_scheduler_utils
from . import custom_callbacks

#######


class ModelsTrainer:
    def __init__(
        self,
        config,
        train_lr_path,
        train_hr_path,
        val_lr_path,
        val_hr_path,
        test_lr_path,
        test_hr_path,
        saving_path,
        verbose=0,
        data_on_memory=0,
    ):
        self.data_name = config.dataset_name

        self.train_lr_path = train_lr_path
        self.train_hr_path = train_hr_path
        train_extension_list = [
            os.path.splitext(e)[1] for e in os.listdir(self.train_hr_path)
        ]
        train_extension = max(set(train_extension_list), key=train_extension_list.count)
        self.train_filenames = sorted(
            [x for x in os.listdir(self.train_hr_path) if x.endswith(train_extension)]
        )

        self.validation_split = config.hyperparam.validation_split
        if val_hr_path is None or val_lr_path is None:
            self.val_lr_path = train_lr_path
            self.val_hr_path = train_hr_path

            self.val_filenames = self.train_filenames[
                int(len(self.train_filenames) * (1 - self.validation_split )) :
            ]
            self.train_filenames = self.train_filenames[
                : int(len(self.train_filenames) * (1 - self.validation_split))
            ]
        else:
            self.val_lr_path = val_lr_path
            self.val_hr_path = val_hr_path

            val_extension_list = [
                os.path.splitext(e)[1] for e in os.listdir(self.val_hr_path)
            ]
            val_extension = max(set(val_extension_list), key=val_extension_list.count)
            self.val_filenames = sorted(
                [x for x in os.listdir(self.val_hr_path) if x.endswith(val_extension)]
            )

        self.test_lr_path = test_lr_path
        self.test_hr_path = test_hr_path
        test_extension_list = [
            os.path.splitext(e)[1] for e in os.listdir(self.test_hr_path)
        ]
        test_extension = max(set(test_extension_list), key=test_extension_list.count)
        self.test_filenames = sorted(
            [x for x in os.listdir(self.test_hr_path) if x.endswith(test_extension)]
        )

        self.crappifier_method = config.used_dataset.crappifier
        self.scale_factor = config.used_dataset.scale
        self.lr_patch_size_x = config.used_dataset.patch_size_x
        self.lr_patch_size_y = config.used_dataset.patch_size_y
        self.datagen_sampling_pdf = config.hyperparam.datagen_sampling_pdf

        if "rotation" in config.hyperparam.data_augmentation:
            self.rotation = True
        if "horizontal_flip" in config.hyperparam.data_augmentation:
            self.horizontal_flip = True
        if "vertical_flip" in config.hyperparam.data_augmentation:
            self.vertical_flip = True
        if len(config.hyperparam.data_augmentation) != 0 and (
            not self.rotation or not self.horizontal_flip or not self.vertical_flip
        ):
            raise ValueError("Data augmentation values are not well defined.")

        self.model_name = config.model_name
        self.num_epochs = config.hyperparam.num_epochs
        self.batch_size = config.hyperparam.batch_size
        self.learning_rate = config.hyperparam.lr
        self.discriminator_learning_rate = config.hyperparam.discriminator_lr
        self.optimizer_name = config.hyperparam.optimizer
        self.discriminator_optimizer = config.hyperparam.discriminator_optimizer
        self.lr_scheduler_name = config.hyperparam.scheduler
        self.discriminator_lr_scheduler = config.hyperparam.discriminator_lr_scheduler

        self.test_metric_indexes = config.hyperparam.test_metric_indexes

        self.additional_folder = config.hyperparam.additional_folder
        self.seed = config.hyperparam.seed

        self.verbose = verbose
        self.data_on_memory = data_on_memory

        save_folder = "scale" + str(self.scale_factor)

        if self.additional_folder:
            save_folder += "_" + self.additional_folder

        self.saving_path = saving_path
        self.config = config

        os.makedirs(self.saving_path, exist_ok=True)
        utils.save_yaml(
            self.config,
            os.path.join(self.saving_path, "train_configuration.yaml"),
        )

        utils.set_seed(self.seed)
        # To calculate the input and output shape and the actual scale factor 
        (
            _,
            train_input_shape,
            train_output_shape,
            actual_scale_factor,
        ) = datasets.TFDataset(
            filenames=self.train_filenames,
            hr_data_path=self.train_hr_path,
            lr_data_path=self.train_lr_path,
            scale_factor=self.scale_factor,
            crappifier_name=self.crappifier_method,
            lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            validation_split=0.1,
            batch_size=self.batch_size,
            rotation=self.rotation,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            verbose=self.verbose
        )

        self.input_data_shape = train_input_shape
        self.output_data_shape = train_output_shape

        if self.scale_factor is None or self.scale_factor != actual_scale_factor:
            self.scale_factor = actual_scale_factor
            utils.update_yaml(
                os.path.join(self.saving_path, "train_configuration.yaml"),
                "actual_scale_factor",
                actual_scale_factor,
            )
            if self.verbose > 0:
                print(
                    "Actual scale factor that will be used is: {}".format(
                        self.scale_factor
                    )
                )


        print("\n" + "-" * 10)
        print(
            "{} model will be trained with the next configuration".format(
                self.model_name
            )
        )
        print("Dataset: {}".format(self.data_name))
        print("\tTrain wf path: {}".format(train_lr_path))
        print("\tTrain gt path: {}".format(train_hr_path))
        print("\tVal wf path: {}".format(val_lr_path))
        print("\tVal gt path: {}".format(val_hr_path))
        print("\tTest wf path: {}".format(test_lr_path))
        print("\tTest gt path: {}".format(test_hr_path))
        print("Preprocessing info:")
        print("\tScale factor: {}".format(self.scale_factor))
        print("\tCrappifier method: {}".format(self.crappifier_method))
        print("\tPatch size: {} x {}".format(self.lr_patch_size_x, self.lr_patch_size_y))
        print("Training info:")
        print("\tEpochs: {}".format(self.num_epochs))
        print("\tBatchsize: {}".format(self.batch_size))
        print("\tGen learning rate: {}".format(self.learning_rate))
        print("\tDisc learning rate: {}".format(self.discriminator_learning_rate))
        print("\tGen optimizer: {}".format(self.optimizer_name))
        print("\tDisc optimizer: {}".format(self.discriminator_optimizer))
        print("\tGen scheduler: {}".format(self.lr_scheduler_name))
        print("\tDisc scheduler: {}".format(self.discriminator_lr_scheduler))
        print("-" * 10)

    def launch(self):
        self.prepare_data()
        self.train_model()

        # if self.data_name in ['ER', 'MT', 'F-actin']:
        #     dataset_levels = {'ER':6, 'MT':9, 'F-actin':12}
        #     levels = dataset_levels[self.data_name]
        #     for i in range(1, levels):
        #         level_folder = f"level_{i:02d}"
        #         if "level" in self.test_lr_path:
        #             self.test_lr_path = os.path.join(self.test_lr_path[:-9], level_folder)
        #         else:
        #             self.test_lr_path = os.path.join(self.test_lr_path, level_folder)
        #         self.predict_images(result_folder_name=level_folder)
        #         self.eval_model(result_folder_name=level_folder)
        # else:
        #     self.predict_images()
        #     self.eval_model()

        self.predict_images()
        self.eval_model()

        return self.history

    def prepare_data(self):
        raise NotImplementedError("prepare_data() not implemented.")

    def train_model(self):
        raise NotImplementedError("train_model() not implemented.")

    def predict_images(self, result_folder_name=""):
        raise NotImplementedError("predict_images() not implemented")

    def eval_model(self, result_folder_name=""):
        utils.set_seed(self.seed)

        if self.verbose  > 0:
            utils.print_info("eval_model() - self.Y_test", self.Y_test)
            utils.print_info("eval_model() - self.predictions", self.predictions)
            utils.print_info("eval_model() - self.X_test", self.X_test)

        print("The predictions will be evaluated:")
        metrics_dict = metrics.obtain_metrics(
            gt_image_list=self.Y_test,
            predicted_image_list=self.predictions,
            wf_image_list=self.X_test,
            test_metric_indexes=self.test_metric_indexes,
        )

        os.makedirs(os.path.join(self.saving_path, "test_metrics", result_folder_name), exist_ok=True)

        for key in metrics_dict.keys():
            if len(metrics_dict[key]) > 0:
                print("{}: {}".format(key, np.mean(metrics_dict[key])))
                np.save(
                    os.path.join(self.saving_path, "test_metrics", result_folder_name, f"{key}.npy"),
                    metrics_dict[key],
                )


class TensorflowTrainer(ModelsTrainer):
    def __init__(
        self,
        config,
        train_lr_path,
        train_hr_path,
        val_lr_path,
        val_hr_path,
        test_lr_path,
        test_hr_path,
        saving_path,
        verbose=0,
        data_on_memory=0
    ):
        super().__init__(
            config,
            train_lr_path,
            train_hr_path,
            val_lr_path,
            val_hr_path,
            test_lr_path,
            test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory
        )

        tf.config.run_functions_eagerly(False)

        self.library_name = "tensorflow"

    def prepare_data(self):
        utils.set_seed(self.seed)
        if self.data_on_memory:
            if self.verbose > 0:
                print('Data will be loaded on memory for all the epochs the same.')
            X_train, Y_train, actual_scale_factor = datasets.extract_random_patches_from_folder( 
                                                        lr_data_path = self.train_lr_path,
                                                        hr_data_path = self.train_hr_path,
                                                        filenames = self.train_filenames,
                                                        scale_factor = self.scale_factor,
                                                        crappifier_name = self.crappifier_method,
                                                        lr_patch_shape = (self.lr_patch_size_x, self.lr_patch_size_y),
                                                        datagen_sampling_pdf = self.datagen_sampling_pdf)
            X_train = np.expand_dims(X_train, axis=-1)
            Y_train = np.expand_dims(Y_train, axis=-1)

            self.input_data_shape = X_train.shape
            self.output_data_shape = Y_train.shape

            train_generator = datasets.get_train_val_generators(X_data=X_train,
                                                                Y_data=Y_train,
                                                                batch_size=self.batch_size)


            X_val, Y_val, _ = datasets.extract_random_patches_from_folder( 
                                                        lr_data_path = self.val_lr_path,
                                                        hr_data_path = self.val_hr_path,
                                                        filenames = self.val_filenames,
                                                        scale_factor = self.scale_factor,
                                                        crappifier_name = self.crappifier_method,
                                                        lr_patch_shape = (self.lr_patch_size_x, self.lr_patch_size_y),
                                                        datagen_sampling_pdf = self.datagen_sampling_pdf)
            X_val = np.expand_dims(X_val, axis=-1)
            Y_val = np.expand_dims(Y_val, axis=-1)

            self.val_input_data_shape = X_val.shape
            self.val_output_data_shape = Y_val.shape

            val_generator = datasets.get_train_val_generators(X_data=X_val,
                                                             Y_data=Y_val,
                                                             batch_size=self.batch_size)
        else:
            print('Data will be loaded on the fly, each batch new data will be loaded..')

            train_generator, train_input_shape,train_output_shape, actual_scale_factor = datasets.TFDataset(
                filenames=self.train_filenames,
                hr_data_path=self.train_hr_path,
                lr_data_path=self.train_lr_path,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
                validation_split=0.1,
                batch_size=self.batch_size,
                rotation=self.rotation,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                verbose=self.verbose
            )
            
            self.input_data_shape = train_input_shape
            self.output_data_shape = train_output_shape

            # training_images_path = os.path.join(self.saving_path, "special_folder")
            # os.makedirs(training_images_path, exist_ok=True)
            # cont = 0
            # for hr_img, lr_img in train_generator:
            #     for i in range(hr_img.shape[0]):
            #         io.imsave(os.path.join(training_images_path, "hr" + str(cont) + ".tif"), np.array(hr_img[i,...]))
            #         io.imsave(os.path.join(training_images_path, "lr" + str(cont) + ".tif"), np.array(lr_img[i,...]))
            #         if cont > 100:
            #             break
            #         cont += 1
            #     if cont > 100:
            #         break

            val_generator, _, _, _ = datasets.TFDataset(
                filenames=self.val_filenames,
                hr_data_path=self.val_hr_path,
                lr_data_path=self.val_lr_path,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=(self.lr_patch_size_x, self.lr_patch_size_y),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
                validation_split=0.1,
                batch_size=self.batch_size,
                rotation=self.rotation,
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                verbose=self.verbose
            )

        if self.verbose > 0:
            print("input_data_shape: {}".format(self.input_data_shape))
            print("output_data_shape: {}".format(self.output_data_shape))

        if self.scale_factor is None or self.scale_factor != actual_scale_factor:
            self.scale_factor = actual_scale_factor
            utils.update_yaml(
                os.path.join(self.saving_path, "train_configuration.yaml"),
                "actual_scale_factor",
                actual_scale_factor,
            )
            if self.verbose > 0:
                print(
                    "Actual scale factor that will be used is: {}".format(
                        self.scale_factor
                    )
                )

        utils.update_yaml(
            os.path.join(self.saving_path, "train_configuration.yaml"),
            "input_data_shape",
            self.input_data_shape,
        )
        utils.update_yaml(
            os.path.join(self.saving_path, "train_configuration.yaml"),
            "output_data_shape",
            self.output_data_shape,
        )

        self.train_generator = train_generator
        self.val_generator = val_generator

    def train_model(self):

        utils.set_seed(self.seed)

        callbacks = []
        lr_schedule = optimizer_scheduler_utils.select_lr_schedule(
                    library_name=self.library_name,
                    lr_scheduler_name=self.lr_scheduler_name,
                    data_len=self.input_data_shape[0] // self.batch_size,
                    num_epochs=self.num_epochs,
                    learning_rate=self.learning_rate,
                    monitor_loss='val_ssim_loss',
                    name=None,
                    optimizer=None,
                    frequency=None,
                    additional_configuration=self.config,
                    verbose=self.verbose
        )

        if self.lr_scheduler_name in ["CosineDecay", "MultiStepScheduler"]:
            self.optim = optimizer_scheduler_utils.select_optimizer(
                library_name=self.library_name,
                optimizer_name=self.optimizer_name,
                learning_rate=lr_schedule,
                check_point=None,
                parameters=None,
                additional_configuration=self.config,
                verbose=self.verbose
            )
        else:
            self.optim = optimizer_scheduler_utils.select_optimizer(
                library_name=self.library_name,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                check_point=None,
                parameters=None,
                additional_configuration=self.config,
                verbose=self.verbose
            )
            if not lr_schedule is None:
                callbacks.append(lr_schedule)

        model = model_utils.select_model(
            model_name=self.model_name,
            input_shape=self.input_data_shape,
            output_channels=self.output_data_shape[-1],
            scale_factor=self.scale_factor,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            model_configuration=self.config.used_model,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        loss_funct = tf.keras.losses.mean_absolute_error
        eval_metric = tf.keras.losses.mean_squared_error

        model.compile(
            optimizer=self.optim,
            loss=loss_funct,
            metrics=[eval_metric, utils.ssim_loss],
        )

        trainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.trainable_weights]
        )
        nonTrainableParams = np.sum(
            [np.prod(v.get_shape()) for v in model.non_trainable_weights]
        )
        totalParams = trainableParams + nonTrainableParams

        model_checkpoint = tf_ModelCheckpoint(
            os.path.join(self.saving_path, "weights_best.h5"),
            monitor="val_loss",
            verbose=self.verbose,
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(model_checkpoint)

        # callback for early stopping
        earlystopper = EarlyStopping(
            monitor=self.config.model.optim.early_stop.loss,
            patience=self.config.model.optim.early_stop.patience,
            min_delta=0.005,
            mode=self.config.model.optim.early_stop.mode,
            verbose=self.verbose,
            restore_best_weights=True,
        )
        callbacks.append(earlystopper)

        # callback for saving the learning rate
        lr_observer = custom_callbacks.LearningRateObserver()
        callbacks.append(lr_observer)

        for x, y in self.val_generator:
            x_val = x
            y_val = y
            break
        
        plt_saving_path = os.path.join(self.saving_path, "training_images")
        os.makedirs(plt_saving_path, exist_ok=True)
        plot_callback = custom_callbacks.PerformancePlotCallback(
            x_val, y_val, plt_saving_path, frequency=10, is_cddpm=self.model_name=="cddpm"
        )
        callbacks.append(plot_callback)

        if self.verbose > 0:
            print("Model configuration:")
            print(f"\tModel_name: {self.model_name}")
            print(f"\tOptimizer: {self.optim}")
            print(f"\tLR scheduler: {lr_schedule}")
            print(f"\tLoss: {loss_funct}")
            print(f"\tEval: {eval_metric}")
            print(
                "Trainable parameteres: {} \nNon trainable parameters: {} \nTotal parameters: {}".format(
                    trainableParams, nonTrainableParams, totalParams
                )
            )
            # callbacks.append(custom_callbacks.CustomCallback())

        if self.model_name == "cddpm":
            # calculate mean and variance of training dataset for normalization
            model.normalizer.adapt(self.train_generator.map(lambda x, y: x))

        start = time.time()

        print("Training is going to start:")
        
        if self.data_on_memory:
            history = model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=self.num_epochs,
                steps_per_epoch=self.input_data_shape[0]//self.batch_size,
                validation_steps=self.val_input_data_shape[0]//self.batch_size,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=self.num_epochs,
                callbacks=callbacks,
            )

        dt = time.time() - start
        mins, sec = divmod(dt, 60)
        hour, mins = divmod(mins, 60)
        print(
            "\nTime elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)\n"
        )

        model.save_weights(os.path.join(self.saving_path, "weights_last.h5"))
        self.history = history

        os.makedirs(self.saving_path + "/train_metrics", exist_ok=True)

        for key in history.history:
            np.save(
                self.saving_path + "/train_metrics/" + key + ".npy",
                history.history[key],
            )
        np.save(self.saving_path + "/train_metrics/time.npy", np.array([dt]))
        np.save(self.saving_path + "/train_metrics/lr.npy", np.array(lr_observer.epoch_lrs))

    def predict_images(self, result_folder_name=""):

        utils.set_seed(self.seed)
        ground_truths = []
        widefields = []
        predictions = []
        print(f"Prediction of {self.model_name} is going to start:")
        for test_filename in self.test_filenames:
            
            lr_images, hr_images, _ = datasets.extract_random_patches_from_folder(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=[test_filename],
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                datagen_sampling_pdf=1,
            )

            hr_images = np.expand_dims(hr_images, axis=-1)
            lr_images = np.expand_dims(lr_images, axis=-1)

            ground_truths.append(hr_images[0, ...])
            widefields.append(lr_images[0, ...])
            
            if self.model_name == "unet":
                if self.verbose > 0:
                    print("Padding will be added to the images.")
                    print("LR images before padding:")
                    print(
                        "LR images - shape:{} dtype:{}".format(
                            lr_images.shape, lr_images.dtype
                        )
                    )

                height_padding, width_padding = utils.calculate_pad_for_Unet(
                    lr_img_shape=lr_images[0].shape,
                    depth_Unet=self.config.used_model.depth,
                    is_pre=True,
                    scale=self.scale_factor,
                )

                if self.verbose > 0 and (
                    height_padding == (0, 0) and width_padding == (0, 0)
                ):
                    print("No padding has been needed to be added.")

                lr_images = utils.add_padding_for_Unet(
                    lr_imgs=lr_images,
                    height_padding=height_padding,
                    width_padding=width_padding,
                )

            if self.verbose > 0:
                print(
                    "HR images - shape:{} dtype:{}".format(
                        hr_images.shape, hr_images.dtype
                    )
                )
                print(
                    "LR images - shape:{} dtype:{}".format(
                        lr_images.shape, lr_images.dtype
                    )
                )

            if self.config.model.others.positional_encoding:
                lr_images = utils.concatenate_encoding(
                    lr_images,
                    self.config.model.others.positional_encoding_channels,
                )

            optim = optimizer_scheduler_utils.select_optimizer(
                library_name=self.library_name,
                optimizer_name=self.optimizer_name,
                learning_rate=self.learning_rate,
                check_point=None,
                parameters=None,
                additional_configuration=self.config,
            )

            model = model_utils.select_model(
                model_name=self.model_name,
                input_shape=lr_images.shape,
                output_channels=hr_images.shape[-1],
                scale_factor=self.scale_factor,
                datagen_sampling_pdf=self.datagen_sampling_pdf,
                model_configuration=self.config.used_model,
                verbose=self.verbose
            )

            if self.verbose > 0:
                print(model.summary())

            loss_funct = "mean_absolute_error"
            eval_metric = "mean_squared_error"

            model.compile(
                optimizer=optim, loss=loss_funct, metrics=[eval_metric, utils.ssim_loss]
            )

            # Load old weights
            model.load_weights(os.path.join(self.saving_path, "weights_best.h5"))

            aux_prediction = model.predict(lr_images, batch_size=1)

            if self.model_name == "unet":
                aux_prediction = utils.remove_padding_for_Unet(
                    pad_hr_imgs=aux_prediction,
                    height_padding=height_padding,
                    width_padding=width_padding,
                    scale=self.scale_factor,
                )

            # aux_prediction = datasets.normalization(aux_prediction)
            aux_prediction = np.clip(aux_prediction, a_min=0.0, a_max=1.0)

            if len(aux_prediction.shape) == 4:
                predictions.append(aux_prediction[0, ...])
            elif len(aux_prediction.shape) == 3:
                if aux_prediction.shape[-1] == 1:
                    predictions.append(aux_prediction)
                if aux_prediction.shape[0] == 1:
                    predictions.append(np.expand_dims(aux_prediction[0,:,:], -1))

        self.Y_test = ground_truths
        self.predictions = predictions
        self.X_test = widefields

        # assert (np.max(self.Y_test) <= 1.0).all and (np.max(self.predictions) <= 1.0).all and (np.max(self.X_test) <= 1.0).all
        # assert (np.min(self.Y_test) >= 0.0).all and (np.min(self.predictions) >= 0.0).all and (np.min(self.X_test) >= 0.0).all

        if self.verbose > 0:
            utils.print_info("predict_images() - Y_test", self.Y_test)
            utils.print_info("predict_images() - predictions", self.predictions)
            utils.print_info("predict_images() - X_test", self.X_test)

        # Save the predictions
        os.makedirs(os.path.join(self.saving_path, "predicted_images", result_folder_name), exist_ok=True)

        for i, image in enumerate(predictions):

            print(image.shape)

            tf.keras.preprocessing.image.save_img(
                os.path.join(self.saving_path, "predicted_images", result_folder_name, self.test_filenames[i]),
                image,
                data_format=None,
                file_format=None,
            )
        print(
            "Predicted images have been saved in: "
            + self.saving_path
            + "/predicted_images"
        )


class PytorchTrainer(ModelsTrainer):
    def __init__(
        self,
        data_name,
        train_lr_path,
        train_hr_path,
        val_lr_path,
        val_hr_path,
        test_lr_path,
        test_hr_path,
        saving_path,
        verbose=0,
        data_on_memory=0,
    ):
        super().__init__(
            data_name,
            train_lr_path,
            train_hr_path,
            val_lr_path,
            val_hr_path,
            test_lr_path,
            test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory
        )

        self.library_name = "pytorch"

    def prepare_data(self):

        utils.update_yaml(
            os.path.join(self.saving_path, "train_configuration.yaml"),
            "input_data_shape",
            self.input_data_shape,
        )
        utils.update_yaml(
            os.path.join(self.saving_path, "train_configuration.yaml"),
            "output_data_shape",
            self.output_data_shape,
        )

    def train_model(self):
        utils.set_seed(self.seed)

        model = model_utils.select_model(
            model_name=self.model_name,
            input_shape=None,
            output_channels=None,
            scale_factor=self.scale_factor,
            batch_size=self.batch_size,
            lr_patch_size_x=self.lr_patch_size_x,
            lr_patch_size_y=self.lr_patch_size_y,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            learning_rate_g=self.learning_rate,
            learning_rate_d=self.discriminator_learning_rate,
            g_optimizer=self.optimizer_name,
            d_optimizer=self.discriminator_optimizer,
            g_scheduler=self.lr_scheduler_name,
            d_scheduler=self.discriminator_lr_scheduler,
            epochs=self.num_epochs,
            save_basedir=self.saving_path,
            train_hr_path=self.train_hr_path,
            train_lr_path=self.train_lr_path,
            train_filenames=self.train_filenames,
            val_hr_path=self.val_hr_path,
            val_lr_path=self.val_lr_path,
            val_filenames=self.val_filenames,
            crappifier_method=self.crappifier_method,
            model_configuration=self.config,
            verbose=self.verbose,
        )
        
        # Take one batch of data
        data = next(iter(model.train_dataloader()))
        
        if self.verbose > 0:
            print("LR patch shape: {}".format(data["lr"][0][0].shape))
            print("HR patch shape: {}".format(data["hr"][0][0].shape))

            utils.print_info("train_model() - lr", data["lr"])
            utils.print_info("train_model() - hr", data["hr"])

        # Let's define the callbacks that will be used during training
        callbacks = [] 
        
        # First to monitor the LR en each epoch (for validating the scheduler and the optimizer)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

        # Saving checkpoints during training based
        checkpoints = ModelCheckpoint(
            monitor="val_ssim",
            mode="max",
            save_top_k=1,
            every_n_train_steps=5,
            save_last=True,
            filename="{epoch:02d}-{val_ssim:.3f}",
        )
        callbacks.append(checkpoints)

        # Saving plots during training to see the evolution on the performance
        # plt_saving_path = os.path.join(self.saving_path, "training_images")
        # os.makedirs(plt_saving_path, exist_ok=True)
        # plot_callback = custom_callbacks.PerformancePlotCallback_Pytorch(
        #     data["lr"], data["hr"], plt_saving_path, frequency=10
        # )
        # callbacks.append(plot_callback)
        

        os.makedirs(self.saving_path + "/Quality Control", exist_ok=True)
        logger = CSVLogger(self.saving_path + "/Quality Control", name="Logger")

        trainer = Trainer(
            accelerator="gpu",
            devices=-1,
            max_epochs=self.num_epochs,
            logger=logger,
            callbacks=callbacks,
        )

        print(trainer.strategy)

        print("Training is going to start:")
        start = time.time()

        trainer.fit(model)

        # Displaying the time elapsed for training
        dt = time.time() - start
        mins, sec = divmod(dt, 60)
        hour, mins = divmod(mins, 60)
        print(
            "\nTime elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)\n"
        )

        logger_path = os.path.join(self.saving_path + "/Quality Control/Logger")
        all_logger_versions = [
            os.path.join(logger_path, dname) for dname in os.listdir(logger_path)
        ]
        last_logger = all_logger_versions[-1]

        train_csv_path = last_logger + "/metrics.csv"

        if not os.path.exists(train_csv_path):
            print(
                "The path does not contain a csv file containing the loss and validation evolution of the model"
            )
        else:
            with open(train_csv_path, "r") as csvfile:
                csvRead = csv.reader(csvfile, delimiter=",")
                keys = next(csvRead)
                step_idx = keys.index("step")
                keys.remove("step")

                # Initialize the dictionary with empty lists
                train_metrics = {}
                for k in keys:
                    train_metrics[k] = []

                # Fill the dictionary
                for row in csvRead:
                    step = int(row[step_idx])
                    row.pop(step_idx)
                    for i, row_value in enumerate(row):
                        if row_value:
                            train_metrics[keys[i]].append([step, float(row_value)])

                os.makedirs(self.saving_path + "/train_metrics", exist_ok=True)

                # Save the metrics
                for key in train_metrics:
                    values_to_save = np.array([e[1] for e in train_metrics[key]])
                    np.save(
                        self.saving_path + "/train_metrics/" + key + ".npy",
                        values_to_save,
                    )
                np.save(self.saving_path + "/train_metrics/time.npy", np.array([dt]))

        self.history = []
        print("Train information saved.")

    def predict_images(self, result_folder_name=""):
        utils.set_seed(self.seed)
        model = model_utils.select_model(
            model_name=self.model_name,
            scale_factor=self.scale_factor,
            batch_size=self.batch_size,
            save_basedir=self.saving_path,
            model_configuration=self.config,
            datagen_sampling_pdf=self.datagen_sampling_pdf,
            checkpoint=os.path.join(self.saving_path, "best_checkpoint.pth"),
            verbose=self.verbose
        )

        trainer = Trainer(accelerator="gpu", devices=-1)

        dataset = datasets.PytorchDataset(
            hr_data_path=self.test_hr_path,
            lr_data_path=self.test_lr_path,
            filenames=self.test_filenames,
            scale_factor=self.scale_factor,
            crappifier_name=self.crappifier_method,
            lr_patch_shape=None,
            transformations=datasets.ToTensor(),
            datagen_sampling_pdf=self.datagen_sampling_pdf,
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        print("Prediction is going to start:")
        predictions = trainer.predict(model, dataloaders=dataloader)
        predictions = np.array(
            [
                np.expand_dims(np.squeeze(e.detach().numpy()), axis=-1)
                for e in predictions
            ]
        )

        if self.verbose > 0:
            data = next(iter(dataloader))
            utils.print_info("predict_images() - lr", data["lr"])
            utils.print_info("predict_images() - hr", data["hr"])
            utils.print_info("predict_images() - predictions", predictions)

        os.makedirs(os.path.join(self.saving_path, "predicted_images"), exist_ok=True)

        for i, image in enumerate(predictions):
            tf.keras.preprocessing.image.save_img(
                self.saving_path + "/predicted_images/" + self.test_filenames[i],
                image,
                data_format=None,
                file_format=None,
            )
        print(
            "Predicted images have been saved in: "
            + self.saving_path
            + "/predicted_images"
        )

        self.predictions = predictions

        lr_images, hr_images, _ = datasets.extract_random_patches_from_folder(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=self.test_filenames,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                datagen_sampling_pdf=1,
            )

        self.Y_test = np.expand_dims(hr_images, axis=-1)
        self.X_test = np.expand_dims(lr_images, axis=-1)

        if self.verbose > 0:
            utils.print_info("predict_images() - self.Y_test", self.Y_test)
            utils.print_info("predict_images() - self.predictions", self.predictions)
            utils.print_info("predict_images() - self.X_test", self.X_test)

        # assert np.max(self.Y_test[0]) <= 1.0 and np.max(self.predictions[0]) <= 1.0
        # assert np.min(self.Y_test[0]) >= 0.0 and np.min(self.predictions[0]) >= 0.0

def get_model_trainer(
    config,
    train_lr_path, train_hr_path,
    val_lr_path, val_hr_path,
    test_lr_path, test_hr_path,
    saving_path,
    verbose=0,
    data_on_memory=0,
):
    if config.model_name in ["wgan", "esrganplus"]:
        model_trainer = PytorchTrainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path,  val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )
    elif config.model_name in ["rcan", "dfcan", "wdsr", "unet", "cddpm"]:
        model_trainer = TensorflowTrainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path, val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )
    else:
        raise Exception("Not available model.")

    return model_trainer

def train_configuration(
    config,
    train_lr_path, train_hr_path,
    val_lr_path, val_hr_path,
    test_lr_path, test_hr_path,
    saving_path,
    verbose=0,
    data_on_memory=0,
):
    if config.model_name in ["wgan", "esrganplus"]:
        model_trainer = PytorchTrainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path,  val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )
    elif config.model_name in ["rcan", "dfcan", "wdsr", "unet", "cddpm"]:
        model_trainer = TensorflowTrainer(
            config,
            train_lr_path, train_hr_path,
            val_lr_path, val_hr_path,
            test_lr_path, test_hr_path,
            saving_path,
            verbose=verbose,
            data_on_memory=data_on_memory,
        )
    else:
        raise Exception("Not available model.")

    return model_trainer.launch()


def predict_configuration(
    config,
    train_lr_path, train_hr_path,
    val_lr_path, val_hr_path,
    test_lr_path, test_hr_path,
    saving_path,
    verbose=0,
    data_on_memory=0,
):

    if config.dataset_name in ['ER', 'MT', 'F-actin']:
        dataset_levels = {'ER':6, 'MT':9, 'F-actin':12}
        levels = dataset_levels[config.dataset_name]
        for i in range(1, levels+1):
            level_folder = f"level_{i:02d}"
            if "level" in test_lr_path:
                test_lr_path = os.path.join(test_lr_path[:-9], level_folder)
            else:
                test_lr_path = os.path.join(test_lr_path, level_folder)

            if config.model_name in ["wgan", "esrganplus"]:
                model_trainer = PytorchTrainer(
                    config,
                    train_lr_path, train_hr_path,
                    val_lr_path, val_hr_path,
                    test_lr_path, test_hr_path,
                    saving_path,
                    verbose=verbose,
                    data_on_memory=data_on_memory,
                )
            elif config.model_name in ["rcan", "dfcan", "wdsr", "unet", "cddpm"]:
                model_trainer = TensorflowTrainer(
                    config,
                    train_lr_path, train_hr_path,
                    val_lr_path, val_hr_path,
                    test_lr_path, test_hr_path,
                    saving_path,
                    verbose=verbose,
                    data_on_memory=data_on_memory,
                )
            else:
                raise Exception("Not available model.")

            model_trainer.predict_images(result_folder_name=level_folder)
            model_trainer.eval_model(result_folder_name=level_folder)
    else:
        if config.model_name in ["wgan", "esrganplus"]:
            model_trainer = PytorchTrainer(
                config,
                train_lr_path, train_hr_path,
                val_lr_path, val_hr_path,
                test_lr_path, test_hr_path,
                saving_path,
                verbose=verbose,
                data_on_memory=data_on_memory,
            )
        elif config.model_name in ["rcan", "dfcan", "wdsr", "unet", "cddpm"]:
            model_trainer = TensorflowTrainer(
                config,
                train_lr_path, train_hr_path,
                val_lr_path, val_hr_path,
                test_lr_path, test_hr_path,
                saving_path,
                verbose=verbose,
                data_on_memory=data_on_memory,
            )
        else:
            raise Exception("Not available model.")

    model_trainer.predict_images()
    model_trainer.eval_model()


    
