import copy
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath

from torch import nn
import torch

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)

import constants
from base.experiment import GenericExperiment
from base.utils import load_pickle
from base.loss_function import CCCLoss
from trainer import Trainer

from dataset import DataArranger, Dataset
from base.checkpointer import Checkpointer
from models.model import LFAN, CAN, JMT

from base.parameter_control import ResnetParamControl

import os


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = copy.deepcopy(args)
        self.task = args.task
        self.release_count = args.opt__release_count
        self.gradual_release = args.opt__gradual_release
        self.milestone = args.opt__milestone
        self.backbone_mode = "ir"
        self.min_num_epochs = args.min_num_epochs
        self.num_epochs = args.num_epochs
        self.early_stopping = args.early_stopping
        self.load_best_at_each_epoch = args.opt__load_best_at_each_epoch

        self.num_heads = args.num_heads
        self.modal_dim = args.modal_dim
        self.tcn_kernel_size = args.tcn_kernel_size

    def prepare(self):
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.get_modality()  # pass
        self.num_classes = self.args.num_classes
        self.dataset_name = self.args.dataset_name

        if self.dataset_name == constants.C_EXPR_DB and \
                self.args.use_other_class:
            self.num_classes += 1

        if self.args.use_other_class and self.dataset_name != \
                constants.C_EXPR_DB:
            raise NotImplementedError

        self.continuous_label_dim = self.get_selected_continuous_label_dim()

        # self.dataset_info = load_pickle(join(self.dataset_path,
        #                                      "dataset_info.pkl"))

        self.dataset_info = self.load_dataset_info()
        self.data_arranger = self.init_data_arranger()

        if self.calc_mean_std:
            self.calc_mean_std_fn()

        self.mean_std_dict = load_pickle(self.get_mean_std_dict_path())

    def load_dataset_info(self) -> dict:
        ds = self.args.dataset_name

        if ds == constants.MELD:
            dataset_info = {
                split: load_pickle(
                    join(self.dataset_path,
                         'features', f"dataset_info_{ds}_{split}.pkl")
                ) for split in [constants.TRAINSET,
                                constants.VALIDSET,
                                constants.TESTSET]
            }

        elif ds == constants.C_EXPR_DB:
            dataset_info = {
                split: load_pickle(
                    join(self.dataset_path,
                         'features', f"dataset_info_{ds}_{split}.pkl")
                ) for split in [constants.TRAINSET,
                                constants.VALIDSET
                                ]
            }
            # no test set yet.
            dataset_info[constants.TESTSET] = copy.deepcopy(
                dataset_info[constants.VALIDSET])

        elif ds == constants.C_EXPR_DB_CHALLENGE:
            dataset_info = {
                split: load_pickle(
                    join(self.dataset_path,
                         'features', f"dataset_info_{ds}_{split}.pkl")
                ) for split in [constants.TRAINSET]
            }
            # train == valid == test
            dataset_info[constants.VALIDSET] = copy.deepcopy(
                dataset_info[constants.TRAINSET])
            dataset_info[constants.TESTSET] = copy.deepcopy(
                dataset_info[constants.TRAINSET])

        else:
            raise NotImplementedError

        return dataset_info

    def init_data_arranger(self):
        arranger = DataArranger(self.args,
                                self.dataset_info,
                                self.dataset_path,
                                self.fold_to_run,
                                self.folds_dir
                                )
        return arranger

    def run(self):

        # criterion = CCCLoss()
        if self.task == constants.CLASSFICATION:
            criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        else:
            raise NotImplementedError(self.task)

        fold = self.fold_to_run

        # save_path = os.path.join(self.save_path,
        #                          self.exp_id + "_" + self.model_name +
        #                          "_" + self.stamp + "_fold" + str(
        #                              fold) + "_" + self.emotion)
        save_path = self.outd
        os.makedirs(save_path, exist_ok=True)

        checkpoint_filename = join(save_path, "checkpoint.pkl")

        model = self.init_model()

        dataloaders = self.init_dataloader(fold)

        trainer_kwards = {'device': self.device,
                          'emotion': self.emotion,
                          'model_name': self.model_name,
                          'models': model,
                          'save_path': save_path,
                          'fold': fold,
                          'min_epoch': self.min_num_epochs,
                          'max_epoch': self.num_epochs,
                          'early_stopping': self.early_stopping,
                          # 'scheduler': self.scheduler,
                          'learning_rate': self.learning_rate,
                          'min_learning_rate': self.min_learning_rate,
                          'patience': self.patience,
                          'train_batch_size': self.train_batch_size,
                          'eval_batch_size': self.eval_batch_size,
                          'criterion': criterion,
                          'factor': self.factor,
                          'verbose': True,
                          'milestone': self.milestone,
                          'metrics': self.config['metrics'],
                          'load_best_at_each_epoch': self.load_best_at_each_epoch,
                          'save_plot': self.config['save_plot']
                          }


        trainer = Trainer(**trainer_kwards)
        trainer.set_args(self.args)
        trainer.post_set_args()
        trainer.set_number_classes(self.num_classes)
        trainer.init_optimizer_and_scheduler(epoch=0)

        # parameter_controller = ResnetParamControl(
        #     trainer,
        #     gradual_release=self.gradual_release,
        #     release_count=self.release_count,
        #     backbone_mode=["visual", "audio"]  # not used.
        # )

        checkpoint_controller = None
        parameter_controller = None

        # cancelled: no time to revise all this.

        # checkpoint_controller = Checkpointer(checkpoint_filename,
        #                                      trainer,
        #                                      parameter_controller,
        #                                      resume=self.resume
        #                                      )

        # if self.resume:
        #     trainer, parameter_controller = checkpoint_controller.load_checkpoint()
        # else:
        #     checkpoint_controller.init_csv_logger(self.args, self.config)

        if not trainer.fit_finished:
            # changed from .fit to .optimize
            # run: train, valid, test.
            trainer.optimize(dataloaders,
                             parameter_controller=parameter_controller,
                             checkpoint_controller=checkpoint_controller
                             )

        # test_kwargs = {'dataloader_dict': dataloaders,
        #                'epoch': None,
        #                'partition': 'extra'
        #                }

        # trainer.test(checkpoint_controller, predict_only=1, **test_kwargs)

    def run_eval(self, path_model: str = None):

        # criterion = CCCLoss()
        if self.task == constants.CLASSFICATION:
            criterion = nn.CrossEntropyLoss(reduction="mean").to(self.device)

        else:
            raise NotImplementedError(self.task)

        fold = self.fold_to_run

        # save_path = os.path.join(self.save_path,
        #                          self.exp_id + "_" + self.model_name +
        #                          "_" + self.stamp + "_fold" + str(
        #                              fold) + "_" + self.emotion)
        save_path = self.outd
        os.makedirs(save_path, exist_ok=True)

        checkpoint_filename = join(save_path, "checkpoint.pkl")

        model = self.init_model()
        # override the model's weights
        assert os.path.isfile(path_model), path_model
        state_dict = torch.load(path_model, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f'Loaded the weights from {path_model}')

        dataloaders = self.init_dataloader(fold)

        trainer_kwards = {'device': self.device,
                          'emotion': self.emotion,
                          'model_name': self.model_name,
                          'models': model,
                          'save_path': save_path,
                          'fold': fold,
                          'min_epoch': self.min_num_epochs,
                          'max_epoch': self.num_epochs,
                          'early_stopping': self.early_stopping,
                          # 'scheduler': self.scheduler,
                          'learning_rate': self.learning_rate,
                          'min_learning_rate': self.min_learning_rate,
                          'patience': self.patience,
                          'train_batch_size': self.train_batch_size,
                          'eval_batch_size': self.eval_batch_size,
                          'criterion': criterion,
                          'factor': self.factor,
                          'verbose': True,
                          'milestone': self.milestone,
                          'metrics': self.config['metrics'],
                          'load_best_at_each_epoch': self.load_best_at_each_epoch,
                          'save_plot': self.config['save_plot']
                          }

        trainer = Trainer(**trainer_kwards)
        trainer.set_args(self.args)
        trainer.post_set_args()
        trainer.set_number_classes(self.num_classes)
        trainer.init_optimizer_and_scheduler(epoch=0)

        trainer.inference(dataloaders[constants.TESTSET])

    def init_dataset(self, data, continuous_label_dim, mode, fold=None):
        dataset = Dataset(data,
                          continuous_label_dim,
                          self.modality,
                          self.multiplier,
                          self.feature_dimension,
                          self.window_length,
                          mode,
                          mean_std=self.mean_std_dict,
                          time_delay=self.time_delay
                          )
        dataset.set_args(self.args)

        return dataset

    def init_model(self):
        self.init_randomness()
        modality = [modal for modal in self.modality if "continuous_label" not in modal]

        if self.model_name == "LFAN":
            model = LFAN(backbone_settings=self.config['backbone_settings'],
                         output_dim=self.num_classes,
                         task=self.task,
                         modality=modality,
                         example_length=self.window_length,
                         kernel_size=self.tcn_kernel_size,
                         tcn_channel=self.config['tcn']['channels'],
                         modal_dim=self.modal_dim,
                         num_heads=self.num_heads,
                         root_dir=self.load_path,
                         device=self.device
                         )
            model.init()

        elif self.model_name == "CAN":
            model = CAN(task=self.task,
                        modalities=modality,
                        tcn_settings=self.config['tcn_settings'],
                        backbone_settings=self.config['backbone_settings'],
                        output_dim=self.num_classes,
                        root_dir=self.load_path,
                        device=self.device
                        )

        elif self.model_name == "JMT":
            model = JMT(task=self.task,
                        root_dir=self.load_path,
                        modalities=modality,
                        tcn_settings=self.config['tcn_settings'],
                        backbone_settings=self.config['backbone_settings'],
                        output_dim=self.num_classes,
                        device=self.device,
                        model_name='JMT'
                        )

        elif self.model_name == "MT":
            model = JMT(task=self.task,
                        root_dir=self.load_path,
                        modalities=modality,
                        tcn_settings=self.config['tcn_settings'],
                        backbone_settings=self.config['backbone_settings'],
                        output_dim=self.num_classes,
                        device=self.device,
                        model_name='MT'
                        )
        else:
            raise NotImplementedError(self.model_name)

        return model

    def get_modality(self):
        pass

    def get_config(self):
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):

        ds = self.dataset_name

        if ds in [constants.MELD, constants.C_EXPR_DB,
                  constants.C_EXPR_DB_CHALLENGE]:
            return [0]

        else:
            if self.emotion == "arousal":
                dim = [1]
            elif self.emotion == "valence":
                dim = [0]
            else:
                raise ValueError("Unknown emotion!")
            return dim
