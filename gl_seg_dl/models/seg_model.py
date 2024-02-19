import logging

import segmentation_models_pytorch as smp
import torch


class SegModel(torch.nn.Module):
    def __init__(self, input_settings, training_settings, model_name, model_args):
        super().__init__()
        self.model_args = model_args
        self.model_name = model_name

        # extract the inputs
        self.input_settings = input_settings
        self.use_elevation = input_settings['elevation']

        # prepare the logger
        self.logger = logging.getLogger('pytorch_lightning.core')

        # compute the number of input channels based on what variables are used
        num_ch = 4
        if self.use_elevation:
            num_ch += 1
        self.model_args['in_channels'] = num_ch

        # set the number of output channels
        self.model_args['classes'] = 1

        self.logger.info(f'Building Unet with {self.model_args}')
        self.unet = getattr(smp, self.model_name)(**self.model_args)

    def forward(self, batch):
        input_list = []
        img_data = batch['s1_bands']

        # add the S1-bands
        input_list.append(img_data)

        # add the DEM
        if self.use_elevation:
            input_list.append(batch['dem'][:, None, :, :])

        # concatenate all the inputs over channel
        inputs = torch.cat(input_list, dim=1)

        # get the predictions
        preds = self.unet(inputs)

        return preds
