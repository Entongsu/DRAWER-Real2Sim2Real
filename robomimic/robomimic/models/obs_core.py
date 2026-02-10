"""
Contains torch Modules for core observation processing blocks
such as encoders (e.g. EncoderCore, VisualCore, ScanCore, ...)
and randomizers (e.g. Randomizer, CropRandomizer).
"""

import abc
import numpy as np
import textwrap
import random

import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose
import torchvision.transforms.functional as TVF

import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict

# NOTE: this is required for the backbone classes to be found by the `eval` call in the core networks
from robomimic.models.base_nets import *
from robomimic.utils.vis_utils import visualize_image_randomizer
from robomimic.macros import VISUALIZE_RANDOMIZER
"""
================================================
Encoder Core Networks (Abstract class)
================================================
"""


class EncoderCore(BaseNets.Module):
    """
    Abstract class used to categorize all cores used to encode observations
    """

    def __init__(self, input_shape):
        self.input_shape = input_shape
        super(EncoderCore, self).__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid observation encoders
        in a global dict.

        This global dict stores mapping from observation encoder network name to class.
        We keep track of these registries to enable automated class inference at runtime, allowing
        users to simply extend our base encoder class and refer to that class in string form
        in their config, without having to manually register their class internally.
        This also future-proofs us for any additional encoder classes we would
        like to add ourselves.
        """
        ObsUtils.register_encoder_core(cls)


"""
================================================
Visual Core Networks (Backbone + Pool)
================================================
"""


class VisualCore(EncoderCore, BaseNets.ConvBase):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers.
    """

    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network. Defaults
                to "ResNet18Conv".
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool". Defaults to
                "SpatialSoftmax".
            backbone_kwargs (dict): kwargs for the visual backbone network (optional)
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the visual features
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
        """
        super(VisualCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten

        if backbone_kwargs is None:
            backbone_kwargs = dict()

        # add input channel dimension to visual core inputs
        backbone_kwargs["input_channel"] = input_shape[0]

        # extract only relevant kwargs for this specific backbone
        backbone_kwargs = extract_class_init_kwargs_from_dict(
            cls=eval(backbone_class), dic=backbone_kwargs, copy=True)

        # visual backbone
        assert isinstance(backbone_class, str)
        self.backbone = eval(backbone_class)(**backbone_kwargs)

        assert isinstance(self.backbone, BaseNets.ConvBase)

        feat_shape = self.backbone.output_shape(input_shape)
        net_list = [self.backbone]

        # maybe make pool net
        if pool_class is not None:
            assert isinstance(pool_class, str)
            # feed output shape of backbone to pool net
            if pool_kwargs is None:
                pool_kwargs = dict()
            # extract only relevant kwargs for this specific backbone
            pool_kwargs["input_shape"] = feat_shape
            pool_kwargs = extract_class_init_kwargs_from_dict(
                cls=eval(pool_class), dic=pool_kwargs, copy=True)
            self.pool = eval(pool_class)(**pool_kwargs)
            assert isinstance(self.pool, BaseNets.Module)

            feat_shape = self.pool.output_shape(feat_shape)
            net_list.append(self.pool)
        else:
            self.pool = None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        self.feature_dimension = feature_dimension
        if feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)),
                                     feature_dimension)
            net_list.append(linear)

        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(feat_shape)
        # backbone + flat output
        if self.flatten:
            return [np.prod(feat_shape)]
        else:
            return feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)

        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        

        return super(VisualCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(
                self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone),
                               indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg


"""
================================================
Scan Core Networks (Conv1D Sequential + Pool)
================================================
"""


class ScanCore(EncoderCore, BaseNets.ConvBase):
    """
    A network block that combines a Conv1D backbone network with optional pooling
    and linear layers.
    """

    def __init__(
        self,
        input_shape,
        conv_kwargs=None,
        conv_activation="relu",
        pool_class=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=None,
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            conv_kwargs (dict): kwargs for the conv1d backbone network. Should contain lists for the following values:
                out_channels (int)
                kernel_size (int)
                stride (int)
                ...

                If not specified, or an empty dictionary is specified, some default settings will be used.
            conv_activation (str or None): Activation to use between conv layers. Default is relu.
                Currently, valid options are {relu}
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool"
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the network output
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension (note: flatten must be set to True!)
        """
        super(ScanCore, self).__init__(input_shape=input_shape)
        self.flatten = flatten
        self.feature_dimension = feature_dimension

        if conv_kwargs is None:
            conv_kwargs = dict()

        # Generate backbone network
        # N input channels is assumed to be the first dimension
        self.backbone = BaseNets.Conv1dBase(
            input_channel=self.input_shape[0],
            activation=conv_activation,
            **conv_kwargs,
        )
        feat_shape = self.backbone.output_shape(input_shape=input_shape)

        # Create netlist of all generated networks
        net_list = [self.backbone]

        # Possibly add pooling network
        if pool_class is not None:
            # Add an unsqueeze network so that the shape is correct to pass to pooling network
            self.unsqueeze = Unsqueeze(dim=-1)
            net_list.append(self.unsqueeze)
            # Get output shape
            feat_shape = self.unsqueeze.output_shape(feat_shape)
            # Create pooling network
            self.pool = eval(pool_class)(input_shape=feat_shape, **pool_kwargs)
            net_list.append(self.pool)
            feat_shape = self.pool.output_shape(feat_shape)
        else:
            self.unsqueeze, self.pool = None, None

        # flatten layer
        if self.flatten:
            net_list.append(torch.nn.Flatten(start_dim=1, end_dim=-1))

        # maybe linear layer
        if self.feature_dimension is not None:
            assert self.flatten
            linear = torch.nn.Linear(int(np.prod(feat_shape)),
                                     self.feature_dimension)
            net_list.append(linear)

        # Generate final network
        self.nets = nn.Sequential(*net_list)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        if self.feature_dimension is not None:
            # linear output
            return [self.feature_dimension]
        feat_shape = self.backbone.output_shape(input_shape)
        if self.pool is not None:
            # pool output
            feat_shape = self.pool.output_shape(
                self.unsqueeze.output_shape(feat_shape))
        # backbone + flat output
        return [np.prod(feat_shape)] if self.flatten else feat_shape

    def forward(self, inputs):
        """
        Forward pass through visual core.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        return super(ScanCore, self).forward(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(
                self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone),
                               indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        msg = header + '(' + msg + '\n)'
        return msg


"""
================================================
Observation Randomizer Networks
================================================
"""


class Randomizer(BaseNets.Module):
    """
    Base class for randomizer networks. Each randomizer should implement the @output_shape_in,
    @output_shape_out, @forward_in, and @forward_out methods. The randomizer's @forward_in
    method is invoked on raw inputs, and @forward_out is invoked on processed inputs
    (usually processed by a @VisualCore instance). Note that the self.training property
    can be used to change the randomizer's behavior at train vs. test time.
    """

    def __init__(self):
        super(Randomizer, self).__init__()

    def __init_subclass__(cls, **kwargs):
        """
        Hook method to automatically register all valid subclasses so we can keep track of valid observation randomizers
        in a global dict.

        This global dict stores mapping from observation randomizer network name to class.
        We keep track of these registries to enable automated class inference at runtime, allowing
        users to simply extend our base randomizer class and refer to that class in string form
        in their config, without having to manually register their class internally.
        This also future-proofs us for any additional randomizer classes we would
        like to add ourselves.
        """
        ObsUtils.register_randomizer(cls)

    def output_shape(self, input_shape=None):
        """
        This function is unused. See @output_shape_in and @output_shape_out.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward_in(self, inputs):
        """
        Randomize raw inputs if training.
        """
        if self.training:
            randomized_inputs = self._forward_in(inputs=inputs)
            if VISUALIZE_RANDOMIZER:
                num_samples_to_visualize = min(4, inputs.shape[0])
                self._visualize(
                    inputs,
                    randomized_inputs,
                    num_samples_to_visualize=num_samples_to_visualize)
            return randomized_inputs
        else:
            return self._forward_in_eval(inputs)

    def forward_out(self, inputs):
        """
        Processing for network outputs.
        """
        if self.training:
            return self._forward_out(inputs)
        else:
            return self._forward_out_eval(inputs)

    @abc.abstractmethod
    def _forward_in(self, inputs):
        """
        Randomize raw inputs.
        """
        raise NotImplementedError

    def _forward_in_eval(self, inputs):
        """
        Test-time behavior for the randomizer
        """
        return inputs

    @abc.abstractmethod
    def _forward_out(self, inputs):
        """
        Processing for network outputs.
        """
        return inputs

    def _forward_out_eval(self, inputs):
        """
        Test-time behavior for the randomizer
        """
        return inputs

    @abc.abstractmethod
    def _visualize(self,
                   pre_random_input,
                   randomized_input,
                   num_samples_to_visualize=2):
        """
        Visualize the original input and the randomized input for _forward_in for debugging purposes.
        """
        pass


class CropRandomizer(Randomizer):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """

    def __init__(
        self,
        input_shape,
        crop_height=76,
        crop_width=76,
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super(CropRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(
            inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        out, _ = ObsUtils.sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        return TensorUtils.join_dimensions(out, 0, 1)

    def _forward_in_eval(self, inputs):
        """
        Do center crops during eval
        """
        assert len(
            inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        inputs = inputs.permute(*range(inputs.dim() - 3),
                                inputs.dim() - 2,
                                inputs.dim() - 1,
                                inputs.dim() - 3)
        out = ObsUtils.center_crop(inputs, self.crop_height, self.crop_width)
        out = out.permute(*range(out.dim() - 3),
                          out.dim() - 1,
                          out.dim() - 3,
                          out.dim() - 2)
        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_crops)
        out = TensorUtils.reshape_dimensions(inputs,
                                             begin_axis=0,
                                             end_axis=0,
                                             target_dims=(batch_size,
                                                          self.num_crops))
        return out.mean(dim=1)

    def _visualize(self,
                   pre_random_input,
                   randomized_input,
                   num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0,
                                           batch_size,
                                           size=(num_samples_to_visualize, ))
        pre_random_input_np = TensorUtils.to_numpy(
            pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size,
                         self.num_crops))  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(
            randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose(
            (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose(
            (0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(pre_random_input_np,
                                   randomized_input_np,
                                   randomizer_name='{}'.format(
                                       str(self.__class__.__name__)))

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width,
            self.num_crops)
        return msg


class ColorRandomizer(Randomizer):
    """
    Randomly sample color jitter at input, and then average across color jtters at output.
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            brightness (None or float or 2-tuple): How much to jitter brightness. brightness_factor is chosen uniformly
                from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. Should be non negative numbers.
            contrast (None or float or 2-tuple): How much to jitter contrast. contrast_factor is chosen uniformly
                from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. Should be non negative numbers.
            saturation (None or float or 2-tuple): How much to jitter saturation. saturation_factor is chosen uniformly
                from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. Should be non negative numbers.
            hue (None or float or 2-tuple): How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or
                the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. To jitter hue, the pixel
                values of the input image has to be non-negative for conversion to HSV space; thus it does not work
                if you normalize your image to an interval with negative values, or use an interpolation that
                generates negative values before using this function.
            num_samples (int): number of random color jitters to take
        """
        super(ColorRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)

        self.input_shape = input_shape
        self.brightness = [
            max(0, 1 - brightness), 1 + brightness
        ] if type(brightness) in {float, int} else brightness
        self.contrast = [max(0, 1 - contrast), 1 + contrast
                         ] if type(contrast) in {float, int} else contrast
        self.saturation = [
            max(0, 1 - saturation), 1 + saturation
        ] if type(saturation) in {float, int} else saturation
        self.hue = [-hue, hue] if type(hue) in {float, int} else hue
        self.num_samples = num_samples

    @torch.jit.unused
    def get_transform(self):
        """
        Get a randomized transform to be applied on image.

        Implementation taken directly from:

        https://github.com/pytorch/vision/blob/2f40a483d73018ae6e1488a484c5927f2b309969/torchvision/transforms/transforms.py#L1053-L1085

        Returns:
            Transform: Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0],
                                               self.brightness[1])
            transforms.append(
                Lambda(
                    lambda img: TVF.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0],
                                             self.contrast[1])
            transforms.append(
                Lambda(lambda img: TVF.adjust_contrast(img, contrast_factor)))

        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0],
                                               self.saturation[1])
            transforms.append(
                Lambda(
                    lambda img: TVF.adjust_saturation(img, saturation_factor)))

        if self.hue is not None:
            hue_factor = random.uniform(self.hue[0], self.hue[1])
            transforms.append(
                Lambda(lambda img: TVF.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def get_batch_transform(self, N):
        """
        Generates a batch transform, where each set of sample(s) along the batch (first) dimension will have the same
        @N unique ColorJitter transforms applied.

        Args:
            N (int): Number of ColorJitter transforms to apply per set of sample(s) along the batch (first) dimension

        Returns:
            Lambda: Aggregated transform which will autoamtically apply a different ColorJitter transforms to
                each sub-set of samples along batch dimension, assumed to be the FIRST dimension in the inputted tensor
                Note: This function will MULTIPLY the first dimension by N
        """
        return Lambda(lambda x: torch.stack(
            [self.get_transform()(x_) for x_ in x for _ in range(N)]))

    def output_shape_in(self, input_shape=None):
        # outputs are same shape as inputs
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random color jitters for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(
            inputs.shape) >= 3  # must have at least (C, H, W) dimensions

        if isinstance(inputs, np.ndarray):
            inputs = torch.as_tensor(inputs)

        # Make sure shape is exactly 4
        if len(inputs.shape) == 3:
            inputs = torch.unsqueeze(inputs, dim=0)

        # Create lambda to aggregate all color randomizings at once
        transform = self.get_batch_transform(N=self.num_samples)

        return transform(inputs)

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(inputs,
                                             begin_axis=0,
                                             end_axis=0,
                                             target_dims=(batch_size,
                                                          self.num_samples))
        return out.mean(dim=1)

    def _visualize(self,
                   pre_random_input,
                   randomized_input,
                   num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0,
                                           batch_size,
                                           size=(num_samples_to_visualize, ))
        pre_random_input_np = TensorUtils.to_numpy(
            pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size,
                         self.num_samples))  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(
            randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose(
            (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose(
            (0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(pre_random_input_np,
                                   randomized_input_np,
                                   randomizer_name='{}'.format(
                                       str(self.__class__.__name__)))

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, brightness={self.brightness}, contrast={self.contrast}, " \
                       f"saturation={self.saturation}, hue={self.hue}, num_samples={self.num_samples})"
        return msg


class GaussianNoiseRandomizer(Randomizer):
    """
    Randomly sample gaussian noise at input, and then average across noises at output.
    """

    def __init__(
        self,
        input_shape,
        noise_mean=0.0,
        noise_std=0.3,
        limits=None,
        num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            noise_mean (float): Mean of noise to apply
            noise_std (float): Standard deviation of noise to apply
            limits (None or 2-tuple): If specified, should be the (min, max) values to clamp all noisied samples to
            num_samples (int): number of random color jitters to take
        """
        super(GaussianNoiseRandomizer, self).__init__()

        self.input_shape = input_shape
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.limits = limits
        self.num_samples = num_samples

    def output_shape_in(self, input_shape=None):
        # outputs are same shape as inputs
        return list(input_shape)

    def output_shape_out(self, input_shape=None):
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def _forward_in(self, inputs):
        """
        Samples N random gaussian noises for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        out = TensorUtils.repeat_by_expand_at(inputs,
                                              repeats=self.num_samples,
                                              dim=0)

        # Sample noise across all samples
        out = torch.rand(size=out.shape).to(
            inputs.device) * self.noise_std + self.noise_mean + out

        # Possibly clamp
        if self.limits is not None:
            out = torch.clip(out, min=self.limits[0], max=self.limits[1])

        return out

    def _forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = (inputs.shape[0] // self.num_samples)
        out = TensorUtils.reshape_dimensions(inputs,
                                             begin_axis=0,
                                             end_axis=0,
                                             target_dims=(batch_size,
                                                          self.num_samples))
        return out.mean(dim=1)

    def _visualize(self,
                   pre_random_input,
                   randomized_input,
                   num_samples_to_visualize=2):
        batch_size = pre_random_input.shape[0]
        random_sample_inds = torch.randint(0,
                                           batch_size,
                                           size=(num_samples_to_visualize, ))
        pre_random_input_np = TensorUtils.to_numpy(
            pre_random_input)[random_sample_inds]
        randomized_input = TensorUtils.reshape_dimensions(
            randomized_input,
            begin_axis=0,
            end_axis=0,
            target_dims=(batch_size,
                         self.num_samples))  # [B * N, ...] -> [B, N, ...]
        randomized_input_np = TensorUtils.to_numpy(
            randomized_input[random_sample_inds])

        pre_random_input_np = pre_random_input_np.transpose(
            (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        randomized_input_np = randomized_input_np.transpose(
            (0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        visualize_image_randomizer(pre_random_input_np,
                                   randomized_input_np,
                                   randomizer_name='{}'.format(
                                       str(self.__class__.__name__)))

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + f"(input_shape={self.input_shape}, noise_mean={self.noise_mean}, noise_std={self.noise_std}, " \
                       f"limits={self.limits}, num_samples={self.num_samples})"
        return msg


import torch
import torch.nn as nn

from robomimic.models.mlp import mlp1d_bn_relu, mlp_bn_relu, mlp_relu, mlp1d_relu

__all__ = ["PointNet"]


class PointNet(EncoderCore, BaseNets.ConvBase):
    """PointNet for classification.
    Notes:
        1. The original implementation includes dropout for global MLPs.
        2. The original implementation decays the BN momentum.
    """

    def __init__(
            self,
            input_shape,
            in_channels=3,
            local_channels=(64, 64, 64, 128, 1024),
            global_channels=(512, 256),
            use_bn=True,
    ):
        super(PointNet, self).__init__(input_shape=input_shape)

        self.output_feature = global_channels[1]

        self.in_channels = in_channels
        self.out_channels = (local_channels + global_channels)[-1]
        self.use_bn = use_bn

        net_list = []

        if use_bn:
            self.mlp_local = mlp1d_bn_relu(in_channels, local_channels)
            net_list.append(self.mlp_local)
            self.mlp_global = mlp_bn_relu(local_channels[-1], global_channels)
            net_list.append(self.mlp_global)
        else:
            self.mlp_local = mlp1d_relu(in_channels, local_channels)
            self.mlp_global = mlp_relu(local_channels[-1], global_channels)

        self.reset_parameters()
        self.nets = nn.Sequential(*net_list)

    def forward(self, points) -> dict:
        # points: [B, 3, N]; points_feature: [B, C, N], points_mask: [B, N]

        local_feature = self.mlp_local(points)

        global_feature, max_indices = torch.max(local_feature, 2)
        output_feature = self.mlp_global(global_feature)

        return output_feature

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = 0.01

    def output_shape(self, inputs):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        return self.output_feature


class PointNetPolicy(EncoderCore, BaseNets.ConvBase):

    def __init__(
            self,
            input_shape,
            act_size=7,
            use_state=False,
            state_dim=0,  # tool position, tool orientation, open 3 + 4 + 2
            device="cuda",
            feature_transform=False):
        super(PointNetPolicy, self).__init__(input_shape=input_shape)

        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True,
                                 feature_transform=feature_transform)

        self.embed_dim = 64
        if use_state:
            self.embed_dim += state_dim

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(self.embed_dim, act_size)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.device = device
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        # Input shape np.array(T, B, N, 3) torch.tensor(T, B, dim)
        (coords) = x[:, :3, :]

        # coords = torch.tensor(coords).to(self.device).transpose(2, 1).float()
        # logits, trans, trans_feat = self.net(coords)

        x, trans, trans_feat = self.feat(coords)

        x = F.relu(self.bn1(self.fc1(x)))

        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        # x = torch.hstack([x, obs.to(self.device)])

        # x = self.fc4(x)

        # logits = torch.tanh(x)

        return x

    def output_shape(self, inputs):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        return self.embed_dim


from torch.autograd import Variable


class PointNetfeat(nn.Module):

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):

        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:

            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class STN3d(nn.Module):

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):

        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(
                np.array([1, 0, 0, 0, 1, 0, 0, 0, 1
                          ]).astype(np.float32))).view(1,
                                                       9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class DenseConvPolicy(EncoderCore, BaseNets.ConvBase):

    def __init__(
        self,
        input_shape,
        # obs_size,
        act_size,
        layers=[256, 256],
        pcd_normalization=None,
        pcd_scene_scale=0.5,
        emb_size=128,
        dropout=0,
        nonlinearity=torch.nn.ReLU,
        use_state=False,
        state_dim=9,  # tool position, tool orientation, open 3 + 4 + 2

        # scene_encoder_kwargs = None,
        device="cuda"):

        super(DenseConvPolicy, self).__init__(input_shape)

        scene_encoder_kwargs = {
            'local_coord': True,
            # 'encoder': pointnet_local_pool,
            'c_dim': 32,
            # 'encoder_kwargs':
            'hidden_dim': 32,
            # 'plane_type': ['xz', 'xy', 'yz', 'grid'],
            'plane_type': ["grid"],
            # 'grid_resolution': 32,
            'plane_resolution': 128,
            'unet3d': True,
            'unet3d_kwargs': {
                'num_levels': 3,
                'f_maps': 32,
                'in_channels': 32,
                'out_channels': 64,  #32,
                'plane_resolution': 128,
            },
            'unet': False,
            # 'unet_kwargs': {
            #     'depth': 5,
            #     'merge_mode': 'concat',
            #     'start_filts': 32
            # }
        }

        self.emb_size = emb_size
        self.use_state = use_state
        self.device = device

        self.scene_encoder_kwargs = scene_encoder_kwargs

        self.scene_offset = torch.Tensor([[pcd_normalization]
                                          ]).float().to(self.device)
        self.scene_scale = pcd_scene_scale

        # mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        # mc_vis['scene'].delete()
        # Encoder
        self.plane_type = self.scene_encoder_kwargs["plane_type"]
        self.point_encoder = LocalPoolPointnet(
            dim=pt_dim,
            padding=padding,
            grid_resolution=voxel_reso_grid,
            mc_vis=None,
            **self.scene_encoder_kwargs).cuda()

        net_layers = []
        dim = 0
        for key in self.plane_type:
            if key == "grid":
                dim += 128
            else:
                dim += 64

        if len(self.plane_type) == 4:
            layers[0] = 512

        if self.use_state:
            dim += state_dim
        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(nonlinearity())
            dim = layer_size
        if dropout > 0:
            net_layers.append(torch.nn.Dropout(dropout))
        dim = layer_size

        net_layers.append(torch.nn.Linear(dim, act_size))
        self.layers = net_layers
        self.mlp = torch.nn.Sequential(*net_layers).to("cuda")

    def forward(self, x):
        coords = x

        B, N, C = coords.shape

        coords = torch.tensor(coords).to(self.device).float()

        coords = (coords - self.scene_offset.repeat(
            (B, N, 1))) * self.scene_scale  # normalize

        points_embed_all = self.point_encoder(coords)
        all_features = []
        for key in points_embed_all.keys():
            points_embed = points_embed_all[key]
            feat_dim = points_embed.shape[
                1]  #self.scene_encoder_kwargs['unet3d_kwargs']['out_channels']
            if key == "grid":
                fea_grid = points_embed.permute(0, 2, 3, 4, 1)
                flat_fea_grid = fea_grid.reshape(B, -1, feat_dim)
                global_fea1_mean = flat_fea_grid.mean(1)
                global_fea1_max = flat_fea_grid.max(1).values
                all_features.append(global_fea1_max)
                all_features.append(global_fea1_mean)
            else:
                plane_embed = points_embed.permute(0, 2, 3, 1).reshape(
                    B, -1, feat_dim)  # (B, 128**2, 32)
                plane_fea1_mean = plane_embed.mean(1)  # (B, 32x3) = (B, 96)
                plane_fea1_max = plane_embed.max(
                    1).values  # (B, 32x3) = (B, 96)
                all_features.append(plane_fea1_max)
                all_features.append(plane_fea1_mean)

            # plane_fea_emb = torch.hstack([plane_fea1_mean, plane_fea1_max])  # (B, 96x2) = (B, 192)

        point_emb = torch.hstack(all_features)

        if self.use_state:
            mlp_input = torch.hstack([point_emb, obs])
        else:
            mlp_input = point_emb

        logits = self.mlp(mlp_input)

        return logits


class PointCloudRandomizer(Randomizer):

    def __init__(self,
                 input_shape,
                 num_points=1024,
                 num_samples=1,
                 pos_enc=False):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            num_points (int): number of points to sample from input
            num_samples (int): number of random point cloud samples to take
            pos_enc (bool): whether to use positional encoding
        """
        super(PointCloudRandomizer, self).__init__()

        self.num_points = num_points

    def forward_in(self, inputs):

        indices = torch.randint(0, inputs.shape[2],
                                (inputs.shape[0], self.num_points)).to(
                                    inputs.device)

        sampled_inputs = torch.gather(
            inputs, 2,
            indices.unsqueeze(1).expand(-1, inputs.shape[1], -1))

        import sys
        # sys.path.append('../IsaacLab')

        # from tools.visualization_utils import vis_pc, visualize_pcd
        # cam3d = vis_pc(sampled_inputs[0].cpu().numpy().transpose(1, 0), None)
        # visualize_pcd([cam3d])

        return sampled_inputs

    def output_shape_in(self, input_shape=None):
        return list([3, self.num_points])

    def output_shape_out(self, input_shape=None):

        return [input_shape]


class EverythingRandomizer(Randomizer):
    """
    Chains together multiple randomizers to apply a sequence of randomizations to inputs.
    """

    def __init__(
        self,
        input_shape,
        # CropRandomizer
        crop_randomizer=True,
        crop_height=76,
        crop_width=76,
        num_crops=1,
        pos_enc=False,
        # ColorRandomizer
        color_randomizer=True,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        color_num_samples=1,
        # GaussianNoiseRandomizer
        gaussian_randomizer=True,
        noise_mean=0.0,
        noise_std=0.3,
        limits=None,
        noise_num_samples=1,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
        """
        super(EverythingRandomizer, self).__init__()

        assert len(input_shape) == 3  # (C, H, W)

        self.input_shape = input_shape

        self.crop_randomizer = CropRandomizer(
            input_shape=self.input_shape,
            crop_height=crop_height,
            crop_width=crop_width,
            num_crops=num_crops,
            pos_enc=pos_enc,
        ) if crop_randomizer else None

        if self.crop_randomizer is not None:
            print(input_shape)
            self.input_shape = (self.input_shape[0], crop_height, crop_width)
            print(input_shape)

        self.color_randomizer = ColorRandomizer(
            input_shape=self.input_shape,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            num_samples=color_num_samples,
        ) if color_randomizer else None

        self.gaussian_noise_randomizer = GaussianNoiseRandomizer(
            input_shape=self.input_shape,
            noise_mean=noise_mean,
            noise_std=noise_std,
            limits=limits,
            num_samples=noise_num_samples,
        ) if gaussian_randomizer else None

    def output_shape_in(self, input_shape=None):
        output_shape = self.crop_randomizer.output_shape_in(
            input_shape) if self.crop_randomizer is not None else input_shape
        output_shape = self.color_randomizer.output_shape_in(
            output_shape
        ) if self.color_randomizer is not None else output_shape
        output_shape = self.gaussian_noise_randomizer.output_shape_in(
            output_shape
        ) if self.gaussian_noise_randomizer is not None else output_shape
        return list(output_shape)

    def output_shape_out(self, input_shape=None):
        output_shape = self.crop_randomizer.output_shape_out(
            input_shape) if self.crop_randomizer is not None else input_shape
        output_shape = self.color_randomizer.output_shape_out(
            output_shape
        ) if self.color_randomizer is not None else output_shape
        output_shape = self.gaussian_noise_randomizer.output_shape_out(
            output_shape
        ) if self.gaussian_noise_randomizer is not None else output_shape
        return list(output_shape)

    def _forward_in(self, inputs):
        out = self.crop_randomizer._forward_in(
            inputs) if self.crop_randomizer is not None else inputs
        out = self.color_randomizer._forward_in(
            out) if self.color_randomizer is not None else out
        out = self.gaussian_noise_randomizer._forward_in(
            out) if self.gaussian_noise_randomizer is not None else out

        return out

    def _forward_in_eval(self, inputs):
        out = self.crop_randomizer._forward_in_eval(
            inputs) if self.crop_randomizer is not None else inputs
        out = self.color_randomizer._forward_in_eval(
            out) if self.color_randomizer is not None else out
        out = self.gaussian_noise_randomizer._forward_in_eval(
            out) if self.gaussian_noise_randomizer is not None else out

        return out

    def _forward_out(self, inputs):
        out = self.crop_randomizer._forward_out(
            inputs) if self.crop_randomizer is not None else inputs
        out = self.color_randomizer._forward_out(
            out) if self.color_randomizer is not None else out
        out = self.gaussian_noise_randomizer._forward_out(
            out) if self.gaussian_noise_randomizer is not None else out

        return out

    def _visualize(self,
                   pre_random_input,
                   randomized_input,
                   num_samples_to_visualize=2):
        # batch_size = pre_random_input.shape[0]
        # random_sample_inds = torch.randint(0, batch_size, size=(num_samples_to_visualize,))
        # pre_random_input_np = TensorUtils.to_numpy(pre_random_input)[random_sample_inds]
        # randomized_input = TensorUtils.reshape_dimensions(
        #     randomized_input,
        #     begin_axis=0,
        #     end_axis=0,
        #     target_dims=(batch_size, self.num_samples)
        # )  # [B * N, ...] -> [B, N, ...]
        # randomized_input_np = TensorUtils.to_numpy(randomized_input[random_sample_inds])

        # pre_random_input_np = pre_random_input_np.transpose((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        # randomized_input_np = randomized_input_np.transpose((0, 1, 3, 4, 2))  # [B, N, C, H, W] -> [B, N, H, W, C]

        # visualize_image_randomizer(
        #     pre_random_input_np,
        #     randomized_input_np,
        #     randomizer_name='{}'.format(str(self.__class__.__name__))
        # )
        print("Visualizing EverythingRandomizer not implemented yet.")
        return

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(\n"
        msg += "\n  " + self.crop_randomizer.__repr__(
        ) if self.crop_randomizer is not None else ""
        msg += "\n  " + self.color_randomizer.__repr__(
        ) if self.color_randomizer is not None else ""
        msg += "\n  " + self.gaussian_noise_randomizer.__repr__(
        ) if self.gaussian_noise_randomizer is not None else ""
        msg += "\n)"
        return msg


class VisualCoreLanguageConditioned(VisualCore):
    """
    Variant of VisualCore that expects language embedding during forward pass.
    """

    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18ConvFiLM",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
    ):
        """
        Update default backbone class.
        """
        super(VisualCoreLanguageConditioned, self).__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=pool_class,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=pool_kwargs,
            flatten=flatten,
            feature_dimension=feature_dimension,
        )

    def forward(self, inputs, lang_emb=None):
        """
        Update forward pass to pass language embedding through ResNet18ConvFiLM.
        """
        assert lang_emb is not None
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)

        # feed lang_emb through backbone explicitly, and then feed through rest of network
        assert self.backbone is not None
        x = self.backbone(inputs, lang_emb)
        x = self.nets[1:](x)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(
                x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' %
                             (str(self.output_shape(list(
                                 inputs.shape)[1:])), str(list(x.shape)[1:])))
        return x


from typing import Optional, Dict, Tuple, Union, List, Type


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class StateMLP(EncoderCore, BaseNets.ConvBase):

    def __init__(
            self,
            input_shape,
            state_mlp_size=(16, 16),
            state_mlp_activation_fn=nn.ReLU,
    ):
        super(StateMLP, self).__init__(input_shape)
        self.output_dim = state_mlp_size[-1]
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]

        self.state_mlp = nn.Sequential(
            *create_mlp(input_shape[0], self.output_dim, net_arch,
                        state_mlp_activation_fn))
        self.input_shape = input_shape

    def forward(self, x):
        state_feat = self.state_mlp(x)
        return state_feat

    def output_shape_in(self, input_shape=None):
        return list([self.input_shape[0]])

    def output_shape_out(self, input_shape=None):

        return [input_shape]

    def output_shape(self, inputs):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        return self.output_dim
