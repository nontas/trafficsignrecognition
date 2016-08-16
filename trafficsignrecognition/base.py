import numpy as np
from functools import partial

from menpo.base import name_of_callable
from menpo.feature import no_op
from menpofit.visualize import print_progress

from .correlationfilter import CorrelationFilter
from .normalisation import (normalise_norm_array, image_normalisation,
                            create_cosine_mask)


class Detector(object):
    r"""
    Class for training a multi-channel correlation filter object detector.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The training images from which to learn the detector.
    algorithm : ``{'mosse', 'mccf'}``, optional
        If 'mosse', then the Minimum Output Sum of Squared Errors (MOSSE)
        filter [1] will be used. If 'mccf', then the Multi-Channel Correlation
        (MCCF) filter [2] will be used.
    filter_shape : (`int`, `int`), optional
        The shape of the filter.
    features : `callable`, optional
        The holistic dense features to be extracted from the images.
    normalisation : `callable`, optional
        The callable to be used for normalising the images.
    cosine_mask : `bool`, optional
        If ``True``, then a cosine mask (Hanning window) will be applied on the
        images.
    response_covariance : `int`, optional
        The covariance of the Gaussian desired response that will be used during
        training of the correlation filter.
    l : `float`, optional
        Regularization parameter of the correlation filter.
    boundary : ``{'constant', 'symmetric'}``, optional
        Determines the type of padding that will be applied on the images.
    verbose : `bool`, optional
        If ``True``, then a progress bar is printed.

    References
    ----------
    .. [1] D. S. Bolme, J. R. Beveridge, B. A. Draper, and Y. M. Lui. "Visual
        Object Tracking using Adaptive Correlation Filters", IEEE Proceedings
        of International Conference on Computer Vision and Pattern Recognition
        (CVPR), 2010.
    .. [2] H. K. Galoogahi, T. Sim, and Simon Lucey. "Multi-Channel
        Correlation Filters". IEEE Proceedings of International Conference on
        Computer Vision (ICCV), 2013.
    """
    def __init__(self, images, algorithm='mosse', filter_shape=(64, 64),
                 features=no_op, normalisation=normalise_norm_array,
                 cosine_mask=False, response_covariance=2, l=0.01,
                 boundary='symmetric', verbose=True):
        # Assign properties
        self.algorithm = algorithm
        self.features = features
        self.filter_shape = filter_shape
        self.normalisation = normalisation
        self.cosine_mask = cosine_mask
        self.boundary = boundary

        # Create cosine mask if asked
        cosine_mask = None
        if cosine_mask:
            cosine_mask = create_cosine_mask(filter_shape)

        # Prepare data
        wrap = partial(print_progress, prefix='Pre-processing data',
                       verbose=verbose)
        normalized_data = []
        for im in wrap(images):
            im = features(im)
            im = image_normalisation(im, normalisation=normalisation,
                                     cosine_mask=cosine_mask)
            normalized_data.append(im.pixels)

        # Create data array
        normalized_data = np.asarray(normalized_data)

        # Train correlation filter
        self.model = CorrelationFilter(
            normalized_data, algorithm=algorithm, filter_shape=filter_shape,
            response_covariance=response_covariance, l=l, boundary=boundary,
            verbose=verbose)

    @property
    def n_channels(self):
        r"""
        Returns the model's number of channels.

        :type: `int`
        """
        return self.model.n_channels

    def view_spatial_filter(self, figure_id=None, new_figure=False,
                            channels='all', interpolation='bilinear',
                            cmap_name='afmhot', alpha=1., render_axes=False,
                            axes_font_name='sans-serif', axes_font_size=10,
                            axes_font_style='normal', axes_font_weight='normal',
                            axes_x_limits=None, axes_y_limits=None,
                            axes_x_ticks=None, axes_y_ticks=None,
                            figure_size=(10, 8)):
        r"""
        View the multi-channel filter on the spatial domain.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36,
                hanning, hamming, hermite, kaiser, quadric, catrom, gaussian,
                bessel, mitchell, sinc, lanczos}
        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.

        Returns
        -------
        viewer : `ImageViewer`
            The image viewing object.
        """
        return self.model.view_spatial_filter(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def view_frequency_filter(self, figure_id=None, new_figure=False,
                              channels='all', interpolation='bilinear',
                              cmap_name='afmhot', alpha=1., render_axes=False,
                              axes_font_name='sans-serif', axes_font_size=10,
                              axes_font_style='normal', axes_font_weight='normal',
                              axes_x_limits=None, axes_y_limits=None,
                              axes_x_ticks=None, axes_y_ticks=None,
                              figure_size=(10, 8)):
        r"""
        View the multi-channel filter on the frequency domain.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        channels : `int` or `list` of `int` or ``all`` or ``None``
            If `int` or `list` of `int`, the specified channel(s) will be
            rendered. If ``all``, all the channels will be rendered in subplots.
            If ``None`` and the image is RGB, it will be rendered in RGB mode.
            If ``None`` and the image is not RGB, it is equivalent to ``all``.
        interpolation : See Below, optional
            The interpolation used to render the image. For example, if
            ``bilinear``, the image will be smooth and if ``nearest``, the
            image will be pixelated.
            Example options ::

                {none, nearest, bilinear, bicubic, spline16, spline36,
                hanning, hamming, hermite, kaiser, quadric, catrom, gaussian,
                bessel, mitchell, sinc, lanczos}
        cmap_name: `str`, optional,
            If ``None``, single channel and three channel images default
            to greyscale and rgb colormaps respectively.
        alpha : `float`, optional
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the Image as a percentage of the Image's width. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the Image as a percentage of the Image's height. If
            `tuple` or `list`, then it defines the axis limits. If ``None``, then
            the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.

        Returns
        -------
        viewer : `ImageViewer`
            The image viewing object.
        """
        return self.model.view_frequency_filter(
            figure_id=figure_id, new_figure=new_figure, channels=channels,
            interpolation=interpolation, cmap_name=cmap_name, alpha=alpha,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def __str__(self):
        output_str = r"""Correlation Filter Detector
 - Features: {}
 - Channels: {}
 """.format(name_of_callable(self.features), self.n_channels)
        return output_str + self.model.__str__()
