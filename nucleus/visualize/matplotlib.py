from typing import Optional, List

import enum
import numpy as np
import tensorflow as tf


def _parse_cmap(cmap_name=None, image_shape_len=3):
    import matplotlib.cm as cm
    if cmap_name is not None:
        return cm.get_cmap(cmap_name)
    else:
        if image_shape_len == 2:
            # Single channels are viewed in Gray by default
            return cm.gray
        else:
            return None


def _parse_axes_limits(min_x, max_x, min_y, max_y, axes_x_limits,
                       axes_y_limits):
    if isinstance(axes_x_limits, int):
        axes_x_limits = float(axes_x_limits)
    if isinstance(axes_y_limits, int):
        axes_y_limits = float(axes_y_limits)
    if isinstance(axes_x_limits, float):
        pad = (max_x - min_x) * axes_x_limits
        axes_x_limits = [min_x - pad, max_x + pad]
    if isinstance(axes_y_limits, float):
        pad = (max_y - min_y) * axes_y_limits
        axes_y_limits = [min_y - pad, max_y + pad]
    return axes_x_limits, axes_y_limits


def _set_figure_size(fig, figure_size=(19.20, 10.80)):
    if figure_size is not None:
        fig.set_size_inches(np.asarray(figure_size))


def _set_axes_options(ax, render_axes=True, inverted_y_axis=True,
                      axes_font_name='sans-serif', axes_font_size=10,
                      axes_font_style='normal', axes_font_weight='normal',
                      axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                      axes_y_ticks=None, axes_x_label=None, axes_y_label=None,
                      title=None):
    if render_axes:
        # render axes
        ax.set_axis_on()
        # set font options
        for l in (ax.get_xticklabels() + ax.get_yticklabels()):
            l.set_fontsize(axes_font_size)
            l.set_fontname(axes_font_name)
            l.set_fontstyle(axes_font_style)
            l.set_fontweight(axes_font_weight)
        # set ticks
        if axes_x_ticks is not None:
            ax.set_xticks(axes_x_ticks)
        if axes_y_ticks is not None:
            ax.set_yticks(axes_y_ticks)
        # set labels and title
        if axes_x_label is None:
            axes_x_label = ''
        if axes_y_label is None:
            axes_y_label = ''
        if title is None:
            title = ''
        ax.set_xlabel(
            axes_x_label, fontsize=axes_font_size, fontname=axes_font_name,
            fontstyle=axes_font_style, fontweight=axes_font_weight)
        ax.set_ylabel(
            axes_y_label, fontsize=axes_font_size, fontname=axes_font_name,
            fontstyle=axes_font_style, fontweight=axes_font_weight)
        ax.set_title(
            title, fontsize=axes_font_size, fontname=axes_font_name,
            fontstyle=axes_font_style, fontweight=axes_font_weight)
    else:
        # do not render axes
        ax.set_axis_off()
        # also remove the ticks to get rid of the white area
        ax.set_xticks([])
        ax.set_yticks([])

    # set axes limits
    if axes_x_limits is not None:
        ax.set_xlim(np.sort(axes_x_limits))
    if axes_y_limits is None:
        axes_y_limits = ax.get_ylim()
    if inverted_y_axis:
        ax.set_ylim(np.sort(axes_y_limits)[::-1])
    else:
        ax.set_ylim(np.sort(axes_y_limits))


def _set_numbering(ax, centers, render_numbering=True,
                   numbers_horizontal_align='center',
                   numbers_vertical_align='bottom',
                   numbers_font_name='sans-serif', numbers_font_size=10,
                   numbers_font_style='normal', numbers_font_weight='normal',
                   numbers_font_colour='k'):
    if render_numbering:
        for k, p in enumerate(centers):
            ax.annotate(
                str(k), xy=(p[0], p[1]),
                horizontalalignment=numbers_horizontal_align,
                verticalalignment=numbers_vertical_align,
                size=numbers_font_size, family=numbers_font_name,
                fontstyle=numbers_font_style, fontweight=numbers_font_weight,
                color=numbers_font_colour)


class MatplotlibRenderer:

    def __init__(self, figure_id, new_figure):
        if figure_id is not None and new_figure:
            raise ValueError('Conflicting arguments. figure_id cannot be '
                             'specified if the new_figure flag is True.')

        self.figure_id = figure_id
        self.new_figure = new_figure

        self.figure = self.get_figure()

    def get_figure(self):
        import matplotlib.pyplot as plt

        if self.new_figure or self.figure_id is not None:
            self.figure = plt.figure(self.figure_id)
        else:
            self.figure = plt.gcf()

        self.figure_id = self.figure.number

        return self.figure

    def clear_figure(self):
        self.figure.clf()


class MatplotlibImageViewer(MatplotlibRenderer):

    def __init__(self, figure_id, new_figure, pixels, labels):
        super(MatplotlibImageViewer, self).__init__(figure_id, new_figure)
        self.pixels = pixels.numpy()
        self.labels = (
            ['unknown'] if labels is None or len(labels) == 0 else labels
        )
        self.scores = None

    def render(
            self,
            render_labels =True,
            render_scores=True,

            caption_text_color='black',
            caption_text_size=16,

            interpolation='bilinear',
            cmap_name=None,
            alpha=1.,
            render_axes=False,
            axes_font_name='sans-serif',
            axes_font_size=10,
            axes_font_style='normal',
            axes_font_weight='normal',
            axes_x_limits=None,
            axes_y_limits=None,
            axes_x_ticks=None,
            axes_y_ticks=None,
            figure_size=(19.20, 10.80)
    ):
        import matplotlib.pyplot as plt

        # parse colour map argument
        cmap = _parse_cmap(cmap_name=cmap_name,
                           image_shape_len=len(self.pixels.shape))

        # parse axes limits
        axes_x_limits, axes_y_limits = _parse_axes_limits(
            0., self.pixels.shape[1], 0., self.pixels.shape[0], axes_x_limits,
            axes_y_limits)

        # render chw
        plt.imshow(self.pixels, cmap=cmap, interpolation=interpolation,
                   alpha=alpha)

        # store axes object
        ax = plt.gca()

        if render_labels or render_scores:
            labels = []
            if render_labels:
                labels += self.labels

            scores = []
            if render_scores and self.scores is not None:
                scores += self.scores

            caption = ''
            if len(labels) > 0 and len(scores) > 0:
                caption = '\n'.join(
                    f'{l} : {s}' for l, s in zip(labels, scores)
                )
            elif len(labels) > 0:
                caption = '\n'.join(f'{l}' for l in labels)
            elif len(scores) > 0:
                caption = '\n'.join(f'{s}' for s in scores)

            if caption != '':
                ax.set_title(
                    label=caption,
                    fontsize=caption_text_size,
                    color=caption_text_color
                )

        # set axes options
        _set_axes_options(
            ax, render_axes=render_axes, inverted_y_axis=True,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks)

        # set figure _size
        _set_figure_size(self.figure, figure_size)

        return self


class MatplotlibImageCollectionViewer:
    pass


box_default_style1 = dict(
    resolution=(1080, 1920),
    label_color=True,
    render_label=True,
    render_score=True,
    render_axes=True,
    alpha=1.0,
    edge_color='blue',
    face_color='blue',
    fill=False,
    line_style='-',
    line_width=1,
    stroke_line_width=0,
    stroke_foreground='white',
    auto_size_text=True,
    caption_text_color='white',
    caption_text_size=12,
    caption_edge_color='blue',
    caption_face_color='blue',
    caption_box_alpha=1.0,
    caption_position='bottom',
    axes_font_name='sans-serif',
    axes_font_size=10,
    axes_font_style='normal',
    axes_font_weight='normal',
    axes_x_limits=None,
    axes_y_limits=None,
    axes_x_ticks=None,
    axes_y_ticks=None,
    figure_size=(19.20, 10.80)
)

box_default_style2 = dict(
    resolution=(1080, 1920),
    label_color=True,
    render_label=False,
    render_score=False,
    render_axes=True,
    alpha=1.0,
    edge_color='blue',
    face_color='blue',
    fill=False,
    line_style='-',
    line_width=1,
    stroke_line_width=0,
    stroke_foreground='white',
    auto_size_text=True,
    caption_text_color='white',
    caption_text_size=12,
    caption_edge_color='blue',
    caption_face_color='blue',
    caption_box_alpha=1.0,
    caption_position='bottom',
    axes_font_name='sans-serif',
    axes_font_size=10,
    axes_font_style='normal',
    axes_font_weight='normal',
    axes_x_limits=None,
    axes_y_limits=None,
    axes_x_ticks=None,
    axes_y_ticks=None,
    figure_size=(9.6, 5.4)
)


class MatplotlibBoxViewer(MatplotlibRenderer):
    r"""

    Parameters
    ----------
    figure_id
    new_figure
    ijhw
    labels
    scores
    """
    def __init__(
            self,
            figure_id: int,
            new_figure: bool,
            ijhw: tf.Tensor,
            labels: Optional[List[str]] = None,
            scores: Optional[List[float]] = None
    ):
        super(MatplotlibBoxViewer, self).__init__(figure_id, new_figure)
        self.ijhw = ijhw
        self.labels = (
            ['unknown'] if labels is None or len(labels) == 0 else labels
        )
        self.scores = scores

    # TODO[jalabort]: Can we not get resolution from pixels?
    def render(
            self,
            resolution=(1080, 1920),

            label_color_map=None,
            render_labels=True,
            render_scores=True,
            skip_labels=None,

            render_axes=True,
            alpha=1.0,
            edge_color='blue',
            face_color='blue',
            fill=False,
            line_style='-',
            line_width=1,
            stroke_line_width=0,
            stroke_foreground='white',

            auto_size_text=True,
            caption_text_color='white',
            caption_text_size=12,
            caption_edge_color='blue',
            caption_face_color='blue',
            caption_box_alpha=1.0,
            caption_position='bottom',
            axes_font_name='sans-serif',
            axes_font_size=10,
            axes_font_style='normal',
            axes_font_weight='normal',
            axes_x_limits=None,
            axes_y_limits=None,
            axes_x_ticks=None,
            axes_y_ticks=None,
            figure_size=(19.20, 10.80)
    ):
        from matplotlib import patches, patheffects
        import matplotlib.pyplot as plt
        from nucleus.box import tools as box_tools

        if skip_labels and all(label in skip_labels for label in self.labels):
            return self

        if auto_size_text:
            caption_text_size *= figure_size[0] / 19.20

        if label_color_map is not None:
            label = self.labels[0].upper()
            color = label_color_map[label].value
            edge_color = color
            face_color = color
            caption_edge_color = color
            caption_face_color = color

        ijhw = box_tools.scale_coords(coords=self.ijhw, resolution=resolution)

        ijkl = box_tools.ijhw_to_ijkl(ijhw)

        # parse axes limits1)
        min_x, min_y = np.minimum(ijkl[:2], (0, 0))
        max_x, max_y = np.maximum(ijkl[2:4], resolution)
        axes_x_limits, axes_y_limits = _parse_axes_limits(
            min_x, max_x, min_y, max_y, axes_x_limits, axes_y_limits
        )

        # get current axes object
        ax = plt.gca()

        # Create rectangle
        i, j, h, w = ijhw[:4].numpy()
        rect = patches.Rectangle(
            xy=(j, i), width=w, height=h, alpha=alpha,
            edgecolor=edge_color, facecolor=face_color, fill=fill,
            linestyle=line_style, linewidth=line_width
        )

        if stroke_line_width is not 0:
            # Add stroke effect
            rect.set_path_effects([
                patheffects.Stroke(
                    linewidth=stroke_line_width,
                    foreground=stroke_foreground
                ),
                patheffects.Normal()
            ])

        ax.add_patch(rect)

        ax.autoscale()

        if render_labels or render_scores:
            labels = []
            if render_labels:
                labels += self.labels

            scores = []
            if render_scores and self.scores is not None:
                scores += self.scores

            caption = ''
            if len(labels) > 0 and len(scores) > 0:
                caption = '\n'.join(
                    f'{l} : {s}' for l, s in zip(labels, scores)
                )
            elif len(labels) > 0:
                caption = '\n'.join(f'{l}' for l in labels)
            elif len(scores) > 0:
                caption = '\n'.join(f'{s}' for s in scores)

            if caption != '':
                corrector = 0
                if caption_position == 'top':
                    x, y = j + corrector, i - corrector
                    vertical_alignment = 'bottom'
                    horizontal_alignment = 'left'
                else:
                    x, y = j + corrector, i + h + corrector
                    horizontal_alignment = 'left'
                    vertical_alignment = 'top'
                t = ax.text(x, y, caption,
                            verticalalignment=vertical_alignment,
                            horizontalalignment=horizontal_alignment,
                            color=caption_text_color, size=caption_text_size)
                t.set_bbox(dict(facecolor=caption_face_color,
                                edgecolor=caption_edge_color,
                                alpha=caption_box_alpha))

        # set axes options
        _set_axes_options(
            ax, render_axes=render_axes, inverted_y_axis=True,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks
        )

        # set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # set figure _size
        _set_figure_size(self.figure, figure_size)

        return self


# TODO[jalabort]: Is this really needed?
class MatplotlibBoxCollectionViewer:
    pass
