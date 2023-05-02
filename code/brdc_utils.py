import os 
import itertools

import pandas as pd
import numpy as np

import scipy.stats
from scipy.stats._stats_py import _chk_asarray
from lifelines.statistics import multivariate_logrank_test

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.text as mtext
import matplotlib.ticker as mticker

import seaborn as sns

from statannotations.Annotator import Annotator
from adjustText import adjust_text
from textwrap import wrap


params = {
    'pdf.fonttype': 42,
    'axes.unicode_minus': False,
    'font.sans-serif': 'Arial',
    'lines.linewidth': 0.5,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.linewidth': 0.5,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black'
}
mpl.rcParams.update(params)


def sort_custom(item, order, label=None):
    df_for_sort = item.copy()
    if isinstance(item, pd.DataFrame):
        item = df_for_sort[label]
    if not all(sorted(order) == np.unique(item)):
        raise ValueError(
            'order is not equal to {}, please check the argument.'.format(item.name))
    else:
        tmp_order = item.map(dict(zip(order, range(len(order))))).sort_values()
        if isinstance(df_for_sort, pd.DataFrame):
            item = df_for_sort
        return item.loc[tmp_order.index]


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,
                           usetex=False, **self.text_props)
        handlebox.add_artist(title)
        return title


def _heatmap_legend_handle(lut):
    handles = []
    handles_name = []
    for type_ in lut:
        i = 0
        for name in lut[type_]:
            if i == 0:
                handles.extend(['', type_])
                handles_name.extend(['', ''])
            handles.append(Patch(facecolor=lut[type_][name]))
            handles_name.append(name)
            i += 1
    return handles, handles_name


def heatmap(df,
            lut=None,
            z_score=None,
            nan_policy='omit',
            cmap=None,
            vmax=1,
            vmin=-1,
            center=0,
            col_cluster=False,
            row_cluster=False,
            xticklabels=False,
            yticklabels=True,
            xlabel=None,
            ylabel=None,
            figsize=(4, 4),
            legend=True,
            **kwargs):
    """
    Create a heatmap or clustered heatmap of the given data.

    Parameters
    ----------
    df : pandas DataFrame
        The data to plot.
    lut : Dict, optional
        A nested dictionary mapping column or index names to colors. The columns/index names were the key of the outer dictionary, the components of columns/index were the key of in inner dictionary, the hex code of colors were the value of the inner dictionary.
    z_score : int, optional
        The axis along which to standardize the data. Default is None. 0 for row and 1 for columns.
    nan_policy : str, optional
        How to handle missing values when standard the data. Default is 'omit'.
    cmap : str, optional
        The colormap to use for the heatmap.
    vmax : float, optional
        The maximum value to display in the heatmap. Default is 1.
    vmin : float, optional
        The minimum value to display in the heatmap. Default is -1.
    center : float, optional
        The value at which to center the colormap. Default is 0.
    col_cluster : bool, optional
        Whether to cluster the columns of the heatmap. Default is False.
    row_cluster : bool, optional
        Whether to cluster the rows of the heatmap. Default is False.
    xticklabels : bool, optional
        Whether to display tick labels along the x axis. Default is False.
    yticklabels : bool, optional
        Whether to display tick labels along the y axis. Default is True.
    xlabel : str, optional
        The label for the x axis. Default is None.
    ylabel : str, optional
        The label for the y axis. Default is None.
    figsize : Tuple[int, int], optional
        The size of the figure to create. Default is (4, 4).
    legend : bool, optional
        Whether to display a legend. Default is True.
    **kwargs
        Additional keyword arguments passed to sns.clustermap.

    Returns
    -------
    sns.matrix.ClusterGrid
        The ClusterGrid object representing the heatmap.
    """


    if lut:
        columns_unique = pd.unique(pd.Series(df.columns.names).dropna())
        index_unique = pd.unique(pd.Series(df.index.names).dropna())
        lut_name = np.intersect1d(np.union1d(columns_unique, index_unique), np.asarray(list(set(lut.keys()))))

        if lut_name.size == 0:
            raise ValueError("Pleast check lut parameter.")
        else:
            if set(lut_name) & set(columns_unique):
                order = [i for i in columns_unique if i in lut_name and i == i]
                col_colors = df.columns.to_frame()[order].apply(lambda x: x.map(lut[x.name]))
            else:
                col_colors = None

            if set(lut_name) & set(index_unique):
                order = [i for i in index_unique if i in lut_name]
                row_colors = df.index.to_frame()[order].apply(lambda x: x.map(lut[x.name]))
            else:
                row_colors = None
    else:
        col_colors = None
        row_colors = None

    if any((isinstance(col_colors, pd.DataFrame), isinstance(row_colors, pd.DataFrame), col_cluster, row_cluster)):
        fig = sns.clustermap(df,
                            z_score=z_score,
                            cmap=cmap,
                            vmax=vmax,
                            vmin=vmin,
                            center=center, col_cluster=col_cluster, row_cluster=row_cluster,
                            xticklabels=xticklabels, yticklabels=yticklabels, col_colors=col_colors, row_colors=row_colors, figsize=figsize,
                            **kwargs)
        ax = fig.ax_heatmap

        if legend:
            handle, handle_label = _heatmap_legend_handle(lut)
            ax.legend(handle, handle_label, handler_map={str: LegendTitle({'fontsize': 12})}, bbox_to_anchor=(
                1.05, 0.5), bbox_transform=plt.gcf().transFigure, loc='center left', labelspacing=.3, frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

    else:
        if z_score in [0, 1]:
            axis = {0: 1, 1: 0}.get(z_score, None)
            df = df.apply(scipy.stats.zscore, axis=axis, nan_policy=nan_policy)
        else:
            vmax = vmax
            vmin = vmin
            center = center
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax = sns.heatmap(df,
                         cmap=cmap,
                         vmax=vmax,
                         vmin=vmin,
                         center=center,
                         xticklabels=xticklabels, yticklabels=yticklabels,
                         **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax


def combine(f1, f2, element1=None, element2=None, how='inner', part_element=None):
    if not element1:
        ff1 = f1.copy()
    else:
        ff1 = f1[element1].dropna(how='all')

    if part_element:
        if not isinstance(part_element, dict):
            ff1 = ff1[ff1.isin(part_element)]
        else:
            for k, v in part_element.items():
                ff1 = ff1[ff1[k].isin(v)]

    if not element2:
        ff2 = f2.copy().T
    else:
        ff2 = f2.loc[element2].T

    out = pd.merge(ff1, ff2, left_index=True, right_index=True, how=how).set_index(element1, append=True)
    return out


def cateplot(df,
             x=None,
             hue=None,
             order=None,
             hue_order=None,
             title=None,
             palette=None,
             ax=None,
             figsize=(2.5, 3),
             width=0.7,
             category_type=['box', 'strip'],
             inner=None,   # violin_linewidth should > 0
             violin_linewidth=0,
             violinalpha=0.3,
             scale='width', 
             showfliers=False,
             showcaps=False,
             box_pairs='All',
             probs=None,
             box_facecolor='none',
             stripsize=6,
             stripalpha=0.8,
             orient='v',
             log_transform='log2',
             method='ttest',
             one_tile=False,
             dodge=True,
             adjust_axes=True,
             ticklabels_hide=None,
             ticklabels_format=['y'],
             ticklabels_wrap = [],
             spines_hide=['top', 'right', 'bottom'],
             labels_hide=['y'],
             **kwargs):

    df = df.reset_index()
    df, label, tmp_x, y, hue, size = _plotdata_handle(df)

    if isinstance(x, pd.Series):
        x = x.loc[df[tmp_x]]
        df.insert(1, x.name, x.values)
        x = x.name
    else:
        x = tmp_x
    
    if not order:
        order = pd.unique(df[x])
    df = sort_custom(df, order=order, label=x)
    
    if orient == 'h':
        x, y = y, x
    
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if 'violin' in category_type:
        plot = 'violinplot'
        ax = sns.violinplot(data=df,
                            x=x,
                            y=y,
                            hue=hue,
                            order=order,
                            hue_order=hue_order,
                            width=width,
                            palette=palette,
                            linewidth=violin_linewidth,
                            inner=inner,
                            scale=scale,
                            orient=orient,
                            ax=ax,
                            dodge=dodge,
                            **kwargs)
        plt.setp(ax.collections, alpha=violinalpha)

    if 'box' in category_type:
        plot = 'boxplot'
        if not hue or 'strip' in category_type:
            box_props = {
                'boxprops': {
                    'facecolor': box_facecolor,
                    'edgecolor': 'k'
                },
                'medianprops': {
                    'color': 'k'
                },
                'whiskerprops': {
                    'color': 'k'
                },
                'capprops': {
                    'color': 'k'
                }
            }
        else:
                box_props = {}
            
        ax = sns.boxplot(data=df,
                         x=x,
                         y=y,
                         hue=hue,
                         order=order,
                         hue_order=hue_order,
                         width=width,
                         palette=palette,
                         showfliers=showfliers,
                         showcaps=showcaps,
                         orient=orient,
                         ax=ax,
                         dodge=dodge,
                         **box_props,
                         **kwargs)

    if 'strip' in category_type:
        plot = 'stripplot'

        ax = sns.stripplot(data=df,
                           x=x,
                           y=y,
                           order=order,
                           hue=hue,
                           palette=palette,
                           alpha=stripalpha,
                           size=stripsize,
                           orient=orient,
                           dodge=dodge,
                           ax=ax)

    if box_pairs:
        if not probs:
            if isinstance(box_pairs, str) and box_pairs.lower() == 'all':
                if not hue:
                    box_pairs = list(itertools.combinations(np.unique(df[x]), 2))
                else:
                    list1 = df[x].unique().tolist()
                    list2 = df[hue].unique().tolist()
                    box_pairs = [tuple((j, i) for i in list2) for j in list1]
            if not hue:
                probs = [
                    statistic_func(df.set_index(x).loc[list(i), [y]].pipe(
                        dateset_preprocess_for_statistic,
                        prestatistic_method=log_transform),
                        statistic_method=method)[1][0] for i in box_pairs
                ]
            else:            
                probs = [
                    statistic_func(df.set_index([x, hue]).loc[[i], [y]].pipe(
                        dateset_preprocess_for_statistic,
                        prestatistic_method=log_transform),
                        statistic_method=method)[1][0] for i in list1
                ]
            if one_tile:
                probs = [prob/2 for prob in probs]

        add_stats(ax, df, x, y,
                  hue=hue,
                  plot=plot,
                  box_pairs=box_pairs,
                  line_offset=0.1,
                  text_offset=3,
                  probs=probs, 
                  orient=orient)
    
    if not title:
        title = y

    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide,
                   ticklabels_wrap=ticklabels_wrap)

    return ax


def f_oneway_vectorized(*samples, axis=0):

    samples = [np.asarray(sample, dtype=float) for sample in samples]
    num_groups = len(samples)
    alldata = np.concatenate(samples, axis=axis)
    bign = np.sum(~np.isnan(alldata), axis=axis)
    # print('bign', bign)
    offset = np.nanmean(alldata, axis=axis, keepdims=True)
    alldata -= offset

    normalized_ss = square_of_sums(alldata, axis=axis) / bign
    # print('normalized_ss', normalized_ss)
    sstot = sum_of_squares(alldata, axis=axis) - normalized_ss
    ssbn = 0
    for sample in samples:
        ssbn += square_of_sums(sample - offset,
                                axis=axis) / np.sum(~np.isnan(sample), axis=axis)

    from scipy import special
    ssbn -= normalized_ss
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    # print('dfwn', dfwn)
    msb = ssbn / dfbn
    msw = sswn / dfwn
    with np.errstate(divide='ignore', invalid='ignore'):
        f = msb / msw

    prob = special.fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf
    return f, prob


def kruskal_vectorized(*a):
    n = np.asarray(list(map(lambda x: len(x[0]), a)))
    num_groups = len(a)

    alldata = np.concatenate(a, axis=1)
    ranked = scipy.stats.rankdata(alldata, axis=1)
    if np.any(ranked.max(axis=1) == ranked.min(axis=1)):
        raise ValueError('All numbers are identical in kruskal')
    ties = tiecorrect_vectorized(ranked)

    j = np.insert(np.cumsum(n), 0, 0)
    ssbn = 0

    for i in range(num_groups):
        ssbn += square_of_sums(ranked[:, j[i]:j[i + 1]], axis=1) / n[i]

    totaln = np.sum(n, dtype=float)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    df = num_groups - 1
    h /= ties

    return h, scipy.stats.distributions.chi2.sf([h], [df])[0]


def tiecorrect_vectorized(rankvals):
    global idx, tmp, cnt, size
    arr = np.sort(rankvals, axis=1)
    add_list = np.ones((rankvals.shape[0], 1), dtype=bool)
    tmp = np.nonzero(
        np.concatenate((add_list, arr[:, 1:] != arr[:, :-1]), axis=1))
    idx = np.split(tmp[1], np.cumsum(np.unique(tmp[0],
                                               return_counts=True)[1]))[:-1]

    cnt = np.asarray(
        list(map(lambda x: (x[1:] - x[:-1]).astype(np.float64), idx)), dtype='object')
    size = np.float64(arr[0].size)

    return np.ones(
        (1, rankvals.shape[0])
    ) if size < 2 else 1.0 - np.asarray(list(map(lambda x: np.sum(x), cnt**3 - cnt))) / (size**3 - size)


def statistic_func(statistic_values, statistic_method='ttest'):
    if statistic_method == 'ttest':
        statistic_value = scipy.stats.ttest_ind(*statistic_values, axis=1, equal_var=True, nan_policy='omit')
    elif statistic_method == 'ranksums':
        statistic_value = scipy.stats.ranksums(*statistic_values, axis=1, nan_policy='omit')
    
    elif statistic_method == 'anova':
        statistic_value = f_oneway_vectorized(*statistic_values, axis=1)
    elif statistic_method == 'kruskal':
        statistic_value = kruskal_vectorized(*statistic_values)
    return statistic_value


def square_of_sums(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    s = np.nansum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s


def sum_of_squares(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    return np.nansum(a*a, axis)


def dateset_preprocess_for_statistic(df, prestatistic_method=None):
    prestatistic_methods = {'log2': np.log2, 'log10': np.log10}
    if isinstance(df, (pd.Series, pd.DataFrame)):
        group_values = [
            group.values for _, group in df.groupby(df.index)[df.columns[0]]
        ]
        if group_values[0].ndim == 1:
            group_values = list(map(lambda x: x[None, :], group_values))
    if prestatistic_method in prestatistic_methods.keys():
        statistic_data = list(
            map(lambda x: prestatistic_methods[prestatistic_method](x),
                group_values))
    else:
        statistic_data = group_values

    return statistic_data


def add_stats(ax, df, x, y, 
              order=None, 
              hue=None, 
              hue_order=None, 
              box_pairs=None, 
              plot='boxplot',
              probs=None, 
              test_short_name=None, 
              loc='inside', 
              test=None,
              line_offset=0.2,
              line_height=0,
              text_offset=4,
              line_width=0.5,
              line_offset_to_group=0.2,
              fontsize=10,
              verbose=0,
              text_format='star',
              orient='v',
              *args,
              **kwargs):
    annot = Annotator(ax, box_pairs, data=df, x=x, y=y, order=order, hue=hue, hue_order=hue_order, orient=orient, plot=plot)
    annot.configure(test=test, test_short_name=test_short_name, loc=loc, verbose=verbose, line_offset=line_offset, line_height=line_height, line_width=line_width, text_offset=text_offset, fontsize=fontsize, text_format=text_format)
    if probs:
        annot.set_pvalues(pvalues=probs)
    else:
        annot.apply_test()
    annot.annotate(line_offset_to_group=line_offset_to_group)
    return ax


def _plotdata_handle(df, palette=None):
    if isinstance(palette, str):
        palette = [palette]

    if len(df.columns) == 2:
        x, y = df.columns
        label, hue, size = None, None, None

    elif len(df.columns) == 3:
        label, x, y = df.columns
        hue, size = None, None

    elif len(df.columns) == 4:
        label, x, hue, y = df.columns
        size = None

    elif len(df.columns) == 5:
        label, x, y, hue, size = df.columns

    else:
        raise ValueError(
            'The plot data should only contain x, y, group(hue) column')

    if palette and hue and df[hue].nunique() != len(palette):
        raise ValueError(
            "the palettes' number must be equal to hue groups.")

    return df, label, x, y, hue, size


def correlation(x, y):
    index = np.intersect1d(x.dropna().index, y.dropna().index)
    x = x.loc[index]
    y = y.loc[index]

    pearson_corr, pearson_pvalue = scipy.stats.pearsonr(x, y)
    spearman_corr, spearman_pvalue = scipy.stats.spearmanr(x, y)
    return pearson_corr, pearson_pvalue, spearman_corr, spearman_pvalue


def fdr(pvals, method='indep'):
    '''
    Calculate FDR by statsmodels.stats.multitest.fdrcorrection
    :param pvals: array-like 1d
    :param method:
        'i', 'indep', 'p', 'poscorr': Benjamini/Hochberg for independent or positively correlated tests
        'n', 'negcorr': fdr_by (Benjamini/Yekutieli for general or negatively correlated tests

    return:
     FDR: array-like 1d
    '''

    pvalues = pvals.copy()
    from statsmodels.stats import multitest
    indices = False
    if np.isnan(pvalues).any():
        indices = True
        pvals_indices = np.vstack((range(len(pvalues)), pvalues))
        pvals_indices = pvals_indices[:, ~(
            np.isnan(pvals_indices).any(axis=0))]
        ps = pvals_indices[1]
    else:
        ps = pvalues
    fdr = multitest.fdrcorrection(ps, method=method)

    if indices:
        np.put(pvalues, pvals_indices[0].astype(int), fdr[1])
        fdr = (fdr[0], pvalues)
    return fdr


def scatterplot(df,
                title=None,
                palette=None,
                ax=None,
                figsize=(3, 3),
                linewidth=0,
                hue_order=None,
                size=None,
                sizes=None,
                style=None,
                highlight_points=None,
                adjust_axes=True,
                ticklabels_hide=['x'],
                ticklabels_format=['y'],
                ticklabels_wrap=['y'],
                wrap_length=20,
                spines_hide=['top', 'right'],
                labels_hide=None,
                legend='brief',
                text_label=None,
                **kwargs):
    """
    Create a scatter plot of the given data.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    title : str, optional
        The title of the plot. Default is None.
    palette : str, optional
        The color palette to use for the plot. Default is None.
    ax : plt.Axes, optional
        The Axes object to draw the plot on. If not provided, a new one will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if no Axes object is provided. Default is (3, 3).
    linewidth : float, optional
        The width of the lines around the scatter points. Default is 0.
    hue_order : List[str], optional
        The order in which to plot the hue levels. Default is None.
    size : str, optional
        The column in the DataFrame to use for sizing the scatter points. Default is None.
    sizes : Tuple[float, float], optional
        The minimum and maximum size of the scatter points. Default is None.
    style : str, optional
        The column in the DataFrame to use for styling the scatter points. Default is None.
    highlight_points : List[str], optional
        A list of points to highlight on the plot. Default is None.
    adjust_axes : bool, optional
        Whether to adjust the formatting of the axes. Default is True.
    ticklabels_hide : List[str], optional
        The tick labels to hide. Default is ['x'].
    ticklabels_format : List[str], optional
        The format of the tick labels. Default is ['y'].
    ticklabels_wrap : List[str], optional
        The tick labels to wrap. Default is ['y'].
    wrap_length : int, optional
        The maximum length of wrapped tick labels. Default is 20.
    spines_hide : List[str], optional
        The spines to hide. Default is ['top', 'right'].
    labels_hide : List[str], optional
        The axis labels to hide. Default is None.
    legend : str, optional
        How to draw the legend. Default is 'brief'.
    text_label : str, optional
        The column in the DataFrame to use for labeling points on the plot. Default is None.
    **kwargs
        Additional keyword arguments passed to sns.scatterplot.

    Returns
    -------
    plt.Axes
        The Axes object on which the plot was drawn.
    """

    df = df.reset_index()
    df, label, x, y, hue, size = _plotdata_handle(df)

    if sizes and not size:
        size = hue
    elif not sizes and size:
        sizes = (5, 10)

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if not text_label:
        text_label = df.columns[0]

    ax = sns.scatterplot(data=df,
                         x=x,
                         y=y,
                         hue=hue,
                         palette=palette,
                         ax=ax,
                         hue_order=hue_order,
                         size=size,
                         sizes=sizes,
                         style=style,
                         linewidth=linewidth,
                         legend=legend,
                         **kwargs)

    if highlight_points:
        if df.shape[1] == 2:
            annot_df = df.reset_index().set_index(x)
            annot_df.columns = x, y
        else:
            annot_df = df.set_index(text_label)
        adjusttext(annot_df, ax, highlight_points, x, y)

    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   ticklabels_wrap=ticklabels_wrap,
                   wrap_length=wrap_length,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide)
    return ax


def adjusttext(df, ax, highlight_points, x, y):
    texts = [
        ax.text(df.loc[i, x],
                 df.loc[i, y],
                 i.split('_', 1)[-1].replace('_', ' '),
                 fontsize=8) for i in highlight_points if i in df.index
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='black'))


def axes_(ax,
          title,
          ticklabels_hide=None,
          ticklabels_format=None,
          ticklabels_wrap=['y'],
          wrap_length=50,
          spines_hide=None,
          labels_hide=None):
    if title:
        ax.set_title(title, fontsize=11)

    if spines_hide and isinstance(spines_hide, (list, set)):
        for side in spines_hide:
            ax.spines[side].set_visible(False)

    ax.tick_params(axis='both', labelsize=9)
    if ticklabels_hide:
        if 'x' in ticklabels_hide:
            ax.set_xticklabels('')
        if 'y' in ticklabels_hide:
            ax.set_yticklabels('')
    else:
        if ticklabels_format:
            tick_format = mticker.FuncFormatter(format_zero_func)
            sci_formatter = mticker.ScalarFormatter(useMathText=True)
            sci_formatter.set_scientific(True)
            sci_formatter.set_powerlimits((0, 3))

            if 'x' in ticklabels_format:
                ax.xaxis.set_major_formatter(tick_format)
                ax.xaxis.set_major_formatter(sci_formatter)
                
            if 'y' in ticklabels_format:
                ax.yaxis.set_major_formatter(tick_format)
                ax.yaxis.set_major_formatter(sci_formatter)

    if ticklabels_wrap:
        def wrap_func(x): return '\n'.join(wrap(x.get_text(), wrap_length))
        plt.draw()

        if 'x' in ticklabels_wrap:
            labels = ax.get_xticklabels()
            ticks = ax.get_xticks()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.set_xticklabels(list(map(wrap_func, labels)))
        if 'y' in ticklabels_wrap:
            labels = ax.get_yticklabels()
            ticks = ax.get_yticks()
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.set_yticklabels(list(map(wrap_func, labels)))

    if labels_hide:
        if 'x' in labels_hide:
            ax.set_xlabel('')
        if 'y' in labels_hide:
            ax.set_ylabel('')
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

    handle_legend(ax)

    return ax


def format_zero_func(x, pos, s='0f'):
    '''
    change ax ticklables 0.0 to 0
    method: ax.xaxis.set_major_formatter(plt.FuncFormatter(format_zero_func))
    '''

    if x == 0:
        return '0'
    else:
        return '%.{}'.format(s) % int(x)


def handle_legend(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
    return ax


def group_table(df, method='mean', prestatistic_method='log2', one_tile=False, ratio2second=2):
    group = df.groupby(df.index.name, axis=0)
    if method == 'mean':
        param_table = group.mean().T
    elif method == 'median':
        param_table = group.median().T

    prestatistic_methods = {'log2': np.log2, 'log10': np.log10}
    prestatistic_method = prestatistic_methods.get(prestatistic_method, None)
    if prestatistic_method:
        df = prestatistic_method(df)

    tmp = {
        key: value.values.T
        for key, value in df.groupby(df.index.name)
    }
    statistic_df = tmp.values()
    
    n = df.index.nunique()
    if n == 2:
        if param_table.min().min() >= 0:
            param_table['_vs_'.join(tmp.keys())] = param_table.iloc[:, 0] / param_table.iloc[:, 1]
        else:
            param_table['_vs_'.join(tmp.keys())] = param_table.iloc[:, 0] - param_table.iloc[:, 1]
        param_table['ttest'] = statistic_func(statistic_df)[1]
        param_table['ranksums'] = statistic_func(statistic_df, statistic_method='ranksums')[1]

        if one_tile:
            param_table['ttest'] = param_table['ttest'] / 2
            param_table['ranksums'] = param_table['ranksums'] / 2

        param_table['ttest_fdr'] = fdr(param_table['ttest'].values)[1]
        param_table['ranksums_fdr'] = fdr(param_table['ranksums'].values)[1]
    elif n > 2:
        compare_table = ratio_to_second(param_table, n=ratio2second).rename(columns=lambda x: x+'_than_2nd')
        param_table = pd.concat([param_table, compare_table], axis=1)
        param_table['anova'] = statistic_func(statistic_df, statistic_method='anova')[1]
        param_table['kruskal'] = statistic_func(statistic_df, statistic_method='kruskal')[1]
        param_table['anova_fdr'] = fdr(param_table['anova'].values)[1]
        param_table['kruskal_fdr'] = fdr(param_table['kruskal'].values)[1]


    return param_table


def ratio_to_second(x, n=2):
    y = x > (np.sort(x, axis=1)[:, -2].reshape(x.shape[0], 1) * n)
    return pd.DataFrame(y, index=x.index, columns=x.columns)


def lineplot(df,
            title=None,
            palette=None,
            ax=None,
            figsize=(3, 3),
            hue_order=None,
            style=None,
            adjust_axes=True,
            ticklabels_hide=['x'],
            ticklabels_format=['y'],
            ticklabels_wrap=['y'],
            wrap_length=20,
            spines_hide=['top', 'right'],
            labels_hide=None,
            legend='brief',
            text_label=None,
            **kwargs):
    df = df.reset_index()
    df, label, x, y, hue, size = _plotdata_handle(df)

    if size:
        sizes = (df[size].min(), df[size].max())
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)


    if not text_label:
        text_label = df.columns[0]
    ax = sns.lineplot(data=df,
                         x=x,
                         y=y,
                         hue=hue,
                         palette=palette,
                         ax=ax,
                         hue_order=hue_order,
                         size=size,
                         style=style,
                         legend=legend,
                         **kwargs)

    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   ticklabels_wrap=ticklabels_wrap,
                   wrap_length=wrap_length,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide)
    return ax


def barplot(df,
            palette=None,
            title=None,
            orient='v',
            ax=None,
            figsize=(3, 3),
            adjust_axes=True,
            ticklabels_hide=['x'],
            ticklabels_format=['y'],
            spines_hide=['top', 'right'],
            labels_hide=None,
            linewidth=0,
            dodge=False,
            **kwargs):
    """
    Create a bar plot of the given data.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    palette : str, optional
        The color palette to use for the plot. Default is None.
    title : str, optional
        The title of the plot. Default is None.
    orient : str, optional
        The orientation of the plot. Can be either 'v' for vertical or 'h' for horizontal. Default is 'v'.
    ax : plt.Axes, optional
        The Axes object to draw the plot on. If not provided, a new one will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if no Axes object is provided. Default is (3, 3).
    adjust_axes : bool, optional
        Whether to adjust the formatting of the axes. Default is True.
    ticklabels_hide : List[str], optional
        The tick labels to hide. Default is ['x'].
    ticklabels_format : List[str], optional
        The format of the tick labels. Default is ['y'].
    spines_hide : List[str], optional
        The spines to hide. Default is ['top', 'right'].
    labels_hide : List[str], optional
        The axis labels to hide. Default is None.
    linewidth : float, optional
        The width of the lines around the bars. Default is 0.
    dodge : bool, optional 
        When hue nesting is used, whether elements should be shifted along the categorical axis.
    **kwargs
        Additional keyword arguments passed to sns.barplot.

    Returns
    -------
    plt.Axes
        The Axes object on which the plot was drawn.
    """

    
    df = df.reset_index()
    df, label, x, y, hue, args = _plotdata_handle(df, palette)
    if orient == 'h':
        x, y = y, x
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax = sns.barplot(data=df,
                     x=x,
                     y=y,
                     hue=hue,
                     palette=palette,
                     ax=ax,
                     linewidth=linewidth,
                     dodge=dodge,
                     **kwargs)
    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide)
    return ax


def prob_star(pvalue):
    if pvalue > 0.05:
        return 'na'
    elif pvalue > 0.01:
        return '*'
    elif pvalue > 0.001:
        return '**'
    elif pvalue > 1e-4:
        return '***'
    else:
        return '****'


def regplot(df,
            method='spearman',
            scattersize=20,
            scattercolor='black',
            linecolor='red',
            ax=None,
            figsize=(3, 3),
            adjust_axes=True,
            ticklabels_format=['x', 'y'],
            ticklabels_hide=[],
            ticklabels_wrap=[],
            **kwargs):
    """
    Create a regression plot using the specified method.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy (“long-form”) dataframe where each column is a variable and each row is an observation without nan.
    method : str, optional
        The correlation method to use. Can be either 'spearman' or 'pearson'. Default is 'spearman'.
    scattersize : int, optional
        The size of the scatter points. Default is 20.
    scattercolor : str, optional
        The color of the scatter points. Default is 'black'.
    linecolor : str, optional
        The color of the regression line. Default is 'red'.
    ax : matplotlib Axes, optional
        The Axes object to draw the plot on. If not provided, a new one will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if no Axes object is provided. Default is (3, 3).
    adjust_axes : bool, optional
        Whether to adjust the formatting of the axes. Default is True.
    ticklabels_format : List[str], optional
        Whether to format the tick labels. Default is ['x', 'y'].
    ticklabels_hide : List[str], optional
        Whether to hide the tick labels. Default is [].
    ticklabels_wrap : List[str], optional
        Whether to wrap the tick labels to wrap. Default is [].
    **kwargs
        Additional keyword arguments passed to sns.regplot.

    Returns
    -------
    ax: marplotlib Axes
        The Axes object containing the plot.
    """

    df = df.reset_index()
    scatter_kws = {'s': scattersize, 'color': scattercolor}
    line_kws = {'color': linecolor}
    df, label, x, y, hue, size = _plotdata_handle(df)

    if size:
        scatter_kws['size'] = df['size']

    if method.lower() == 'spearman':
        corr, pvalue = scipy.stats.spearmanr(df[x], df[y])

    elif method.lower() == 'pearson':
        corr, pvalue = scipy.stats.pearsonr(df[x], df[y])

    else:
        raise KeyError(
            "method parameter should be one of 'spearman' and 'pearson', please check your input parameter."
        )

    methods_name = {'spearman': ' rho: ', 'pearson': ' corr: '}
    title = ' '.join([
        method.title(), methods_name[method],
        '%.2f' % corr, '\nP-value:',
        '%.2e' % pvalue
    ])

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax = sns.regplot(data=df,
                     x=x,
                     y=y,
                     line_kws=line_kws,
                     scatter_kws=scatter_kws,
                     ax=ax,
                     **kwargs)

    if adjust_axes:
        ax = axes_(ax, title=None, ticklabels_format=ticklabels_format,
                   ticklabels_hide=ticklabels_hide, ticklabels_wrap=ticklabels_wrap)
    ax.set_title(title, horizontalalignment='left', loc='left', fontsize=8)

    return ax


def multi_logrank(df, weightings=None):
    '''
    Cal culate log rank pvalue for multi-groups
    :param df: columns order: group(high & low), state, time;
    :param weightings_option: wilcoxon, tarone-ware, peto, default: None

    return: multivariate-logrank pvalue
    '''

    df.columns = ['group', 'state', 'time']
    return multivariate_logrank_test(df['time'],
                                     df['group'],
                                     df['state'],
                                     weightings=weightings).p_value


def filter_by_quantile(x, up=0.75, bottom=0.25):
    '''
    Filter data by its quantile(IQR);
    :param x: pd.Series 
    :param up: upper quantile
    :param bottom: lower quantile

    return: a filtered series;
    '''

    q1 = x.quantile(bottom)
    q2 = x.quantile(up)
    iqr = q2 - q1
    filter = (x >= q1 - 1.5 * iqr) & (x <= q2 + 1.5 * iqr)
    return x.loc[filter]