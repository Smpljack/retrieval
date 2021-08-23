import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from scipy.interpolate import interp1d

from typhon.plots import profile_p, heatmap, styles, label_axes
from retrieval_data_handler import RetrievalDataHandler
plt.style.use(styles.get('typhon'))


def plot_profiles(fig, ax, data, quantity, xlims=None, xlog=False, subset_colors=None,
                  ref_h2o_vmr=False, moist_anom=False, anom_shading=False):
    """
    Profile plots of H2O VMR and Temperature for a retrieval run, including
    a priori, true and retrieval result.
    :param p_grid:
    :return:
    """
    p = data['a_priori_p']
    if quantity == 'delta_t':
        profile_p(p, np.zeros(p.shape),
                  color='black', linestyle='--',
                  alpha=0.5, linewidth=0.9, ax=ax,
                  )
        profile_p(
            p,
            data[f'a_priori_t'] - data['true_t'],
            label="a priori", ax=ax,
            color=subset_colors['a_priori'],
            linestyle='--',
        )
        profile_p(
            p,
            data[f'retrieved_t'] - data['true_t'],
            label="retrieved", ax=ax,
            color=subset_colors['retrieved'],
            linestyle='--',
        )
    elif False:#quantity == 't':
        profile_p(
            p,
            data[f'true_t'],
            label="True T", ax=ax,
            color=subset_colors['true'],
            linestyle='-',
        )
        profile_p(
            p,
            data[f'a_priori_t'],
            label="Assumed T", ax=ax,
            color=subset_colors['a_priori'],
            linestyle='--',
        )

    else:
        if quantity == 'heating_rates':
            profile_p(p, np.zeros(p.shape),
                      color='black', linestyle='-',
                      alpha=0.5, linewidth=0.9, ax=ax)
        profile_p(
            p,
            data[f'true_{quantity}'],
            label="true", ax=ax, linestyle="-",
            color=subset_colors['true']
        )
        profile_p(
            p,
            data[f'a_priori_{quantity}'],
            label="a priori", ax=ax,
            color=subset_colors['a_priori'],
            linestyle='--'
        )
        profile_p(
            p,
            data[f'retrieved_{quantity}'],
            label="retrieved", linestyle='dotted', ax=ax,
            color=subset_colors['retrieved']
        )

    if ref_h2o_vmr and quantity == 'h2o_vmr':
        profile_p(
            p,
            data[f'true_eml_ref_h2o_vmr'],
            label=" reference \n true", ax=ax,
            linestyle="-.",
            color=subset_colors['true'], alpha=0.6,
        )
        profile_p(
            p,
            data[f'retrieved_eml_ref_h2o_vmr'],
            label=" reference \n retrieved", ax=ax, linestyle="-.",
            color=subset_colors['retrieved'], alpha=0.6,
        )
    if moist_anom and quantity == 'h2o_vmr':
        ax.hlines(
            data['true_moisture_characteristics'].values[()].pmean,
            xmin=0,
            xmax=0.3,
            linestyles='dashed',
            colors='black',
        )
    if xlims is not None:
        ax.set_xlim(xlims)
    if xlog and quantity == 'h2o_vmr':
        ax.set_xscale("log")

    return fig, ax


def plot_box_whisker(fig, ax, data, quantity, xlims=None, data_type='true'):
    if data_type == 'true':
        qdata = data[f'true_{quantity}'].values
    elif data_type == 'retrieved':
        qdata = data[f'retrieved_{quantity}'].values
    elif data_type == 'retrieved-true':
        if quantity == 'h2o_vmr':
            qdata = (data[f'retrieved_{quantity}'].values - data[f'true_{quantity}'].values) / \
                     data[f'true_{quantity}'].values
        else:
            qdata = data[f'retrieved_{quantity}'].values - data[f'true_{quantity}'].values
    elif data_type == 'retrieved-smoothed':
        qdata = data[f'retrieved_{quantity}'].values - data[f'smoothed_{quantity}'].values
    elif data_type == 'smoothed-true':
            qdata = data[f'smoothed_{quantity}'].values - data[f'true_{quantity}'].values
    means = np.nanmedian(qdata, axis=0)
    percentile_10 = np.nanpercentile(qdata, 10, axis=0)
    percentile_90 = np.nanpercentile(qdata, 90, axis=0)
    percentile_25 = np.nanpercentile(qdata, 25, axis=0)
    percentile_75 = np.nanpercentile(qdata, 75, axis=0)
    # ax.fill_betweenx(p_grid, maxs, mins, facecolor='grey', alpha=0.5, color='grey')
    ax.fill_betweenx(
        data['a_priori_p'].values[0, :],
        percentile_10,
        percentile_90,
        facecolor='orange', alpha=0.5, color='orange',
    )
    ax.fill_betweenx(
        data['a_priori_p'].values[0, :],
        percentile_25,
        percentile_75,
        facecolor='red', alpha=0.5, color='red',
    )
    if xlims is not None:
        ax.set_xlim(xlims)
        if xlims[0] == -xlims[1]:
            ax.plot(
                np.zeros(len(data['a_priori_p'].values[0, :])),
                data['a_priori_p'].values[0, :],
                linestyle='--', color='black', linewidth=1.0, alpha=0.7,
            )
    profile_p(data['a_priori_p'].values[0, :], means, color='black', ax=ax)
    # ax.set_xlabel(f'{get_quantity_label(quantity)}')
    # if set_xlog(quantity):
    #     ax.set_xscale('log')
    return fig, ax


def plot_hist(fig, ax, data, quantity, xlims, subsets, xlabel=None, ylabel=None,
              data_label=None, xlog=False, norm=False, nbins=None, mean_line=False, subset_colors=None):
    for subset in subsets:
        bins = np.linspace(xlims[0], xlims[1], num=nbins)
        if xlog and quantity == 'strength':
            # bins = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), num=nbins)
            data[f'{subset}_{quantity}'] = np.log10(data[f'{subset}_{quantity}'])
        n, _, _ = ax.hist(
            data[f'{subset}_{quantity}'],
            histtype='step', density=norm, bins=bins, label=f'{subset}', linewidth=2.0,
            color=subset_colors[subset],
        )
        if mean_line:
            ax.vlines(np.nanmean(data[f'{subset}_{quantity}']), 0, np.max(n),
                      linestyles='dashed', alpha=0.5, linewidth=2.0,
                      color=subset_colors[subset])
    # if xlog and quantity == 'strength':
    #     ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_avg_kernel(fig, ax, data, subset, quantity, xlims=None):
    colormap = plt.cm.plasma
    ax.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, len(data['a_priori_p']))])
    for irow in range(len(data['a_priori_p'])):
        profile_p(
            data['a_priori_p'],
            data[f'{subset}_averaging_kernel_{quantity}'][irow, :],
            ax=ax
        )
    profile_p(
        data['a_priori_p'],
        np.sum(data[f'{subset}_averaging_kernel_{quantity}'], axis=1),
        ax=ax, color="black", linewidth=1.5
    )
    profile_p(
        data['a_priori_p'],
        np.ones(len(data['a_priori_p'])),
        linestyle="--", color="gray", ax=ax
    )
    if xlims is not None:
        ax.set_xlim(xlims)
    # ax.set_xlabel(f'{get_quantity_label(quantity)}')
    return fig, ax


def plot_resolution(fig, ax, data, quantity, xlims=None):
    profile_p(data['a_priori_p'], data[f'true_resolution_{quantity}']/1000., ax=ax, color="black")
    if xlims is not None:
        ax.set_xlim(xlims)
    # ax.set_xlabel(f'{get_quantity_label(quantity)}')
    return fig, ax


def plot_2d_hist(fig, ax, xdata, ydata, xlims=None, ylims=None,
                 xlog=False, ylog=False, xlabel=None, ylabel=None, nbins=10, bisectrix=False):
    if xlims is None:
        xlims = [np.nanmin(xdata), np.nanmax(xdata)]
    if ylims is None:
        ylims = [np.nanmin(ydata), np.nanmax(ydata)]
    bins = [np.linspace(xlims[0], xlims[1], num=nbins),
            np.linspace(ylims[0], ylims[1], num=nbins)]
    if xlog:
        bins[0] = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), num=nbins)
    if ylog:
        bins[1] = np.logspace(np.log10(ylims[0]), np.log10(ylims[1]), num=nbins)
    xdata = xdata[~np.isnan(xdata)]
    ydata = ydata[~np.isnan(ydata)]
    print(xlabel)
    print(ylabel)
    print(xdata.max())
    print(ydata.max())
    h = heatmap(xdata, ydata,
                bins=bins, ax=ax,
                color="black",
                linewidth=0.2,
                range=[xlims, ylims],
                cmap='temperature',
                bisectrix=bisectrix,
                vmin=0,
                vmax=70,
                )
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax, h


def panel_plot_per_quantity(quantities, data, plot_funcs, xlims=False, xlabels=None,
                            sharey=True, fig=None, axs=None, irow=None, icol=None, **kwargs):
    """
    :param quantities:
    :param quantities_data:
    :param plot_func:
    :param kwargs:
    :return:
    """
    if fig is None and axs is None:
        fig, axs = plt.subplots(ncols=len(quantities), sharey=sharey)
        axs_snip = axs
    elif irow is not None:
        axs_snip = np.copy(axs[irow, :])
    elif icol is not None:
        axs_snip = np.copy(axs[:, icol])
    else:
        axs_snip = np.copy(axs)
    if type(axs_snip) is not np.ndarray:
        axs_snip = [axs_snip]
    for quantity, data, ax, plot_func in zip(quantities, data, axs_snip, plot_funcs):
        fig, ax = plot_func(fig, ax, data, quantity, xlims[quantity], **kwargs)
        # if xlims:
        #     ax.set_xlim(xlims[quantity])
        if xlabels is not None:
            ax.set_xlabel(xlabels[quantity])
    # axs[2].legend(bbox_to_anchor=(4.9, 0.96))

    return fig, axs


def format_yaxis_p_grid(axs, p_grid, xaxis=False):
    """
    Format the vertical y-axis on a p-grid to the standard format I want to use.
    :param axs:
    :param p_grid:
    :return:
    """
    [axs[i].set_yscale('log') for i in range(len(axs))]
    [axs[i].set_ylim([p_grid.max()+1000, 8000.]) for i in range(len(axs))]
    [axs[i].set_yticks(
        [10 ** 5, 0.8 * 10 ** 5, 0.6 * 10 ** 5, 0.4 * 10 ** 5, 0.2 * 10 ** 5, 0.1 * 10 ** 5])
        for i in range(len(axs))]
    [axs[i].set_yticklabels(['1000', '800', '600', '400', '200', '100']) for i in range(len(axs))]
    if xaxis:
        [axs[i].set_xscale('log') for i in range(len(axs))]
        [axs[i].set_xlim([p_grid.max() + 1000, 8000.]) for i in range(len(axs))]
        [axs[i].set_xticks(
            [10 ** 5, 0.8 * 10 ** 5, 0.6 * 10 ** 5, 0.4 * 10 ** 5, 0.2 * 10 ** 5, 0.1 * 10 ** 5])
            for i in range(len(axs))]
        [axs[i].set_xticklabels(['1000', '800', '600', '400', '200', '100']) for i in range(len(axs))]
    #[axs[i].set_yticklabels([]) for i in range(1, len(axs))]
    return axs


def anomaly_shading(ax, p, x1, x2, pos_anomaly, interpolate=False):
    ax.fill_betweenx(
        p,
        x1,
        x2,
        where=pos_anomaly,
        interpolate=interpolate,
        facecolor='royalblue',
        alpha=0.8,
    )
    ax.fill_betweenx(
        p,
        x1,
        x2,
        where=~pos_anomaly,
        interpolate=interpolate,
        facecolor='darkorange',
        alpha=0.8,
    )
    return ax


def _main():
    import argparse
    plot_types = ['profiles', 'box_whisker', 'moist_char_heatmap', 'moist_char_hist', 'eml_diff_hist',
                  'averaging_kernel', '2d_hist', 'error_covariance', 'moist_char_diff_hist',
                  'retrieved_true_heatmap']
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", help="Path of project directory")
    parser.add_argument("plot_type", choices=plot_types, help=f"Supported plot_types: {plot_types}")
    parser.add_argument("--batch_start", type=int, default=0, help="batch field start index")
    parser.add_argument("--batch_end", type=int, default=1, help="batch field end index")
    parser.add_argument("-ss", "--subsets", nargs='+', choices=['a_priori', 'true', 'retrieved'],
                        help="Specify the type of box-whisker plot")
    parser.add_argument("--quantities", choices=['h2o_vmr', 't', 'rh', 'delta_t', 'log_h2o_vmr', 'jacobian_h2o_vmr',
                                                 'jacobian_t', 'averaging_kernel_log_h2o_vmr', 'averaging_kernel_t',
                                                 'eml_strength', 'eml_width', 'true_smoothed', 'heating_rates'],
                        nargs='+', help="Specify the quantities for which you want a panel plot")
    parser.add_argument("-f", "--filter", choices=['converged', 'all', 'eml_true', 'eml_retrieved'],
                        default=['converged'],
                        nargs='+', help='Filter data for specific criteria')
    parser.add_argument("-cm", "--categorical_measure", choices=['sensitivity', 'specificity', 'accuracy', 'mcc'],
                        help="Specify the categorical eml measure to plot")
    parser.add_argument('-xlog', '--xlog', action='store_true')
    parser.add_argument('-ylog', '--ylog', action='store_true')
    parser.add_argument('-norm', '--normalize', action='store_true', help='normalize histogram')
    parser.add_argument('-ref_h2o', '--ref_h2o_vmr', action='store_true',
                        help='show reference h2o vmr in h2o vmr profile plot.')
    parser.add_argument('-moist_anom', '--moist_anom', action='store_true',
                        help='show moisture anomaly characteristics in h2o profile')
    parser.add_argument('-bins', '--bins', type=int, default=10, help='amount of bins for histogram.')
    parser.add_argument('-mcx', '--moist_char_x', nargs='+', type=str, choices=[
        'pmin', 'pmax', 'zmin', 'zmax', 'strength', 'pwidth', 'zwidth', 'pmean', 'zmean', 'heating_rate_mean',
        'heating_rate_min', 'heating_rate_anomaly_mean', 'heating_rate_anomaly_min', 'heating_rate_anomaly_max',
        'heating_rate_max',
    ])
    parser.add_argument('-mcy', '--moist_char_y', type=str, nargs='+', choices=[
        'pmin', 'pmax', 'zmin', 'zmax', 'strength', 'pwidth', 'zwidth', 'pmean', 'zmean', 'heating_rate_mean',
        'heating_rate_min', 'heating_rate_anomaly_mean', 'heating_rate_anomaly_min', 'heating_rate_anomaly_max',
        'heating_rate_max',
    ])
    parser.add_argument('-re', '--relative_error', action='store_true',
                        help='Plot relative error instead of absolute.')
    parser.add_argument('-rhr', '--ref_heating_rate', action='store_true', help='load reference heating rate.')
    parser.add_argument('--anomaly_shading', action='store_true',
                        help='Plot shading for moisture anomalies in profile plot',)
    parser.add_argument('-bw', '--box_whisker', choices=['true', 'retrieved', 'retrieved-true', 'retrieved-smoothed',
                                                         'smoothed-true'],
                        default='true', help='Type of data box whisker plot shows.')
    args = parser.parse_args()

    if args.plot_type not in plot_types:
        print(f'Plot type incorrect. Options are {plot_types}. \nUse the -h option to get help.')
        return

    if args.plot_type == 'profiles':
        load_quantities = list(np.copy(args.quantities))
        if 'heating_rates' in args.quantities:
            load_quantities.remove('heating_rates')
        if 'delta_t' in args.quantities:
            load_quantities.remove('delta_t')
        if args.moist_anom and 't' not in load_quantities:
            load_quantities.append('t')
        retrieval_data = RetrievalDataHandler(
            project_path=args.project_path,
            config_path='datahandler_setup.yaml',
            quantities=load_quantities,
            subsets=['a_priori', 'true', 'retrieved'],
            filters=['converged'],
        )

        if args.ref_h2o_vmr:
            retrieval_data.calc_reference_h2o_vmr(subset='true')
            retrieval_data.calc_reference_h2o_vmr(subset='retrieved')
        if 'heating_rates' in args.quantities:
            retrieval_data.load_heating_rates(retrieval_data.subsets, ref=args.ref_heating_rate)
        if args.moist_anom:
           retrieval_data.calc_eml_characteristics()
        data = [retrieval_data.data] * len(args.quantities)
        plot_setup = yaml.safe_load(open('plot_setup.yaml'))
        for ibatch in range(args.batch_start, args.batch_end):
            plot_funcs = [plot_profiles] * len(args.quantities)
            fig, axs = panel_plot_per_quantity(
                args.quantities,
                [entry.isel(nprofile=ibatch) for entry in data],
                plot_funcs,
                xlims=plot_setup['xlims'],
                xlabels=plot_setup['xlabels'],
                xlog=args.xlog,
                subset_colors=plot_setup['subset_colors'],
                ref_h2o_vmr=args.ref_h2o_vmr,
                moist_anom=args.moist_anom,
            )
            if args.anomaly_shading:
                for ax in axs:
                    if ax.get_xlabel() == plot_setup['xlabels']['h2o_vmr']:
                        top_range = retrieval_data.data['a_priori_p'].values[ibatch, :] < 20000.
                        top_range &= retrieval_data.data['a_priori_p'].values[ibatch, :] > 10000.
                        top_p_ind = np.absolute(retrieval_data.data['true_h2o_vmr'].values[ibatch, :] -
                                                retrieval_data.data['true_eml_ref_h2o_vmr'].values[ibatch, :]
                                                )[top_range].argmin()
                        p_ind = retrieval_data.data['a_priori_p'][ibatch, :].values < 90000.
                        p_ind &= retrieval_data.data['a_priori_p'][ibatch, :].values > \
                                 retrieval_data.data['a_priori_p'].values[ibatch, :][top_range][top_p_ind]
                        pos_anomaly = (retrieval_data.data['true_h2o_vmr'].values[ibatch, :] -
                                       retrieval_data.data['true_eml_ref_h2o_vmr'].values[ibatch, :]) > 0
                        anomaly_shading(
                            ax,
                            retrieval_data.data['a_priori_p'].values[ibatch][p_ind],
                            retrieval_data.data['true_h2o_vmr'].values[ibatch][p_ind],
                            retrieval_data.data['true_eml_ref_h2o_vmr'].values[ibatch][p_ind],
                            pos_anomaly[p_ind],
                            interpolate=True,
                        )
                        ax.fill_between(
                            plot_setup['xlims']['h2o_vmr'],
                            [101500., 101500.],
                            [90000., 90000.],
                            facecolor='gray',
                            alpha=0.6,
                        )
                        ax.fill_between(
                            plot_setup['xlims']['h2o_vmr'],
                            [10000., 10000.],
                            [1000., 1000.],
                            facecolor='gray',
                            alpha=0.6,
                        )
                    elif ax.get_xlabel() == plot_setup['xlabels']['heating_rates']:
                        new_p = np.arange(retrieval_data.data['a_priori_p'].values[ibatch].max(),
                                          retrieval_data.data['a_priori_p'].values[ibatch].min(),
                                          -10)
                        new_true_h2o_vmr = np.interp(
                                              new_p[::-1],
                                              retrieval_data.data['a_priori_p'].values[ibatch][::-1],
                                              retrieval_data.data['true_h2o_vmr'].values[ibatch][::-1],
                                          )[::-1]
                        new_ref_h2o_vmr = np.interp(
                                              new_p[::-1],
                                              retrieval_data.data['a_priori_p'].values[ibatch][::-1],
                                              retrieval_data.data['true_eml_ref_h2o_vmr'].values[ibatch][::-1],
                                          )[::-1]
                        top_range = new_p < 20000.
                        top_range &= new_p > 10000.
                        top_p_ind = np.absolute(new_true_h2o_vmr[top_range] - new_ref_h2o_vmr[top_range]).argmin()
                        p_ind = new_p < 90000.
                        p_ind &= new_p > new_p[top_range][top_p_ind]
                        pos_anomaly = (new_true_h2o_vmr - new_ref_h2o_vmr) > 0
                        anomaly_shading(
                            ax,
                            new_p[p_ind],
                            np.interp(new_p[::-1],
                                      retrieval_data.data['a_priori_p'].values[ibatch][::-1],
                                      retrieval_data.data['true_heating_rates'].values[ibatch][::-1],
                                      )[::-1][p_ind],
                            0,
                            pos_anomaly[p_ind]
                        )
                        ax.fill_between(
                            plot_setup['xlims']['heating_rates'],
                            [101500., 101500.],
                            [90000., 90000.],
                            facecolor='gray',
                            alpha=0.6,
                        )
                        ax.fill_between(
                            plot_setup['xlims']['heating_rates'],
                            [10000., 10000.],
                            [1000., 1000.],
                            facecolor='gray',
                            alpha=0.6,
                        )
            axs = format_yaxis_p_grid(axs, retrieval_data.data['a_priori_p'][ibatch])
            # handles, labels = axs[-1].get_legend_handles_labels()
            # lgd = axs[-1].legend(handles, labels,
            #                      #bbox_to_anchor=(4.9, 0.96))
            #                      )
            label_axes(axs, labels=['(a)', '(b)'], loc=(-0.15, 0.97))
            plt.savefig(
                os.path.join(
                    args.project_path,
                    f"plots/{args.plot_type}_ibatch{ibatch}_{'_'.join(args.quantities)}.pdf"
                ),
                # bbox_extra_artists=(lgd,),
                bbox_inches='tight',
            )
            plt.show()
    elif args.plot_type == 'box_whisker':
        load_quantities = list(np.copy(args.quantities))
        if args.box_whisker == 'retrieved-smoothed' or args.box_whisker == 'smoothed-true':
            load_quantities.append('averaging_kernel_log_h2o_vmr')
            load_quantities.append('averaging_kernel_t')
        retrieval_data = RetrievalDataHandler(
            project_path=args.project_path,
            config_path='datahandler_setup.yaml',
            quantities=load_quantities,
            subsets=['true', 'retrieved', 'a_priori'],
        )
        if args.box_whisker == 'retrieved-smoothed' or args.box_whisker == 'smoothed-true':
            retrieval_data.calc_smoothed()
        plot_setup = yaml.safe_load(open('plot_setup.yaml'))
        plot_funcs = [plot_box_whisker] * len(args.quantities)
        data = [retrieval_data.data] * len(args.quantities)
        fig, axs = panel_plot_per_quantity(args.quantities, data, plot_funcs,
                                           xlims=plot_setup['xlims'],
                                           xlabels=plot_setup['xlabels'],
                                           data_type=args.box_whisker,
                                           )
        axs = format_yaxis_p_grid(axs, retrieval_data.data['a_priori_p'][0])
        label_axes(axs, labels=['a', 'b', 'c'], loc=(-0.1, 0.97))
        plt.savefig(
            os.path.join(
                args.project_path,
                f"plots/{args.plot_type}_{args.box_whisker}_{'_'.join(args.quantities)}.pdf"
            )
        )
        plt.show()
        print('please')

    elif args.plot_type == 'moist_char_hist':
        retrieval_data = RetrievalDataHandler(
            project_path=args.project_path,
            config_path='datahandler_setup.yaml',
            quantities=['h2o_vmr', 't'],
            filters=['converged'],
            subsets=['true', 'retrieved'],
        )
        retrieval_data.calc_eml_characteristics(z_in_km=True, p_in_hPa=True)
        data = [
            {
                f'{subset}_{quantity}': np.concatenate(
                    [
                        getattr(moist_char, quantity) for moist_char in
                        retrieval_data.data[f'{subset}_moisture_characteristics'].values
                    ]
                )
                for subset in ['true', 'retrieved']
            }
            for quantity in args.moist_char_x
        ]
        plot_setup = yaml.safe_load(open('plot_setup.yaml'))
        fig, axs = panel_plot_per_quantity(quantities=args.moist_char_x,
                                           data=data,
                                           plot_funcs=[plot_hist] * len(data),
                                           sharey=False,
                                           xlims=plot_setup['xlims'],
                                           xlabels=plot_setup['xlabels'],
                                           subsets=['true', 'retrieved'],
                                           norm=args.normalize,
                                           nbins=args.bins,
                                           mean_line=True,
                                           xlog=args.xlog,
                                           subset_colors=plot_setup['subset_colors'])
        plt.savefig(
            os.path.join(
                args.project_path,
                f"plots/{args.plot_type}_norm-{args.normalize}_{'_'.join(args.moist_char_x)}.pdf"
            )
        )
        plt.show()

    elif args.plot_type == 'moist_char_diff_hist':
        retrieval_data = RetrievalDataHandler(
            project_path=args.project_path,
            config_path='datahandler_setup.yaml',
            quantities=['h2o_vmr', 't'],
            filters=['converged'],
            subsets=['true', 'retrieved'],
        )
        retrieval_data.calc_eml_characteristics(z_in_km=True, p_in_hPa=True)
        retrieval_data.reduce_to_matching_moist_anomalies()
        data = [
            {
                f'retrieved-true_{quantity}': np.concatenate(
                    [getattr(moist_char, quantity) for moist_char in
                        retrieval_data.data[f'retrieved_moisture_characteristics'].values]
                ) -
                np.concatenate(
                    [getattr(moist_char, quantity) for moist_char in
                     retrieval_data.data[f'true_moisture_characteristics'].values]
                )
                if not args.relative_error else
                (
                    np.concatenate(
                        [getattr(moist_char, quantity) for moist_char in
                         retrieval_data.data[f'retrieved_moisture_characteristics'].values]
                    ) -
                    np.concatenate(
                        [getattr(moist_char, quantity) for moist_char in
                         retrieval_data.data[f'true_moisture_characteristics'].values]
                    )
                ) /
                np.concatenate(
                    [getattr(moist_char, quantity) for moist_char in
                     retrieval_data.data[f'true_moisture_characteristics'].values]
                )
            }
            for quantity in args.moist_char_x
        ]
        plot_setup = yaml.safe_load(open('plot_setup.yaml'))
        fig, axs = panel_plot_per_quantity(quantities=args.moist_char_x,
                                           data=data,
                                           plot_funcs=[plot_hist] * len(data),
                                           sharey=False,
                                           xlims=plot_setup['xlims'],
                                           xlabels=plot_setup['xlabels'],
                                           subsets=['retrieved-true'],
                                           norm=args.normalize,
                                           nbins=args.bins,
                                           mean_line=True,
                                           xlog=args.xlog,
                                           subset_colors=plot_setup['subset_colors'])
        #
        plt.savefig(
            os.path.join(
                args.project_path,
                f"plots/{args.plot_type}_norm-{args.normalize}_{'_'.join(args.moist_char_x)}_"
                f"relative_{args.relative_error}.pdf"
            )
        )
        plt.show()
    elif args.plot_type == 'moist_char_heatmap':
        retrieval_data = RetrievalDataHandler(
            project_path=args.project_path,
            config_path='datahandler_setup.yaml',
            quantities=['h2o_vmr', 't'],
            filters=['converged'],
            subsets=args.subsets,
        )
        retrieval_data.load_heating_rates(subsets=['retrieved', 'true'])
        retrieval_data.calc_eml_characteristics(z_in_km=True, p_in_hPa=False)
        plot_setup = yaml.safe_load(open('plot_setup.yaml'))
        for subset in args.subsets:
            fig, axs = plt.subplots()
            fig, axs, h = plot_2d_hist(
                fig,
                axs,
                xdata=np.concatenate(
                    [
                        getattr(moist_char, args.moist_char_x[0]) for moist_char in
                        retrieval_data.data[f'{subset}_moisture_characteristics'].values
                    ]
                ),
                ydata=np.concatenate(
                    [
                        getattr(moist_char, args.moist_char_y[0]) for moist_char in
                        retrieval_data.data[f'{subset}_moisture_characteristics'].values
                    ]
                ),
                xlims=plot_setup['xlims'][args.moist_char_x[0]],
                ylims=plot_setup['xlims'][args.moist_char_y[0]],
                xlabel=f'{plot_setup["xlabels"][args.moist_char_x[0]]}',
                ylabel=f'{plot_setup["xlabels"][args.moist_char_y[0]]}',
                nbins=args.bins,
                xlog=args.xlog,
                ylog=args.ylog,
            )
            # axs = format_yaxis_p_grid([axs], retrieval_data.data['a_priori_p'][0])[0]
            label_axes([axs], labels=['a'], loc=(-0.1, 0.97))
            plt.savefig(
                os.path.join(
                    args.project_path,
                    f"plots/{args.plot_type}_{subset}_{args.moist_char_x[0]}_vs_{args.moist_char_y[0]}.pdf"
                )
            )
            plt.show()
    elif args.plot_type == 'retrieved_true_heatmap':
        retrieval_data = RetrievalDataHandler(
            project_path=args.project_path,
            config_path='datahandler_setup.yaml',
            quantities=['h2o_vmr', 't'],
            filters=['converged'],
            subsets=['retrieved', 'true'],
        )
        retrieval_data.load_heating_rates(subsets=['retrieved', 'true'])
        retrieval_data.calc_eml_characteristics(z_in_km=True, p_in_hPa=True)
        retrieval_data.reduce_to_matching_moist_anomalies()
        plot_setup = yaml.safe_load(open('plot_setup.yaml'))
        fig, axs = plt.subplots()
        fig, axs = plot_2d_hist(
            fig,
            axs,
            xdata=np.concatenate(
                [
                    getattr(moist_char, args.moist_char_x[0]) for moist_char in
                    retrieval_data.data[f'true_moisture_characteristics'].values
                ]
            ),
            ydata=np.concatenate(
                [
                    getattr(moist_char, args.moist_char_y[0]) for moist_char in
                    retrieval_data.data[f'retrieved_moisture_characteristics'].values
                ]
            ),
            xlims=plot_setup['xlims'][args.moist_char_x[0]],
            ylims=plot_setup['xlims'][args.moist_char_y[0]],
            xlabel=f'true {plot_setup["xlabels"][args.moist_char_x[0]]}',
            ylabel=f'retrieved {plot_setup["xlabels"][args.moist_char_y[0]]}',
            nbins=args.bins,
            xlog=args.xlog,
            ylog=args.ylog,
            bisectrix=True,
        )
        plt.savefig(
            os.path.join(
                args.project_path,
                f"plots/{args.plot_type}_true_retrieved_{args.moist_char_x[0]}_vs_{args.moist_char_y[0]}.pdf"
            )
        )
        plt.show()

    return fig, axs


if __name__ == "__main__":
    _main()
