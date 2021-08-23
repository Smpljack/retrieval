import numpy as np
import xarray as xr
import os
from scipy.linalg import inv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import peak_widths
import yaml

from pyarts import xml
from typhon.physics import vmr2relative_humidity, e_eq_mixed_mk, vmr2specific_humidity, density
from typhon.math import integrate_column
# import konrad


class RetrievalDataHandler:
    """
    Class to handle output of an IASI-OEM retrieval.
    """
    def __init__(self, project_path=None, config_path=None,
                 quantities=None, subsets=None, filters=[None]):
        self.project_path = project_path
        self.quantities = quantities
        self.subsets = subsets
        self.filters = filters
        self.config = yaml.safe_load(open(config_path))
        self.add_dependent_vars()
        self.quantities.sort(key=self.quantity_keys)
        self.create_dataset()
        self.filter_data(filters)

    def filter_data(self, filters):
        filter_dict = {
            'converged': 0,
            'max_iterations': 1,
            'max_gamma': 2,
            'error': 9,
            'high_start_cost': 99,
        }
        # Initialize filter_bool: True=not_filtered, False=filtered
        filter_bool = np.array(self.data.dims['nprofile']*[True])
        if 'converged' in filters:
            filter_bool &= self.data['retrieved_oem_diagnostics'][:, 0].astype(int) == \
                           filter_dict['converged']
        if 'retrieved_h2o_vmr' in self.data.variables.keys():
            # Require retrieved h2o_vmr to be less than 5 % to avoid unphysical runaway cases.
            filter_bool &= np.all(self.data['retrieved_h2o_vmr'].values < 0.05, axis=1)
        # if 'retrieved_rh' in self.data.variables.keys():
        #     is_tropo = self.data['a_priori_p'][0] > 10000.
        #     filter_bool &= np.all(self.data['retrieved_rh'].values[:, is_tropo] <= 120, axis=1)
        self.data = self.data.isel(nprofile=filter_bool)

    def create_dataset(self):
        # Saved quantities that are specific to a subset
        self.data = xr.Dataset(
            data_vars={
                f'{subset}_{quantity}': (
                    self.config['quantity_dims'][quantity],
                    self.load_data(quantity, subset)
                )
                for quantity in self.quantities
                for subset in self.subsets#['true', 'retrieved', 'a_priori']
                if f'{subset}_{quantity}' in self.config['saved_vars']
            },
            attrs={'project_path': self.project_path}
        )
        # Calculate not saved quantities
        for quantity in self.quantities:
            for subset in self.subsets:
                if quantity == 'averaging_kernel_log_h2o_vmr' and subset == 'retrieved' \
                        or quantity == 'averaging_kernel_t' and subset == 'retrieved':
                    continue
                if f'{subset}_{quantity}' in self.config['calc_vars']:
                    print(f'calculating {subset} {quantity}')
                    self.data = self.data.assign(
                        variables={
                            f'{subset}_{quantity}': (
                                self.config['quantity_dims'][quantity],
                                self.load_data(quantity, subset)
                            )
                        }
                    )

    def add_dependent_vars(self):
        for quantity, dependencies in self.config['quantity_dependencies'].items():
            if quantity in self.quantities:
                [self.append_quantity(dep_q) for dep_q in dependencies]
        self.append_quantity('p') # Always load pressure grid
        self.append_quantity('z') # Always load altitude grid
        self.append_quantity('oem_diagnostics') # Always load oem diagnostics

    def append_quantity(self, quantity):
        if quantity not in self.quantities:
            self.quantities.append(quantity)

    @staticmethod
    def quantity_keys(quantity):
        sorted_quantities = [
            'p', 'z', 'h2o_vmr', 'log_h2o_vmr', 't', 'ts', 'oem_diagnostics', 'covariance_h2o_vmr', 'covariance_t',
            'covariance_y', 'rh', 'ybatch', 'jacobian_h2o_vmr', 'jacobian_t', 'jacobian_ts', 'averaging_kernel_log_h2o_vmr',
            'averaging_kernel_t', 'resolution_log_h2o_vmr', 'resolution_t', 'heating_rates',
        ]
        quantity_keys = {quantity: key for key, quantity in enumerate(sorted_quantities)}
        return quantity_keys[quantity]

    @staticmethod
    def subset_trans(subset):
        subset_trans = {
            'retrieved': 'retrieval_output',
            'true': 'observations',
            'a_priori': 'a_priori',
        }
        return subset_trans[subset]

    def load_data(self, quantity, subset):
        if f'{subset}_{quantity}' in self.config['saved_vars']:
            data = xml.load(os.path.join(
                self.project_path, f"{self.subset_trans(f'{subset}')}/{quantity}.xml"))
        else:
            data = self.calc_quantity(quantity, subset)
        return np.array(data)

    def calc_quantity(self, quantity, subset):
        calc_funcs = {
            'rh': self.calc_rh,
            'log_h2o_vmr': self.calc_log_h2o_vmr,
            # 'eml_characteristics': self.calc_eml_characteristics,
            'jacobian_h2o_vmr': self.calc_jacobian_h2o_vmr,
            'jacobian_t': self.calc_jacobian_t,
            'jacobian_ts': self.calc_jacobian_ts,
            'averaging_kernel_log_h2o_vmr': self.calc_averaging_kernel_log_h2o_vmr,
            'averaging_kernel_t': self.calc_averaging_kernel_t,
            # 'heating_rates': self.calc_heating_rates,
            'resolution_t': self.calc_vertical_resolution_t,
            'resolution_log_h2o_vmr': self.calc_vertical_resolution_log_h2o_vmr,
        }
        return calc_funcs[quantity](subset)

    def plev2phlev(self, plev):
        """Convert full-level to half-level pressure"""
        f = interp1d(
            x=np.arange(plev.size),
            y=np.log(plev),
            fill_value="extrapolate",
        )
        return np.exp(f(np.arange(-0.5, plev.size)))

    # def heating_rate(self, p, t, h2o_vmr, kind='net'):
    #     ph = self.plev2phlev(p)
    #     atm = konrad.atmosphere.Atmosphere(ph)
    #     atm['plev'][:] = p
    #     atm['T'][:] = t
    #     atm['H2O'][:] = h2o_vmr
    #     rrtmg = konrad.radiation.RRTMG()
    #     rrtmg.update_heatingrates(
    #         atm,
    #         surface=konrad.surface.SlabOcean.from_atmosphere(atm),
    #         cloud=konrad.cloud.ClearSky.from_atmosphere(atm)
    #     )
    #     return rrtmg[f'{kind}_htngrt'][0]

    # def calc_heating_rates(self, subset):
    #     print('calc heating rates...')
    #     heating_rates = np.array(
    #         [
    #             self.heating_rate(p, t, h2o_vmr) for p, t, h2o_vmr in
    #             zip(self.data[f'a_priori_p'].values,
    #                 self.data[f'{subset}_t'].values,
    #                 self.data[f'{subset}_h2o_vmr'].values)
    #         ]
    #     )
    #     return heating_rates
    def calc_jacobian_ts(self, subset):
            if subset is not 'a_priori':
                jacobian_ts = [
                    K[:, 0] for K in
                    xml.load(
                        os.path.join(
                            self.project_path,
                            f"{self.subset_trans(f'{subset}')}/jacobian.xml"
                        )
                    )
                ]
            return jacobian_ts

    def calc_jacobian_t(self, subset):
        if subset is not 'a_priori':
            jacobian_t_vmr = [
                K[:, 1:139] for K in
                    xml.load(
                        os.path.join(
                            self.project_path,
                            f"{self.subset_trans(f'{subset}')}/jacobian.xml"
                        )
                    )
            ]
        return jacobian_t_vmr

    def calc_jacobian_h2o_vmr(self, subset):
        if subset is not 'a_priori':
            jacobian_h2o_vmr = [
                K[:, 139:] for K in
                 xml.load(
                     os.path.join(
                         self.project_path,
                         f"{self.subset_trans(f'{subset}')}/jacobian.xml"
                     )
                 )
            ]
        return jacobian_h2o_vmr

    def calc_rh(self, subset):
        rh = [
            vmr2relative_humidity(vmr, p, t, e_eq=e_eq_mixed_mk) * 100
            for vmr, p, t in
            zip(
                self.data[f'{subset}_h2o_vmr'].values,
                self.data[f'a_priori_p'].values,
                self.data[f'{subset}_t'].values,
            )
        ]
        return rh

    def calc_log_h2o_vmr(self, subset):
        rh = [np.log(h2o_vmr) for h2o_vmr in self.data[f'{subset}_h2o_vmr'].values]
        return rh

    @staticmethod
    def averaging_kernel(K, Sa, Sy):
        return inv(K.T @ inv(Sy) @ K + inv(Sa)) @ K.T @ inv(Sy) @ K

    def calc_averaging_kernel_t(self, subset):
        if subset is not 'a_priori':
            A = [
                self.averaging_kernel(
                    K,
                    self.data['a_priori_covariance_t'].values,
                    self.data['a_priori_covariance_y'].values,
                )
                if np.isnan(K).sum() == 0 else np.nan
                for K in self.data[f'{subset}_jacobian_t'].values
            ]
        return A

    def calc_averaging_kernel_log_h2o_vmr(self, subset):
        print('calc averaging kernel h2o')
        if subset is not 'a_priori':
            A = [
                self.averaging_kernel(
                    K,
                    self.data['a_priori_covariance_h2o_vmr'].values,
                    self.data['a_priori_covariance_y'].values,
                )
                if np.isnan(K).sum() == 0 else np.nan
                for K in self.data[f'{subset}_jacobian_h2o_vmr'].values
            ]
        return A

    @staticmethod
    def vertical_resolution(z, A):
        z_interp = interp1d(np.arange(0, len(z)), z, fill_value="extrapolate")
        fwhm = []
        for irow in range(len(z)):
            peak = np.array([A[irow, :].argmax()])
            pws = peak_widths(
                x=A[irow, :],
                peaks=peak,
                rel_height=0.5)
            fwhm_bounds = z_interp([peak - pws[0] / 2, peak + pws[0] / 2])
            fwhm.append(float(fwhm_bounds[1] - fwhm_bounds[0]))
        return fwhm

    def calc_vertical_resolution_log_h2o_vmr(self, subset):
        res = [
            self.vertical_resolution(z, A) for z, A in
            zip(self.data['a_priori_z'].values,
                self.data[f'{subset}_averaging_kernel_log_h2o_vmr'].values)
        ]
        return res

    def calc_vertical_resolution_t(self, subset):
        res = [
            self.vertical_resolution(z, A) for z, A in
            zip(self.data['a_priori_z'].values,
                self.data[f'{subset}_averaging_kernel_t'].values)
        ]
        return res

    def calc_smoothed(self):
        smoothed_quantities = ['log_h2o_vmr', 't']
        for quantity in smoothed_quantities:
            smoothed = [
                self.smoothed_truth(A, true, a_priori) for A, true, a_priori in
                zip(self.data[f'true_averaging_kernel_{quantity}'].values,
                    self.data[f'true_{quantity}'].values,
                    self.data[f'a_priori_{quantity}'].values)
                        ]
            self.data = self.data.assign(
                variables={
                    f'smoothed_{quantity}': (self.config['quantity_dims'][quantity], np.array(smoothed)),
                }
            )
        if 'true_h2o_vmr' in self.data.variables:
            self.data = self.data.assign(
                variables={
                    f'smoothed_h2o_vmr': (self.config['quantity_dims']['h2o_vmr'],
                                          np.exp(self.data['smoothed_log_h2o_vmr'].values)),
                }
            )
        if 'true_rh' in self.data.variables:
            self.data = self.data.assign(
                variables={
                    f'smoothed_rh': (self.config['quantity_dims']['rh'],
                                     vmr2relative_humidity(
                                         self.data['smoothed_h2o_vmr'].values,
                                         self.data['a_priori_p'].values,
                                         self.data['smoothed_t'].values,
                                         e_eq=e_eq_mixed_mk,
                                     ) * 100.,
                                     ),
                }
            )

    @staticmethod
    def smoothed_truth(avg_kernel, true, a_priori):
        return avg_kernel @ (true - a_priori) + a_priori

    def calc_reference_h2o_vmr(self, subset):
        ref_h2o_vmr = [
            self.reference_h2o_vmr_profile(vmr, p, z) for vmr, p, z in
            zip(
                self.data[f'{subset}_h2o_vmr'].values,
                self.data['a_priori_p'].values,
                self.data['a_priori_z'].values,
            )
        ]
        self.data = self.data.assign(
            variables={
                f'{subset}_eml_ref_h2o_vmr': (self.config['quantity_dims']['h2o_vmr'], np.array(ref_h2o_vmr))
            }
        )

    def calc_eml_characteristics(self, **kwargs):
        for subset in self.subsets:
            self.calc_reference_h2o_vmr(subset)
            try:
                eml_characteristics = np.array(
                    [
                        self.eml_characteristics(
                            vmr, ref_vmr, t, p, z,
                            heating_rate=heating_rate,
                            ref_heating_rate=ref_heating_rate,
                            min_eml_p_width=self.config['moisture_characteristics']['anomaly_p_width_min'],
                            min_eml_strength=self.config['moisture_characteristics']['anomaly_iwv_strength_min'],
                            p_min=self.config['moisture_characteristics']['p_min'],
                            p_max=self.config['moisture_characteristics']['p_max'],
                            **kwargs,
                        )
                        for vmr, ref_vmr, t, p, z, heating_rate, ref_heating_rate in
                        zip(
                            self.data[f'{subset}_h2o_vmr'].values,
                            self.data[f'{subset}_eml_ref_h2o_vmr'].values,
                            self.data[f'{subset}_t'].values,
                            self.data[f'a_priori_p'].values,
                            self.data[f'a_priori_z'].values,
                            self.data[f'{subset}_heating_rates'].values,
                            self.data[f'{subset}_ref_heating_rates'].values,
                        )
                    ]
                )
            except KeyError:
                eml_characteristics = np.array(
                    [
                        self.eml_characteristics(
                            vmr, ref_vmr, t, p, z,
                            min_eml_p_width=self.config['moisture_characteristics']['anomaly_p_width_min'],
                            min_eml_strength=self.config['moisture_characteristics']['anomaly_iwv_strength_min'],
                            p_min=self.config['moisture_characteristics']['p_min'],
                            p_max=self.config['moisture_characteristics']['p_max'],
                            **kwargs,
                        )
                        for vmr, ref_vmr, t, p, z, in
                        zip(
                            self.data[f'{subset}_h2o_vmr'].values,
                            self.data[f'{subset}_eml_ref_h2o_vmr'].values,
                            self.data[f'{subset}_t'].values,
                            self.data[f'a_priori_p'].values,
                            self.data[f'a_priori_z'].values,
                        )
                    ]
                )
            self.data = self.data.assign(
                variables={
                    f'{subset}_moisture_characteristics': (['nprofile'], eml_characteristics),
                }
            )

    def reference_h2o_vmr_profile(self, h2o_vmr, p, z):
        is_tropo = p > self.config['moisture_characteristics']['tropo_p_min']
        popt, _ = curve_fit(
            self.ref_profile_opt_func,
            z[is_tropo],
            np.log(h2o_vmr[is_tropo]),
            bounds=([-np.inf, -np.inf, np.log(h2o_vmr[0]) - 0.01],
                    [np.inf, np.inf, np.log(h2o_vmr[0]) + 0.01])
        )
        ref_profile = np.exp(np.polyval(popt, z))

        return ref_profile

    @staticmethod
    def ref_profile_opt_func(z, a, b, c):
        return a * z ** 2 + b * z + c

    def eml_characteristics(self, h2o_vmr, ref_h2o_vmr, t, p, z,
                            heating_rate=None,
                            ref_heating_rate=None,
                            min_eml_p_width=1000.,
                            min_eml_strength=0.000001,
                            p_min=10000.,
                            p_max=90000.,
                            z_in_km=False,
                            p_in_hPa=False,
                            ):
        p_ind = p > p_min
        p_ind &= p < p_max
        p = p[p_ind]
        z = z[p_ind]
        t = t[p_ind]
        h2o_vmr = h2o_vmr[p_ind]
        ref_h2o_vmr = ref_h2o_vmr[p_ind]
        anomaly = h2o_vmr - ref_h2o_vmr
        moist = anomaly > 0
        ss = (np.nonzero(moist[1:] != moist[:-1])[0] + 1)
        if moist[0]: # Add limit at index 0, if there is an anomaly
            ss = np.concatenate([[0], ss])
        if moist[-1]: # Add limit at the index -1, if there is an anomaly
            ss = np.concatenate([ss, [len(moist) - 1]])
        ss = ss.reshape(-1, 2)
        if ss.shape[0] == 0:
            return MoistureCharacteristics()
        if ss[0, 0] == 0: # Drop anomaly, if it's bound by upper pressure (lower z limit)
            ss = np.delete(ss, axis=0, obj=[0])
        if ss.shape[0] == 0:
            return MoistureCharacteristics()
        if ss[-1, -1] == len(moist) - 1: # Drop anomaly, if it's bound by lower pressure (upper z limit)
            ss = np.delete(ss, axis=0, obj=[-1])
        eml_pressure_widths = p[ss[:, 0]] - p[ss[:, 1]]
        if not np.any(eml_pressure_widths > min_eml_p_width):
            return MoistureCharacteristics()
        ss = ss[eml_pressure_widths > min_eml_p_width, :]
        eml_pressure_widths = eml_pressure_widths[eml_pressure_widths > min_eml_p_width]
        eml_height_widths = z[ss[:, 1]] - z[ss[:, 0]]
        # Integrate VMR over the inversion layers, relative to humidity at layer bottom.
        abs_humidity = vmr2specific_humidity(h2o_vmr) * density(p, t)
        abs_humidity_ref = vmr2specific_humidity(ref_h2o_vmr) * density(p, t)
        eml_strengths = np.array(
            [
                integrate_column(
                    y=(anomaly)[ss[ieml, 0]:ss[ieml, 1]],
                    x=z[ss[ieml, 0]:ss[ieml, 1]])
                for ieml in range(ss.shape[0])
            ]
        ) / eml_height_widths
        if not np.any(eml_strengths > min_eml_strength):
            return MoistureCharacteristics()
        ss = ss[eml_strengths > min_eml_strength, :]
        eml_pressure_widths = eml_pressure_widths[eml_strengths > min_eml_strength]
        eml_height_widths = eml_height_widths[eml_strengths > min_eml_strength]
        eml_strengths = eml_strengths[eml_strengths > min_eml_strength]
        eml_inds = [np.arange(start, stop) for start, stop in ss]
        anomaly_p_means = np.array(
            [self.anomaly_position(p[eml_ind], anomaly[eml_ind]) for eml_ind in eml_inds]
        )
        anomaly_z_means = np.array(
            [self.anomaly_position(z[eml_ind], anomaly[eml_ind]) for eml_ind in eml_inds]
        )
        if heating_rate is not None:
            heating_rate = heating_rate[p_ind]
            ref_heating_rate = ref_heating_rate[p_ind]
            heating_rate_means = np.array(
                [np.mean(heating_rate[ss[ieml, 0]:ss[ieml, 1]]) for ieml in range(ss.shape[0])]
            )
            heating_rate_min = np.array(
                [np.min(heating_rate[ss[ieml, 0]:ss[ieml, 1]]) for ieml in range(ss.shape[0])]
            )
            heating_rate_max = np.array(
                [np.max(heating_rate[ss[ieml, 0]:ss[ieml, 1]]) for ieml in range(ss.shape[0])]
            )
            heating_rate_anom_means = np.array(
                [np.mean((ref_heating_rate - heating_rate)[ss[ieml, 0]:ss[ieml, 1]]) for ieml in range(ss.shape[0])]
            )
            heating_rate_anom_min = np.array(
                [np.min((ref_heating_rate - heating_rate)[ss[ieml, 0]:ss[ieml, 1]]) for ieml in range(ss.shape[0])]
            )
            heating_rate_anom_max = np.array(
                [np.max((ref_heating_rate-heating_rate)[ss[ieml, 0]:ss[ieml, 1]]) for ieml in range(ss.shape[0])]
            )
        else:
            heating_rate_means = np.nan * np.ones(ss.shape[0])
            heating_rate_min = np.nan * np.ones(ss.shape[0])
            heating_rate_max = np.nan * np.ones(ss.shape[0])
            heating_rate_anom_means = np.nan * np.ones(ss.shape[0])
            heating_rate_anom_min = np.nan * np.ones(ss.shape[0])
            heating_rate_anom_max = np.nan * np.ones(ss.shape[0])
        characteristics = MoistureCharacteristics(
            pmin=p[ss[:, 1]],
            pmax=p[ss[:, 0]],
            zmin=z[ss[:, 0]],
            zmax=z[ss[:, 1]],
            strength=eml_strengths,
            pwidth=eml_pressure_widths,
            zwidth=eml_height_widths,
            pmean=anomaly_p_means,
            zmean=anomaly_z_means,
            heating_rate_mean=heating_rate_means,
            heating_rate_min=heating_rate_min,
            heating_rate_max=heating_rate_max,
            heating_rate_anomaly_mean=heating_rate_anom_means,
            heating_rate_anomaly_min=heating_rate_anom_min,
            heating_rate_anomaly_max=heating_rate_anom_max,
        )
        if z_in_km:
            characteristics.to_km()
        if p_in_hPa:
            characteristics.to_hpa()
        return characteristics

    @staticmethod
    def anomaly_position(grid, anomaly):
        anomaly_f = interp1d(
            grid,
            anomaly,
        )
        new_grid = np.linspace(
            grid.min(),
            grid.max(),
            num=len(grid)*1000,
        )
        if np.diff(grid)[0] < 0:
            new_grid = new_grid[::-1]
        anomaly_regrid = anomaly_f(new_grid)
        anomaly_position = np.average(new_grid, weights=anomaly_regrid)
        return anomaly_position

    def load_heating_rates(self, subsets, ref=True):
        for subset in subsets:
            self.data = self.data.assign(
                variables={
                    f'{subset}_heating_rates': (
                        self.config['quantity_dims']['heating_rates'],
                        self.load_data('heating_rates', subset)
                    )
                }
            )
        # Also load reference h2o vmr profile heating rates
        if ref:
            self.data = self.data.assign(
                variables={
                    'true_ref_heating_rates': (
                        self.config['quantity_dims']['heating_rates'],
                        self.load_data('ref_heating_rates', 'true')
                    ),
                    'retrieved_ref_heating_rates': (
                        self.config['quantity_dims']['heating_rates'],
                        self.load_data('ref_heating_rates', 'retrieved')
                    )
                }
            )

    def reduce_to_matching_moist_anomalies(self):
        anomaly_match_mask = np.array([False] * self.data.dims['nprofile'])
        overlap_ind_list = []
        for iprofile, (moist_anom_true, moist_anom_retrieved) in enumerate(zip(
                self.data['true_moisture_characteristics'].values,
                self.data['retrieved_moisture_characteristics'].values
        )):
            overlap_ind = self.find_pres_overlap(moist_anom_true, moist_anom_retrieved)
            overlap_ind_list.append(overlap_ind)
            if len(overlap_ind.shape) < 2: # No matching anomalies, throw away.
                continue
            elif len(overlap_ind[:, 0]) > len(set(overlap_ind[:, 0])) or \
                    len(overlap_ind[:, 0]) > len(set(overlap_ind[:, 0])): # Multiple matching anomalies, throw away.
                continue
            else:
                anomaly_match_mask[iprofile] = True
        self.data = self.data.isel(nprofile=anomaly_match_mask) # Filter out all profiles without any overlaping anomalies.
        true_matching_anomaly_ind = [ind[:, 0] for ind in np.array(overlap_ind_list)[anomaly_match_mask]]
        retrieved_matching_anomaly_ind = [ind[:, 1] for ind in np.array(overlap_ind_list)[anomaly_match_mask]]
        _ = [anom[ind] for anom, ind in zip(
            self.data[f'true_moisture_characteristics'].values,
            true_matching_anomaly_ind,
        )
             ]
        _ = [anom[ind] for anom, ind in zip(
            self.data[f'retrieved_moisture_characteristics'].values,
            retrieved_matching_anomaly_ind,
        )
             ]

    def find_pres_overlap(self, moist_anom1, moist_anom2):
        overlap_ind = []
        for i, (pmin1, pmax1, pmean1) in enumerate(zip(moist_anom1.pmin, moist_anom1.pmax, moist_anom1.pmean)):
            for j, (pmin2, pmax2, pmean2) in enumerate(zip(moist_anom2.pmin, moist_anom2.pmax, moist_anom2.pmean)):
                if pmax1 > pmean2 > pmin1 and pmax2 > pmean1 > pmin2:
                    overlap_ind.append((i, j))
        return np.array(overlap_ind)


class MoistureCharacteristics:

    def __init__(self,
                 pmin=np.array([np.nan]),
                 pmax=np.array([np.nan]),
                 zmin=np.array([np.nan]),
                 zmax=np.array([np.nan]),
                 strength=np.array([np.nan]),
                 pwidth=np.array([np.nan]),
                 zwidth=np.array([np.nan]),
                 zmean=np.array([np.nan]),
                 pmean=np.array([np.nan]),
                 heating_rate_mean=np.array([np.nan]),
                 heating_rate_min=np.array([np.nan]),
                 heating_rate_max=np.array([np.nan]),
                 heating_rate_anomaly_mean=np.array([np.nan]),
                 heating_rate_anomaly_min=np.array([np.nan]),
                 heating_rate_anomaly_max=np.array([np.nan]),
                 ):
        self.pmin = pmin
        self.pmax = pmax
        self.zmin = zmin
        self.zmax = zmax
        self.strength = strength
        self.pwidth = pwidth
        self.zwidth = zwidth
        self.pmean = pmean
        self.zmean = zmean
        self.heating_rate_mean = heating_rate_mean
        self.heating_rate_min = heating_rate_min
        self.heating_rate_max = heating_rate_max
        self.heating_rate_anomaly_mean = heating_rate_anomaly_mean
        self.heating_rate_anomaly_min = heating_rate_anomaly_min
        self.heating_rate_anomaly_max = heating_rate_anomaly_max

    def __getitem__(self, index):
        return self.__init__(
                  pmin=np.array([self.pmin[index]]).reshape(len(index)),
                  pmax=np.array([self.pmax[index]]).reshape(len(index)),
                  zmin=np.array([self.zmin[index]]).reshape(len(index)),
                  zmax=np.array([self.zmax[index]]).reshape(len(index)),
                  strength=np.array([self.strength[index]]).reshape(len(index)),
                  pwidth=np.array([self.pwidth[index]]).reshape(len(index)),
                  zwidth=np.array([self.zwidth[index]]).reshape(len(index)),
                  zmean=np.array([self.zmean[index]]).reshape(len(index)),
                  pmean=np.array([self.pmean[index]]).reshape(len(index)),
                  heating_rate_mean=np.array([self.heating_rate_mean[index]]).reshape(len(index)),
                  heating_rate_min=np.array([self.heating_rate_min[index]]).reshape(len(index)),
                  heating_rate_max=np.array([self.heating_rate_max[index]]).reshape(len(index)),
                  heating_rate_anomaly_mean=np.array([self.heating_rate_anomaly_mean[index]]).reshape(len(index)),
                  heating_rate_anomaly_min=np.array([self.heating_rate_anomaly_min[index]]).reshape(len(index)),
                  heating_rate_anomaly_max=np.array([self.heating_rate_anomaly_max[index]]).reshape(len(index)),
        )

    def to_hpa(self):
        self.pmin /= 100.
        self.pmax /= 100.
        self.pwidth /= 100.
        self.pmean /= 100.

    def to_km(self):
        self.zmin /= 1000.
        self.zmax /= 1000.
        self.zwidth /= 1000.
        self.zmean /= 1000.

    def get_max_strength_anomaly(self):
        max_str_anomaly = self.__getitem__(index=self.strength.argmax())
        return max_str_anomaly
