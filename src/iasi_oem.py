import os
import numpy as np
from scipy.linalg import inv
from scipy.interpolate import interp1d

from pyarts.workspace import arts_agenda
from pyarts import xml
from typhon.physics import wavenumber2frequency, frequency2wavenumber, \
    constants, planck, radiance2planckTb, moist_lapse_rate, relative_humidity2vmr, e_eq_mixed_mk


def setup_retrieval_paths(project_path):
    """
    Setup the directory infrastructure for the specific retrieval project.
    """
    if not os.path.isdir(project_path):
        os.mkdir(project_path)
    for dir_name in ["plots", "sensor", "a_priori", "observations", "retrieval_output"]:
        if not os.path.isdir(os.path.join(project_path, dir_name)):
            os.mkdir(os.path.join(project_path, dir_name))
    os.chdir(project_path)


def load_abs_lookup(ws, abs_lookup_path=None, use_abs_lookup_f_grid=False):
    """
    Load existing absorption lookup table or create one in case it does
    not exist yet for given batch of atmospheres and frequency range.
    :param ws: ARTS workspace object
    :param hitran_split_artscat5_path: path to hitran_split_artscat5 catalogue.
    :param atm_batch_path: path to batch of atmospheres to calculate absorption lookup table for
    :param f_ranges: frequency range
    :param abs_lookup_path: if already available, provide lookup table path here to avoid new calculation
    :return:
    """
    if abs_lookup_path is not None:
        ws.ReadXML(ws.abs_lookup, abs_lookup_path)
        if use_abs_lookup_f_grid:
            ws.f_grid = ws.abs_lookup.value.f_grid.data
        ws = abs_setup(ws)
        ws.abs_lookupAdapt()
    else:
        ws = abs_setup(ws)
        ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
            basename="spectroscopy/Artscat/"
        )
        ws.abs_lines_per_speciesSetCutoff(option="ByLine", value=750e9)
        ws.abs_lines_per_speciesCompact()
        ws.abs_xsec_agenda_checkedCalc()
        ws.lbl_checkedCalc()
    return ws


def setup_sensor(ws, f_backend_width, srf_gaussian_width=wavenumber2frequency(50),
                 f_ranges=None, add_frequencies=None, f_grid=None, wmrf_weights=None,
                 bt_spectrum=None, f_per_channel=1, sensor_los=180., save_grids=True):
    """
    Setup the sensor properties, mainly including the frequency grid.
    :param ws: ARTS workspace
    :param f_ranges: sensor frequency ranges as array of arrays that represent different spectral regions.
    :param f_backend_width: frequency interval between channels
    :param add_frequencies: array of frequencies to add.
    :return
    """
    ws.sensor_pos = np.array([[850e3]])  # 850km
    # ws.sensor_time = np.array([0.0])
    ws.sensor_los = np.array([[sensor_los]])  # 180 = nadir viewing
    if f_ranges is not None:
        ws.f_backend = np.concatenate([np.arange(freq[0],
                                                 freq[1],
                                                 f_backend_width) for freq in f_ranges])
    else:
        ws.f_backend = np.array([])
    if add_frequencies is not None:
        ws.f_backend = np.unique(np.sort(np.append(ws.f_backend.value, add_frequencies)))
    # Sensor settings
    ws.FlagOn(ws.sensor_norm)
    ws.AntennaOff()
    if wmrf_weights is not None:
        ws = setup_wmrf_response(ws, f_grid, wmrf_weights, bt_spectrum)
    else:
        ws = setup_gaussian_response(ws, f_backend_width, f_per_channel, srf_gaussian_width, f_grid)
    if save_grids:
        ws.WriteXML("binary", ws.f_backend, "sensor/f_backend.xml")
        ws.WriteXML("binary", ws.f_grid, "sensor/f_grid.xml")
    return ws


def setup_gaussian_response(ws, f_backend_width, f_per_channel=1, srf_gaussian_width=wavenumber2frequency(50),
                            f_grid=None):
    if f_grid is not None:
        ws.f_grid = f_grid
    else:
        f_interval_n = int((ws.f_backend.value.max() - ws.f_backend.value.min()) / f_backend_width)
        ws.VectorNLinSpace(
        ws.f_grid,
        f_interval_n * f_per_channel + 41 * f_per_channel,
        ws.f_backend.value.min() - wavenumber2frequency(500.),
        ws.f_backend.value.max() + wavenumber2frequency(500.)
    )
    ws.backend_channel_responseGaussian(fwhm=np.array([srf_gaussian_width]))
    ws.sensor_responseInit()
    ws.sensor_responseBackend()
    return ws


def setup_wmrf_response(ws, f_grid, weights, bt_spectrum=None):
    ws.f_grid = f_grid
    ws.f_grid = np.sort(ws.f_grid.value)
    ws.wmrf_weights = weights
    iasi_f_backend = wavenumber2frequency(np.arange(645.25, 2760.5, 0.25) * 100.)
    if bt_spectrum is not None:
        channel_sel_ind = np.where(np.logical_and(
            np.isin(iasi_f_backend, ws.f_backend.value), # Index to select desired channels.
            ~np.isnan(bt_spectrum)  # Check for nan values in the spectrum and throw out these channels.
        ))[0]
    else:
        channel_sel_ind = np.where(np.isin(iasi_f_backend, ws.f_backend.value))[0] # Index to select desired channels.
    ws.f_backend = iasi_f_backend
    ws.ArrayOfIndexSet(ws.wmrf_channels, channel_sel_ind.tolist())
    ws.WMRFSelectChannels()
    ws.sensor_responseInit()
    ws.sensor_responseWMRF()
    return ws


def load_generic_settings(ws, lookup_table=False):
    """
    Load generic agendas and set generic settings.
    :param ws: ARTS workspace
    :return: ws
    """
    ws.execute_controlfile("general/general.arts")
    ws.execute_controlfile("general/continua.arts")
    ws.execute_controlfile("general/agendas.arts")
    ws.execute_controlfile("general/planet_earth.arts")

    # (standard) emission calculation
    ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)

    # cosmic background radiation
    ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)

    @arts_agenda
    def iy_surface_agenda_PY(ws):
        ws.SurfaceBlackbody()
        ws.iySurfaceRtpropCalc()

    @arts_agenda
    def abs_xsec_agenda_no_lines(ws):
        ws.Ignore(ws.abs_nlte)
        ws.abs_xsec_per_speciesInit()
        ws.abs_xsec_per_speciesAddConts()

    @arts_agenda
    def propmat_clearsky_agenda_lines(ws):
        ws.Ignore(ws.rtp_mag)
        ws.Ignore(ws.rtp_los)
        ws.propmat_clearskyInit()
        ws.propmat_clearskyAddXsecAgenda()
        ws.propmat_clearskyAddLines()

    ws.Copy(ws.iy_surface_agenda, iy_surface_agenda_PY)
    # standard surface agenda (i.e., make use of surface_rtprop_agenda)
    # ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)

    ws.Copy(ws.surface_rtprop_agenda, ws.surface_rtprop_agenda__Specular_NoPol_ReflFix_SurfTFromt_surface)

    # clearsky agenda
    if lookup_table:
        ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__LookUpTable)
        ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)
    else:
        ws.Copy(ws.propmat_clearsky_agenda, propmat_clearsky_agenda_lines)
        ws.Copy(ws.abs_xsec_agenda, abs_xsec_agenda_no_lines)
    # sensor-only path
    ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)
    # no refraction
    ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)



    ws.stokes_dim = 1
    ws.atmosphere_dim = 1  # 1D VAR
    ws.jacobian_quantities = []
    ws.iy_unit = "PlanckBT"
    ws.cloudboxOff()
    ws.surface_scalar_reflectivity = np.array([0.00])  # nominal albedo for surface
    return ws


def iasi_observation(ws, atm_batch_path=None, ybatch_start=None, ybatch_n=None,
                     iasi_obs_path=None, iasi_obs_data=None, add_measurement_noise=True):
    """
    Forward simulate IASI observations for given batch of atmospheres.
    :param ws: ARTS workspace
    :param atm_batch_path: Path to batch of atmospheres to do forward simulations for
    :param ybatch_start: Start index for given batch of atmospheres
    :param ybatch_n: Amount of atmospheres within batch to do forward simulations for, starting at ybatch_start
    :param iasi_obs_path: You may load already available batch of spectra instead of doing forward simulation.
    :param add_measurement_noise: Boolean to decide whether or not to add gaussian noise on top of spectrum.
    :return:
    """
    if iasi_obs_path:
        ws.ReadXML(ws.ybatch, iasi_obs_path)
    elif iasi_obs_data is not None:
        ws.Copy(ws.ybatch, iasi_obs_data)
    else:
        ws.ReadXML(ws.batch_atm_fields_compact, atm_batch_path)
        ws = abs_setup(ws)
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.abs_xsec_agenda_checkedCalc()

        ws.VectorCreate("t_surface_vector")
        ws.NumericCreate("t_surface_numeric")

        @arts_agenda
        def ybatch_calc_agenda(ws):
            ws.Extract(ws.atm_fields_compact,
                       ws.batch_atm_fields_compact,
                       ws.ybatch_index)
            ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
            ws.Extract(ws.z_surface, ws.z_field, 0)
            ws.Extract(ws.t_surface, ws.t_field, 0)
            ws.Copy(ws.surface_props_names, ["Skin temperature"])
            ws.VectorExtractFromMatrix(ws.t_surface_vector, ws.t_surface, 0, "row")
            ws.Extract(ws.t_surface_numeric, ws.t_surface_vector, 0)
            ws.Tensor3SetConstant(ws.surface_props_data, 1, 1, 1, ws.t_surface_numeric)
            ws.jacobianInit()
            ws.jacobianAddSurfaceQuantity(
                g1=ws.lat_grid, g2=ws.lon_grid, quantity="Skin temperature")
            ws.jacobianAddTemperature(g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
            ws.jacobianAddAbsSpecies(species="H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                                     unit="rel",
                                     g1=ws.p_grid, g2=ws.lat_grid, g3=ws.lon_grid)
            ws.jacobianClose()
            ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
            ws.atmgeom_checkedCalc()
            ws.cloudbox_checkedCalc()
            ws.sensor_checkedCalc()
            ws.yCalc()

        ws.Copy(ws.ybatch_calc_agenda, ybatch_calc_agenda)
        if ybatch_start is not None:
            ws.IndexSet(ws.ybatch_start, ybatch_start)
        ws.IndexSet(ws.ybatch_n, ybatch_n)  # Amount of atmospheres
        ws.ybatchCalc()
        # Add measurement noise to synthetic observation
        if add_measurement_noise:
            for i in range(ybatch_n):
                ws.ybatch.value[i].data[ws.f_backend.value < wavenumber2frequency(175000)] += \
                    np.array([np.random.normal(loc=0.0, scale=0.1,
                                               size=np.sum(ws.f_backend.value < wavenumber2frequency(175000))
                                               )])[0, :]
                ws.ybatch.value[i].data[ws.f_backend.value >= wavenumber2frequency(175000)] += \
                    np.array([np.random.normal(loc=0.0, scale=0.2,
                                               size=np.sum(ws.f_backend.value >= wavenumber2frequency(175000))
                                               )])[0, :]
        ws.WriteXML("binary", ws.ybatch_jacobians,
                    f"observations/jacobian_{ybatch_start}-"
                    f"{ybatch_start + ybatch_n}.xml")
        ws.WriteXML("binary", ws.batch_atm_fields_compact, f"observations/atm_fields.xml")

    ws.WriteXML("binary", ws.ybatch,
                f"observations/ybatch_{ybatch_start}-{ybatch_start + ybatch_n}.xml")
    return ws


def abs_setup(ws):
    """
    Setup absorption species for the RT calculation.
    :param ws: ARTS workspace
    :return:
    """
    # define absorbing species and load lines for given frequency range from HITRAN
    ws.abs_speciesSet(species=["H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                               "O2, O2-CIAfunCKDMT100",
                               "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
                               "O3",
                               "CO2-626, CO2-627, CO2-628, CO2-636, CO2-637, CO2-638, "
                               "CO2-727, CO2-728, CO2-828, CO2-837, CO2-838, CO2-CKDMT252",
                               "CH4",
                               "N2O",
                               ])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-O2",
                                           value=0.2095,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-N2",
                                           value=0.7808,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-CO2",
                                           value=0.0004,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-CH4",
                                           value=0.00000182,
                                           condensibles=["abs_species-H2O"])
    ws.batch_atm_fields_compactAddConstant(name="abs_species-N2O",
                                           value=0.00000033,
                                           condensibles=["abs_species-H2O"])
    return ws


def setup_apriori_state(ws, a_priori_atm_batch_path, batch_ind,
                        moist_adiabat=False, t_surface=None, t_surface_std=0.,
                        stratospheric_temperature=None, RH=None):
    """
    Setup the a priori state based on a batch of atmospheres and a batch index.
    Generally, the retrieval is performed for a batch of of atmospheres,
    so the a priori state also is a batch of atmospheres.
    @param ws: ARTS workspace
    @param a_priori_atm_batch_path: Path to batch of a priori atmospheres
    @param batch_ind: The index of the atmosphere to use for current a priori.
    @param moist_adiabat: Boolean to decide whether or not use a moist adiabatic temperature profile
    based on the surface temperature as the a priori temperature profile.
    @param t_surface: Option to manually set the a priori surface temperature
    @param t_surface_std: Artificially add gaussian noise to the surface temperature a priori state. Useful
    to represent uncertainty in Ts that would in reality exist.
    @param stratospheric_temperature: A full temperature profile, of which to extract the
    tropopause and above layers, which fitted to the a priori temperature profile. Useful when using moist adiabat.
    @param RH: You may provide an a priori relative humidity profile, based on which together with the a priori
    temperature profile, the a priori H2O VMR profile is determined.
    @return:
    """
    ws.ReadXML(ws.batch_atm_fields_compact, a_priori_atm_batch_path)
    ws = abs_setup(ws)
    ws.Extract(ws.atm_fields_compact,
               ws.batch_atm_fields_compact,
               int(batch_ind))
    ws.AtmFieldsAndParticleBulkPropFieldFromCompact()
    ws.lat_grid = []
    ws.lon_grid = []
    if t_surface is not None:
        ws.t_surface = t_surface
    else:
        ws.Extract(ws.t_surface, ws.t_field, 0)
    ws.Extract(ws.z_surface, ws.z_field, 0)
    # ws.AtmFieldsCalc()
    if moist_adiabat:
        ws = t_profile_to_moist_adiabat(ws, t_surface_std, stratospheric_temperature)
        ws.Extract(ws.t_surface, ws.t_field, 0)
        ws.Extract(ws.z_surface, ws.z_field, 0)
    if RH is not None:
        ws = h2o_vmr_from_rh(ws, RH)
    ws.AbsInputFromAtmFields()
    ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
    ws.atmgeom_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()
    snames = ["Skin temperature"]
    sdata = np.array([ws.t_surface.value])
    ws.Copy(ws.surface_props_names, snames)
    ws.Copy(ws.surface_props_data, sdata)
    return ws


def setup_retrieval_quantities(ws, retrieval_quantities, cov_cross=None, cov_h2o_vmr=None,
                               cov_t=None, cov_t_surface=None):
    """
    Setup the retrieval quantities.
    @param ws: ARTS workspace
    @param retrieval_quantities: Currently supported: t_surface, Temperature, H2O
    @param cov_cross: List of dictionaries with keys 'S', 'i', 'j' to optionally provide
    cross covariances between the main covariance blocks (i,j index the blocks).
    @param cov_h2o_vmr: covariance matrix for H2O VMR in logarithmic units
    @param cov_t: covariance matrix for temperature profile
    @param cov_t_surface: covariance matrix for surface temperature. Matrix of size (1,1).
    @return:
    """
    if "t_surface_python" in retrieval_quantities:
        ws = get_transmittance(ws)
        ws.MatrixCreate("cov_t_surface")
        ws.MatrixCreate("cov_y_t_surface")
        ws.cov_t_surface = cov_t_surface
        ws.cov_y_t_surface = 0.1 ** 2 * np.diag(np.ones(ws.f_backend.value.shape))
        ws.WriteXML("ascii", ws.cov_y_t_surface.value, "a_priori/covariance_y.xml")
        ws.WriteXML("ascii", cov_t_surface, "a_priori/covariance_t_surface.xml")

    else:
        ws.retrievalDefInit()
        if "t_surface" in retrieval_quantities:
            ws.retrievalAddSurfaceQuantity(
                g1=ws.lat_grid, g2=ws.lon_grid, quantity="Skin temperature")
            ws.covmat_sxAddBlock(block=cov_t_surface)

        if "Temperature" in retrieval_quantities:
            ws.retrievalAddTemperature(
                g1=ws.p_grid,
                g2=ws.lat_grid,
                g3=ws.lon_grid)
            ws.covmat_sxAddBlock(block=cov_t)

        if "H2O" in retrieval_quantities:
            ws.retrievalAddAbsSpecies(species="H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252",
                                      unit="vmr",
                                      g1=ws.p_grid,
                                      g2=ws.lat_grid,
                                      g3=ws.lon_grid)
            ws.jacobianSetFuncTransformation(transformation_func="log")
            ws.covmat_sxAddBlock(block=cov_h2o_vmr)

        if cov_cross is not None:
            for S_dict in cov_cross:
                ws.covmat_sxAddBlock(block=S_dict["S"], i=S_dict["i"], j=S_dict["j"])
        cov_y = np.diag(np.ones(ws.f_backend.value.size))
        low_noise_ind = ws.f_backend.value < wavenumber2frequency(175000)
        high_noise_ind = ws.f_backend.value >= wavenumber2frequency(175000)
        cov_y[low_noise_ind, low_noise_ind] *= 0.1**2
        cov_y[high_noise_ind, high_noise_ind] *= 0.2**2
        ws.covmat_seAddBlock(block=cov_y)
        ws.retrievalDefClose()
        ws.WriteXML("ascii", cov_y, "a_priori/covariance_y.xml")
        ws.WriteXML("ascii", ws.covmat_sx, "a_priori/covmat_sx.xml")
    return ws


def save_atm_state(ws, dirname):
    """
    Save current atmospheric state within ARTS workspace to dirname.
    @param ws: ARTS workspace
    @param dirname: directory path
    @return:
    """
    ws.WriteXML("ascii", ws.vmr_field, f"{dirname}/vmr.xml")
    ws.WriteXML("ascii", ws.t_field, f"{dirname}/temperature.xml")
    ws.WriteXML("ascii", ws.t_surface, f"{dirname}/t_surface.xml")
    ws.WriteXML("ascii", ws.p_grid, f"{dirname}/p.xml")


def save_z_p_grids(ws, dirname):
    """
    Save current p and z grid to dirname.
    @param ws: ARTS workspace
    @param dirname: directory path
    @return:
    """
    ws.WriteXML("ascii", ws.p_grid, f"{dirname}/p.xml")
    ws.WriteXML("ascii", [ws.z_field.value[:, 0, 0]], f"{dirname}/z.xml")


def t_profile_to_moist_adiabat(ws, t_surface_std, stratospheric_temperature, eps=1.0):
    """
    Turn current temperature profile in the troposphere (up to 100 hPa) in ARTS workspace
    (ws.t_field) to moist adiabat based on current ws.t_surface and let the moist adiabat
    smoothly transition into the provided stratospheric_temperature profile provided.
    @param ws: ARTS workspace
    @param t_surface_std: Gaussian noise to add to current surface temperature,
    before determining temperature profile
    @param stratospheric_temperature: Full temperature profile, into which the current profile ws.t_field
    transitions in about 100 hPa and above.
    @return:
    """
    # eps is surface emissivity.
    t_moist = ws.t_surface.value[0, :] / eps + np.random.normal(0., t_surface_std)
    for p, z_diff in zip(ws.p_grid.value[1:], np.diff(ws.z_field.value[:, 0, 0])):
        try:
            t_moist = np.append(t_moist,
                                t_moist[-1] -
                                moist_lapse_rate(p, t_moist[-1]) * z_diff)
        except Exception:
            break
    x_smooth = np.arange(4000., 16000., 10)
    tanh_smooth = interp1d(x_smooth, (0.5 + 0.5 * np.tanh(0.0005 * (x_smooth - 10000.))))
    smooth_ind = np.where((ws.p_grid.value < 15000.) & (ws.p_grid.value > 5000.))[0]
    t_smooth = t_moist[smooth_ind] * tanh_smooth(ws.p_grid.value[smooth_ind]) + \
               stratospheric_temperature[smooth_ind] * (1 - tanh_smooth(ws.p_grid.value[smooth_ind]))

    ws.t_field.value[:, 0, 0] = np.append(np.append(t_moist[:smooth_ind[0]], t_smooth),
                                          stratospheric_temperature[smooth_ind[-1]+1:])
    return ws


def h2o_vmr_from_rh(ws, RH):
    """
    Set h2o_vmr field in ARTS workspace to provided Relative Humidity (RH) profile.
    @param ws: ARTS workspace
    @param RH: Relative Humidity profile
    @return:
    """
    new_vmr = relative_humidity2vmr(RH, ws.p_grid.value, ws.t_field.value[:, 0, 0],
                                    e_eq=e_eq_mixed_mk)
    # new_vmr[ws.p_grid.value < 10000.] = new_vmr[np.argmin(np.abs(ws.p_grid.value - 10000.))]
    ws.vmr_field.value[0, :, 0, 0] = np.copy(new_vmr)
    return ws


def retrieve_ybatch_for_a_priori_batch(ws, retrieval_batch_indices, a_priori_batch_indices, a_priori_atm_batch_path,
                                       Sa_T=None, Sa_h2o=None, Sa_cross=None, Sa_t_surface=None, t_surface=None,
                                       t_surface_std=None, stratospheric_temperature=None,
                                       retrieval_quantities=['Temperature', 'H2O', 't_surface'], moist_adiabat=False,
                                       RH=None, inversion_method="lm", max_iter=15, gamma_start=10,
                                       gamma_dec_factor=2.0, gamma_inc_factor=2.0, gamma_upper_limit=1e20,
                                       gamma_lower_limit=1.0,
                                       gamma_upper_convergence_limit=99.0):
    """
    Wrapping function to conduct retrieval calculations for given batch of a priori states and batch of spectra.
    @param ws: ARTS workspace
    @param retrieval_batch_indices: 1D array of batch indices to use of the ws.ybatch spectra
    @param a_priori_batch_indices: 1D array of batch indices to use of the batch of a priori
    atmospheres stored in a_priori_atm_batch_path. Must be same length as a_priori_batch_indices.
    @param a_priori_atm_batch_path: Path to batch of atmospheric profiles to use as a priori states.
    @param Sa_T: covariance matrix for temperature profile
    @param Sa_h2o:covariance matrix for H2O VMR in logarithmic units
    @param Sa_cross: List of dictionaries with keys 'S', 'i', 'j' to optionally provide
    cross covariances between the main covariance blocks (i,j index the blocks).
    @param Sa_t_surface: covariance matrix for surface temperature. Matrix of size (1,1).
    @param t_surface: Optionally provide a fix a priori surface temperature
    @param t_surface_std: gaussian noise added to a priori surface temperature, before a priori
    temperature and H2O profiles are determined.
    @param stratospheric_temperature: A full temperature profile, of which to extract the
    tropopause and above layers, which fitted to the a priori temperature profile. Useful when using moist adiabat.
    @param retrieval_quantities:Currently supported: t_surface, Temperature, H2O
    @param moist_adiabat: Boolean to decide whether or not use a moist adiabatic temperature profile
    @param RH: You may provide an a priori relative humidity profile, based on which together with the a priori
    temperature profile, the a priori H2O VMR profile is determined.
    @param inversion_method: OEM inversion method, currently supported:
    'li' (linear), 'gn' (Gauss-Newton), 'lm' (Levenberg-Marquardt)
    @param max_iter: Maximum amount of OEM iterations
    @param gamma_start: For LM inversion, Gamma start parameter
    @param gamma_dec_factor: For LM inversion, Gamma decrease factor after successful reduction of cost function.
    @param gamma_inc_factor: For LM inversion, Gamma increase factor after unsuccessful reduction of cost function.
    @param gamma_upper_limit: For LM inversion, Gamma limit. This is an additional stop criterion. Convergence
    is not considered until there has been one succesful iteration having a gamma <= this value.
    @param gamma_lower_limit: For LM inversion, lower Gamma treshold. If the threshold is passed, gamma is set to zero.
    If gamma must be increased from zero, gamma is set to this value.
    @param gamma_upper_convergence_limit: For LM inversion, maximum allowed Gamma value. If the value is passed,
    the inversion is halted.
    @return:
    """
    retrieved_h2o_vmr = []
    retrieved_t = []
    retrieved_ts = []
    retrieved_y = []
    retrieved_jacobian = []
    apriori_h2o_vmr = []
    apriori_t = []
    apriori_ts = []
    p_grid = []
    z_grid = []
    oem_diagnostics = []
    for retr_batch_ind, apriori_batch_ind in zip(retrieval_batch_indices, a_priori_batch_indices):
        ws.y = np.copy(ws.ybatch.value[int(retr_batch_ind - ws.ybatch_start.value)].data)
        ws = setup_apriori_state(ws,
                                 a_priori_atm_batch_path=a_priori_atm_batch_path,
                                 batch_ind=apriori_batch_ind,
                                 moist_adiabat=moist_adiabat,
                                 t_surface=t_surface,
                                 t_surface_std=t_surface_std,
                                 stratospheric_temperature=stratospheric_temperature,
                                 RH=RH)
        ws = setup_retrieval_quantities(ws,
                                        retrieval_quantities=retrieval_quantities,
                                        cov_t_surface=Sa_t_surface,
                                        cov_t=Sa_T,
                                        cov_h2o_vmr=Sa_h2o,
                                        cov_cross=Sa_cross,
                                        )
        apriori_h2o_vmr.append(np.copy(ws.vmr_field.value[0, :, 0, 0]))
        apriori_t.append(np.copy(ws.t_field.value[:, 0, 0]))
        apriori_ts.append(np.copy(ws.surface_props_data.value[0][0, 0]))
        p_grid.append(np.copy(ws.p_grid.value))
        z_grid.append(np.copy(ws.z_field.value[:, 0, 0]))
        print(f"Retrieving batch profile {retr_batch_ind + 1}.")
        print(f"Profile {retr_batch_ind - retrieval_batch_indices[0] + 1} "
              f"out of {len(retrieval_batch_indices)} in this job.")
        # try:
        ws = oem_retrieval(ws, inversion_method, max_iter, gamma_start,
                           gamma_dec_factor, gamma_inc_factor, gamma_upper_limit,
                           gamma_lower_limit, gamma_upper_convergence_limit)
        # except Exception:
        #     print(ws.oem_errors.value)
        #     retrieved_h2o_vmr.append(np.nan * np.ones(ws.vmr_field.value[0, :, 0, 0].shape))
        #     retrieved_t.append(np.nan * np.ones(ws.t_field.value[:, 0, 0].shape))
        #     retrieved_ts.append(np.nan * np.ones(ws.surface_props_data.value[0].shape))
        #     retrieved_y.append(np.nan * np.ones(ws.y.value.shape))
        #     retrieved_jacobian.append(np.nan * np.ones((len(ws.y.value), len(ws.p_grid.value) * 2 + 1)))
        #     oem_diagnostics.append(np.nan * np.ones(5))
        #     print(retrieved_t[-1])
        #     continue
        retrieved_h2o_vmr.append(np.copy(ws.vmr_field.value[0, :, 0, 0]))
        retrieved_t.append(np.copy(ws.t_field.value[:, 0, 0]))
        retrieved_ts.append(ws.surface_props_data.value[0][0, 0])
        retrieved_y.append(np.copy(ws.yf.value))
        retrieved_jacobian.append(np.copy(ws.jacobian.value))
        oem_diagnostics.append(ws.oem_diagnostics.value)
    ws.WriteXML("ascii", retrieved_h2o_vmr, f"retrieval_output/h2o_vmr_batch_profiles_"
                                            f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", retrieved_t, f"retrieval_output/temperature_batch_profiles_"
                                      f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", np.array(retrieved_ts), f"retrieval_output/surface_temperature_batch_profiles_"
                                                 f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("binary", retrieved_y, f"retrieval_output/ybatch_profiles_"
                                       f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("binary", retrieved_jacobian, f"retrieval_output/jacobian_batch_profiles_"
                                              f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", oem_diagnostics, f"retrieval_output/oem_diagnostics_batch_profiles_"
                                          f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", apriori_h2o_vmr, f"a_priori/h2o_vmr_batch_profiles_"
                                          f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", apriori_t, f"a_priori/temperature_batch_profiles_"
                                    f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", np.array(apriori_ts), f"a_priori/surface_temperature_batch_profiles_"
                                               f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", p_grid, f"a_priori/p_grid_batch_profiles_"
                                 f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")
    ws.WriteXML("ascii", z_grid, f"a_priori/z_grid_batch_profiles_"
                                 f"{retrieval_batch_indices[0]}-{retrieval_batch_indices[-1] + 1}.xml")


def oem_retrieval(ws, inversion_method="lm", max_iter=20, gamma_start=1000,
                  gamma_dec_factor=2.0, gamma_inc_factor=2.0, gamma_upper_limit=1e20,
                  gamma_lower_limit=1.0, gamma_upper_convergence_limit=99.0):
    """
    Wrapper function to conduct the oem retrieval.
    @param ws: ARTS workspace
    @param inversion_method: OEM inversion method, currently supported:
    'li' (linear), 'gn' (Gauss-Newton), 'lm' (Levenberg-Marquardt)
    @param max_iter: Maximum amount of OEM iterations
    @param gamma_start: For LM inversion, Gamma start parameter
    @param gamma_dec_factor: For LM inversion, Gamma decrease factor after successful reduction of cost function.
    @param gamma_inc_factor: For LM inversion, Gamma increase factor after unsuccessful reduction of cost function.
    @param gamma_upper_limit: For LM inversion, Gamma limit. This is an additional stop criterion. Convergence
    is not considered until there has been one succesful iteration having a gamma <= this value.
    @param gamma_lower_limit: For LM inversion, lower Gamma treshold. If the threshold is passed, gamma is set to zero.
    If gamma must be increased from zero, gamma is set to this value.
    @param gamma_upper_convergence_limit: For LM inversion, maximum allowed Gamma value. If the value is passed,
    the inversion is halted.
    @return:
    """
    # define inversion iteration as function object within python
    @arts_agenda
    def inversion_agenda(ws):
        ws.Ignore(ws.inversion_iteration_counter)
        # print('before' + str(ws.x.value[0]))
        # ws.x.value[0] = ws.surface_props_data.value[0]
        # print('after' + str(ws.x.value[0]))
        ws.x2artsAtmAndSurf()
        # ws.Extract(ws.t_surface, ws.surface_props_data, 0)
        # ws.Extract(ws.surface_skin_t, ws.surface_props_data, 0)
        # print(ws.surface_skin_t.value)
        # to be safe, rerun checks dealing with atmosph.
        # Allow negative vmr? Allow temperatures < 150 K and > 300 K?
        ws.atmfields_checkedCalc(
            # negative_vmr_ok=1,
            bad_partition_functions_ok=1,
        )
        ws.atmgeom_checkedCalc()
        ws.abs_xsec_agenda_checkedCalc()
        ws.yCalc()  # calculate yf and jacobian matching x
        ws.Copy(ws.yf, ws.y)
        ws.jacobianAdjustAndTransform()

    ws.Copy(ws.inversion_iterate_agenda, inversion_agenda)

    ws.xaStandard()  # a_priori vector is current state of retrieval fields in ws, but transformed
    ws.x = np.array([])  # create empty vector for retrieved state vector?
    ws.yf = np.array([])  # create empty vector for simulated TB?
    ws.jacobian = np.array([[]])
    ws.oem_errors = []
    ws.OEM(method=inversion_method,
           max_iter=max_iter,
           display_progress=1,
           max_start_cost=1e5,
           # start value for gamma, decrease/increase factors,
           # upper limit for gamma, lower gamma limit which causes gamma=0
           # Upper gamma limit, above which no convergence is accepted
           lm_ga_settings=np.array([gamma_start, gamma_dec_factor, gamma_inc_factor, gamma_upper_limit,
                                    gamma_lower_limit, gamma_upper_convergence_limit]))
    ws.x2artsAtmAndSurf()  # convert from ARTS coords back to user-defined grid
    return ws


#####################################################################################################
# REMAINING FUNCTIONS WORK MOSTLY INDEPENDENT OF THE ARTS WORKSPACE AND DON'T NEED TO BE CONSIDERED #
#####################################################################################################


def radiance2planck_bt_wavenumber(r, wavenumber):
    c = constants.speed_of_light
    k = constants.boltzmann
    h = constants.planck
    return h / k * c * wavenumber / np.log(np.divide(2 * h * c ** 2 * wavenumber ** 3, r) + 1)


def get_transmittance(ws):
    ws.MatrixCreate("t_surface_a")
    ws.t_surface_a = np.copy(ws.t_surface.value)
    ws.jacobianOff()
    ws.yCalc()
    ws.MatrixCreate("planck_a")
    ws.MatrixPlanck(ws.planck_a, ws.stokes_dim, ws.f_backend.value, ws.t_surface_a.value[0, 0])
    ws.VectorCreate("transmittance")
    if ws.iy_unit.value == "PlanckBT":
        ws.transmittance = planck(ws.f_backend.value, ws.y.value) / ws.planck_a.value[:, 0] / \
                           (1 - ws.surface_scalar_reflectivity.value)
    else:
        ws.transmittance = ws.y.value / ws.planck_a.value[:, 0] / (1 - ws.surface_scalar_reflectivity.value)
    return ws


def oem_t_surface(ws, ybatch_indices, max_iter):
    Sa = ws.cov_t_surface.value
    Sy = ws.cov_y_t_surface.value
    ws.MatrixCreate("jacobian_Ts")
    retrieved_Ts = []
    for obs in np.array(ws.ybatch.value)[ybatch_indices]:
        ws.y = np.copy(obs)
        ws.t_surface.value = np.copy(ws.t_surface_a.value)
        not_converged = True
        print(f"t_surface_apriori={ws.t_surface.value}")
        iter_n = 0
        while not_converged:
            iter_n += 1
            ws.jacobian_Ts = np.array([planck_derivative_T(ws.f_backend.value, ws.t_surface.value[0, 0]) *
                                       ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value)]).T
            ws.t_surface = gauss_newton_t_surface(ws, Sa, Sy)
            print(f"iter{iter_n}, t_surface={ws.t_surface.value}")
            print(f"iter{iter_n}, cost={eval_cost_function(ws, Sa, Sy)}")
            if iter_n == max_iter:
                break
        retrieved_Ts.append(np.array([np.copy(ws.t_surface.value[0, 0])]))
    ws.WriteXML("ascii", ws.jacobian_Ts, "retrieval_output/jacobian_t_surface.xml")
    ws.WriteXML("ascii", retrieved_Ts, "retrieval_output/retrieved_t_surface.xml")


def planck_derivative_T(f, T):
    c = constants.speed_of_light
    k = constants.boltzmann
    h = constants.planck
    return 2 * h ** 2 * f ** 4 / (c ** 2 * k * T ** 2) * np.exp(h * f / (k * T)) / (np.exp(h * f / (k * T)) - 1) ** 2


def gauss_newton_t_surface(ws, Sa, Sy):
    yi = radiance2planckTb(ws.f_backend.value, planck(ws.f_backend.value, ws.t_surface.value[0, 0]) *
                           ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value))
    return \
        ws.t_surface_a.value + inv(inv(Sa) + ws.jacobian_Ts.value.T @ inv(Sy) @ ws.jacobian_Ts.value) @ \
        ws.jacobian_Ts.value.T @ inv(Sy) @ \
        (ws.y.value - yi +
         ws.jacobian_Ts.value @ (ws.t_surface.value[:, 0] - ws.t_surface_a.value[:, 0]))


def eval_cost_function(ws, Sa, Sy):
    yi = radiance2planckTb(ws.f_backend.value, planck(ws.f_backend.value, ws.t_surface.value[0, 0]) *
                           ws.transmittance.value * (1 - ws.surface_scalar_reflectivity.value))
    return (ws.y.value - yi).T @ inv(Sy) @ (ws.y.value - yi) + \
           (ws.t_surface.value[:, 0] - ws.t_surface_a.value[:, 0]) ** 2 @ inv(Sa)


# def plot_retrieval_results():


def corr_length_cov(z, trpp=12.5e3):
    """Return correlation lengths for given altitudes.
    Parameters:
        z (np.array): Height levels [m]
        trpp (float): Tropopause height [m]

    Returns:
        np.array: Correlation length for each heigth level.

    """
    f = np.poly1d(np.polyfit(x=(0, trpp), y=(2.5e3, 10e3), deg=1))
    cl = f(z)
    cl[z > trpp] = 10e3

    return cl


def covmat_cross(covmat1, covmat2, z_grid, corr_height=1500.):
    """
    Return cross-covariance block for given H2O and Temperature
    covariance matrices. The cross-covariances drop exponentially
    with height (1/e at corr_height) and correlation length approach
    is used for determining non-diagonal entries.
    """
    S = np.zeros((covmat1.shape[0], covmat2.shape[0]))
    if np.any(np.array(S.shape) == 1):
        S = np.array([np.sqrt(covmat1[0, 0] * covmat2[0, 0]) * np.exp(-1 / corr_height * z_grid[i])
                      for i in range(len(z_grid))]).reshape(1, len(z_grid))
        return S

    S[np.diag_indices_from(S)] = [np.sqrt(covmat1[0, 0] * covmat2[0, 0]) * np.exp(-1 / corr_height * z_grid[i])
                                  for i in range(len(z_grid))]
    cl = corr_length_cov(z_grid)
    for i in range(S.shape[1]):
        for j in range(S.shape[0]):
            cl_mean = (cl[i] + cl[j]) / 2
            s = (S[j, j] + S[i, i]) / 2
            S[i, j] = s * np.exp(-np.abs(z_grid[i] - z_grid[j]) / cl_mean)

    return S


def a_priori_cov(z, trpp=12.5e3, strp=25e3, pblp=2e3):
    """Return a priori covariance values for given altitudes.

    Parameters:
        z (np.array): Height levels [m]
        trpp (float): Tropopause height [m]
        strp (float): Stratopuase height [m]

    Returns:
        np.array: A priori covariance for each height level.

    """
    f = np.poly1d(np.polyfit(x=(trpp, strp), y=(1, 0.25), deg=1))
    apc = f(z)
    apc[z < trpp] = 1
    apc[z > strp] = 0.25
    if pblp > 0:
        f = np.poly1d(np.polyfit(x=(0, pblp), y=(0.1, 1), deg=1))
        apc[z < pblp] = f(z[z < pblp])

    return apc


def covmat_water_vapor(z):
    """Calculate the a priori covariance matrix for water vapor."""

    S = np.identity(z.size) * a_priori_cov(z)
    cl = corr_length_cov(z)
    for i in range(S.shape[1]):
        for j in range(S.shape[0]):
            cl_mean = (cl[i] + cl[j]) / 2
            s = (S[j, j] + S[i, i]) / 2
            S[i, j] = s * np.exp(-np.abs(z[i] - z[j]) / cl_mean)

    return S

def covmat_temperature(z, p, atm_batch_path):
    """Calculate the a priori covariance matrix for temperature."""
    atm_batch = interpolate_profiles(p_grid=p, atm_batch_path=atm_batch_path)
    T_profile_stack = np.array([np.squeeze(atm.get('T')) for atm in atm_batch]).T

    cov = np.cov(T_profile_stack)
    S = cov * np.eye(len(cov))
    # S = np.eye(len(cov)) * np.mean(np.diag(S)) # FIXME: Quick change to constant mean covariance

    cl = corr_length_cov(z)
    for i in range(S.shape[1]):
        for j in range(S.shape[0]):
            cl_mean = (cl[i] + cl[j]) / 2
            s = (S[j, j] + S[i, i]) / 2
            S[i, j] = s * np.exp(-np.abs(z[i] - z[j]) / cl_mean)

    return S

def interpolate_profiles(p_grid, atm_batch_path):
    """
    Read batch of atmospheric fields and interpolate them onto
    given p_grid.

    Parameters:
        p_grid (np.array): Desired pressure grid.
        atm_batch_path: File path of atmospheric data batch.

    Returns:
        list: batch of atmospheric fields, interpolated on new p_grid.
    """
    atm_batch = xml.load(atm_batch_path)
    for atm_field in atm_batch:
        p = atm_field.grids[1]
        for var in atm_field.grids[0]:
            var_data = atm_field.get(var)
            f = interp1d(p, np.squeeze(var_data), fill_value='extrapolate', kind='nearest')
            atm_field.set(key=var, data=f(p_grid).reshape(var_data.shape))
        atm_field.grids[1] = p_grid

    return atm_batch


def save_covariances(z, p, atm_batch_path, abs_species=None, fmt='binary', path='a_priori/'):
    """Calculate and save common a priori covariances.

    Supported quantities:
        * Temperature (T)
        * Water Vapor ((abs_species_H2O), (H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252))

    Parameters:
        abs_species (list): List of absorption species.
        fmt (str): Output format: 'ascii' or 'binary' (default).

    """
    cov_functions = {
        'T': covmat_temperature(z, p, atm_batch_path),
        'abs_species-H2O': covmat_water_vapor(z),
        'H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252': covmat_water_vapor(z),
        }

    if abs_species is None:
        abs_species = cov_functions.keys()

    for species in abs_species:
        S = cov_functions[species]
        species_strings = {
            'T': 't',
            'H2O, H2O-SelfContCKDMT252, H2O-ForeignContCKDMT252': 'h2o_vmr',
            'abs_species-H2O': 'h2o_vmr',
        }
        xml.save(S, path + 'covariance_{}.xml'.format(species_strings[species]), format=fmt)