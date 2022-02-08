#!/usr/bin/env python
"""
Read ROQ posterior and calculate full likelihood at same parameter space points.
"""

import numpy as np
import bilby


outdir = "outdir"
label = "full"

np.random.seed(170808)

duration = 4
sampling_frequency = 1024

injection_parameters = dict(
    chirp_mass=36.0,
    mass_ratio=0.1,
    a_1=0.8,
    a_2=0.3,
    tilt_1=0.0,
    tilt_2=0.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=0.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=20.0,
    minimum_frequency=20.0,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    waveform_arguments=waveform_arguments,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

ifos.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 3,
)

ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "theta_jn",
    "psi",
    "ra",
    "dec",
    "phi_12",
    "phi_jl",
    "luminosity_distance",
]:
    priors[key] = injection_parameters[key]
del priors["mass_1"], priors["mass_2"]
priors["chirp_mass"] = bilby.core.prior.Uniform(30, 40, latex_label="$\\mathcal{M}$")
priors["mass_ratio"] = bilby.core.prior.Uniform(0.05, 0.25, latex_label="$q$")
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1,
    injection_parameters["geocent_time"] + 0.1,
    latex_label="$t_c$",
    unit="s",
)

waveform_generator.waveform_arguments["waveform_approximant"] = "IMRPhenomXP"

likelihood_1 = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

result_1 = bilby.run_sampler(
    likelihood=likelihood_1,
    priors=priors,
    sampler="nestle",
    nlive=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=f"{label}_XP",
    save="hdf5",
)

waveform_generator.waveform_arguments["waveform_approximant"] = "IMRPhenomXPHM"

likelihood_2 = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)

sample_file = f"{result_1.outdir}/{result_1.label}_result.hdf5"

result_2 = bilby.run_sampler(
    likelihood=likelihood_2,
    priors=priors,
    sampler="fake_sampler",
    sample_file=sample_file,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=f"{label}_XPHM",
    save="hdf5",
    plot=True,
    verbose=True,
)

bilby.core.result.plot_multiple([result_1, result_2])
