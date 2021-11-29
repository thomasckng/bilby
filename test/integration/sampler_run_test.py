import multiprocessing
multiprocessing.set_start_method("fork")  # noqa

import unittest
import pytest
import shutil
import sys

import bilby
import numpy as np


class TestRunningSamplers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        bilby.core.utils.command_line_args.bilby_test_mode = False
        self.x = np.linspace(0, 1, 11)
        self.model = lambda x, m, c: m * x + c
        self.injection_parameters = dict(m=0.5, c=0.2)
        self.sigma = 0.1
        self.y = self.model(self.x, **self.injection_parameters) + np.random.normal(
            0, self.sigma, len(self.x)
        )
        self.likelihood = bilby.likelihood.GaussianLikelihood(
            self.x, self.y, self.model, self.sigma
        )

        self.priors = bilby.core.prior.PriorDict()
        self.priors["m"] = bilby.core.prior.Uniform(0, 5, boundary="periodic")
        self.priors["c"] = bilby.core.prior.Uniform(-2, 2, boundary="reflective")
        bilby.core.utils.check_directory_exists_and_if_not_mkdir("outdir")

    def tearDown(self):
        del self.likelihood
        del self.priors
        bilby.core.utils.command_line_args.bilby_test_mode = False
        shutil.rmtree("outdir")

    def _run_sampler(self, sampler, **kwargs):
        kwargs["resume"] = kwargs.get("resume", False)
        for npool in [1, 2]:
            _ = bilby.run_sampler(
                likelihood=self.likelihood,
                priors=self.priors,
                sampler=sampler,
                save=False,
                npool=npool,
                **kwargs,
            )

    def test_run_cpnest(self):
        self._run_sampler(sampler="cpnest", nlive=100)

    def test_run_dnest4(self):
        self._run_sampler(
            sampler="dnest4",
            max_num_levels=2,
            num_steps=10,
            new_level_interval=10,
            num_per_step=10,
            thread_steps=1,
            num_particles=50,
        )

    def test_run_dynesty(self):
        self._run_sampler(sampler="dynesty", nlive=100)

    def test_run_dynamic_dynesty(self):
        self._run_sampler(
            sampler="dynamic_dynesty",
            nlive_init=100,
            nlive_batch=100,
            dlogz_init=1.0,
            maxbatch=0,
            maxcall=100,
            bound="single",
        )

    def test_run_emcee(self):
        self._run_sampler(sampler="emcee", iterations=1000, nwalkers=10)

    def test_run_kombine(self):
        self._run_sampler(
            sampler="kombine",
            iterations=2000,
            nwalkers=20,
            autoburnin=False,
        )

    def test_run_nestle(self):
        self._run_sampler(sampler="nestle", nlive=100)

    def test_run_nessai(self):
        self._run_sampler(
            sampler="nessai",
            nlive=100,
            poolsize=1000,
            max_iteration=1000,
        )

    def test_run_pypolychord(self):
        pytest.importorskip("pypolychord")
        self._run_sampler(sampler="pypolychord", nlive=100)

    def test_run_ptemcee(self):
        self._run_sampler(
            sampler="ptemcee",
            nsamples=100,
            nwalkers=50,
            burn_in_act=1,
            ntemps=1,
            frac_threshold=0.5,
        )

    @pytest.mark.skipif(sys.version_info[1] <= 6, reason="pymc3 is broken in py36")
    def test_run_pymc3(self):
        self._run_sampler(
            sampler="pymc3",
            draws=50,
            tune=50,
            n_init=250,
        )

    def test_run_pymultinest(self):
        self._run_sampler(sampler="pymultinest", nlive=100)

    def test_run_PTMCMCSampler(self):
        self._run_sampler(
            sampler="PTMCMCsampler",
            Niter=101,
            burn=2,
            isave=100,
        )

    def test_run_ultranest(self):
        # run using NestedSampler (with nlive specified)
        self._run_sampler(sampler="ultranest", nlive=100, resume="overwrite")

        # run using ReactiveNestedSampler (with no nlive given)
        self._run_sampler(sampler='ultranest', resume="overwrite")

    def test_run_bilby_mcmc(self):
        self._run_sampler(sampler="bilby_mcmc", nsamples=200, printdt=1)


if __name__ == "__main__":
    unittest.main()
