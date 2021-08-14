import numpy as np

from ..utils import logger
from .dynesty import Dynesty, _log_likelihood_wrapper, _prior_transform_wrapper


class DynamicDynesty(Dynesty):
    """
    bilby wrapper of `dynesty.DynamicNestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.DynamicNestedSampler`, see
    documentation for that class for further help. Under Other Parameter below,
    we list commonly all kwargs and the bilby defaults.

    Parameters
    ==========
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments

    Other Parameters
    ------==========
    bound: {'none', 'single', 'multi', 'balls', 'cubes'}, ('multi')
        Method used to select new points
    sample: {'unif', 'rwalk', 'slice', 'rslice', 'hslice'}, ('rwalk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int
        Number of walks taken if using `sample='rwalk'`, defaults to `ndim * 5`
    verbose: Bool
        If true, print information information about the convergence during
    check_point: bool,
        If true, use check pointing.
    check_point_delta_t: float (600)
        The approximate checkpoint period (in seconds). Should the run be
        interrupted, it can be resumed from the last checkpoint. Set to
        `None` to turn-off check pointing
    n_check_point: int, optional (None)
        The number of steps to take before check pointing (override
        check_point_delta_t).
    resume: bool
        If true, resume run from checkpoint (if available)
    """
    default_kwargs = dict(bound='multi', sample='rwalk',
                          verbose=True,
                          check_point_delta_t=600,
                          first_update=None,
                          npdim=None, rstate=None, queue_size=None, pool=None,
                          use_pool=None,
                          logl_args=None, logl_kwargs=None,
                          ptform_args=None, ptform_kwargs=None,
                          enlarge=None, bootstrap=None, vol_dec=0.5, vol_check=2.0,
                          facc=0.5, slices=5,
                          walks=None, update_interval=0.6,
                          nlive_init=500, maxiter_init=None, maxcall_init=None,
                          dlogz_init=0.01, logl_max_init=np.inf, nlive_batch=500,
                          wt_function=None, wt_kwargs=None, maxiter_batch=None,
                          maxcall_batch=None, maxiter=None, maxcall=None,
                          maxbatch=0, stop_function=None, stop_kwargs=None,
                          use_stop=True, save_bounds=True,
                          print_progress=True, print_func=None, live_points=None,
                          maxmcmc=1000, nact=5, mcmc_scale="normal", adapt_tscale=100,
                          )

    @property
    def external_sampler_name(self):
        return 'dynesty'

    @property
    def sampler_function_kwargs(self):
        keys = ['nlive_init', 'maxiter_init', 'maxcall_init', 'dlogz_init',
                'logl_max_init', 'nlive_batch', 'wt_function', 'wt_kwargs',
                'maxiter_batch', 'maxcall_batch', 'maxiter', 'maxcall',
                'maxbatch', 'stop_function', 'stop_kwargs', 'use_stop',
                'save_bounds', 'print_progress', 'print_func', 'live_points']
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler(self):
        import dynesty
        if self._sampler is None:
            self._sampler = dynesty.DynamicNestedSampler(
                loglikelihood=_log_likelihood_wrapper,
                prior_transform=_prior_transform_wrapper,
                ndim=self.ndim, **self.sampler_init_kwargs
            )
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        self._sampler = sampler

    def run_sampler(self):
        self.kwargs["nlive"] = self.kwargs["nlive_init"]
        return super(DynamicDynesty, self).run_sampler()

    def _run_external_sampler_with_checkpointing(self):
        return super(DynamicDynesty, self)._run_external_sampler_with_checkpointing()

    def _check_converged(self):
        from dynesty.dynamicsampler import stopping_function
        if self.kwargs.get("stop_function", None) is None:
            stop_function = stopping_function
        else:
            stop_function = stopping_function
        stop = stop_function(
            self.sampler.results, self.kwargs["stop_kwargs"],
            rstate=np.random,
            M=map,
            return_vals=False
        )
        return stop

    def _run_nested_wrapper(self, kwargs):
        """ Wrapper function to run_nested

        This wrapper catches exceptions related to different versions of
        dynesty accepting different arguments.

        Parameters
        ==========
        kwargs: dict
            The dictionary of kwargs to pass to run_nested

        """
        logger.debug("Calling run_nested with sampler_function_kwargs {}"
                     .format(kwargs))
        kwargs["maxbatch"] += 1
        self.sampler.run_nested(**kwargs)

    def write_current_state(self):
        super(DynamicDynesty, self).write_current_state()
        self.sampler.sampler.pool = self.pool
        if self.kwargs["rstate"] is None:
            self.sampler.sampler.rstate = np.random
        else:
            self.sampler.sampler.rstate = self.kwargs["rstate"]
        if self.sampler.sampler.pool is not None:
            self.sampler.sampler.M = self.sampler.pool.map

    def _print_func(
            self, results, niter, ncall=None, dlogz=None, nbatch=0, stop_val=None,
            logl_min=-np.inf, logl_max=np.inf,
            *args, **kwargs
    ):
        """ Replacing status update for dynesty.result.print_func """

        # Extract results at the current iteration.
        (worst, ustar, vstar, loglstar, logvol, logwt,
         logz, logzvar, h, nc, worst_it, boundidx, bounditer,
         eff, delta_logz) = results

        # Adjusting outputs for printing.
        if delta_logz > 1e6:
            delta_logz = np.inf
        if 0. <= logzvar <= 1e6:
            logzerr = np.sqrt(logzvar)
        else:
            logzerr = np.nan
        if logz <= -1e6:
            logz = -np.inf
        if loglstar <= -1e6:
            loglstar = -np.inf

        if self.use_ratio:
            key = 'logz-ratio'
        else:
            key = 'logz'

        # Constructing output.
        string = []
        string.append("bound:{:d}".format(bounditer))
        string.append("nc:{:3d}".format(nc))
        string.append("ncall:{:.1e}".format(ncall))
        string.append("eff:{:0.1f}%".format(eff))
        string.append("{}={:0.2f}+/-{:0.2f}".format(key, logz, logzerr))
        if dlogz is None:
            string.append("logl:{:.3e}<{:.3e}<{:.3e}".format(logl_min, loglstar, logl_max))
            string.append("nbatch:{:d}".format(nbatch))
            string.append("stop:{:.3f}".format(stop_val))
        else:
            string.append("dlogz:{:0.3f}>{:0.2g}".format(delta_logz, dlogz))

        self.pbar.set_postfix_str(" ".join(string), refresh=False)
        self.pbar.update(niter - self.pbar.n)
