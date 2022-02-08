from copy import deepcopy

import numpy as np

from .base_sampler import Sampler
from ..result import read_in_result
from ..utils.log import logger
from ..utils.plotting import latex_plot_format


class FakeSampler(Sampler):
    r"""
    A "fake" sampler that evaluates the likelihood at a list of
    configurations read from a posterior data file.

    See base class for parameters. Added parameters are described below.

    Weights are calculated for each posterior sample as

    .. math::

        w_{i} = \frac{
            {\cal L}_{\rm new}(d | \theta_{i}) \pi_{\rm new}(\theta_{i})
        }{
            {\cal L}_{\rm old}(d | \theta_{i}) \pi_{\rm old}(\theta_{i})
        }.

    Here :math:`{\cal L}` is the likelihood and :math`\pi` is the prior
    distribution.

    Following this, the Bayes factor comparing the two models is computed as

    .. math::

        \ln BF = \left< w_{i} \right> +/- \sigma_{w}. \\
        \sigma^{2}_{w} = \frac{1}{N}
            \frac{\left< w^{2}_{i} - \left< w_{i} \right>^{2}\right>}
            {\left< w_{i} \right>^{2}}.

    The posterior samples are rejection sampled, i.e., kept with probability
    :math:`w_{i}`.

    Parameters
    ==========
    sample_file: str
        A string pointing to the posterior data file to be loaded.
    """

    default_kwargs = dict(verbose=True)

    def __init__(
        self,
        likelihood,
        priors,
        sample_file,
        outdir="outdir",
        label="label",
        plot=False,
        **kwargs,
    ):
        kwargs["use_ratio"] = False
        super(FakeSampler, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            plot=plot,
            skip_import_verification=True,
            **kwargs,
        )
        self._read_parameter_list_from_file(sample_file)
        self.result.outdir = outdir
        self.result.label = label
        self.changed_priors = self._old_result.priors != priors

    def _read_parameter_list_from_file(self, sample_file):
        """Read a list of sampling parameters from file.

        The sample_file should be in bilby posterior HDF5 format.
        """
        self._old_result = read_in_result(filename=sample_file)
        self.result = deepcopy(self._old_result)

    def run_sampler(self):
        """Compute the likelihood for the list of parameter space points."""
        likelihood_ratios = list()
        ln_weights = list()
        posterior = self.result.posterior[self.priors.keys()]
        old_ln_likelihoods = self._old_result.log_likelihood_evaluations

        for ii in np.arange(posterior.shape[0]):
            sample = posterior.iloc[ii]

            self.likelihood.parameters = sample.to_dict()
            old_ln_prob = old_ln_likelihoods[ii]
            if self.result.use_ratio:
                new_ln_prob = self.likelihood.log_likelihood_ratio()
            else:
                new_ln_prob = self.likelihood.log_likelihood()
            likelihood_ratios.append(new_ln_prob)
            sample.log_likelihood = new_ln_prob
            if self.changed_priors:
                old_ln_prob += self._old_result.priors.ln_prob(sample)
                new_ln_prob += self.priors.ln_prob(sample)
            ln_weights.append(new_ln_prob - old_ln_prob)

            logger.debug(
                f"iteration: {ii}: old ln L: {old_ln_prob}, new ln L: {old_ln_prob}, "
                f"delta ln L: {new_ln_prob - old_ln_prob}"
            )
        ln_weights = np.array(ln_weights)
        max_ln_ratio = max(ln_weights)
        normalized = np.exp(ln_weights - max_ln_ratio)
        ln_bayes_factor = np.log(np.mean(normalized)) + max_ln_ratio
        variance = (
            (np.mean(normalized**2) - np.mean(normalized) ** 2)
            / np.mean(normalized) ** 2
            / len(ln_weights)
        )

        self.result.log_likelihood_evaluations = np.array(likelihood_ratios)

        keep = normalized > np.random.uniform(0, 1, len(normalized))
        self.result.posterior = self.result.posterior[keep]

        logger.info(
            f"Resampling has ln Bayes Factor: {ln_bayes_factor:.2f} +/- {variance ** 0.5:.2f}."
        )
        logger.info(f"Keeping {sum(keep)} of {len(keep)} samples.")

        if not all(
            [self.result.log_bayes_factor is np.nan, self.result.log_evidence is np.nan]
        ):
            self.result.log_evidence += ln_bayes_factor
            self.result.log_bayes_factor += ln_bayes_factor
            self.result.log_evidence_err = (
                self.result.log_evidence_err**2 + variance
            ) ** 0.5

        if self.plot:
            self.make_comparison_histograms()

        self.result.sampling_time = None

        return self.result

    @latex_plot_format
    def make_comparison_histograms(self):
        """
        Plot comparison histograms of the new and old log likelihoods.
        """
        import matplotlib.pyplot as plt

        ln_l_old = self._old_result.log_likelihood_evaluations
        ln_l_new = self.result.log_likelihood_evaluations
        old_label = self._old_result.label
        new_label = self.label

        _label = "\\ln {{\\cal L}}_{{\\rm {}}}"

        plt.figure()
        plt.hist(ln_l_old, bins=50, label=old_label, histtype="step")
        plt.hist(ln_l_new, bins=50, label=new_label, histtype="step")
        plt.xlabel(r"$\ln {\cal L}$")
        plt.legend(loc=2)
        plt.savefig(f"{self.outdir}/{self.label}_logl.pdf")
        plt.close()

        plt.figure()
        delta_logl = ln_l_new - ln_l_old
        plt.hist(delta_logl, bins=50)
        plt.xlabel(f"${_label.format(new_label)} - {_label.format(old_label)}$")
        plt.savefig(f"{self.outdir}/{self.label}_delta_logl.pdf")
        plt.close()

        plt.figure()
        delta_logl = np.abs(delta_logl)
        bins = np.logspace(np.log10(delta_logl.min()), np.log10(delta_logl.max()), 25)
        plt.hist(delta_logl, bins=bins)
        plt.xscale("log")
        plt.xlabel(f"$|{_label.format(new_label)}  - {_label.format(old_label)}|")
        plt.savefig(f"{self.outdir}/{self.label}_abs_delta_logl.pdf")
        plt.close()
