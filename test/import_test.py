"""
Verify that imports of Bilby and associated sampler package behave as expected.

This script does two things:
- tests the number of packages imported with a top-level :code:`import bilby`
  statement.
- tests that all of the implemented samplers can be initialized.
  The :code:`FakeSampler` is omitted as that doesn't require importing
  any package.
"""

import sys

import bilby  # noqa

unique_packages = set(sys.modules)

unwanted = {
    "lal", "lalsimulation", "matplotlib",
    "h5py", "dill", "tqdm", "tables", "deepdish", "corner",
}

for filename in ["sampler_requirements.txt", "optional_requirements.txt"]:
    with open(filename, "r") as ff:
        packages = ff.readlines()
        for package in packages:
            package = package.split(">")[0].split("<")[0].split("=")[0].strip()
            unwanted.add(package)

if not unique_packages.isdisjoint(unwanted):
    raise ImportError(
        f"{' '.join(unique_packages.intersection(unwanted))} imported with Bilby"
    )

bilby.core.utils.logger.setLevel("ERROR")
IMPLEMENTED_SAMPLERS = bilby.core.sampler.IMPLEMENTED_SAMPLERS
likelihood = bilby.core.likelihood.Likelihood(dict())
priors = bilby.core.prior.PriorDict(dict(a=bilby.core.prior.Uniform(0, 1)))
for sampler in IMPLEMENTED_SAMPLERS:
    if sampler == "fake_sampler":
        continue
    sampler_class = IMPLEMENTED_SAMPLERS[sampler]
    sampler = sampler_class(likelihood=likelihood, priors=priors)
