import copy
import threading
import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

from mach3sbitools.utils import get_logger

from .parameter_handler import ParameterHandler
from .sample_handler import SampleHandler

logger = get_logger("tutorial")


# ---------------------------------------------------------------------------
# Thread / process-safe handler wrapper
# ---------------------------------------------------------------------------


class _ThreadSafeHandlers:
    """Wraps existing handler instances, providing per-thread copies.

    Accepts already-constructed handlers (as in a notebook context) and
    ensures each thread or subprocess gets its own independent copy via
    :func:`copy.deepcopy`, so mutable numpy state is never shared.

    The main thread always operates on the originals — no copy is made unless
    a second thread or a worker process requests access.

    :param ph: Fully initialised parameter handler.
    :param sh: Fully initialised sample handler.
    """

    def __init__(self, ph: ParameterHandler, sh: SampleHandler) -> None:
        self._source_ph = ph
        self._source_sh = sh
        self._local = threading.local()

    def _ensure(self) -> None:
        if not hasattr(self._local, "ph"):
            logger.debug("Copying handlers for thread %s", threading.get_ident())
            self._local.ph = copy.deepcopy(self._source_ph)
            self._local.sh = copy.deepcopy(self._source_sh)
            # Re-link the sample handler's parameter handler to the local copy
            self._local.sh.parameter_handler = self._local.ph

    @property
    def ph(self) -> ParameterHandler:
        self._ensure()
        return cast(ParameterHandler, self._local.ph)

    @property
    def sh(self) -> SampleHandler:
        self._ensure()
        return cast(SampleHandler, self._local.sh)

    # On pickling into a worker process, ship only the source handlers.
    # Each worker then gets its own deepcopy on first access.
    def __getstate__(self) -> dict:
        return {
            "_source_ph": self._source_ph,
            "_source_sh": self._source_sh,
        }

    def __setstate__(self, state: dict) -> None:
        self._source_ph = state["_source_ph"]
        self._source_sh = state["_source_sh"]
        self._local = threading.local()


# ---------------------------------------------------------------------------
# Blackbox Op
# ---------------------------------------------------------------------------


class _PoissonLogLikeOp(Op):
    """Thin pytensor ``Op`` that evaluates the Poisson log-likelihood.

    PyMC's graph cannot differentiate through :class:`SampleHandler`, so this
    ``Op`` is intentionally gradient-free.  Pair it with a gradient-free step
    method such as :class:`pymc.DEMetropolisZ` or :class:`pymc.Metropolis`.

    Handler access is routed through :class:`_ThreadSafeHandlers`, so the
    ``Op`` is safe to pickle into worker processes or call from multiple
    threads concurrently.

    :param ph: Fully initialised parameter handler.
    :param sh: Fully initialised sample handler.
    """

    def __init__(self, ph: ParameterHandler, sh: SampleHandler) -> None:
        super().__init__()
        self._handlers = _ThreadSafeHandlers(ph, sh)

    def make_node(self, *inputs: pt.Variable) -> Apply:
        (theta,) = inputs
        theta = pt.as_tensor_variable(theta)
        return Apply(self, [theta], [pt.dscalar()])

    def perform(self, node: Apply, inputs: Sequence[Any], outputs: list) -> None:
        (theta,) = inputs
        ph = self._handlers.ph
        sh = self._handlers.sh
        ph.set_parameter_values(theta.astype(float))
        sh.reweight()
        loglike = sh.get_likelihood()
        # Must be a 0-d ndarray — pytensor rejects plain scalars
        outputs[0][0] = np.array(
            loglike if np.isfinite(loglike) else -1e30,
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------


class MCMCSampler:
    """Build and run an MCMC model over the physics parameter space.

    Accepts already-constructed handlers, as is typical in a notebook context.
    Internally wraps them in :class:`_ThreadSafeHandlers` so that
    ``cores > 1`` is safe to pass to :meth:`sample` without any shared-state
    issues across threads or worker processes.

    Priors
    ------
    * **Flat** parameters receive a :class:`pymc.Uniform` prior bounded by
      the parameter's ``bounds``.
    * **Gaussian** parameters are modelled jointly as a
      :class:`pymc.MvNormal` using the full covariance sub-block from
      :class:`ParameterHandler`, so inter-parameter correlations are
      respected.  Hard bounds are enforced via a :class:`pymc.Potential`.

    :param sample_handler: Fully initialised sample handler.
    :param parameter_handler: Fully initialised parameter handler.
    """

    def __init__(
        self,
        sample_handler: SampleHandler,
        parameter_handler: ParameterHandler,
    ) -> None:
        self._ph = parameter_handler
        self._sh = sample_handler

        self._flat_idx: list[int] = [
            i for i in range(self._ph.n_params) if self._ph.get_is_flat(i)
        ]
        self._gauss_idx: list[int] = [
            i for i in range(self._ph.n_params) if not self._ph.get_is_flat(i)
        ]

        logger.info(
            "MCMCSampler: %d flat, %d Gaussian parameters",
            len(self._flat_idx),
            len(self._gauss_idx),
        )

        self._loglike_op = _PoissonLogLikeOp(parameter_handler, sample_handler)
        self.model: pm.Model = self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> pm.Model:
        ph = self._ph

        with pm.Model() as model:
            param_vars: list[pt.TensorVariable | None] = [None] * ph.n_params

            # ── Flat / uniform priors ──────────────────────────────────
            for i in self._flat_idx:
                lo, hi = ph.get_bounds(i)
                param_vars[i] = pm.Uniform(
                    ph.get_name(i),
                    lower=lo,
                    upper=hi,
                    initval=ph.get_nominal(i),
                )

            # ── Correlated Gaussian prior (MvNormal) ───────────────────
            if self._gauss_idx:
                g_idx = np.array(self._gauss_idx)
                mu_g = ph.nominal_values[g_idx]
                cov_g = ph.covariance_matrix[np.ix_(g_idx, g_idx)]
                lo_g = np.array([ph.get_bounds(i)[0] for i in self._gauss_idx])
                hi_g = np.array([ph.get_bounds(i)[1] for i in self._gauss_idx])

                gauss_vec = pm.MvNormal(
                    "gauss_params",
                    mu=mu_g,
                    cov=cov_g,
                    initval=mu_g,
                    shape=len(self._gauss_idx),
                )

                # Hard bounds — return -inf outside the allowed region
                pm.Potential(
                    "gauss_bounds",
                    pt.switch(
                        pt.all(pt.ge(gauss_vec, lo_g) & pt.le(gauss_vec, hi_g)),
                        pt.as_tensor(0.0, dtype="float64"),
                        pt.as_tensor(-np.inf, dtype="float64"),
                    ),
                )

                # Named deterministics so individual params appear in the trace
                for k, i in enumerate(self._gauss_idx):
                    param_vars[i] = pm.Deterministic(ph.get_name(i), gauss_vec[k])

            # ── Assemble full θ vector and attach likelihood ───────────
            theta = pt.stack(cast(list[pt.TensorVariable], param_vars))
            pm.Potential("poisson_loglike", self._loglike_op(theta))

        return model

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        draws: int = 2_000,
        tune: int = 1_000,
        chains: int = 2,
        cores: int | None = None,
        step: pm.step_methods.arraystep.BlockedStep | None = None,
        random_seed: int | Sequence[int] | None = 42,
        **kwargs,
    ):
        """Run MCMC and return an ArviZ-compatible ``InferenceData`` trace.

        Uses :class:`pymc.DEMetropolisZ` by default — it adapts the proposal
        covariance from chain history, which works well for the correlated
        Gaussian block.  Pass ``step=pm.Metropolis()`` to fall back to
        independent Metropolis-Hastings.

        :param draws: Number of posterior draws per chain.
        :param tune: Number of tuning (burn-in) steps per chain.
        :param chains: Number of independent chains.
        :param cores: Number of parallel processes.  Safe to set ``> 1``
            thanks to :class:`_ThreadSafeHandlers`.  Defaults to PyMC's
            own default (usually ``min(chains, cpu_count)``).
        :param step: PyMC step method.  Defaults to
            :class:`pymc.DEMetropolisZ`.
        :param random_seed: Seed(s) forwarded to :func:`pymc.sample`.
        :param kwargs: Forwarded verbatim to :func:`pymc.sample`.
        :returns: ArviZ ``InferenceData`` trace.
        """
        logger.info("Starting MCMC: %d draws, %d tune, %d chains", draws, tune, chains)

        with self.model:
            if step is None:
                step = pm.DEMetropolisZ()

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in exp",
                    category=RuntimeWarning,
                )
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    step=step,
                    random_seed=random_seed,
                    **kwargs,
                )

        return trace

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_posterior_array(self, trace) -> np.ndarray:
        """Flatten the trace into a ``(n_samples, n_params)`` array.

        Samples are stacked chain-first so the ordering is deterministic.

        :param trace: ArviZ ``InferenceData`` object returned by
            :meth:`sample`.
        :returns: Posterior samples, shape ``(n_chains * n_draws, n_params)``.
        :rtype: np.ndarray
        """
        ph = self._ph
        posterior = trace.posterior
        n_total = posterior.sizes["chain"] * posterior.sizes["draw"]

        out = np.empty((n_total, ph.n_params))
        for i in range(ph.n_params):
            out[:, i] = posterior[ph.get_name(i)].values.reshape(n_total)

        return out

    def summary(self, trace) -> None:
        """Print a quick ArviZ summary for all named parameters.

        :param trace: ArviZ ``InferenceData`` object returned by
            :meth:`sample`.
        """
        import arviz as az

        var_names = [self._ph.get_name(i) for i in range(self._ph.n_params)]
        print(az.summary(trace, var_names=var_names))
