import pytest
import torch
import numpy as np
from scipy import stats

from mach3sbitools.inference import InferenceHandler
from dummy_simulator import DummySimulator

N_SAMPLES = 10000


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def dummy_sim():
    return DummySimulator(None)


@pytest.fixture(scope="session")
def nominal_observation(dummy_sim):
    """
    A realistic observation: Poisson(1) draw cast to float32,
    matching the dtype the network was trained on.
    """
    rng = np.random.default_rng(42)
    return rng.poisson(lam=1, size=len(dummy_sim.get_data_bins())).astype(np.float32).tolist()

@pytest.fixture(scope="session")
def trained_handler(prior_save, dummy_data_dir, posterior_config, training_config):
    """A fully trained handler, created once for the whole session."""
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    handler.load_training_data()
    handler.create_posterior(posterior_config)
    handler.train_posterior(training_config)
    return handler


@pytest.fixture(scope="session")
def samples(trained_handler, nominal_observation):
    """Posterior samples conditioned on a realistic Poisson(1) observation."""
    return trained_handler.sample_posterior(N_SAMPLES, nominal_observation)


# ── Setup / guard tests ────────────────────────────────────────────────────────

def test_no_dataset_load(prior_save):
    """load_training_data should raise if set_dataset was never called."""
    handler = InferenceHandler(prior_save)
    with pytest.raises(ValueError):
        handler.load_training_data()


def test_no_posterior_train_without_create(prior_save, dummy_data_dir, training_config):
    """train_posterior should raise if create_posterior was never called."""
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    handler.load_training_data()
    with pytest.raises(ValueError):
        handler.train_posterior(training_config)


def test_set_dataset(prior_save, dummy_data_dir):
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    assert handler.dataset is not None


def test_load_training_data(prior_save, dummy_data_dir):
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    handler.load_training_data()
    assert handler._tensor_dataset is not None


def test_create_posterior(prior_save, dummy_data_dir, posterior_config):
    handler = InferenceHandler(prior_save)
    handler.set_dataset(dummy_data_dir)
    handler.load_training_data()
    handler.create_posterior(posterior_config)
    assert handler.inference is not None


def test_train(trained_handler):
    assert trained_handler._density_estimator is not None


# ── Output shape / type ────────────────────────────────────────────────────────

def test_total_samples(samples):
    assert len(samples) == N_SAMPLES


def test_samples_shape(samples, trained_handler):
    n_params = trained_handler.prior.n_params
    assert samples.shape == (N_SAMPLES, n_params), (
        f"Expected ({N_SAMPLES}, {n_params}), got {samples.shape}"
    )


def test_samples_are_tensor(samples):
    assert isinstance(samples, torch.Tensor)


def test_samples_finite(samples):
    assert torch.all(torch.isfinite(samples)), "Posterior samples contain NaN or Inf"


def test_samples_within_prior_bounds(samples, trained_handler):
    lower = trained_handler.prior.prior_data.lower_bounds.cpu()
    upper = trained_handler.prior.prior_data.upper_bounds.cpu()
    s = samples.cpu()
    assert torch.all(s >= lower), "Some samples are below prior lower bounds"
    assert torch.all(s <= upper), "Some samples are above prior upper bounds"


# ── Posterior quality ──────────────────────────────────────────────────────────

def test_posterior_mean_near_nominal(samples, trained_handler):
    """
    The dummy simulator output is independent of theta, so the posterior
    should not update strongly away from the prior nominal.
    """
    nominals = trained_handler.prior.prior_data.nominals.cpu()
    posterior_mean = samples.cpu().mean(dim=0)
    prior_std = trained_handler.prior.variance.cpu().sqrt()
    deviation = (posterior_mean - nominals).abs()
    assert torch.all(deviation < 3 * prior_std), (
        f"Posterior mean deviates >3σ from nominal:\n"
        f"  nominals:       {nominals}\n"
        f"  posterior mean: {posterior_mean}\n"
        f"  deviations:     {deviation}"
    )


def test_posterior_is_narrower_than_prior(samples, trained_handler):
    prior_var = trained_handler.prior.variance.cpu()
    posterior_var = samples.cpu().var(dim=0)
    n_narrower = (posterior_var < prior_var).sum().item()
    n_params = trained_handler.prior.n_params
    assert n_narrower >= n_params // 2, (
        f"Expected posterior narrower than prior for ≥half of parameters, "
        f"got {n_narrower}/{n_params}"
    )


def test_posterior_reproducible_with_same_seed(trained_handler, nominal_observation):
    torch.manual_seed(42)
    samples_a = trained_handler.sample_posterior(100, nominal_observation)

    torch.manual_seed(42)
    samples_b = trained_handler.sample_posterior(100, nominal_observation)

    assert torch.allclose(samples_a, samples_b), (
        "Posterior sampling is not reproducible with the same seed"
    )


# ── Checkpoint / save-load ─────────────────────────────────────────────────────

def test_save_and_load_posterior(trained_handler, tmp_path, prior_save,
                                  posterior_config, nominal_observation):
    ckpt_path = tmp_path / "density_estimator.pt"
    torch.save(trained_handler._density_estimator.state_dict(), ckpt_path)

    # Infer correct dims from the trained handler's tensor dataset
    theta_dim = trained_handler._tensor_dataset.tensors[0].shape[1]
    x_dim = trained_handler._tensor_dataset.tensors[1].shape[1]

    loaded_handler = InferenceHandler(prior_save)
    loaded_handler.create_posterior(posterior_config)
    density_estimator = loaded_handler.inference._build_neural_net(
        torch.zeros(2, theta_dim),
        torch.zeros(2, x_dim),
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    density_estimator.load_state_dict(state_dict)
    density_estimator.eval()
    loaded_handler._density_estimator = density_estimator

    original_samples = trained_handler.sample_posterior(500, nominal_observation).cpu().numpy()
    loaded_samples = loaded_handler.sample_posterior(500, nominal_observation).cpu().numpy()

    for i in range(original_samples.shape[1]):
        ks_stat, _ = stats.ks_2samp(original_samples[:, i], loaded_samples[:, i], method='asymp')
        assert ks_stat < 0.1, (
            f"Parameter {i}: loaded posterior differs from original (KS={ks_stat:.3f})"
        )

def test_x_dtype_matches_training(trained_handler, nominal_observation):
    """
    The dtype of the observation passed to sample_posterior must match
    the dtype of x in the tensor dataset the network was trained on.
    """
    trained_x_dtype = trained_handler._tensor_dataset.tensors[1].dtype
    x_tensor = torch.tensor([nominal_observation])
    assert x_tensor.dtype == trained_x_dtype, (
        f"Observation dtype {x_tensor.dtype} does not match "
        f"training data dtype {trained_x_dtype}. "
        f"Fix nominal_observation fixture or sample_posterior dtype cast."
    )

def test_x_shape_matches_training(trained_handler, nominal_observation):
    """
    The shape of a single observation must match the x event_shape
    the network was trained on.
    """
    trained_x_shape = trained_handler._tensor_dataset.tensors[1].shape[1:]
    x_tensor = torch.tensor([nominal_observation])
    assert x_tensor.shape[1:] == trained_x_shape, (
        f"Observation shape {x_tensor.shape[1:]} does not match "
        f"training x shape {trained_x_shape}."
    )