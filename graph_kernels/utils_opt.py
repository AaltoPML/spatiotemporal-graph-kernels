import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np

from . import utils


gpflow.config.set_default_float(tf.float64)
f64 = gpflow.utilities.to_default_float


def optimize_ada_natgrad(gprocess, train_X, train_y, test_X, test_y, n_iter, learning_rate=1e-2,
                         transformer=None):
    decayed_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, 5000, 0.5, staircase=False, name=None
    )
    optimizer = tf.optimizers.Adam(decayed_lr)

    gpflow.set_trainable(gprocess.q_mu, False)
    gpflow.set_trainable(gprocess.q_sqrt, False)
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)

    result = {}
    for epoch in range(n_iter):
        elbo = -utils.training_step(
            train_X, train_y, optimizer, gprocess, natgrad_opt).numpy()

        mape = utils.evaluate_mape(test_X, test_y, gprocess, transformer)
        mae = utils.evaluate_mae(test_X, test_y, gprocess, transformer)
        result[epoch] = {
            "ELBO": elbo,
            "MAPE": mape,
            "MAE": mae,
        }
        print(f"{epoch}:\tELBO: {elbo:.5f}\tMAPE: {mape:.10f}\tMAE: {mae:.10f}")

    return result, gprocess


def optimize_lbfgs_b(gprocess, train_X, train_y, test_X, test_y, n_iter, transformer=None, compile=False):
    optimizer = gpflow.optimizers.Scipy()
    loss_fn = gprocess.training_loss_closure((train_X, train_y), compile=compile)
    callback = utils.Callback(
        gprocess, train_X, train_y, test_X, test_y,
        transformer=transformer)
    optimizer.minimize(
        loss_fn,
        variables=gprocess.trainable_variables,
        compile=compile,
        options=dict(disp=True, maxiter=n_iter),
        step_callback=callback,
    )

    mape = utils.evaluate_mape(test_X, test_y, gprocess, transformer=transformer)
    mae = utils.evaluate_mae(test_X, test_y, gprocess, transformer=transformer)
    elbo = loss_fn()
    result = {"ELBO": elbo.numpy(), "MAPE": mape, "MAE": mae}
    return result, gprocess


def evaluate_kernel_svgp(kernel, train_X, train_y, test_X, test_y, graph, transformer=None,
                         dump_everything=False, dump_directory=None, optimizer_name="Adam", n_iter=2000,
                         mean_function=None, compile=False):
    # Optimizer = Adam or LBFGS
    if mean_function is None:
        mean_function = utils.ConstantArray(len(graph.nodes()))

    gprocess = gpflow.models.SVGP(
        kernel, gpflow.likelihoods.Gaussian(),
        inducing_variable=train_X, mean_function=mean_function, whiten=True, q_diag=False)
    gpflow.set_trainable(gprocess.inducing_variable, False)
    gprocess.likelihood.variance.assign(1e-2)
    if optimizer_name == "Adam":
        result, gprocess = optimize_ada_natgrad(
            gprocess, train_X, train_y,
            test_X, test_y, n_iter=n_iter, transformer=transformer)
    elif optimizer_name == "LBFGS":
        result, gprocess = optimize_lbfgs_b(
            gprocess, train_X, train_y, test_X, test_y, n_iter=n_iter, transformer=transformer,
            compile=compile)
    else:
        raise ValueError("Supported optimizers: Adam & LBFGS")
    return result, gprocess


def initialize_hmc_helpers(model):
    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters,
    )
    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=10, step_size=0.01
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=10, target_accept_prob=0.75, adaptation_rate=0.1
    )
    return hmc_helper, hmc, adaptive_hmc


def evaluate_kernel_mcmc(kernel, train_X, train_y, test_X, test_y, graph,
                         mean_function=None, transformer=None,
                         dump_everything=False, dump_directory=None, optimizer_name="Adam", n_iter=2000,
                         num_burnin_steps=300, num_samples=500, full_mcmc=False):
    if mean_function is None:
        mean_function = utils.ConstantArray(len(graph.nodes()))
    model = gpflow.models.GPR((train_X, train_y), kernel, mean_function, noise_variance=0.01)

    model.likelihood.variance.prior = tfd.Normal(f64(0.0), f64(1e-2))
    for var in kernel.trainable_parameters:
        var.prior = tfd.Gamma(f64(1.0), f64(1.0))
    optimizer = gpflow.optimizers.Scipy()
    loss_fn = model.training_loss_closure(compile=False)
    callback = utils.Callback(
        model, train_X, train_y, test_X, test_y,
        loss_fn=loss_fn, transformer=transformer)
    optimizer.minimize(
        loss_fn, model.trainable_variables, compile=False,
        options=dict(disp=True, maxiter=n_iter), callback=callback)
    if full_mcmc:
        hmc_helper, hmc, adaptive_hmc = initialize_hmc_helpers(model)
        samples, traces = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )
        print("Acceptance rate:", traces.is_accepted.numpy().mean())

        r_hat = tfp.mcmc.potential_scale_reduction(samples)
        print("R-hat diagnostic (per latent variable):", r_hat.numpy())

        #parameter_samples = hmc_helper.convert_to_constrained_values(samples)
        f_samples = utils.get_hmc_sample(num_samples, samples, hmc_helper, model, test_X)
        y_pred = np.median(f_samples, 0)
        mape = utils.evaluate_mape_predictions(y_pred, test_y, transformer)
        mae = utils.evaluate_mae_predictions(y_pred, test_y, transformer)
        elbo = loss_fn()
    else:
        mape = utils.evaluate_mape(test_X, test_y, model, transformer)
        mae = utils.evaluate_mae(test_X, test_y, model, transformer)
        elbo = loss_fn()
    return {"ELBO": elbo.numpy(), "MAPE": mape, "MAE": mae}, model
