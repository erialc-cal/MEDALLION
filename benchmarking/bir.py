""" From explainer.py in https://github.com/vu-minh/mlteam-lime-for-tsne/blob/master/ original BIR repo
Correction in the code: 
- my_kl_divergence_bh() requires num_threads and compute_error with updated sklearn 
- correct rotation for predictino_error

"""
# minhvu
# 15/11/2019
# explain the samples with a linear models

import numpy as np
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence

from sklearn.linear_model import ElasticNet, MultiTaskElasticNetCV


def rotate_matrix_1(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def optimize_linear_model(X, Y, linear_model, space, random_state):
    @use_named_args(space)
    def objective(rotation, alpha, l1_ratio):
        R = rotate_matrix_1(rotation)
        Y_rotated = R.dot(Y.T).T

        linear_model.set_params(alpha=alpha, l1_ratio=l1_ratio)
        linear_model.fit(X, Y_rotated)

        r_square_score = linear_model.score(X, Y_rotated)
        print("score: ", r_square_score)
        return 1.0 - r_square_score

    res_gp = gp_minimize(objective, space, n_calls=50, random_state=random_state)
    return res_gp.x, res_gp.fun


def explain_samples(x_samples, y_samples, linear_model):
    linear_model = ElasticNet()

    space = [
        Real(0, 180, name="rotation"),
        Real(0.1, 2.0, name="alpha"),
        Real(0.5, 1.0, name="l1_ratio"),
    ]
    (rotation, alpha, l1_ratio), score = optimize_linear_model(
        x_samples, y_samples, linear_model, space, random_state=42
    )
    best_score = 1.0 - score

    print(f"[DEBUG]: Best rotation: {rotation} degree with score = {best_score:.3f}")
    print(f"[DEBUG]: alpha = {alpha:.3f}, l1_ratio = {l1_ratio:.3f}")

    R = rotate_matrix_1(rotation)
    y_rotated = R.dot(y_samples.T).T
    linear_model.set_params(alpha=alpha, l1_ratio=l1_ratio)
    linear_model.fit(x_samples, y_rotated)

    best_W = linear_model.coef_
    return best_W, best_score, rotation


def explain_samples_with_cv(
    x_samples, y_samples, linear_model, find_rotation=True, use_weight=True
):
    """Naive explainer by a linear regression (LR) model.
    Note that, we do not use the intercept, so we should standardize the input.
    Args:
        x_samples: [n_samples x D]
        y_samples: [n_samples x 2]
        find_rotation: bool default to True: find the best rotation of `y_samples`
            that minimizes the `scores` of the linear model.
    Returns:
        weights of LR model (representing the importance of each pixel)
    """

    # greedy search for best rotation angle
    best_score = -np.inf
    best_W = None
    best_angle = 0
    best_alpha = None
    best_l1_ratio = None

    linear_model = MultiTaskElasticNetCV(cv=5)

    for d in np.arange(0, 90, 1) if find_rotation else [0]:
        R = rotate_matrix_1(d)
        y_rotated = R.dot(y_samples.T).T

        reg = linear_model.fit(x_samples, y_rotated)
        score = reg.score(x_samples, y_rotated)

        if score >= best_score:
            best_score = score
            best_angle = d
            best_W = reg.coef_
            best_alpha = reg.alpha_
            best_l1_ratio = reg.l1_ratio_

    print(f"[DEBUG]: Best rotation: {best_angle} degree with score = {best_score:.3f}")
    print(f"[DEBUG]: alpha = {best_alpha:.3f}, l1_ratio = {best_l1_ratio:.3f}")

    return best_W, best_score, best_angle


""" From sampling.py in https://github.com/vu-minh/mlteam-lime-for-tsne/blob/master/ original BIR repo"""
# minhvu
# 13/11/2019
# functions for generate samples

import math
# import numpy as np
from scipy.spatial.distance import cdist


def sample_around(x, sigma=0.1, n_samples=1):
    """Sample a new point around the given datapoint `x`
    `x_sampled = x + epsilon`
    epsilon is a isotropic gaussian noise with variance `sigma`
    """
    epsilon = (sigma ** 2) * np.random.randn(n_samples, len(x.shape))
    return x + epsilon


def sample_with_global_noise(x, data, factor=0.5, n_samples=1):
    """Sample with global noise from the multivariate gaussian of HD `data` (denoted as `X`).
    global noise `epsilon` ~ MVN(X.mu, X.sigma^2)
    Given the input point `x`, return `x + epsilon`
    """
    mean = data.mean(axis=0)  # global mean of data according each feature
    sigma = data.var(axis=0)  # sigma is variance, which is square of std
    epsilon = np.random.multivariate_normal(mean=mean, cov=np.diag(sigma), size=n_samples)
    return x + factor * epsilon


def perturb_image(x, replace_rate=(1, 4)):
    """Random remove some pixels in the input image `x`
    The percent of removed pixels is in `replace_rate`.
    """
    replace_rate = 0.1 * np.random.randint(*replace_rate)
    mask = np.random.choice([0, 1], size=x.shape, p=((1 - replace_rate), replace_rate))
    x_new = x.copy()
    x_new[mask.astype(np.bool)] = 0
    return x_new.reshape(1, -1)


def remove_blob(x, n_remove=1):
    """Randomly remove some blobs with random size from the input image `x`
    """
    img_size = int(math.sqrt(len(x)))
    x_new = x.copy().reshape(img_size, img_size)
    for _ in range(n_remove):
        col1, col2 = np.random.randint(0, img_size, size=2)
        row1, row2 = np.random.randint(0, img_size, size=2)
        x_new[min(row1, row2) : max(row1, row2), min(col1, col2) : max(col1, col2)] = 0
    return x_new.reshape(1, -1)


def generate_samples_HD(
    selected_point, sampling_method="sample_around", sigma_HD=1.0, n_samples=100
):
    """Generate `n_samples` "around" the `selected_point`
    """
    sampling_func = {
        "sample_around": partial(sample_around, sigma=sigma_HD),
        "perturb_image": partial(perturb_image, replace_rate=(1, 9)),
        "remove_blob": partial(remove_blob, n_remove=2),
    }[sampling_method]
    return [sampling_func(selected_point) for _ in range(n_samples)]


def generate_samples_SMOTE(selected_id, X, k_nearbors=10, n_samples=100):
    """Mimic SMOTE to generate sample between `selected_id`
        and one of its `k_neighbors`
    """
    # distance from the selected points to all other points in HD
    dist = cdist(X[selected_id].reshape(1, -1), X)
    neighbor_ids = np.argsort(dist)[0, 1 : k_nearbors + 1]
    print("[DEBUG] List neighbors: ", neighbor_ids)

    n_samples_per_pair = n_samples // k_nearbors
    x_samples = []
    for neighbor_idx in neighbor_ids:
        diff = X[neighbor_idx] - X[selected_id]
        for _ in range(n_samples_per_pair):
            alpha = np.clip(np.random.rand(), a_min=1e-3, a_max=1.0)
            x_sample = X[selected_id] + alpha * diff
            x_samples.append(x_sample.reshape(1, -1))

    print("[DEBUG] Generated samples: ", len(x_samples))
    return x_samples

""" From modified_tsne.py in https://github.com/vu-minh/mlteam-lime-for-tsne/blob/master/ original BIR repo"""
# minh
# 15/11/2019
################################################################################
# modify TSNE to customize kl_loss and gradient_descent functions
# Note: the `modified_gradient_descent` is not used yet.
# only change the kl_loss function for both exact and barnes_hut method
################################################################################

from time import time

# import numpy as np
import sklearn  # note that sklearn does not auto import submodule
import sklearn.manifold  # so we have to import the desired submodules manually
from sklearn.manifold import TSNE
from sklearn.manifold import _barnes_hut_tsne
from scipy import linalg
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform


MACHINE_EPSILON = np.finfo(np.double).eps


def my_kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    skip_num_points=0,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.
    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.
    P : array, shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    compute_error: bool (optional, default:True)
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    # print("[DEBUG] my_kl_divergence")

    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    if compute_error:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    # grad = np.ndarray((n_samples, n_components), dtype=params.dtype)

    # 15/11/19, minh: hardcode init gradient to zero,
    # to make sure the points with indices below `skip_num_points` do not move.
    # np.ndarray() inits an array with supersmall value like 1e-300,
    # but the accumulate of N points after thousands iterations is considerable.
    grad = np.zeros((n_samples, n_components), dtype=params.dtype)

    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def my_kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    num_threads=1, 
    compute_error=True,
    verbose=False,
):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2)

    Parameters
    ----------
    params : array, shape (n_params,)
        Unraveled embedding.

    P : csr sparse matrix, shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized.

    degrees_of_freedom : float
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float (default: 0.5)
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int (optional, default:0)
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int
        Verbosity level.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : array, shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
    )

    # 15/11/2019 minh: fix kl_divergence with barnes_hut
    # the original implementation of sklearn does not use `skip_num_points`
    # set gradient of points with indices below `skip_num_points` to zero
    # in order to fix their position in the embedding.
    grad[:skip_num_points] = 0.0

    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad

def modified_gradient_descent(
    objective,
    p0,
    it,
    n_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished." % (i + 1, grad_norm)
                    )
                break

    return p, error, i

def modifiedTSNE(**kwargs):
    # force using "exact" method to test gradient
    # 15/11/2019: no need to for exact method since we have customize for barnes_hut
    # kwargs.update({"method": "exact"})
    # sklearn.manifold._t_sne._gradient_descent = modified_gradient_descent
    sklearn.manifold._t_sne._kl_divergence = my_kl_divergence
    sklearn.manifold._t_sne._kl_divergence_bh = my_kl_divergence_bh
    tsne = sklearn.manifold.TSNE(**kwargs)
    return tsne


# backtrack: /opt/anaconda3/lib/python3.6/site-packages/sklearn/manifold/_t_sne.py

""" From sample_tsne.py in https://github.com/vu-minh/mlteam-lime-for-tsne/blob/master/ original BIR repo"""
# minhvu
# 08/11/2019
# Query-blackbox like function for tsne
# Goal: given an input X, approximate an embedding y without fully rerun tsne

import os
# import math
import joblib
from copy import deepcopy
# from time import time
from functools import partial

# import numpy as np
# from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def tsne_sample_embedded_points(
    X,
    selected_idx,
    n_samples=100,
    n_neighbors_SMOTE=10,
    tsne_hyper_params={},
    early_stop_hyper_params={},
    sampling_method="sample_around",
    log_dir="",
    force_recompute=False,
    batch_mode=False,
):
    """Estimate sampled embedding in LD
    Args:
        X: input data [N, D]
        n_samples: number of points to sample around a selected point
        n_neighbors_SMOTE: number of neighbors in HD to perform oversampling w. SMOTE
        sigma_HD: variance for noise in HD
        sigma_LD: variance for noise in LD
        tsne_hyper_params: dict of tsne hyper-params.
        early_stop_hyper_params: dict of hyper-params for quick re-run t-SNE with early-stop
    Returns:
        Y: original embedding (Nx2)
        sample_ids: list of indices of sampled points
        s_samples: list of `n_samples` points HD
        y_samples: list of `n_samples` embedded points in LD
    """
    assert isinstance(selected_idx, int), "`selected_idx` must be a valid index"
    N, D = X.shape

    # first, run tsne with base `perplexity` to obtain the "base model"
    log_name_pattern = (
        # f"-id{selected_idx}"
        f"-perp{tsne_hyper_params['perplexity']}"
        f"-seed{tsne_hyper_params['random_state']}"
    )
    log_name_Y = f"{log_dir}/Y{log_name_pattern}.Z"
    if os.path.exists(log_name_Y) and not force_recompute:
        Y = joblib.load(log_name_Y)
        print("[DEBUG] Reuse: ", log_name_Y)
    else:
        tsne = modifiedTSNE(init="pca", **tsne_hyper_params)  # pca init for more stable
        Y = tsne.fit_transform(X)
        joblib.dump(Y, log_name_Y)
        print("[DEBUG] Initial tsne model: ", tsne)

    # log_name_samples = f"{log_dir}/{sampling_method}{n_samples}{log_name_pattern}.Z"
    log_name_samples = f"{log_dir}/id{selected_idx}-n{n_samples}{log_name_pattern}.Z"
    if os.path.exists(log_name_samples) and not force_recompute:
        x_samples, y_samples = joblib.load(log_name_samples)
        print("[DEBUG] Reuse: ", log_name_samples)
    else:

        # generate samples in HD
        # x_samples = generate_samples_HD(selected_point=X[selected_idx])
        x_samples = generate_samples_SMOTE(
            selected_idx, X, k_nearbors=n_neighbors_SMOTE, n_samples=n_samples
        )
        x_samples = np.array(x_samples).reshape(-1, D)

        # update hyper-params for quick re-run tsne
        # note: copy the params since the `dict.update` has side effect
        tsne_hyper_params_for_rerun = deepcopy(tsne_hyper_params)
        tsne_hyper_params_for_rerun.update(early_stop_hyper_params)

        if batch_mode:
            tick = time()
            y_samples = query_blackbox_tsne(
                X=X,
                Y=Y,
                query_idx=selected_idx,
                query_points=x_samples,
                tsne_hyper_params=tsne_hyper_params_for_rerun,
            )
            print(f"[DEBUG] Query-blackbox for {n_samples} points in {time() - tick:.3f} s")
        else:
            y_samples = []
            for i, x_sample in enumerate(x_samples):
                # ask tsne for the corresponding embedding of input sample
                tick = time()
                y_sample = query_blackbox_tsne(
                    X=X,
                    Y=Y,
                    query_idx=selected_idx,
                    query_points=x_sample.reshape(1, D),
                    tsne_hyper_params=tsne_hyper_params_for_rerun,
                )
                print(f"[DEBUG] Query-blackbox for {i+1}th point in {time() - tick:.3f} s")
                y_samples.append(y_sample)
            y_samples = np.array(y_samples).reshape(-1, Y.shape[1])

        # save the sampled points for latter use
        joblib.dump((x_samples, y_samples), log_name_samples)
    return Y, x_samples, y_samples


def query_blackbox_tsne(
    X, Y, query_idx=None, query_points=None, sigma_LD=1.0, tsne_hyper_params={}
):
    """Query-blackbox-like function for sample tsne embedding.
    Given the original `tsne` model (represented by (X, Y) pair),
    predict the embedding for the new `query_points`.
    `query_points` is an array of input query points in HD, of shape [, D].
    """
    assert query_idx is not None and query_points is not None
    N = X.shape[0]

    # sample proposal embedded position for the input `query_idx`
    # y_proposal = np.zeros((1, 2))
    y_proposals = sample_around(Y[query_idx], sigma=sigma_LD, n_samples=len(query_points))

    # append new samples in HD and LD into X and Y
    assert X.shape[1] == query_points.shape[1], "query_points.shape and X.shape should match"
    assert Y.shape[1] == y_proposals.shape[1], "y_proposals.shape and Y.shape should match"
    X_new = np.concatenate([X, query_points], axis=0)
    Y_new = np.concatenate([Y, y_proposals], axis=0)

    # approximate the original tsne model by a new one with initialized embedding
    # given by the original model
    tsne = modifiedTSNE(init=Y_new, **tsne_hyper_params)
    Y_with_samples = tsne._fit(X_new, skip_num_points=N)

    # debug to see if the original embedding does not change (or change a little bit)
    # TODO: discuss if it is necessary to make the initial Y does not change
    if tsne_hyper_params.get("verbose", 0) > 0:
        diff = Y - Y_with_samples[:N]
        print("[DEBUG] Check if the embedding for N original points does not change")
        print("Norm of diff: ", np.linalg.norm(diff))
        print("Sum square error: ", 0.5 * np.sum(np.dot(diff.T, diff)))

    return Y_with_samples[N:]

""" From utils.py """
def rotate_matrix(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def prediction_error(X, W, Y, rot_deg):
    R = rotate_matrix(rot_deg)
    diff = X @ W.T - Y @ R
    mse = np.linalg.norm(diff, axis=1) / X.shape[0]
    return mse
