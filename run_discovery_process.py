import numpy as np
import json
import os
import multiprocessing

def norm_pdf(x, m, s=1.0):
    return np.exp(-0.5 * ((x - m) / s)**2) / (np.sqrt(2 * np.pi) * s)

def compute_trial(args):
    n, case_mu, a_true, alpha, sigma, seed = args
    np.random.seed(seed)
    
    # 1. Generate synthetic data
    z = np.random.binomial(1, a_true, n)
    X = np.zeros(n)
    X[z == 0] = np.random.normal(0, 1, np.sum(z == 0))
    X[z == 1] = np.random.normal(case_mu, 1, np.sum(z == 1))
    
    # Generate massive test set for exact GE
    n_test = 2000
    z_test = np.random.binomial(1, a_true, n_test)
    X_test = np.zeros(n_test)
    X_test[z_test == 0] = np.random.normal(0, 1, np.sum(z_test == 0))
    X_test[z_test == 1] = np.random.normal(case_mu, 1, np.sum(z_test == 1))
    
    # True entropy
    p0_train = (1-a_true)*norm_pdf(X, 0) + a_true*norm_pdf(X, case_mu)
    S_n = -np.mean(np.log(p0_train + 1e-15))
    
    p0_test = (1-a_true)*norm_pdf(X_test, 0) + a_true*norm_pdf(X_test, case_mu)
    S = -np.mean(np.log(p0_test + 1e-15))
    
    # Parameter grid (skip a=0 and a=1 to avoid issues with Dirichlet limit)
    a_vals = np.linspace(0.001, 0.999, 150)
    mu_vals = np.linspace(-6, 6, 200)
    A, MU = np.meshgrid(a_vals, mu_vals)
    
    # Prior log probability
    log_prior = (alpha - 1.0) * np.log(A * (1 - A)) - 0.5 * (MU / sigma)**2
    
    # Evaluate log likelihood over grid
    # log_p shape: (n, len(mu_vals), len(a_vals))
    log_p = np.zeros((n, len(mu_vals), len(a_vals)))
    for i, x in enumerate(X):
        p_x_w = (1 - A) * norm_pdf(x, 0) + A * norm_pdf(x, MU)
        log_p[i] = np.log(p_x_w + 1e-15)
        
    total_log_likelihood = np.sum(log_p, axis=0)
    unnorm_log_post = total_log_likelihood + log_prior
    
    # Normalize
    log_post_shifted = unnorm_log_post - np.max(unnorm_log_post)
    post = np.exp(log_post_shifted)
    post /= np.sum(post)
    
    # Predictions
    predictive_density = np.zeros(n)
    V_w = np.zeros(n)
    inv_predictive_density = np.zeros(n)
    
    for i in range(n):
        p_i = np.exp(log_p[i])
        predictive_density[i] = np.sum(p_i * post)
        inv_predictive_density[i] = np.sum((1.0 / (p_i + 1e-15)) * post)
        
        expected_log_p = np.sum(log_p[i] * post)
        expected_log_p_sq = np.sum((log_p[i]**2) * post)
        V_w[i] = expected_log_p_sq - expected_log_p**2

    # Information Criteria
    T_n = -1.0 / n * np.sum(np.log(predictive_density + 1e-15))
    waic = T_n + np.mean(V_w)
    iscv = np.mean(np.log(inv_predictive_density + 1e-15))
    
    # AIC (Requires MLE)
    mle_idx = np.argmax(total_log_likelihood)
    a_mle, mu_mle = A.flat[mle_idx], MU.flat[mle_idx]
    max_log_lik = total_log_likelihood.flat[mle_idx]
    T_n_mle = -max_log_lik / n
    aic = T_n_mle + 2.0 / n  # d=2
    
    # DIC (Requires posterior mean)
    a_mean = np.sum(A * post)
    mu_mean = np.sum(MU * post)
    # Dev at mean
    p_mean_eval = (1 - a_mean) * norm_pdf(X, 0) + a_mean * norm_pdf(X, mu_mean)
    D_theta_mean = -2.0 * np.sum(np.log(p_mean_eval + 1e-15))
    # Expected Dev
    expected_D = -2.0 * np.sum(total_log_likelihood * post)
    p_D = expected_D - D_theta_mean
    dic = (D_theta_mean + 2 * p_D) / (2 * n)  # Scale to per-sample error
    
    # Generalization Error
    batch_size = 1000
    pred_dens_test = np.zeros(n_test)
    for b in range(0, n_test, batch_size):
        end_idx = min(b + batch_size, n_test)
        X_batch = X_test[b:end_idx]
        X_expanded = X_batch[:, np.newaxis, np.newaxis]
        p_batch = (1 - A) * norm_pdf(X_expanded, 0) + A * norm_pdf(X_expanded, MU)
        pred_dens_test[b:end_idx] = np.sum(p_batch * post, axis=(1, 2))
        
    ge = -1.0 / n_test * np.sum(np.log(pred_dens_test + 1e-15))
    
    return {
        'n': n,
        'GE': ge - S,
        'ISCV': iscv - S_n,
        'WAIC': waic - S_n,
        'AIC': aic - S_n,
        'DIC': dic - S_n
    }


def run_case_parallel(case_mu, a_true, alpha, sigma, n_values, trials):
    print(f"Running Case: mu={case_mu}")
    
    results_by_n = {n: {'GE': [], 'ISCV': [], 'WAIC': [], 'AIC': [], 'DIC': []} for n in n_values}
    
    tasks = []
    for n in n_values:
        for t in range(trials):
            tasks.append((n, case_mu, a_true, alpha, sigma, hash(f"{n}_{case_mu}_{t}") % 2**32))
            
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        trial_results = pool.map(compute_trial, tasks)
        
    for res in trial_results:
        n = res['n']
        for k in ['GE', 'ISCV', 'WAIC', 'AIC', 'DIC']:
            results_by_n[n][k].append(n * res[k])  # Save n * Error
            
    # Calculate means
    final_res = {}
    for n in n_values:
        final_res[n] = {
            k: float(np.mean(results_by_n[n][k])) for k in ['GE', 'ISCV', 'WAIC', 'AIC', 'DIC']
        }
    return final_res

if __name__ == "__main__":
    n_values = [5, 10, 20, 40, 80, 160, 320, 640] # Removed 1280 to save time
    trials = 30
    
    alpha = 0.3
    sigma = 10.0
    a_true = 0.5
    
    cases = {
        "1": {"mu": 4.0}, # Regular: True is separable, lambda=d/2=1.0
        "2": {"mu": 0.0}, # Singular: True is at singularity, lambda=alpha/2=0.15
        "3": {"mu": 1.0}, # Discovery: Transitions from singular to regular
    }
    
    full_results = {}
    for case_id, conf in cases.items():
        res = run_case_parallel(conf["mu"], a_true, alpha, sigma, n_values, trials)
        full_results[case_id] = res
        
    with open('discovery_process_results.json', 'w') as f:
        json.dump(full_results, f, indent=4)
        
    print("Done generating data.")
