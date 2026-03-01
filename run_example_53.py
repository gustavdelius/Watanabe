import numpy as np
import scipy.stats as stats
import pandas as pd
import json
import multiprocessing
import os
import gc

def norm_pdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s)**2) / (np.sqrt(2 * np.pi) * s)

def log_prior(a, b, c):
    # a ~ Beta(0.5, 0.5) => prior \propto (a(1-a))^{-0.5}
    # b, c ~ N(0, B^2) with B=10
    if a <= 0 or a >= 1:
        return -np.inf
    lp_a = -0.5 * np.log(a * (1 - a))
    lp_b = -0.5 * (b / 10.0)**2
    lp_c = -0.5 * (c / 10.0)**2
    return lp_a + lp_b + lp_c

def sample_mcmc(X, iters=3000, burn_in=1000):
    n = len(X)
    a, b, c = 0.5, 0.1, -0.1
    
    # Pre-allocate
    samples_a = np.zeros(iters)
    samples_b = np.zeros(iters)
    samples_c = np.zeros(iters)
    
    # initial log likelihood
    p = a * norm_pdf(X, b, 1.0) + (1 - a) * norm_pdf(X, c, 1.0)
    ll = np.sum(np.log(p + 1e-15))
    lp = log_prior(a, b, c)
    
    # Steps
    step_a, step_b, step_c = 0.1, 0.2, 0.2
    
    for i in range(iters):
        # Propose
        a_new = a + np.random.normal(0, step_a)
        b_new = b + np.random.normal(0, step_b)
        c_new = c + np.random.normal(0, step_c)
        
        if 0 < a_new < 1:
            p_new = a_new * norm_pdf(X, b_new, 1.0) + (1 - a_new) * norm_pdf(X, c_new, 1.0)
            ll_new = np.sum(np.log(p_new + 1e-15))
            lp_new = log_prior(a_new, b_new, c_new)
            
            # Accept/reject
            if np.log(np.random.rand()) < (ll_new + lp_new - ll - lp):
                a, b, c = a_new, b_new, c_new
                ll, lp = ll_new, lp_new
                
        samples_a[i] = a
        samples_b[i] = b
        samples_c[i] = c
        
    return samples_a[burn_in:], samples_b[burn_in:], samples_c[burn_in:]

def calc_criteria(X, X_test, q_X, q_X_test, samples_a, samples_b, samples_c):
    n = len(X)
    
    # 1. True Distribution Entropy
    S_n = -np.mean(np.log(q_X + 1e-15))
    S = -np.mean(np.log(q_X_test + 1e-15))
    
    # Vectorized likelihood evaluation
    # Shape: (num_samples, n)
    diff_sq_b = (X[None, :] - samples_b[:, None])**2
    p_b = np.exp(-0.5 * diff_sq_b) / np.sqrt(2 * np.pi)
    
    diff_sq_c = (X[None, :] - samples_c[:, None])**2
    p_c = np.exp(-0.5 * diff_sq_c) / np.sqrt(2 * np.pi)
    
    p_train = samples_a[:, None] * p_b + (1 - samples_a[:, None]) * p_c + 1e-15
    log_p_train = np.log(p_train)
    
    # Predictive density
    pred_train = np.mean(p_train, axis=0)
    
    # Training loss (empirical loss)
    T_n = -np.mean(np.log(pred_train))
    
    # Functional Variance
    V_n = np.mean(np.var(log_p_train, axis=0))
    
    # WAIC
    waic = T_n + V_n
    
    # --- DIC ---
    mean_a, mean_b, mean_c = np.mean(samples_a), np.mean(samples_b), np.mean(samples_c)
    p_mean = mean_a * norm_pdf(X, mean_b, 1.0) + (1 - mean_a) * norm_pdf(X, mean_c, 1.0) + 1e-15
    L_mean = np.mean(np.log(p_mean))
    
    E_L = np.mean(np.mean(log_p_train, axis=1)) 
    dic = L_mean - 2 * E_L
    
    # --- AIC ---
    best_idx = np.argmax(np.sum(log_p_train, axis=1))
    best_a, best_b, best_c = samples_a[best_idx], samples_b[best_idx], samples_c[best_idx]
    
    p_best = best_a * norm_pdf(X, best_b, 1.0) + (1 - best_a) * norm_pdf(X, best_c, 1.0) + 1e-15
    L_best = np.mean(np.log(p_best))
    
    aic = -L_best + 3.0 / n
    
    # --- Generalization Error (GE) ---
    diff_sq_b_test = (X_test[None, :] - samples_b[:, None])**2
    p_b_test = np.exp(-0.5 * diff_sq_b_test) / np.sqrt(2 * np.pi)
    
    diff_sq_c_test = (X_test[None, :] - samples_c[:, None])**2
    p_c_test = np.exp(-0.5 * diff_sq_c_test) / np.sqrt(2 * np.pi)
    
    p_test = samples_a[:, None] * p_b_test + (1 - samples_a[:, None]) * p_c_test + 1e-15
    pred_test = np.mean(p_test, axis=0)
    
    ge = -np.mean(np.log(pred_test))
    
    # ISCV 
    inv_p = 1.0 / p_train
    expected_inv_p = np.mean(inv_p, axis=0)
    iscv = np.mean(np.log(expected_inv_p))
    
    return ge - S, iscv - S_n, aic - S_n, dic - S_n, waic - S_n

def run_single_trial(args):
    case_num, t, n = args
    np.random.seed(42 + t * 100 + case_num)
    
    # Test set logic common across cases
    # Reduced test set size slightly to save memory
    X_test = np.zeros(10000) 
    
    if case_num == 1:
        # (1) Regular and realizable
        a0, b0, c0 = 0.5, 2.0, -2.0
        z = np.random.binomial(1, a0, n)
        X = np.where(z == 1, np.random.normal(b0, 1.0, n), np.random.normal(c0, 1.0, n))
        z_test = np.random.binomial(1, a0, len(X_test))
        X_test[:] = np.where(z_test == 1, np.random.normal(b0, 1.0, len(X_test)), np.random.normal(c0, 1.0, len(X_test)))
        q_X = a0 * norm_pdf(X, b0, 1.0) + (1-a0) * norm_pdf(X, c0, 1.0)
        q_X_test = a0 * norm_pdf(X_test, b0, 1.0) + (1-a0) * norm_pdf(X_test, c0, 1.0)
        
    elif case_num == 2:
        # (2) Regular and unrealizable 
        a0, sigma, b0, c0 = 0.5, 0.8, 2.0, -2.0
        z = np.random.binomial(1, a0, n)
        X = np.where(z == 1, np.random.normal(b0, sigma, n), np.random.normal(c0, sigma, n))
        z_test = np.random.binomial(1, a0, len(X_test))
        X_test[:] = np.where(z_test == 1, np.random.normal(b0, sigma, len(X_test)), np.random.normal(c0, sigma, len(X_test)))
        q_X = a0 * norm_pdf(X, b0, sigma) + (1-a0) * norm_pdf(X, c0, sigma)
        q_X_test = a0 * norm_pdf(X_test, b0, sigma) + (1-a0) * norm_pdf(X_test, c0, sigma)
        
    elif case_num == 3:
        # (3) Nonregular and realizable
        X = np.random.normal(0, 1.0, n)
        X_test = np.random.normal(0, 1.0, len(X_test))
        q_X = norm_pdf(X, 0, 1.0)
        q_X_test = norm_pdf(X_test, 0, 1.0)
        
    elif case_num == 4:
        # (4) Nonregular and unrealizable
        sigma = 0.8
        X = np.random.normal(0, sigma, n)
        X_test = np.random.normal(0, sigma, len(X_test))
        q_X = norm_pdf(X, 0, sigma)
        q_X_test = norm_pdf(X_test, 0, sigma)
        
    elif case_num == 5:
        # (5) Delicate case
        a0, sigma, b0, c0 = 0.5, 0.95, 0.5, -0.5
        z = np.random.binomial(1, a0, n)
        X = np.where(z == 1, np.random.normal(b0, sigma, n), np.random.normal(c0, sigma, n))
        z_test = np.random.binomial(1, a0, len(X_test))
        X_test[:] = np.where(z_test == 1, np.random.normal(b0, sigma, len(X_test)), np.random.normal(c0, sigma, len(X_test)))
        q_X = a0 * norm_pdf(X, b0, sigma) + (1-a0) * norm_pdf(X, c0, sigma)
        q_X_test = a0 * norm_pdf(X_test, b0, sigma) + (1-a0) * norm_pdf(X_test, c0, sigma)
        
    elif case_num == 6:
        # (6) Unbalanced case
        a0, b0, c0 = 0.01, 2.0, -2.0
        z = np.random.binomial(1, a0, n)
        X = np.where(z == 1, np.random.normal(b0, 1.0, n), np.random.normal(c0, 1.0, n))
        z_test = np.random.binomial(1, a0, len(X_test))
        X_test[:] = np.where(z_test == 1, np.random.normal(b0, 1.0, len(X_test)), np.random.normal(c0, 1.0, len(X_test)))
        q_X = a0 * norm_pdf(X, b0, 1.0) + (1-a0) * norm_pdf(X, c0, 1.0)
        q_X_test = a0 * norm_pdf(X_test, b0, 1.0) + (1-a0) * norm_pdf(X_test, c0, 1.0)
        
    samples_a, samples_b, samples_c = sample_mcmc(X, iters=5000, burn_in=1000)
    res = calc_criteria(X, X_test, q_X, q_X_test, samples_a, samples_b, samples_c)
    
    # Explicit garbage collection due to massive matrices in calc_criteria
    del X_test
    del q_X_test
    del samples_a
    del samples_b
    del samples_c
    gc.collect()
    
    return res

def run_experiment(case_num, n=100, trials=100):
    results = {'G': [], 'ISCV': [], 'AIC': [], 'DIC': [], 'WAIC': []}
    
    args_list = [(case_num, t, n) for t in range(trials)]
    
    # Restrict drastically to prevent system hangs
    # max 2 processes, maxtasksperchild 5 to constantly recycle memory
    with multiprocessing.Pool(processes=2, maxtasksperchild=5) as pool:
        trial_results = pool.map(run_single_trial, args_list, chunksize=1)
        
    for res in trial_results:
        ge, iscv, aic, dic, waic = res
        results['G'].append(np.float64(ge))
        results['ISCV'].append(np.float64(iscv))
        results['AIC'].append(np.float64(aic))
        results['DIC'].append(np.float64(dic))
        results['WAIC'].append(np.float64(waic))
        
    summary = {}
    for k in results:
        summary[f'{k}_mean'] = float(np.mean(results[k]))
        summary[f'{k}_std'] = float(np.std(results[k]))
        
    return summary

if __name__ == "__main__":
    cases = {
        "1": "Regular Realizable",
        "2": "Regular Unrealizable",
        "3": "Nonreg. Realizable",
        "4": "Nonreg. Unrealizable",
        "5": "Delicate",
        "6": "Unbalanced"
    }
    
    all_results = {}
    for c in range(1, 7):
        print(f"Running Case {c}: {cases[str(c)]}...")
        res = run_experiment(c, n=100, trials=100)
        all_results[str(c)] = {"name": cases[str(c)], **res}
        # pretty print
        print(f"  G:    {res['G_mean']:.4f} \pm {res['G_std']:.4f}")
        print(f"  ISCV: {res['ISCV_mean']:.4f} \pm {res['ISCV_std']:.4f}")
        print(f"  AIC:  {res['AIC_mean']:.4f} \pm {res['AIC_std']:.4f}")
        print(f"  DIC:  {res['DIC_mean']:.4f} \pm {res['DIC_std']:.4f}")
        print(f"  WAIC: {res['WAIC_mean']:.4f} \pm {res['WAIC_std']:.4f}")
        
    with open("example_53_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("Done! Results saved to example_53_results.json")
