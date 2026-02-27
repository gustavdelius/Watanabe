import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os

# Create an output directory
os.makedirs("figures", exist_ok=True)

# ----------------- Model 1 (Eq 1.11, Figures 1.2 & 1.3) -----------------
def generate_data_model1(n, num_datasets=12, seed=42):
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=1.0, size=(num_datasets, n))

def posterior_model1(x, a_range, sigma_range):
    A, Sigma = np.meshgrid(a_range, sigma_range)
    x_reshaped = x.reshape(-1, 1, 1)
    log_likelihood_all = -0.5 * np.log(2 * np.pi * Sigma**2) - 0.5 * ((x_reshaped - A) / Sigma)**2
    log_likelihood = np.sum(log_likelihood_all, axis=0)
    
    log_posterior = log_likelihood - np.max(log_likelihood)
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior)
    return A, Sigma, posterior

def plot_model1(n, filename, num_datasets=12):
    data = generate_data_model1(n, num_datasets, seed=n)
    a_vals = np.linspace(-1, 1, 100)
    sigma_vals = np.linspace(0.01, 2.0, 100)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i in range(num_datasets):
        ax = axes[i // 4, i % 4]
        A, Sigma, post = posterior_model1(data[i], a_vals, sigma_vals)
        ax.imshow(post, origin='lower', extent=[-1, 1, 0.01, 2.0], aspect='auto', cmap='gray_r')
        ax.plot(0, 1, 'ws', markersize=6, markeredgecolor='k')
        
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 1, 2])
        if i // 4 == 2:
            ax.set_xlabel('a')
        if i % 4 == 0:
            ax.set_ylabel(r'$\sigma$')
            
    plt.savefig(f"figures/{filename}", bbox_inches='tight')
    plt.close()

# ----------------- Model 2 (Eq 1.12, Figures 1.4 & 1.5) -----------------
def generate_data_model2(n, num_datasets=12, seed=42):
    np.random.seed(seed)
    data = []
    for _ in range(num_datasets):
        components = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        means = np.where(components == 0, 0.0, 0.3)
        sample = np.random.normal(loc=means, scale=1.0)
        data.append(sample)
    return np.array(data)

def posterior_model2(x, a_range, b_range):
    A, B = np.meshgrid(a_range, b_range)
    x_reshaped = x.reshape(-1, 1, 1)
    p1 = np.exp(-0.5 * x_reshaped**2) / np.sqrt(2 * np.pi)
    p2 = np.exp(-0.5 * (x_reshaped - B)**2) / np.sqrt(2 * np.pi)
    px = (1 - A) * p1 + A * p2
    log_likelihood = np.sum(np.log(px + 1e-15), axis=0)
        
    log_posterior = log_likelihood - np.max(log_likelihood)
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior)
    return A, B, posterior

def plot_model2(n, filename, num_datasets=12):
    data = generate_data_model2(n, num_datasets, seed=n)
    a_vals = np.linspace(0.01, 0.99, 100)
    b_vals = np.linspace(0.01, 0.99, 100)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i in range(num_datasets):
        ax = axes[i // 4, i % 4]
        A, B, post = posterior_model2(data[i], a_vals, b_vals)
        ax.imshow(post, origin='lower', extent=[0.01, 0.99, 0.01, 0.99], aspect='auto', cmap='gray_r')
        ax.plot(0.5, 0.3, 'ws', markersize=6, markeredgecolor='k')
        
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        if i // 4 == 2:
            ax.set_xlabel('a')
        if i % 4 == 0:
            ax.set_ylabel('b')
            
    plt.savefig(f"figures/{filename}", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Figure 1.2 (Model 1, n=10)...", flush=True)
    plot_model1(10, "Figure_1.2.png")
    
    print("Generating Figure 1.3 (Model 1, n=50)...", flush=True)
    plot_model1(50, "Figure_1.3.png")
    
    print("Generating Figure 1.4 (Model 2, n=100)...", flush=True)
    plot_model2(100, "Figure_1.4.png")
    
    print("Generating Figure 1.5 (Model 2, n=1000)...", flush=True)
    plot_model2(1000, "Figure_1.5.png")
    
    print("All figures generated successfully.", flush=True)
