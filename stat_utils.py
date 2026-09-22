from sklearn.metrics import cohen_kappa_score
import numpy as np

def get_kappa_ci(x, y, weights=None, n_boot=1000, seed=42):
    """Calculates Cohen's Kappa, 95% CI via bootstrapping, and Observed Agreement."""
    np.random.seed(seed)
    n = len(x)
    kappa = cohen_kappa_score(x, y, weights=weights)
    po = (x == y).mean() * 100

    # bootstrapping for 95% CI
    boot_kappas = []
    x_arr, y_arr = np.array(x), np.array(y)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_k = cohen_kappa_score(
            x_arr[idx], y_arr[idx], weights=weights
        )
        boot_kappas.append(boot_k)

    ci_lower, ci_upper = np.percentile(boot_kappas, [2.5, 97.5])
    return kappa, ci_lower, ci_upper, po
