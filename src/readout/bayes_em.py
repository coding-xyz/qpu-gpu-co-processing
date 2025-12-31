import torch
from .bayes_core import bayes_posterior_from_templates

def bayes_em_fit(X, n_iter=30, anchor_mu0=None, anchor_strength=1e3, init_sigma=0.5, device=None):
    '''
    Bayes-EM for two-state Gaussian trace model:
      p(x | y=s) = N(mu_s(t), sigma^2 I)

    Optionally "weakly anchor" mu0 toward a calibrated template anchor_mu0.
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.as_tensor(X, dtype=torch.float32, device=device)  # (N,T,2)
    N, T, C = X.shape

    y = torch.randint(0, 2, (N,), device=device)
    mu0_hat = X[y==0].mean(dim=0, keepdim=True)
    mu1_hat = X[y==1].mean(dim=0, keepdim=True)
    sigma = torch.tensor(float(init_sigma), device=device)

    anchor = None
    if anchor_mu0 is not None:
        anchor = torch.as_tensor(anchor_mu0, dtype=torch.float32, device=device).unsqueeze(0)

    for _ in range(n_iter):
        mu0 = mu0_hat.repeat(N,1,1)
        mu1 = mu1_hat.repeat(N,1,1)

        post, _ = bayes_posterior_from_templates(X, mu0, mu1, sigma)
        w0 = post[:,0].view(N,1,1)
        w1 = post[:,1].view(N,1,1)

        denom0 = w0.sum(dim=0, keepdim=True) + 1e-9
        denom1 = w1.sum(dim=0, keepdim=True) + 1e-9

        mu0_new = (w0 * X).sum(dim=0, keepdim=True) / denom0
        mu1_new = (w1 * X).sum(dim=0, keepdim=True) / denom1

        if anchor is not None and anchor_strength > 0:
            lam = float(anchor_strength)
            mu0_new = (mu0_new + lam * anchor) / (1.0 + lam)

        diff0 = X - mu0_new
        diff1 = X - mu1_new
        mse = (w0 * diff0 * diff0 + w1 * diff1 * diff1).mean()
        sigma = torch.sqrt(mse + 1e-9)

        mu0_hat, mu1_hat = mu0_new, mu1_new

    return mu0_hat.squeeze(0).detach().cpu(), mu1_hat.squeeze(0).detach().cpu(), float(sigma.detach().cpu())
