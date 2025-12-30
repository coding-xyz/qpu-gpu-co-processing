import torch

def hmm_forward_backward_log(em_logp, logA, logpi):
    B,T,K = em_logp.shape
    log_alpha = torch.empty((B,T,K), device=em_logp.device, dtype=em_logp.dtype)
    log_alpha[:,0,:] = logpi.view(1,K) + em_logp[:,0,:]
    for t in range(1,T):
        log_alpha[:,t,:] = em_logp[:,t,:] + torch.logsumexp(
            log_alpha[:,t-1,:].unsqueeze(-1) + logA.unsqueeze(0),
            dim=1
        )
    logZ = torch.logsumexp(log_alpha[:,T-1,:], dim=-1)

    log_beta = torch.zeros((B,T,K), device=em_logp.device, dtype=em_logp.dtype)
    for t in range(T-2, -1, -1):
        log_beta[:,t,:] = torch.logsumexp(
            logA.unsqueeze(0) + em_logp[:,t+1,:].unsqueeze(1) + log_beta[:,t+1,:].unsqueeze(1),
            dim=2
        )

    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=-1).unsqueeze(-1)
    gamma = torch.exp(log_gamma)

    xi = None
    if T > 1:
        log_xi = (
            log_alpha[:,:-1,:].unsqueeze(-1)
            + logA.unsqueeze(0).unsqueeze(0)
            + em_logp[:,1:,:].unsqueeze(2)
            + log_beta[:,1:,:].unsqueeze(2)
        )
        log_xi = log_xi - torch.logsumexp(log_xi.view(B, T-1, K*K), dim=-1).view(B,T-1,1,1)
        xi = torch.exp(log_xi)

    return gamma, xi, logZ

def emission_logp_from_templates(X, mu0, mu1, sigma):
    B,T,_ = X.shape
    mu = torch.stack([mu0, mu1], dim=0)          # (K,T,2)
    mu = mu.unsqueeze(0).expand(B,-1,-1,-1)      # (B,K,T,2)
    Xb = X.unsqueeze(1)                          # (B,1,T,2)
    diff = Xb - mu
    var = sigma**2
    ll = -0.5 * (diff*diff/var + torch.log(2*torch.pi*var)).sum(dim=-1)  # (B,K,T)
    return ll.transpose(1,2)  # (B,T,K)

@torch.no_grad()
def fit_hmm_templates_fixed(X, mu0, mu1, n_iter=30, init_sigma=0.5, init_p01=1e-3, init_p10=1e-2, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu0 = torch.as_tensor(mu0, dtype=torch.float32, device=device)
    mu1 = torch.as_tensor(mu1, dtype=torch.float32, device=device)

    sigma = torch.tensor(float(init_sigma), device=device)
    p01 = torch.tensor(float(init_p01), device=device).clamp(1e-8, 0.2)
    p10 = torch.tensor(float(init_p10), device=device).clamp(1e-8, 0.5)

    for _ in range(n_iter):
        A = torch.stack([torch.stack([1-p01, p01]), torch.stack([p10, 1-p10])])
        A = A / A.sum(dim=-1, keepdim=True)
        logA = torch.log(A)
        logpi = torch.log(torch.tensor([0.5,0.5], device=device))

        em_logp = emission_logp_from_templates(X, mu0, mu1, sigma)
        gamma, xi, _ = hmm_forward_backward_log(em_logp, logA, logpi)

        xi_sum = xi.sum(dim=(0,1)) + 1e-9
        A_new = xi_sum / xi_sum.sum(dim=-1, keepdim=True)
        p01 = A_new[0,1].clamp(1e-8, 0.2)
        p10 = A_new[1,0].clamp(1e-8, 0.5)

        mu = torch.stack([mu0, mu1], dim=0)  # (K,T,2)
        mu_bt = mu.permute(1,0,2).unsqueeze(0).expand(X.shape[0],-1,-1,-1)  # (B,T,K,2)
        diff = X.unsqueeze(2) - mu_bt
        sq = (diff*diff).sum(dim=-1)  # (B,T,K)
        mse = (gamma * sq).mean()
        sigma = torch.sqrt(mse/2.0 + 1e-9)
    return float(sigma.detach().cpu()), float(p01.detach().cpu()), float(p10.detach().cpu())

@torch.no_grad()
def hmm_classify(X, mu0, mu1, sigma, p01=1e-3, p10=1e-2, prior=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu0 = torch.as_tensor(mu0, dtype=torch.float32, device=device)
    mu1 = torch.as_tensor(mu1, dtype=torch.float32, device=device)
    sigma = torch.tensor(float(sigma), device=device)

    A = torch.tensor([[1-p01, p01],[p10, 1-p10]], device=device, dtype=torch.float32)
    A = A / A.sum(dim=-1, keepdim=True)
    logA = torch.log(A)

    if prior is None:
        pi = torch.tensor([0.5,0.5], device=device)
    else:
        pi = torch.as_tensor(prior, dtype=torch.float32, device=device); pi = pi/pi.sum()
    logpi = torch.log(pi)

    em_logp = emission_logp_from_templates(X, mu0, mu1, sigma)
    gamma, _, _ = hmm_forward_backward_log(em_logp, logA, logpi)
    return gamma[:,-1,:]
