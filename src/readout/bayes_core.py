import torch

def loglik_gaussian_trace(x, mu, sigma):
    diff = x - mu
    var = sigma**2
    return -0.5 * (diff*diff/var + torch.log(2*torch.pi*var)).sum(dim=(-1,-2))

@torch.no_grad()
def bayes_posterior_from_templates(x, mu0, mu1, sigma, prior=None):
    device = x.device
    if prior is None:
        prior = torch.tensor([0.5, 0.5], device=device)
    prior = prior / prior.sum()

    ll0 = loglik_gaussian_trace(x, mu0, sigma)
    ll1 = loglik_gaussian_trace(x, mu1, sigma)

    logp0 = torch.log(prior[0]) + ll0
    logp1 = torch.log(prior[1]) + ll1
    logZ = torch.logsumexp(torch.stack([logp0, logp1], dim=-1), dim=-1)

    post0 = torch.exp(logp0 - logZ)
    post1 = torch.exp(logp1 - logZ)
    return torch.stack([post0, post1], dim=-1), (ll0, ll1)
