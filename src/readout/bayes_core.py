import torch
import numpy as np

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


def bayes_init(X_train, y_train):
    '''
    初始化贝叶斯方法的模板（mu0, mu1）和标准差（sigma）。
    
    计算每个类别的信号模板，并估计噪声水平（标准差）。
    '''
    # 计算每个类别的模板（均值）
    mu0 = np.array(X_train[y_train == 0].mean(axis=0, keepdims=True), np.float32)  # 类别 0 的模板
    mu1 = np.array(X_train[y_train == 1].mean(axis=0, keepdims=True), np.float32)  # 类别 1 的模板

    # 计算每个类别的标准差（sigma）
    # 对于类别 0
    diff0 = X_train[y_train == 0] - mu0  # 类别 0 样本与模板的差异
    sigma0 = np.sqrt(np.mean(diff0 ** 2))  # 计算类别 0 的标准差

    # 对于类别 1
    diff1 = X_train[y_train == 1] - mu1  # 类别 1 样本与模板的差异
    sigma1 = np.sqrt(np.mean(diff1 ** 2))  # 计算类别 1 的标准差

    # 选择两者的均值作为最终的 sigma
    sigma = (sigma0 + sigma1) / 2

    return mu0[0], mu1[0], sigma