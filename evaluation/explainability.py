import torch
from torch import autograd


def saliency(model, x, y_ohe, clip=False, zsw=None):
    #import pdb;pdb.set_trace()
    x.require_grad = True
    x.retain_grad()

    if clip:

        image_features = model.encode_image(x)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        logits = image_features@zsw

        #logits = model.forward_batch(x)
        y_pred = torch.sum(logits * y_ohe)
    else:
        inputs = model(x)
        y_pred = torch.sum(inputs * y_ohe)
    y_pred.backward()
    phi = x.grad

    return phi


def input_gradient(model, x, y):
    return saliency(model, x, y) * x


def smoothgrad(model, x, y, nb_samples=50, sigma=0.2):
    x_noisy = x.unsqueeze(1)
    x_noisy = x_noisy.repeat(1, nb_samples, 1, 1, 1)

    epsilon = torch.normal(torch.zeros(1, nb_samples, *x_noisy.shape[2:]), sigma).cuda()
    x_noisy = x_noisy + epsilon

    y_noisy = y.unsqueeze(1)
    y_noisy = y_noisy.repeat(1, nb_samples)

    sgs = None
    for i in range(len(x)):
        sg = torch.mean(saliency(model, x_noisy[i], y_noisy[i]), 0, keepdim=True)
        sgs = torch.cat([sgs, sg], 0) if sgs is not None else sg

    return sgs


def integrad(model, x, y, nb_samples=50, clip=False):
    alphas = torch.linspace(0.0, 1.0, nb_samples)[None, :, None, None, None].cuda()

    x_integrated = x.unsqueeze(1)
    x_integrated = x_integrated.repeat(1, nb_samples, 1, 1, 1)
    x_integrated = x_integrated * alphas

    y_integrated = y.unsqueeze(1)
    y_integrated = y_integrated.repeat(1, nb_samples)

    igs = None
    for i in range(len(x)):
        ig = x[i] * torch.mean(saliency(model, x_integrated[i],
                               y_integrated[i]), 0, keepdim=True, clip=clip)
        igs = torch.cat([igs, ig], 0) if igs is not None else ig

    return igs
