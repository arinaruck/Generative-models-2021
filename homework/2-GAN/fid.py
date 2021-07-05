import torch
import torch.nn as nn
from scipy import linalg
import numpy as np
from tqdm import tqdm
from utils import permute_labels

CLF_HIDDEN = 2048


def make_inception_feature_extractor(device):
    classifier = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    classifier.dropout = nn.Identity()
    classifier.fc = nn.Identity()
    classifier.eval()
    classifier.to(device)
    return classifier


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


@torch.no_grad()
def calculate_activation_statistics(dataloader, model, classifier, device, ATTRIBUTE_IDX):
    classifier.eval()
    batch_size = dataloader.batch_size
    examples = len(dataloader) * batch_size
    input_acts = np.zeros((examples, CLF_HIDDEN))
    output_acts = np.zeros((examples, CLF_HIDDEN))

    for i, (image, label) in enumerate(tqdm(dataloader, leave=False, desc="fid")):
        input_img = image.to(device)
        label = label[:, ATTRIBUTE_IDX].to(device)
        new_label = permute_labels(label)
        output_img = model.generate(input_img, new_label)
        input_act = classifier(input_img)
        output_act = classifier(output_img)
        input_acts[i * batch_size: (i + 1) * batch_size] = input_act.cpu().numpy()
        output_acts[i * batch_size: (i + 1) * batch_size] = output_act.cpu().numpy()

    mu1, sigma1 = input_acts.mean(axis=0), np.cov(input_acts, rowvar=False)
    mu2, sigma2 = output_acts.mean(axis=0), np.cov(output_acts, rowvar=False)
    return mu1, sigma1, mu2, sigma2


@torch.no_grad()
def calculate_fid(dataloader, model, classifier, device, ATTRIBUTE_IDX):
    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, model, classifier, device, ATTRIBUTE_IDX)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()