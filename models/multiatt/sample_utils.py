import torch
from torch import autograd
from torch.nn import functional as F
from torch.autograd import Variable
def gumble_softmax(logits, temperature):
    '''
    get gumble softmax sample from logits
    :param logits:
    :param temperature:
    :return:
    '''
    noise = torch.rand(logits.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = autograd.Variable(noise)
    noise = noise.cuda()
    x = (logits + noise) / temperature
    x = F.softmax(x.view(-1, x.size()[-1]), dim=-1)
    y_soft = x.view_as(logits)
    # return x.view_as(logits)
    y =  y_soft
    #
    # _, k = y_soft.data.max(-1)
    # y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
    # y = Variable(y_hard - y_soft.data) + y_soft

    return y

def kl_divergence_loss(prior_logits, posterior_logits):
    '''
    Calculating the KL divergence between prior latent distribution and posterior latent distribution
    :param prior_logits: shape: batch_size x latent_ct
    :param posterior_logits: shape: batch_size x latent_ct
    :return:
    '''
    batch_size = prior_logits.size()[0]
    prior_prob = F.softmax(prior_logits.view(-1, prior_logits.size()[-1]), dim=-1)
    posterior_prob = F.softmax(posterior_logits.view(-1, posterior_logits.size()[-1]), dim=-1)

    log_ratio = torch.log(posterior_prob) - torch.log(prior_prob)
    kl_dis = posterior_prob*log_ratio
    kl_dis_sum = torch.sum(kl_dis, dim=-1)
    kl_dis_loss = torch.mean(kl_dis_sum)
    return kl_dis_loss



def get_hard_mask(z, return_ind=False):
    '''
        -z: torch Tensor where each element probablity of element
        being selected
        -args: experiment level config
        returns: A torch variable that is binary mask of z >= .5
    '''
    masked = torch.ge(z, 0.5).float()
    return masked

