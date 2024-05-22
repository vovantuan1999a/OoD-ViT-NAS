import torch

import torch.nn.functional as F 
# from . import measure
from . import indicator
from ..p_utils import get_layer_metric_array_dss,get_layer_metric_array
import torch.nn as nn
from jacobian import JacobianReg
reg = JacobianReg() # Jacobian regularization
lambda_JR = 0.01 # hyperparameter

lossfunc = nn.CrossEntropyLoss().cuda()

@indicator('dss', bn=False, mode='param')
def compute_dss_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device
    # print(inputs)
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    # print("input o day:",inputs)
    output = net.forward(inputs)
    torch.sum(output).backward()

    def dss(layer):
        if layer._get_name() == 'PatchembedSuper':
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples or isinstance(layer,
                                                                                                       nn.Linear) and layer.out_features == layer.in_features and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(
                    torch.norm(layer.samples['weight'].grad, 'nuc') * torch.norm(layer.samples['weight'], 'nuc'))
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer,
                      nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
            else:
                return torch.zeros_like(layer.samples['weight'])
        elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            if layer.weight.grad is not None:
                return torch.abs(layer.weight.grad * layer.weight)
            else:
                return torch.zeros_like(layer.weight)
        else:
            return torch.tensor(0).to(device)
    grads_abs = get_layer_metric_array_dss(net, dss, mode)

    nonlinearize(net, signs)

    return grads_abs


@indicator('AutoProxA', bn=False, mode='param')
def compute_AutoProxA_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()

    def AutoProxA(layer):
        if layer._get_name() == 'PatchembedSuper':  # [Xinda] ‘PatchembedSuper’ is a class in model/module/embedding_super.py, it uses a conv2D to convert the image into embeddings
            if layer.sampled_weight.grad is not None:
                return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
            else:
                return torch.zeros_like(layer.sampled_weight)  # [Xinda] this subsection is useless, because when 'mode' = 'param', get_layer_metric_array_dss only compute the proxy for nn.Linear, which does not exist in PatchembedSuper.
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples or isinstance(layer,
                                                                                                       nn.Linear) and layer.out_features == layer.in_features and layer.samples:  # [Xinda] if 'qkv' exists in the name of the layer, it is considered 'MSA' layer, the proxy is computed based on Eq.(4)
            if layer.samples['weight'].grad is not None:
                return torch.norm(layer.samples['weight'].grad, p=1)  # [Xinda] L1-norm of the gradient of the weights, see Equation 2 & Figure 7 in AAAI24 paper
            else:
                return torch.zeros_like(layer.samples['weight'])
        if isinstance(layer,
                      nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:  # [Xinda] if 'qkv' does not exist in the name of the layer, it is considered 'MLP' layers, the proxy is computed based on Eq.(5)
            if layer.samples['weight'].grad is not None:
                return torch.sum(torch.sigmoid(layer.samples['weight'].grad))/(torch.numel(layer.samples['weight'].grad) + 1e-9)
            else:
                return torch.zeros_like(layer.samples['weight'])
        elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:  # [Xinda] for some other MLP layers, the proxy is also computed based on Eq.(5), the difference is only about "layer.samples['weight'].grad" and "layer.weight.grad"
            if layer.weight.grad is not None:
                return torch.sum(torch.sigmoid(layer.weight.grad))/(torch.numel(layer.weight.grad) + 1e-9)
            else:
                return torch.zeros_like(layer.weight)
        else:  # otherwise, it does not contribute to the proxy
            return torch.tensor(0).to(device)
    grads_abs = get_layer_metric_array_dss(net, AutoProxA, mode)  # [Xinda] I kept the name 'get_layer_metric_array_dss' because it is simpler

    nonlinearize(net, signs)

    return grads_abs


import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import copy
# from . import measure
from ..p_utils import adj_weights, get_layer_metric_array_adv_feats


def fgsm_attack(net, image, target, epsilon):
    perturbed_image = image.detach().clone()
    perturbed_image.requires_grad = True
    net.zero_grad()

    logits = net(perturbed_image)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    
    sign_data_grad = perturbed_image.grad.sign_()
    perturbed_image = perturbed_image - epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


@indicator('croze', bn=False, mode='param')
def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None, search_space='AutoFormer'):

    device = inputs.device
    origin_inputs, origin_outputs = inputs, targets
    
    cos_loss = nn.CosineSimilarity(dim=0)
    ce_loss = nn.CrossEntropyLoss()
    
    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])


    advnet = copy.deepcopy(net)
    
    # keep signs of all params
    signs = linearize(net)
    adv_signs = linearize(advnet)
    
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.float()
    advnet.float()
    feats = {}
    
    def forward_hook(module, data_input, data_output):
        # print('---day')
        # feats.append(data_output)
        mod_name = module.__class__.__name__
        feats[mod_name] = data_output
        # fea = data_output[0].detach()
        # fea = fea.reshape(fea.shape[0], -1)
        # corr = torch.corrcoef(fea)
        # corr[torch.isnan(corr)] = 0
        # corr[torch.isinf(corr)] = 0
        # values = torch.linalg.eig(corr)[0]
        # # result = np.real(np.min(values)) / np.real(np.max(values))
        # result = torch.min(torch.real(values))
        # result_list.append(result)

    for modules in net.modules():
        modules.register_forward_hook(forward_hook)
    output = net.forward(origin_inputs.float())
    output.retain_grad()

    advnet = adj_weights(advnet, origin_inputs.float(), origin_outputs, 2.0, loss_maximize=True)
    advinput = fgsm_attack(advnet, origin_inputs.float(), origin_outputs, 0.01)

    advnet.train()
    adv_feats = {}
    def forward_hook_adv(module, data_input, data_output):
        # print('---day')
        mod_name = module.__class__.__name__
        adv_feats[mod_name] = data_output
        # fea = data_output[0].detach()
        # fea = fea.reshape(fea.shape[0], -1)
        # corr = torch.corrcoef(fea)
        # corr[torch.isnan(corr)] = 0
        # corr[torch.isinf(corr)] = 0
        # values = torch.linalg.eig(corr)[0]
        # # result = np.real(np.min(values)) / np.real(np.max(values))
        # result = torch.min(torch.real(values))
        # result_list.append(result)
    check_len = 0
    for name, modules in advnet.named_modules():
        check_len+=1
        modules.register_forward_hook(forward_hook_adv)
    adv_outputs = advnet.forward(advinput.detach())
    adv_outputs.retain_grad()
    # print('check len--------:',check_len)
    loss = ce_loss(output, origin_outputs) + ce_loss(adv_outputs, origin_outputs)
    loss.backward() 

    def croze(layer, layer_adv, feat, feat_adv):
        if layer.samples['weight'].grad is not None:
            w_sim = (1+cos_loss(layer_adv.samples['weight'].grad, layer.samples['weight'])).sum()
            sim = (torch.abs(cos_loss(layer_adv.samples['weight'].grad, layer.samples['weight'].grad))).sum()
            feat_sim = (1+cos_loss(feat_adv, feat)).sum()
            return torch.abs(w_sim * sim * feat_sim)
        else:
            return torch.zeros_like(layer.samples['weight'])

    grads_abs = get_layer_metric_array_adv_feats(net, advnet, feats, adv_feats, croze, mode, search_space) 

    # apply signs of all params
    nonlinearize(net, signs)
    nonlinearize(advnet, adv_signs)
    
    del feats, adv_feats
    del advnet
    
    return grads_abs


@indicator('jacobian', bn=False, mode='param')
def compute_synflow_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None, search_space='AutoFormer'):

    device = inputs.device
    origin_inputs, origin_outputs = inputs, targets
    origin_inputs.requires_grad = True
    
    cos_loss = nn.CosineSimilarity(dim=0)
    ce_loss = nn.CrossEntropyLoss()
    
    #convert params to their abs. Keep sign for converting it back.
    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    #convert to orig values
    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])


    
    # keep signs of all params
    signs = linearize(net)
    
    # Compute gradients with input of 1s 
    net.zero_grad()
    net.float()
    
    
    output = net.forward(origin_inputs.float())
    output.retain_grad()
    R = reg(origin_inputs.float(), output)
    loss = ce_loss(output, origin_outputs) + lambda_JR*R
    loss.backward() 

     

    # apply signs of all params
    nonlinearize(net, signs)
    
    
    return R.item()

@indicator('meco', bn=False, mode='param')
def get_score_Meco_8x8_opt_weight_result(net, x, target,mode, split_data=1, loss_fn=None, search_space='AutoFormer'):
    result_list = []
   
    def forward_hook(module, data_input, data_output):
        fea = data_output[0].detach()
        fea = fea.reshape(fea.shape[0], -1)
        random_indices_8_a = torch.randperm(fea.shape[0])[:8]  # Get 8 random indices
        random_tensor_8_a_fea = fea[random_indices_8_a]
        corr = torch.corrcoef(random_tensor_8_a_fea)
        # print(corr.shape)
        corr[torch.isnan(corr)] = 0
        corr[torch.isinf(corr)] = 0
        values = torch.linalg.eig(corr)[0]
        # result = np.real(np.min(values)) / np.real(np.max(values))
        result = (fea.shape[0]/8)*torch.min(torch.real(values))
        result_list.append(result)
  
    for name, modules in net.named_modules():
        modules.register_forward_hook(forward_hook)

    N = x.shape[0]
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data
        y = net(x[st:en])
    
    results = torch.tensor(result_list)
    results = results[torch.logical_not(torch.isnan(results))]
    v = torch.sum(results)
    result_list.clear()
    return v.item()