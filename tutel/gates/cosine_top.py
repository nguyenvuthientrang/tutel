# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, proj_dim=256, init_t=0.5, form=3, epsw=1e-4, epsx=1e-4, **options):
        super(CosineTopKGate, self).__init__()
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

        self.form = form
        self.epsw = epsw
        self.epsx = epsx

        print("==========Perturbing using form {} with eps_w {} and eps_x {}==========".format(self.form, self.epsw, self.epsx))

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            cosine_projector = self.cosine_projector.float()
            sim_matrix = self.sim_matrix.float()
        else:
            cosine_projector = self.cosine_projector
            sim_matrix = self.sim_matrix

        Wx = cosine_projector(x)
        epsx = torch.ones_like(Wx) * self.epsx
        epsw = torch.ones_like(sim_matrix) * self.epsw

        if self.form == 2:
            logits = torch.matmul(self._normalize_add(Wx, p=2.0, dim=1, pertube_eps=epsx),
                                  self._normalize_add(sim_matrix, p=2.0, dim=1, pertube_eps=epsw))           
        if self.form == 3:
            logits = torch.matmul(F.normalize(Wx, p=2.0, dim=1, eps=self.epsx),
                                  self._normalize_add(sim_matrix, p=2.0, dim=1, pertube_eps = epsw))        
            
        # logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
        #                       F.normalize(sim_matrix, dim=0))
        # logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        # logits = logits * logit_scale
        return logits
    
    def _normalize_add(self, input, p: float = 2.0, dim: int = 1, pertube_eps = 1e-4): 

        denum = input.norm(p, dim, keepdim=True).expand_as(input) + pertube_eps
        return input / denum


Gate = CosineTopKGate
