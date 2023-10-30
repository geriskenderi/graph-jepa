import torch

def hyperbolic_dist(gt, target):
    bs, _, _ = gt.size()
    gt, target = gt.view(bs, -1), target.view(bs, -1) # Bs x Num_targets x D
    nom  = 2 * (torch.linalg.norm(gt-target, dim=1)**2)
    denom = ((1-torch.linalg.norm(gt, dim=1)**2)*(1-torch.linalg.norm(target, dim=1)**2))
    hdist = torch.acosh(1. + nom/denom)
    
    return hdist.mean()

def unithyper_geodesic(p):
  x = p[:, 0]
  y = p[:, 1]
  result = (y * torch.log(torch.abs(y * torch.sqrt(x**2 + y**2) + torch.abs(y) * x)) + (x * torch.sqrt(x**2 + y**2) / torch.abs(y))) / 2
  return result