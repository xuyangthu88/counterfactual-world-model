"""Change the horizontal coordinate frame for an entire factual situation."""
import math

import torch


def rotate_layouts(d, probability):
    x=d['x']
    b=len(x)
    angle=(2*torch.rand(b,device=x.device)-1)*math.pi
    angle=angle*(torch.rand(b,device=x.device)<probability)
    c,s=angle.cos(),angle.sin()
    zero,one=torch.zeros_like(c),torch.ones_like(c)
    rotation=torch.stack([c,s,zero,-s,c,zero,zero,zero,one],-1).reshape(b,3,3)

    def vectors(v):
        return torch.einsum('b...d,bde->b...e',v.unflatten(-1,(-1,3)),rotation).flatten(-2)

    out=dict(d)
    for key in ('x','goal','fact','reset','pose','native_pose','native_off','off'):
        if key in d:
            out[key]=vectors(d[key])
    for key in ('a','fa'):
        flag,shift=d[key].split([1,6],-1)
        out[key]=torch.cat([flag,vectors(shift)],-1)
    return out
