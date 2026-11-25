"""Portable inference shared by the native Python 3.10 simulator workers."""
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def graph_predict(p,x):
    a=np.broadcast_to(x[:,None],(len(x),3,x.shape[-1]))
    k=np.broadcast_to(np.array([-1,0,1],np.float32)[None,:,None],(len(x),3,1))
    h=np.concatenate((a,k),axis=-1)
    for i in [0,2,4]:
        h=h@p[f'g{i}'].T+p[f'g{i+1}']
        if i<4:h=h*sigmoid(h)
    prior=np.concatenate((a[...,14:16],a[...,16:17]/.47),axis=-1)*a[...,17:18]
    mu=(h[...,:3]+prior)*sigmoid(h[...,3:4])
    return np.concatenate((mu,sigmoid(h[...,4:5])),axis=-1)


def ensemble_predict(p,x):
    h=(x-p['xm'])/p['xs']
    for i in range(4):
        h=h@p[f'ew{i}']+p[f'eb{i}']
        if i<3:h=np.maximum(h,0.)
    mu,lv=np.split(h,2,axis=-1)
    lv=p['max_logvar']-np.logaddexp(0,p['max_logvar']-lv)
    lv=p['min_logvar']+np.logaddexp(0,lv-p['min_logvar'])
    return mu*p['ys']+p['ym'],np.exp(lv*.5)*p['ys']
