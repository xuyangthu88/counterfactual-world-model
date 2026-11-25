"""Select policies from observed factual histories, then write choices before native execution."""
import argparse
import json
import numpy as np
import torch
from .fit import data,restore,pairs,subset
from .model import preference
from .native import OUT,DOMAINS

@torch.no_grad()
def select(history):
    net=restore(history=history)
    tag='history-' if history else ''
    results={}
    for domain in DOMAINS:
        d=data(domain,'observe','cpu',history=history)
        i,j=pairs(d['a'].size(1),'cpu')
        choices={key:[] for key in ['marginal','mean','law','kl']}
        for ix in torch.arange(len(d['x'])).split(32):
            row=subset(d,ix)
            o=net(row)
            _,_,p=net.contrasts(o,i,j,row['role'][...,0])
            for key,v in preference(net,o,i,j,p).items():
                winner=v.argmax(1)
                if history:
                    winner=torch.where(row['alive'].bool(),winner,0)
                choices[key].extend(winner.tolist())
        design=json.loads((OUT/(tag+'bank.json' if history else 'expanded-bank.json')).read_text())
        table=np.array(design['score' if history else 'screen_score'])[design['ids']]
        best=int(table.mean(1).argmax())
        choices['population']=[best]*len(d['x'])
        choices['actor']=[0]*len(d['x'])
        results[domain]=choices
    (OUT/(tag+'choices.json')).write_text(json.dumps(results))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',action='store_true')
    torch.set_num_threads(2)
    select(parser.parse_args().history)
