"""Add the change in a learned displacement to each native motion step."""
import torch

from whatif.effects.model import WIDTH
from whatif.mechanism.model import mlp
from whatif.shortstep.model import ShortWorld


class CumulativeWorld(ShortWorld):
    def __init__(self):
        super().__init__()
        self.displacement=mlp(2*WIDTH+2,3,64)
        torch.nn.init.zeros_(self.displacement[-1].weight)
        torch.nn.init.zeros_(self.displacement[-1].bias)

    def cumulative_displacement(self,state,initial,time):
        now,start=state[:,:,0],initial[:,:,0]
        context=torch.cat([start,time],-1)
        displacement=self.displacement(torch.cat([now,context],-1))
        origin=self.displacement(torch.cat([start,context],-1))
        return time[...,1:2]*(displacement-origin)
