"""Check the lab's math against TinyLM and accumulation/recomputation gradients."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
import train
from data import get_dataloader
from model import TinyLM
from worker import LabLM


def main():
    torch.set_num_threads(1)
    train.seed_everything(42)
    reference = TinyLM().cuda()
    loader=get_dataloader('train',32,128,0,42)
    x,y=next(iter(loader)); x,y=x.cuda(),y.cuda()
    _,ref_loss=reference(x,y)
    ref_loss.backward()
    results=[]
    for micro,checkpointing in ((32,False),(8,False),(32,True),(8,True)):
        model=LabLM().cuda()
        model.load_state_dict(reference.state_dict())
        model.activation_checkpointing=checkpointing
        total=0.0
        for offset in range(0,32,micro):
            _,loss=model(x[offset:offset+micro],y[offset:offset+micro])
            (loss/(32//micro)).backward()
            total+=loss.item()/(32//micro)
        error=0.0
        for p,q in zip(model.parameters(),reference.parameters()):
            error=max(error,(p.grad-q.grad).abs().max().item())
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-3,atol=2e-6)
        assert abs(total-ref_loss.item())<2e-5
        results.append(dict(microbatch=micro,activation_checkpointing=checkpointing,
                            loss=total,max_gradient_absolute_error=error))
        del model
    print(json.dumps(dict(reference_loss=ref_loss.item(),checks=results),indent=2))


if __name__=='__main__': main()
