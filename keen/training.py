"""Training, validation, checkpointing and resuming with local JSONL data."""
import argparse, json, random, math
from pathlib import Path
import numpy as np
from tinygrad import Tensor, nn
from keen.model import Config, KeenModel, load_model, save_model
from keen.tokenizer import encode

def examples(path):
    result=[]
    for n,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        p, r = row['prompt'].strip(), row['response'].strip()
        if not p or not r: raise ValueError(f'Empty example at line {n}')
        result.append(encode('Usuário: '+p+'\nKeen: '+r, bos=True, eos=True))
    if len(result)<2: raise ValueError('Need at least two examples for validation')
    return result

def batch(rows, context, batch_size, rng):
    # Pack each independently to avoid crossing document boundaries. Predict all bytes.
    xs, ys = [], []
    for _ in range(batch_size):
        seq = rng.choice(rows)
        start = rng.randrange(max(1, len(seq)-context))
        part = seq[start:start+context+1]
        # Pad target with ignore_index (-1); input padding is never scored.
        xs.append((part[:-1]+[258]*context)[:context]); ys.append((part[1:]+[-1]*context)[:context])
    return Tensor(np.array(xs,dtype=np.int32)), Tensor(np.array(ys,dtype=np.int32))

def evaluate(model, rows, batches=4, batch_size=4):
    rng=random.Random(111); losses=[]
    for _ in range(batches):
        x,y=batch(rows,model.config.context,batch_size,rng)
        losses.append(float(model(x).sparse_categorical_crossentropy(y).numpy()))
    loss=sum(losses)/len(losses)
    return {'loss':loss, 'perplexity':math.exp(min(loss,20))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='datasets/personality.jsonl'); ap.add_argument('--output',default='models/keen-1'); ap.add_argument('--resume'); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--lr',type=float,default=0.001); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--eval-every',type=int,default=25); args=ap.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1: ap.error('steps, batch-size and eval-every must be positive')
    Tensor.manual_seed(args.seed); rng=random.Random(args.seed)
    rows=examples(args.data); rng.shuffle(rows); split=max(1,int(len(rows)*0.9)); train,val=rows[:split],rows[split:]
    model=load_model(args.resume) if args.resume else KeenModel(Config())
    opt=nn.optim.Adam(nn.state.get_parameters(model), lr=args.lr)
    print('Initial validation:',evaluate(model,val),flush=True)
    with Tensor.train():
        for step in range(1,args.steps+1):
            x,y=batch(train,model.config.context,args.batch_size,rng)
            opt.zero_grad(); loss=model(x).sparse_categorical_crossentropy(y); loss.backward(); opt.step()
            if step % args.eval_every == 0 or step == args.steps:
                print(f'step={step} train_loss={float(loss.numpy()):.4f} val={evaluate(model,val)}',flush=True)
                save_model(model,args.output)
    (Path(args.output)/'training.json').write_text(json.dumps({'dataset':str(args.data),'steps':args.steps,'seed':args.seed,'resume':args.resume,'validation':evaluate(model,val)},indent=2),encoding='utf-8')
if __name__=='__main__': main()
