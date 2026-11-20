"""Evaluate held-out cross entropy and emit samples for manual behavioral review."""
import argparse, json, random
from keen.model import load_model
from keen.training import examples, evaluate
from keen.inference import Engine

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='models/keen-1'); p.add_argument('--data',default='datasets/personality.jsonl'); p.add_argument('--seed',type=int,default=42); p.add_argument('--samples',type=int,default=2); p.add_argument('--max-tokens',type=int,default=32); a=p.parse_args()
    rows=examples(a.data); random.Random(a.seed).shuffle(rows); val=rows[max(1,int(len(rows)*0.9)):]
    model=load_model(a.model); print(json.dumps({'validation':evaluate(model,val),'note':'Perplexity does not measure factuality, reasoning, safety or personality.'},ensure_ascii=False))
    engine=Engine(model)
    for prompt in ['O que é coragem?', 'Como devo lidar com um erro?', 'Qual é o sentido da vida?'][:a.samples]:
        response=''.join(engine.text_chunks('Usuário: '+prompt+'\nKeen: ',a.max_tokens,0))
        print(json.dumps({'prompt':prompt,'response':response,'review':'manual'},ensure_ascii=False))
if __name__=='__main__': main()
