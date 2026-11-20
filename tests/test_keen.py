import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from tinygrad import Tensor
from keen.tokenizer import encode, decode, BOS, EOS
from keen.model import Config, KeenModel, save_model, load_model
from keen.memory import Memory
from keen.inference import Engine


def test_tokenizer_unicode():
    s='Keen 🧠: reflexão e ação'
    assert decode(encode(s,bos=True,eos=True)) == s
    assert encode(s,bos=True,eos=True)[0]==BOS and encode(s,eos=True)[-1]==EOS

def test_causal_and_checkpoint(tmp_path):
    Tensor.manual_seed(7)
    model=KeenModel(Config(context=16,width=16,heads=4,layers=1))
    a=Tensor(np.array([[1,2,3,4]],dtype=np.int32)); b=Tensor(np.array([[1,2,3,5]],dtype=np.int32))
    out=model(a).numpy(); changed=model(b).numpy()
    assert out.shape==(1,4,259)
    np.testing.assert_allclose(out[:,:3,:],changed[:,:3,:],rtol=1e-5,atol=1e-5)
    save_model(model,tmp_path); restored=load_model(tmp_path)
    np.testing.assert_allclose(out,restored(a).numpy(),rtol=1e-5,atol=1e-5)
    assert len(Engine(restored).embeddings(['olá'])[0])==16

def test_training_updates_weights():
    from tinygrad import nn
    from keen.training import batch
    import random
    Tensor.manual_seed(9)
    model=KeenModel(Config(context=16,width=16,heads=4,layers=1))
    optimizer=nn.optim.Adam(nn.state.get_parameters(model), lr=0.001)
    before=model.output.weight.numpy().copy()
    x,y=batch([encode('Usuário: oi\nKeen: olá',bos=True,eos=True)],16,1,random.Random(1))
    with Tensor.train():
        optimizer.zero_grad()
        loss=model(x).sparse_categorical_crossentropy(y)
        loss.backward(); optimizer.step()
    assert np.isfinite(float(loss.numpy()))
    assert not np.array_equal(before,model.output.weight.numpy())

def test_memory_isolation_and_persistence(tmp_path):
    path=tmp_path/'db.sqlite'
    store=Memory(path); store.put('a','city','Moro em São Paulo'); store.put('b','city','Moro em Salvador')
    store.append_pair('a','c','oi','olá')
    store=Memory(path)
    assert store.recall('a','São Paulo')==[('city','Moro em São Paulo')]
    assert store.recall('b','São Paulo')==[]
    assert store.history('b','c')==[]
    assert store.history('a','c')==[('user','oi'),('assistant','olá')]
    store.put('a','city','Moro em Recife'); assert store.recall('a','Recife')
    assert store.delete('a','city')==1 and store.recall('a','Recife')==[]

def test_http_inference_stream_and_auth(monkeypatch,tmp_path):
    monkeypatch.setenv('KEEN_API_KEYS',json.dumps({'a'*20:'alice','b'*20:'bob'}))
    monkeypatch.setenv('KEEN_DB_PATH',str(tmp_path/'db.sqlite'))
    monkeypatch.setenv('KEEN_MODEL_DIR','models/keen-1')
    from keen.api import app
    with TestClient(app) as c:
        a={'Authorization':'Bearer '+'a'*20}; b={'Authorization':'Bearer '+'b'*20}
        assert c.get('/v1/health').status_code==200
        assert c.get('/v1/models').status_code==401
        assert c.get('/v1/models',headers=a).json()['data'][0]['id']=='keen-1-experimental'
        assert c.get('/v1/models',headers={'Authorization':'Bearer invalid'}).status_code==401
        assert c.put('/v1/memories/city',headers=a,json={'content':'cidade Curitiba'}).status_code==200
        assert app.state.memory.recall('alice','Curitiba') and not app.state.memory.recall('bob','Curitiba')
        emb=c.post('/v1/embeddings',headers=a,json={'input':['olá','mundo']}); assert emb.status_code==200 and len(emb.json()['data'])==2
        body={'messages':[{'role':'user','content':'Olá'}],'max_tokens':2,'temperature':0,'conversation_id':'test'}
        response=c.post('/v1/chat/completions',headers=a,json=body); assert response.status_code==200, response.text
        assert 'content' in response.json()['choices'][0]['message']
        assert len(app.state.memory.history('alice','test'))==2 and not app.state.memory.history('bob','test')
        body['stream']=True
        stream=c.post('/v1/chat/completions',headers=a,json=body); assert stream.status_code==200
        events=[x[6:] for x in stream.text.split('\n\n') if x.startswith('data: ')]
        assert events[-1]=='[DONE]'
        parsed=[json.loads(x) for x in events[:-1]]
        assert parsed[0]['choices'][0]['delta']['role']=='assistant'
        assert parsed[-1]['choices'][0]['finish_reason'] in ('stop','length')
        assert len(app.state.memory.history('alice','test'))==4
        completion=c.post('/v1/completions',headers=b,json={'prompt':'Oi','max_tokens':1,'temperature':0}); assert completion.status_code==200
        assert c.delete('/v1/conversations/test',headers=b).json()['deleted_turns']==0
        assert c.delete('/v1/conversations/test',headers=a).json()['deleted_turns']==4

def test_refuses_missing_auth(monkeypatch,tmp_path):
    monkeypatch.delenv('KEEN_API_KEYS',raising=False); monkeypatch.delenv('KEEN_API_KEY',raising=False)
    from keen.api import app
    with pytest.raises(RuntimeError,match='KEEN_API_KEY'):
        with TestClient(app): pass
