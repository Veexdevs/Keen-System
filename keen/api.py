"""HTTP API for Keen-1. Authentication is mandatory except for health."""
import json, os, time, hmac, re, threading, uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from keen.model import load_model
from keen.inference import Engine, GenerationState
from keen.memory import Memory

MODEL_ID = 'keen-1-experimental'

class Message(BaseModel):
    role: Literal['system','user','assistant']
    content: str = Field(min_length=1,max_length=4000)
class Generation(BaseModel):
    model: str = MODEL_ID
    max_tokens: int = Field(default=64,ge=1,le=128)
    temperature: float = Field(default=0.8,ge=0,le=2)
    top_k: int = Field(default=40,ge=1,le=259)
    stream: bool = False
    @field_validator('model')
    @classmethod
    def model_supported(cls, value):
        if value != MODEL_ID: raise ValueError('Unknown model')
        return value
class ChatRequest(Generation):
    messages: list[Message] = Field(min_length=1,max_length=20)
    conversation_id: str | None = Field(default=None,min_length=1,max_length=80,pattern=r'^[a-zA-Z0-9_-]+$')
class CompletionRequest(Generation):
    prompt: str = Field(min_length=1,max_length=4000)
class EmbeddingRequest(BaseModel):
    model: str = MODEL_ID
    input: str | list[str]
    @field_validator('model')
    @classmethod
    def model_supported(cls,value):
        if value != MODEL_ID: raise ValueError('Unknown model')
        return value
    @field_validator('input')
    @classmethod
    def validate_input(cls,value):
        items=[value] if isinstance(value,str) else value
        if not 1 <= len(items) <= 16 or any(not s or len(s)>4000 for s in items): raise ValueError('Provide 1-16 nonempty strings, max 4000 characters each')
        return value
class MemoryInput(BaseModel):
    content: str = Field(min_length=1,max_length=2000)

@asynccontextmanager
async def lifespan(app):
    raw = os.getenv('KEEN_API_KEYS')
    if raw:
        keys=json.loads(raw)
        if not isinstance(keys,dict) or not keys or any(not isinstance(k,str) or len(k)<16 or not isinstance(v,str) or not v or len(v)>80 for k,v in keys.items()):
            raise RuntimeError('KEEN_API_KEYS must map keys of at least 16 characters to nonempty tenant IDs')
    else:
        key=os.getenv('KEEN_API_KEY','')
        if len(key)<16: raise RuntimeError('Set KEEN_API_KEY (at least 16 characters) or KEEN_API_KEYS')
        keys={key:'default'}
    app.state.keys=keys
    app.state.engine=Engine(load_model(os.getenv('KEEN_MODEL_DIR','models/keen-1')))
    app.state.memory=Memory(os.getenv('KEEN_DB_PATH','data/keen.sqlite3'))
    app.state.rate=defaultdict(deque); app.state.rate_lock=threading.Lock()
    yield

app=FastAPI(title='Keen API',version='1.0.0',lifespan=lifespan,description='API experimental de inferência local; modelo pequeno e não confiável.')

def tenant(authorization: str | None = Header(default=None)):
    token = authorization[7:] if authorization and authorization.startswith('Bearer ') else ''
    identity = next((v for k,v in app.state.keys.items() if hmac.compare_digest(k.encode(),token.encode())), None)
    if identity is None: raise HTTPException(401, 'Invalid API key', headers={'WWW-Authenticate':'Bearer'})
    now=time.monotonic()
    with app.state.rate_lock:
        q=app.state.rate[identity]
        while q and q[0] <= now-60: q.popleft()
        if len(q)>=int(os.getenv('KEEN_RATE_LIMIT','30')): raise HTTPException(429, 'Rate limit exceeded')
        q.append(now)
    return identity

@app.get('/v1/health')
def health(): return {'status':'ok','model':MODEL_ID}
@app.get('/v1/models')
def models(_=Depends(tenant)): return {'object':'list','data':[{'id':MODEL_ID,'object':'model','owned_by':'keen'}]}

@app.post('/v1/embeddings')
def embeddings(req: EmbeddingRequest, _=Depends(tenant)):
    items=[req.input] if isinstance(req.input,str) else req.input
    return {'object':'list','model':MODEL_ID,'data':[{'object':'embedding','index':i,'embedding':v} for i,v in enumerate(app.state.engine.embeddings(items))]}

@app.put('/v1/memories/{key}')
def put_memory(key: str, body: MemoryInput, who=Depends(tenant)):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',key): raise HTTPException(400,'Invalid memory key')
    app.state.memory.put(who,key,body.content); return {'key':key,'status':'stored'}
@app.delete('/v1/memories/{key}')
def delete_memory(key: str, who=Depends(tenant)):
    return {'deleted':bool(app.state.memory.delete(who,key))}
@app.delete('/v1/conversations/{conversation_id}')
def delete_conversation(conversation_id: str, who=Depends(tenant)):
    return {'deleted_turns':app.state.memory.delete_conversation(who,conversation_id)}

def stream_result(req, prompt, kind, who=None, conversation=None, user=None):
    request_id=('chatcmpl-' if kind=='chat' else 'cmpl-')+uuid.uuid4().hex
    created=int(time.time())
    def events():
        parts=[]
        state=GenerationState()
        if kind=='chat':
            yield 'data: '+json.dumps({'id':request_id,'object':'chat.completion.chunk','created':created,'model':MODEL_ID,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})+'\n\n'
        for piece in app.state.engine.text_chunks(prompt,req.max_tokens,req.temperature,req.top_k,state):
            parts.append(piece)
            choice={'index':0,'delta':{'content':piece},'finish_reason':None} if kind=='chat' else {'index':0,'text':piece,'finish_reason':None}
            yield 'data: '+json.dumps({'id':request_id,'object':'chat.completion.chunk' if kind=='chat' else 'text_completion','created':created,'model':MODEL_ID,'choices':[choice]},ensure_ascii=False)+'\n\n'
        if conversation: app.state.memory.append_pair(who,conversation,user,''.join(parts))
        choice={'index':0,'delta':{},'finish_reason':state.finish_reason} if kind=='chat' else {'index':0,'text':'','finish_reason':state.finish_reason}
        yield 'data: '+json.dumps({'id':request_id,'object':'chat.completion.chunk' if kind=='chat' else 'text_completion','created':created,'model':MODEL_ID,'choices':[choice]})+'\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

def complete(req,prompt,kind,who=None,conversation=None,user=None):
    if req.stream: return stream_result(req,prompt,kind,who,conversation,user)
    state=GenerationState()
    text=''.join(app.state.engine.text_chunks(prompt,req.max_tokens,req.temperature,req.top_k,state))
    if conversation: app.state.memory.append_pair(who,conversation,user,text)
    choice={'index':0,'message':{'role':'assistant','content':text},'finish_reason':state.finish_reason} if kind=='chat' else {'index':0,'text':text,'finish_reason':state.finish_reason}
    return {'id':('chatcmpl-' if kind=='chat' else 'cmpl-')+uuid.uuid4().hex,'object':'chat.completion' if kind=='chat' else 'text_completion','created':int(time.time()),'model':MODEL_ID,'choices':[choice]}

@app.post('/v1/completions')
def completions(req: CompletionRequest, who=Depends(tenant)):
    return complete(req,req.prompt,'completion')

@app.post('/v1/chat/completions')
def chat(req: ChatRequest, who=Depends(tenant)):
    if req.messages[-1].role != 'user': raise HTTPException(400,'Last message must be from user')
    user=req.messages[-1].content
    history=app.state.memory.history(who,req.conversation_id) if req.conversation_id else []
    relevant=app.state.memory.recall(who,user) if req.conversation_id else []
    # Follow the same format used in training; context is clipped to the model window by the engine.
    sections=['Memória: '+v for _,v in relevant]
    sections += [('Usuário: ' if role=='user' else 'Keen: ')+content for role,content in history]
    labels={'user':'Usuário: ','assistant':'Keen: ','system':'Sistema: '}
    sections += [labels[m.role]+m.content for m in req.messages]
    prompt='\n'.join(sections)+'\nKeen: '
    return complete(req,prompt,'chat',who,req.conversation_id,user)
