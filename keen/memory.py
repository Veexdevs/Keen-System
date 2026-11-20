"""Tenant-scoped durable SQLite memory. The tenant comes only from the API key."""
import sqlite3, time, re
from pathlib import Path

class Memory:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY, tenant TEXT NOT NULL, conversation TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS turns_lookup ON turns(tenant, conversation, id);
            CREATE TABLE IF NOT EXISTS memories (tenant TEXT NOT NULL, memory_key TEXT NOT NULL, content TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(tenant, memory_key));''')
    def connect(self):
        return sqlite3.connect(self.path, timeout=10)
    def history(self, tenant, conversation, limit=12):
        with self.connect() as db:
            rows = db.execute('SELECT role,content FROM turns WHERE tenant=? AND conversation=? ORDER BY id DESC LIMIT ?', (tenant, conversation, limit)).fetchall()
        return list(reversed(rows))
    def append_pair(self, tenant, conversation, user, assistant):
        with self.connect() as db:
            db.executemany('INSERT INTO turns(tenant,conversation,role,content,created) VALUES(?,?,?,?,?)', [(tenant,conversation,'user',user,time.time()),(tenant,conversation,'assistant',assistant,time.time())])
    def put(self, tenant, key, content):
        with self.connect() as db:
            db.execute('INSERT INTO memories VALUES(?,?,?,?) ON CONFLICT(tenant,memory_key) DO UPDATE SET content=excluded.content,updated=excluded.updated', (tenant,key,content,time.time()))
    def delete(self, tenant, key):
        with self.connect() as db: return db.execute('DELETE FROM memories WHERE tenant=? AND memory_key=?', (tenant,key)).rowcount
    def delete_conversation(self, tenant, conversation):
        with self.connect() as db: return db.execute('DELETE FROM turns WHERE tenant=? AND conversation=?', (tenant,conversation)).rowcount
    def recall(self, tenant, query, limit=3):
        with self.connect() as db:
            rows = db.execute('SELECT memory_key,content FROM memories WHERE tenant=? ORDER BY updated DESC LIMIT 500', (tenant,)).fetchall()
        terms = set(re.findall(r'\w+', query.lower()))
        scored = [(len(terms & set(re.findall(r'\w+', (k+' '+v).lower()))), k, v) for k,v in rows]
        return [(k,v) for score,k,v in sorted(scored, reverse=True) if score > 0][:limit]
