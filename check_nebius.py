"""Run this first: python check_nebius.py  (tests chat + embeddings + lists models)"""
import os
from openai import OpenAI

c = OpenAI(base_url=os.getenv("NEBIUS_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
           api_key=os.environ["NEBIUS_API_KEY"])
chat = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
emb = os.getenv("EMBED_MODEL", "gemini-embedding-001")

try:
    print("Models available:", [m.id for m in c.models.list().data][:40])
except Exception as e:
    print("Could not list models:", e)

r = c.chat.completions.create(model=chat, messages=[{"role": "user", "content": "Say hi in 3 words"}])
print("CHAT OK:", r.choices[0].message.content)
e = c.embeddings.create(model=emb, input=["hello"])
print("EMBED OK, dims:", len(e.data[0].embedding))
