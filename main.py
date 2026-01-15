import os
import fitz
import faiss
import numpy as np
import pickle
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from mistralai import Mistral

app = FastAPI()

# Clients API
# Note : On garde Mistral pour les embeddings (gratuit/rapide) et OpenAI pour le Chat
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_mistral = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

class AskRequest(BaseModel):
    question: str
    chat_history: List[dict]
    user_profile: dict

class SubstanciaEngine:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.load_index()

    def get_embeddings(self, texts):
        res = client_mistral.embeddings.create(model="mistral-embed", inputs=texts)
        return [np.array(i.embedding, dtype="float32") for i in res.data]

    def load_index(self):
        if os.path.exists("faiss_index.idx"):
            self.index = faiss.read_index("faiss_index.idx")
            with open("chunks_meta.pkl", "rb") as f:
                self.chunks = pickle.load(f)

    def ingest(self, paths):
        self.chunks = []
        for path in paths:
            doc = fitz.open(path)
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if len(text) > 50:
                    self.chunks.append({"text": text, "source": os.path.basename(path), "page": i+1})
            doc.close()
        
        embeddings = np.vstack(self.get_embeddings([c["text"] for c in self.chunks]))
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, "faiss_index.idx")
        with open("chunks_meta.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        return f"✅ {len(self.chunks)} sections indexées."

engine = SubstanciaEngine()

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    os.makedirs("temp", exist_ok=True)
    paths = []
    for f in files:
        path = f"temp/{f.filename}"
        with open(path, "wb") as buffer: shutil.copyfileobj(f.file, buffer)
        paths.append(path)
    return {"status": engine.ingest(paths)}

@app.post("/ask")
async def ask(request: AskRequest):
    if engine.index is None:
        return {"answer": "Aucun document chargé."}

    # 1. Recherche Vectorielle
    q_emb = np.vstack(engine.get_embeddings([request.question])).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idx = engine.index.search(q_emb, 5)
    
    # Seuil de fiabilité (0.7)
    sources = [engine.chunks[i] for s, i in zip(scores[0], idx[0]) if s >= 0.7]
    
    if not sources:
        return {"answer": "Aucune source fiable trouvée. Pourriez-vous préciser votre question ?"}

    context_text = "\n".join([f"Source: {s['source']}, p.{s['page']}\nContenu: {s['text']}" for s in sources])
    profile_str = str(request.user_profile)

    # 2. Prompting ChatGPT
    system_prompt = f"""
    Tu es un Professeur Expert Universitaire.
    PROFIL ÉLÈVE : {profile_str}
    
    CONSIGNE STRICTE :
    - Utilise UNIQUEMENT le contexte suivant : {context_text}
    - Si tu ne trouves pas la réponse, dis : "Aucune source fiable trouvée..."
    - Formatage : LaTeX pour les maths ($inline$, $$bloc$$).
    
    STRUCTURE DE RÉPONSE :
    # Titre : [Sujet de la question]
    ## 1. Introduction
    ## 2. Développement (Détails techniques, matrices, tableaux)
    ## 3. Points clés
    ## 4. Exercice (1-2 questions + 1 Quiz vrai/faux + Correction)
    ## 5. Sources (Nom du fichier, Page)
    """

    response = client_openai.chat.completions.create(
        model="gpt-4-turbo", # ou "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ]
    )

    return {"answer": response.choices[0].message.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
