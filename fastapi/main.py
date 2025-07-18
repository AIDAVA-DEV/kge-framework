from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import rdflib
from rdflib import Graph
import torch
import numpy as np
from src.TorusE import TorusE
import os

app = FastAPI(title="impute-code")

# Global variables
model = None
entity2id = None
relation2id = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default parameters
RDF_FILE = "patient_icd_data.ttl"
PREDICATE = "hasCode"
RANGE_CODES = ["A00.0", "B01.9", "C50.9", "D50.0", "E11.9", "F32.0", "G40.9", "I10", "J45.9", "K29.7", "M54.9", "N39.0", "R51", "J30.1", "E78.0"]

# Pydantic models
class TrainRequest(BaseModel):
    predicate: str = PREDICATE
    range_codes: List[str] = RANGE_CODES

class ImputeRequest(BaseModel):
    patient_id: str
    predicate: str = PREDICATE
    range_codes: List[str] = RANGE_CODES
    k: int = 5

def load_rdf_and_prepare_data(rdf_file_path: str, predicate: str):
    g = Graph()
    g.parse(rdf_file_path, format="turtle")
    
    patient_ns = rdflib.Namespace("http://example.org/patient/")
    icd_ns = rdflib.Namespace("http://hl7.org/fhir/icd/")
    sphn = rdflib.Namespace("http://www.sphn.ch/v1/")
    
    patients = set()
    codes = set()
    triples = []
    
    for s, p, o in g:
        if p == sphn[predicate]:
            patient_id = str(s).split("/")[-1]
            code = str(o).split("/")[-1].replace("-", ".")
            patients.add(patient_id)
            codes.add(code)
            triples.append((patient_id, predicate, code))
    
    all_entities = list(patients | codes)
    entity2id = {ent: idx for idx, ent in enumerate(all_entities)}
    relation2id = {predicate: 0}
    
    triple_indices = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]
    
    np.random.seed(42)
    np.random.shuffle(triple_indices)
    return triple_indices, len(all_entities), len(relation2id), entity2id, relation2id

def predict_top_k_entities(model, source: str, predicate: str, 
                          entity2id: dict, relation2id: dict, range_entities: List[str], 
                          device: torch.device, k: int = 5):
    model.eval()
    with torch.no_grad():
        if source not in entity2id:
            raise HTTPException(status_code=400, detail=f"Source entity {source} not found.")
        if predicate not in relation2id:
            raise HTTPException(status_code=400, detail=f"Predicate {predicate} not found.")
        h_idx = torch.tensor([entity2id[source]], device=device)
        r_idx = torch.tensor([relation2id[predicate]], device=device)
        
        range_indices = [entity2id[ent] for ent in range_entities if ent in entity2id]
        if not range_indices:
            raise HTTPException(status_code=400, detail="No valid range entities found.")
        t_idx = torch.tensor(range_indices, device=device)
        
        h_idx = h_idx.repeat(len(range_indices))
        r_idx = r_idx.repeat(len(range_indices))
        
        batch_size = 1
        num_samples = len(range_indices)
        scores = model.predict(h_idx, r_idx, t_idx, batch_size, num_samples)
        
        scores = scores.squeeze(0)
        top_k_indices = torch.argsort(scores, descending=False)[:k]
        top_k_scores = scores[top_k_indices]
        
        max_score = torch.max(scores).item()
        top_k_scores = max_score - top_k_scores
        
        id2entity = {idx: ent for ent, idx in entity2id.items()}
        top_k_entities = [id2entity[range_indices[idx]] for idx in top_k_indices]
        
        return list(zip(top_k_entities, top_k_scores.tolist()))

@app.post("/train")
async def train_model(file: UploadFile = File(...), request: TrainRequest = TrainRequest()):
    global model, entity2id, relation2id
    
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        train_triples, num_entities, num_relations, entity2id, relation2id = load_rdf_and_prepare_data(
            temp_file_path, request.predicate)
        
        emb_dim = 50
        lr = 1e-2
        train_batch_size = 16
        num_epochs = 50
        
        model = TorusE(num_entities, num_relations, device, emb_dim=emb_dim, lr=lr)
        train_losses = model._train(train_triples, train_batch_size=train_batch_size, num_epochs=num_epochs)
        
        os.remove(temp_file_path)
        
        return {"status": "Model trained successfully", "last_train_loss": float(train_losses[-1])}
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/impute")
async def impute_codes(request: ImputeRequest):
    global model, entity2id, relation2id
    
    if model is None or entity2id is None or relation2id is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
    
    try:
        top_k_predictions = predict_top_k_entities(
            model, request.patient_id, request.predicate, entity2id, relation2id, 
            request.range_codes, device, k=request.k
        )
        return [
            {"icd_code": icd_code, "score": score}
            for icd_code, score in top_k_predictions
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))