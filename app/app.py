"""
app/app.py — Optional FastAPI prediction endpoint.

Install extras:  pip install fastapi uvicorn
Run:             uvicorn app.app:app --reload
"""

# Uncomment to activate:
#
# import pickle
# from pathlib import Path
# import pandas as pd
# from fastapi import FastAPI
# from pydantic import BaseModel
#
# MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
# app = FastAPI(title="Home Credit Default Risk — Ganesh")
#
# with open(MODELS_DIR / "training_results.pkl", "rb") as f:
#     _results = pickle.load(f)
#
#
# class Application(BaseModel):
#     features: dict
#
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
#
#
# @app.post("/predict")
# def predict(application: Application):
#     # Wire up a fitted LightGBM or XGBoost model here
#     return {"default_probability": None}

print("app.py — uncomment FastAPI block and install fastapi/uvicorn to serve.")
