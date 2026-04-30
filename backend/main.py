from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from services.ptx_parser import PTXParser
from services.feature_extractor import FeatureExtractor

app = FastAPI(
    title="KernelSight API",
    description="PTX-level static analysis and ML-based performance prediction for CUDA kernels.",
    version="0.1.0"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PTXRequest(BaseModel):
    ptx_code: str

@app.post("/api/analyze")
def analyze_ptx(request: PTXRequest):
    """
    Parses PTX code, extracts analytical instruction counts, 
    and generates the feature vector used for ML constraint analysis.
    """
    parser = PTXParser()
    counts = parser.parse_text(request.ptx_code)
    
    extractor = FeatureExtractor()
    features = extractor.extract_features(counts)
    
    return {
        "instruction_counts": counts,
        "features": features,
        "status": "success"
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
