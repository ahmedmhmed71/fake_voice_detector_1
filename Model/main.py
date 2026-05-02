from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
from predict import predict_audio

app = FastAPI()

# ✅ حل مشكلة CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ عرض صفحة الموقع
@app.get("/")
def home():
    return FileResponse("index.html")

# ✅ API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_audio(file_path)

    return result
