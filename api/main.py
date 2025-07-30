from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="FilmFusion Recommendation API",
    description="Serve personalized and content-based recommendations"
)
app.include_router(router)

# For Uvicorn:
# uvicorn src.api.main:app --reload
