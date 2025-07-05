from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

BAAIModel = SentenceTransformer("BAAI/bge-large-en-v1.5")
allMiniModel = SentenceTransformer("all-MiniLM-L6-v2")

class SimilarityInput(BaseModel):
    resume: str
    job_description: str

app = FastAPI()

@app.post("/calculate_similarity/")
async def calculate_similarity(input_data: SimilarityInput):
    print(input_data.resume,input_data.job_description)

    # encode both texts to get their embeddings
    resume_embedding1 = BAAIModel.encode(input_data.resume, convert_to_tensor=True)
    job_embedding1 = BAAIModel.encode(input_data.job_description, convert_to_tensor=True)

    resume_embedding2 = allMiniModel.encode(input_data.resume, convert_to_tensor=True)
    job_embedding2 = allMiniModel.encode(input_data.job_description, convert_to_tensor=True)

    # compute cosine similarity
    BAAI_model_similarity_score = util.pytorch_cos_sim(resume_embedding1, job_embedding1).item()
    all_mini_model_similarity_score = util.pytorch_cos_sim(resume_embedding2, job_embedding2).item()

    # calculate the average of both of the cosine similarities
    combined_similarity = (BAAI_model_similarity_score + all_mini_model_similarity_score) / 2

    return {"similarity_score": combined_similarity}


