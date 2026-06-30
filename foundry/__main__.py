"""Allow running: python -m foundry.main  (delegates to uvicorn)."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("foundry.main:app", host="0.0.0.0", port=8000, reload=False)
