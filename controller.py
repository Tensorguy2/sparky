import subprocess
import os
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
import uvicorn

API_KEY = "d-voice-key"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI(title="PS Voice Controller API", dependencies=[Depends(get_api_key)])

process = None

@app.get("/status")
async def get_status():
    global process
    if process and process.poll() is None:
        return {"running": True, "pid": process.pid}
    return {"running": False, "pid": None}

@app.post("/start")
async def start_server():
    global process
    if process and process.poll() is None:
        return JSONResponse({"message": "Server is already running"}, status_code=400)
    
    try:
        # Assuming this script is run from the project root
        process = subprocess.Popen(["python", "src/server.py"])
        return {"message": "Server started"}
    except Exception as e:
        return JSONResponse({"message": f"Failed to start: {e}"}, status_code=500)

@app.post("/stop")
async def stop_server():
    global process
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        process = None
        return {"message": "Server stopped"}
    return JSONResponse({"message": "Server is not running"}, status_code=200)

@app.post("/restart")
async def restart_server():
    await stop_server()
    return await start_server()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
