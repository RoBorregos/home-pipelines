import asyncio
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()
clients = set()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        clients.remove(ws)

async def run_script():
    process = await asyncio.create_subprocess_exec(
        "python", "-u", "-m", "uvicorn", "script_notebook:app", "--port", "8000",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    while True:
        line = await process.stdout.readline()
        if not line:
            break
        line = line.decode().strip()
        print(line)
        for ws in clients.copy():
            try:
                await ws.send_text(line)
            except:
                clients.remove(ws)

async def main():
    # Create task to run the script
    asyncio.create_task(run_script())

    # Start uvicorn asynchronously
    config = uvicorn.Config(app, host="127.0.0.1", port=9000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
