import asyncio
from fastapi import FastAPI, WebSocket
import uvicorn
import json
import signal

app = FastAPI()
clients = set()
process = None

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global process
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await asyncio.sleep(1)
            data = json.loads(await ws.receive_text())
            if data["action"] == "stop":
                if process:
                    process.kill()
                    await ws.send_text("Process terminated.")
                asyncio.create_task(restart_script())
                
                
    finally:
        clients.remove(ws)

async def run_script():
    global process
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

async def restart_script():
    global process
    if process and process.returncode is None:
        process.send_signal(signal.SIGTERM)
        await process.wait()
    await asyncio.sleep(1)
    asyncio.create_task(run_script())

async def main():
    # Create task to run the script
    asyncio.create_task(run_script())

    # Start uvicorn asynchronously
    config = uvicorn.Config(app, host="127.0.0.1", port=9000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
