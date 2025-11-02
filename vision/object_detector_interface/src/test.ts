const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = (msg) => console.log("LOG:", msg.data);

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: "run",
    tags: ["test"]
  }));
};
