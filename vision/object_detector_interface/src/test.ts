export const ws = new WebSocket("ws://localhost:8000/ws");

export const wsOnMessage = ws.onmessage = (msg) => console.log("LOG:", msg.data);

export const wsOnOpen =
ws.onopen = (tag) => {
  ws.send(JSON.stringify({
    action: "run",
    tags: [tag]
  }));
};
