import { ServerWebSocket } from "bun";

interface WebSocketData {
  id: string;
}

const clients = new Set<ServerWebSocket<WebSocketData>>();

const server = Bun.serve({
  port: 5060,
  hostname: "0.0.0.0",
  fetch(req, server) {
    // Upgrade HTTP connection to WebSocket
    const success = server.upgrade(req, {
      data: {
        id: crypto.randomUUID()
      }
    });

    if (success) {
      return undefined;
    }

    return new Response("WebSocket upgrade failed", { status: 400 });
  },
  websocket: {
    open(ws) {
      clients.add(ws);
      console.log(`Client ${ws.data.id} connected. Total clients: ${clients.size}`);
    },

    message(ws, message) {
      const messageStr = typeof message === 'string' ? message : message.toString();
      console.log(`Broadcasting message from ${ws.data.id}: ${messageStr}`);

      // Broadcast to all clients except the sender
      for (const client of clients) {
        if (client !== ws && client.readyState === 1) {
          client.send(messageStr);
        }
      }
    },

    close(ws) {
      clients.delete(ws);
      console.log(`Client ${ws.data.id} disconnected. Total clients: ${clients.size}`);
    },

    error(ws, error) {
      console.error(`WebSocket error for client ${ws.data.id}:`, error);
      clients.delete(ws);
    }
  }
});

console.log(`WebSocket broadcast server running on port ${server.port}`);
console.log(`Connect to: ws://0.0.0.0:${server.port}`);
console.log(`For local connections use: ws://localhost:${server.port}`);
