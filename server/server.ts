import { ServerWebSocket } from "bun";

interface WebSocketData {
  id: string;
}

// Paths to SSL certificate files - update these paths to your actual certificate locations
const CERT_PATH = "/etc/letsencrypt/live/mcrouter.paoloose.site/fullchain.pem";
const KEY_PATH = "/etc/letsencrypt/live/mcrouter.paoloose.site/privkey.pem";

const clients = new Set<ServerWebSocket<WebSocketData>>();

const server = Bun.serve({
  port: 5060,
  hostname: "0.0.0.0",
  tls: {
    key: Bun.file(KEY_PATH),
    cert: Bun.file(CERT_PATH),
  },
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

console.log(`WebSocket broadcast server running on port ${server.port} with HTTPS`);
console.log(`Connect to: wss://0.0.0.0:${server.port}`);
console.log(`For local connections use: wss://localhost:${server.port}`);
console.log(`Certificate path: ${CERT_PATH}`);
console.log(`Private key path: ${KEY_PATH}`);
