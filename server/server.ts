import { ServerWebSocket } from "bun";

interface WebSocketData {
  id: string;
}

// Paths to SSL certificate files - update these paths to your actual certificate locations
const CERT_PATH = "/etc/letsencrypt/live/mcrouter.paoloose.site/fullchain.pem";
const KEY_PATH = "/etc/letsencrypt/live/mcrouter.paoloose.site/privkey.pem";

const clients = new Set<ServerWebSocket<WebSocketData>>();

const server = Bun.serve({
  port: 6060,
  hostname: "0.0.0.0",
  tls: {
    key: Bun.file(KEY_PATH),
    cert: Bun.file(CERT_PATH),
  },
  fetch(req, server) {
    // Set CORS headers for all responses
    const headers = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
      // Allow connections from any origin to the WebSocket server
      "Content-Security-Policy": "default-src 'self'; connect-src *;"
    };

    // Handle OPTIONS requests (CORS preflight)
    if (req.method === "OPTIONS") {
      return new Response(null, { headers });
    }

    // Upgrade HTTP connection to WebSocket
    const success = server.upgrade(req, {
      data: {
        id: crypto.randomUUID()
      },
      headers // Add headers to the WebSocket upgrade response
    });

    if (success) {
      return undefined;
    }

    // If not a WebSocket request, return a standard response with CORS headers
    return new Response("WebSocket endpoint", {
      status: 200,
      headers
    });
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
console.log(`Connect to: wss://mcrouter.paoloose.site:${server.port}`);
console.log(`For local connections use: wss://localhost:${server.port}`);
console.log(`Certificate path: ${CERT_PATH}`);
console.log(`Private key path: ${KEY_PATH}`);
