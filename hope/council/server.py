"""Agentes como servidores separados (Hope «fragmentada por servidores»).

Cada agente puede correr en su propio proceso o máquina y atender pedidos HTTP
con JSON; el coordinador los usa igual que a los locales mediante ``RemoteAgent``.
Solo la biblioteca estándar. Por defecto escucha en 127.0.0.1: exponerlo en la red
es decisión del usuario (y pasa por el guardián como ``network.access``).

    POST /solve  {"topic": "...", "problem": "..."}  →  {"agent", "text", "answer", "seconds", "bytes"}
    GET  /perfil                                      →  perfil del agente
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hope.council.agents import Agent, AgentProfile, Candidate


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Levanta el servidor en un hilo y lo devuelve (``server.server_port`` tiene el puerto real)."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                          # noqa: N802
            if self.path == "/perfil":
                self._send(200, agent.describe())
            else:
                self._send(404, {"error": "ruta desconocida"})

        def do_POST(self):                                         # noqa: N802
            if self.path != "/solve":
                return self._send(404, {"error": "ruta desconocida"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                c = agent.solve(str(req["topic"]), str(req["problem"]))
                self._send(200, {"agent": c.agent, "text": c.text, "answer": c.answer,
                                 "seconds": c.seconds, "bytes": c.bytes_generated})
            except Exception as e:                                 # noqa: BLE001
                self._send(500, {"error": str(e)[:300]})

        def log_message(self, *args):                              # silencioso
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class RemoteAgent(Agent):
    """Agente que vive en otro proceso o máquina; misma interfaz que un ``Agent`` local."""

    def __init__(self, url: str, timeout_s: float = 300.0):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s
        with urllib.request.urlopen(self.url + "/perfil", timeout=timeout_s) as r:
            self.profile = AgentProfile(**json.loads(r.read()))
        self.solver = None

    def solve(self, topic: str, problem: str) -> Candidate:
        req = urllib.request.Request(self.url + "/solve", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"topic": topic, "problem": problem}).encode())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            d = json.loads(r.read())
        return Candidate(d["agent"], d["text"], d["answer"], d["seconds"], d["bytes"], {"remote": self.url})
