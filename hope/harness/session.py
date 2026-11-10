"""Sesión del harness F9: Hope y Qwen trabajan en paralelo sobre el mismo pedido y se comparan.

Ciclo de una tarea (``HarnessSession.task``):

1. **Contexto**: listado de la carpeta, archivos que el pedido menciona (leídos a través del
   guardián) y la nota más parecida del cuaderno. Lo que está fuera de la carpeta queda
   pendiente de aprobación humana y no se lee.
2. **En paralelo**: Hope responde con ``BudgetedResponder`` (velocidad objetivo, p. ej. 200
   tokens/s; el tiempo sobrante se usa para revisar con el consejo y la base de conocimiento) y
   Qwen responde como maestro (cliente OpenAI-compatible, también con permiso del guardián).
3. **Comparación**, la misma vara para los dos (ninguno se aprueba a sí mismo):
   crítica CRE/HCRI de los pasos, consenso (¿coinciden?), pruebas en el sandbox sobre una copia
   de la carpeta con las ediciones de cada uno (si hay comando de pruebas), respuesta esperada
   (si se conoce) y, si Qwen respondió con un episodio, su verificación con herramientas.
   **Gana la respuesta verificada**; si las dos lo están y coinciden, gana Hope.
4. **Registro** en PESAM de todo: pedido, contexto, ambas respuestas, veredictos, pruebas,
   elección y acciones sobre la carpeta.
5. **Episodio de entrenamiento**: si las respuestas difieren y la elegida pasó la verificación,
   la diferencia se guarda como episodio (``episodios.jsonl``) con la respuesta descartada y sus
   errores. Sin verificación no se guarda nada para entrenar (§10.1: el maestro no es la autoridad).
6. Las ediciones de la respuesta elegida quedan **propuestas** con su diff; se escriben solo si la
   persona las acepta (``apply``).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hope.constraints.critic import critique
from hope.council.coordinator import VERDICT_SCORE, _norm
from hope.epistemic import EpistemicStatus
from hope.harness.tools import Edit, FolderTools, HarnessDenied, NeedsApproval
from hope.runtime.budget import SpeedTarget
from hope.runtime.toolcalls import MAX_OUT, ToolRunner
from hope.tools.catalog import ToolDenied

EDIT_RE = re.compile(r"```editar[ \t:]+([^\n`]+?)[ \t]*\n(.*?)```", re.DOTALL)
PATH_RE = re.compile(r"(?<![\w/.~-])((?:~|\.{1,2})?/?[\w./-]*\w\.\w{1,8})(?![\w/])")
RESULT_RE = re.compile(r"Resultado:\s*(.+)")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")

FORMAT = (
    "Formato de respuesta: si hay cálculos, un paso por línea «Paso n: expresión = valor»; al final, "
    "una línea «Resultado: …». Para cambiar un archivo de la carpeta, escribí el contenido completo "
    "nuevo en un bloque\n```editar ruta/relativa\n<contenido>\n```\no, para cambiar solo una parte, un bloque "
    "buscar/reemplazar:\nruta/relativa\n<<<<<<< BUSCAR\n<líneas actuales, únicas en el archivo>\n=======\n"
    "<líneas nuevas>\n>>>>>>> REEMPLAZAR\nNo se escribe nada sin que la persona acepte el cambio.")
HOPE_TOOLS = ("Herramientas de carpeta: ⟦leer: ruta⟧, ⟦listar: ruta⟧, ⟦buscar: patrón | ruta⟧, "
              "⟦ejecutar: comando⟧ (en el sandbox, sobre una copia), ⟦estructura: ruta⟧, ⟦definicion: nombre⟧, "
              "⟦referencias: nombre⟧, ⟦tipos: ruta.py⟧, ⟦recuperar: descripción⟧ (código parecido del proyecto).")
QWEN_SYSTEM = (
    "Sos Qwen y trabajás junto a Hope, una red pequeña que aprende, sobre una carpeta local. Tu "
    "respuesta se compara con la de Hope y se verifica con herramientas: no inventes datos ni "
    "archivos; si falta información, decilo. Respondé en español. " + FORMAT)


def extract_answer(text: str) -> str | None:
    m = list(RESULT_RE.finditer(text or ""))
    return m[-1].group(1).strip() if m else None


def extract_edits(text: str) -> dict[str, str]:
    return {m.group(1).strip(): m.group(2) for m in EDIT_RE.finditer(text or "")}


def topic_for(request: str) -> str:
    """Tema aproximado para la base de conocimiento y el historial del consejo."""
    r = request.lower()
    if re.search(r"\.(py|js|ts|c|cpp|h|rs|go|java|sh|toml|ya?ml|json)\b|funci[oó]n|prueba|test|c[oó]digo|error|bug",
                 r):
        return "código"
    if re.search(r"\d\s*[-+*/×^]\s*\d|cu[aá]nto es|calcul|ohm|voltios|amper", r):
        return "matemática"
    return "general"


# ---------------------------------------------------------------------------
@dataclass
class HarnessToolRunner(ToolRunner):
    """Las herramientas de siempre más las de carpeta, para las llamadas ⟦…⟧ de Hope."""
    folder: FolderTools | None = None
    pending: list = field(default_factory=list)          # ids de solicitudes que quedaron pendientes

    def _editor_for(self, root: str):
        """Herramientas de editor sobre la carpeta del harness: cada lectura pasa por ``FolderTools._check``."""
        if self.folder is None:
            return super()._editor_for(root)
        from hope.tools.editor import EditorTools
        f = self.folder
        return EditorTools(f.resolve(root) if f.inside(root) else f.root,
                           lambda p, op: f._check("file.read", {"path": p, "op": op}))

    def _dispatch(self, name: str, arg: str) -> str:
        from hope.runtime.toolcalls import EDITOR_TOOLS
        if self.folder is not None and name in EDITOR_TOOLS:
            try:
                return super()._dispatch(name, arg)
            except NeedsApproval as e:
                self.pending.append(e.request_id)
                return f"pendiente de aprobación humana (solicitud {e.request_id})"
            except HarnessDenied as e:
                raise ToolDenied(str(e)) from e
        if self.folder is None or name not in ("leer", "listar", "buscar", "ejecutar"):
            return super()._dispatch(name, arg)
        f = self.folder
        try:
            if name == "leer":
                return f.read(arg.strip() or ".", max_bytes=MAX_OUT)
            if name == "listar":
                return "\n".join(f.list(arg.strip() or "."))
            if name == "buscar":
                pattern, _, root = arg.partition("|")
                hits = f.search(pattern.strip(), root.strip() or ".")
                return "\n".join(f"{h['path']}:{h['line']}: {h['text']}" for h in hits) or "sin coincidencias"
            r = f.run(arg)
            return (f"código {r.returncode}{' (tiempo agotado)' if r.timed_out else ''}\n"
                    + (r.stdout + r.stderr)[-MAX_OUT:])
        except NeedsApproval as e:
            self.pending.append(e.request_id)
            return f"pendiente de aprobación humana (solicitud {e.request_id})"
        except HarnessDenied as e:
            raise ToolDenied(str(e)) from e


@dataclass
class Side:
    """Una de las dos respuestas (Hope o Qwen) con su verificación."""
    source: str
    text: str = ""
    answer: str | None = None
    seconds: float = 0.0
    error: str | None = None
    meta: dict = field(default_factory=dict)
    verdict: str = ""
    issues: list[str] = field(default_factory=list)
    edits: dict[str, str] = field(default_factory=dict)
    tests: dict | None = None
    correct: bool | None = None
    episode_status: str | None = None
    verified: bool = False
    score: float = 0.0

    def key(self) -> str:
        """Lo que se compara: el resultado y, si hay, las ediciones."""
        edits = json.dumps({k: v.strip() for k, v in sorted(self.edits.items())}, ensure_ascii=False)
        return _norm(self.answer) + ("|" + edits if self.edits else "")


@dataclass
class TaskResult:
    id: str
    request: str
    topic: str
    hope: Side
    qwen: Side | None
    chosen: str | None                 # "hope" | "qwen" | None
    agree: bool
    reasons: list[str]
    context_files: list[str]
    pending_approvals: list[str]
    proposed_edits: list[int]          # índices en session.edits
    episode_saved: bool = False

    @property
    def differ(self) -> bool:
        return self.qwen is not None and not self.agree

    @property
    def winner(self) -> Side | None:
        return {"hope": self.hope, "qwen": self.qwen}.get(self.chosen or "")


class HarnessSession:
    def __init__(self, folder: FolderTools, responder=None, teacher=None, *, pesam, catalog=None,
                 knowledge=None, notebook=None, target: SpeedTarget | None = None,
                 episodes_path: str | Path, test_command: str | None = None, max_context_bytes: int = 6000):
        self.folder, self.responder, self.teacher = folder, responder, teacher
        self.pesam, self.catalog = pesam, catalog
        self.knowledge, self.notebook = knowledge, notebook
        self.target = target or SpeedTarget(tokens_per_s=200.0)
        self.episodes_path = Path(episodes_path)
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.test_command, self.max_context_bytes = test_command, max_context_bytes
        self.edits: list[Edit | None] = []
        self.history: list[TaskResult] = []
        self.tool_pending: list[str] = []
        self.coder = None                    # hope.runtime.autodepurar.Coder para /autodepurar

    def runner_factory(self, catalog, memory=None) -> HarnessToolRunner:
        """Para ``build_local_council(..., runner_factory=session.runner_factory)``."""
        return HarnessToolRunner(catalog, memory, folder=self.folder, pending=self.tool_pending,
                                 teacher=self.teacher)          # ⟦preguntar_maestro⟧ con el maestro de la sesión

    # --- contexto ---------------------------------------------------------------------
    def context_for(self, request: str) -> tuple[str, list[str], list[str]]:
        parts, files, pending = [f"Carpeta de trabajo: {self.folder.root.name}/"], [], []
        try:
            parts.append("Contenido: " + ", ".join(self.folder.list(".")[:60]))
        except (NeedsApproval, HarnessDenied):
            pass
        budget = self.max_context_bytes
        for token in dict.fromkeys(m.group(1) for m in PATH_RE.finditer(request)):
            p = self.folder.resolve(token)
            if not p.is_file():
                continue
            try:
                text = self.folder.read(token, max_bytes=max(budget, 200))
            except NeedsApproval as e:
                pending.append(e.request_id)
                parts.append(f"(«{token}» está fuera de la carpeta: pendiente de aprobación {e.request_id})")
                continue
            except HarnessDenied as e:
                parts.append(f"(«{token}»: {e})")
                continue
            files.append(self.folder.rel(p))
            parts.append(f"--- {self.folder.rel(p)} ---\n{text}")
            budget -= len(text.encode())
            if budget <= 0:
                break
        if self.notebook is not None:
            note = self.notebook.lookup(request)
            if note is not None:
                parts.append(f"Nota del cuaderno («{note.concept}»): {note.text[:600]}")
        return "\n".join(parts), files, pending

    # --- las dos respuestas -------------------------------------------------------------
    def _ask_hope(self, topic: str, request: str, context: str) -> Side:
        if self.responder is None:
            return Side("hope", error="sin modelo de Hope cargado")
        t0 = time.perf_counter()
        try:
            r = self.responder.respond(topic, request, self.target, context=f"{context}\n{HOPE_TOOLS}\n{FORMAT}")
        except Exception as e:                                          # noqa: BLE001
            return Side("hope", error=f"{type(e).__name__}: {e}"[:300], seconds=time.perf_counter() - t0)
        return Side("hope", r.text, r.answer, r.elapsed_s, meta={
            "agente": r.agent, "presupuesto_s": r.budget_s, "rondas": [asdict(x) for x in r.rounds],
            "corte": r.stop_reason, "tokens_por_s": r.effective_tokens_per_s})

    def _ask_qwen(self, request: str, context: str) -> Side:
        from hope.teacher.episode import Episode, EpisodeParseError, parse_episode
        t0 = time.perf_counter()
        msgs = [{"role": "system", "content": QWEN_SYSTEM},
                {"role": "user", "content": f"{context}\n\nPedido: {request}"}]
        try:
            reply = self.teacher.ask(msgs, purpose="harness")
        except Exception as e:                                          # noqa: BLE001 (incluye TeacherDenied)
            return Side("qwen", error=f"{type(e).__name__}: {e}"[:300], seconds=time.perf_counter() - t0)
        side = Side("qwen", reply.text, extract_answer(reply.text), time.perf_counter() - t0,
                    meta={"modelo": reply.model, "uso": reply.usage})
        if "```editar" not in reply.text and "{" in reply.text:
            try:                                                        # respondió con un episodio JSON
                ep = Episode(parse_episode(reply.text), source_model=reply.model)
            except EpisodeParseError:
                ep = None
            if ep is not None:
                if self.catalog is not None:
                    from hope.tools.verify import verify_episode
                    verify_episode(self.catalog, ep)
                    side.episode_status = ep.verification.get("status")
                side.text, side.answer = ep.training_text(), str(ep.data.get("resultado"))
                side.meta["episodio"] = ep.data
                # Verificar los pasos prueba que son coherentes, no que respondan a ESTE pedido.
                asked = set(NUM_RE.findall(request))
                if asked and not asked <= set(NUM_RE.findall(str(ep.data.get("problema", "")))):
                    side.meta["otro_problema"] = True
        return side

    # --- verificación ---------------------------------------------------------------------
    def _evaluate(self, side: Side, request: str, expected: str | None, test_command: str | None) -> None:
        if side.error:
            side.verdict, side.verified, side.score = "sin_respuesta", False, -1.0
            return
        crit = critique(request, side.text, side.answer)
        side.verdict, side.issues = crit.verdict, [i.detail for i in crit.issues]
        if side.meta.get("otro_problema"):
            side.issues.insert(0, "el episodio resuelve otro problema, no el pedido")
        side.edits = extract_edits(side.text)
        if "<<<<<<<" in side.text:                          # bloques buscar/reemplazar → contenido completo
            from hope.tools.diff_edits import EditError
            try:
                for e in self.folder.propose_search_replace(side.text, source=side.source):
                    side.edits.setdefault(e.rel, e.new)
            except EditError as e:
                side.issues.append(f"edición buscar/reemplazar inválida: {e}")
            except (NeedsApproval, HarnessDenied) as e:
                side.issues.append(f"edición buscar/reemplazar sin permiso: {e}")
        if test_command and side.edits:
            try:
                r = self.folder.run(test_command, overlay=side.edits)
                side.tests = {"ok": r.ok, "codigo": r.returncode, "salida": (r.stdout + r.stderr)[-2000:],
                              "tiempo_agotado": r.timed_out}
            except (NeedsApproval, HarnessDenied) as e:
                side.tests = {"ok": False, "codigo": None, "salida": str(e)}
        if expected is not None:
            from hope.teacher.cycle import answers_match
            side.correct = answers_match(side.answer, expected)
        if side.tests is not None:
            side.verified = side.tests["ok"] and side.verdict != "errores"
        elif side.correct is not None:
            side.verified = side.correct and side.verdict != "errores"
        else:
            side.verified = side.verdict == "ok"
        if side.episode_status == EpistemicStatus.REFUTADO.value or side.meta.get("otro_problema"):
            side.verified = False
        side.score = (3.0 if side.verified else 0.0) + VERDICT_SCORE.get(side.verdict, 0.0) \
            + (1.0 if side.tests and side.tests["ok"] else 0.0) + (1.0 if side.correct else 0.0)

    @staticmethod
    def _choose(hope: Side, qwen: Side | None) -> tuple[str | None, bool, list[str]]:
        if qwen is None:
            return ("hope" if not hope.error else None), False, ["sin maestro: solo Hope"]
        if hope.error and qwen.error:
            return None, False, ["ninguno respondió"]
        agree = not hope.error and not qwen.error and bool(hope.answer or hope.edits) and hope.key() == qwen.key()
        if agree:
            return "hope", True, ["coinciden" + (" y están verificadas" if hope.verified else " (sin verificar)")]
        if hope.verified != qwen.verified:
            w = "hope" if hope.verified else "qwen"
            return w, False, [f"difieren: gana {w} porque su respuesta está verificada"]
        w = "hope" if hope.score > qwen.score or (hope.score == qwen.score and hope.verified) else "qwen"
        why = "las dos están verificadas" if hope.verified else "ninguna está verificada: se muestra la de mayor puntaje"
        return w, False, [f"difieren; {why} → {w}"]

    # --- tarea completa --------------------------------------------------------------------
    def task(self, request: str, *, expected: str | None = None, test_command: str | None = None) -> TaskResult:
        tid = uuid.uuid4().hex[:10]
        topic = topic_for(request)
        n_pending = len(self.tool_pending)
        context, files, pending = self.context_for(request)
        # Hope y Qwen al mismo tiempo. Qwen (que solo hace una llamada HTTP) va en otro hilo; Hope
        # queda en este, porque la base de conocimiento (SQLite) solo se usa desde el hilo que la creó.
        with ThreadPoolExecutor(max_workers=1) as pool:
            fq = pool.submit(self._ask_qwen, request, context) if self.teacher is not None else None
            hope = self._ask_hope(topic, request, context)
            qwen = fq.result() if fq is not None else None
        pending += self.tool_pending[n_pending:]
        command = test_command or self.test_command
        for side in filter(None, (hope, qwen)):
            self._evaluate(side, request, expected, command)
        chosen, agree, reasons = self._choose(hope, qwen)
        result = TaskResult(tid, request, topic, hope, qwen, chosen, agree, reasons, files, pending, [])

        winner = result.winner
        if winner is not None:                                       # ediciones: propuestas, no escritas
            for rel, content in winner.edits.items():
                try:
                    self.edits.append(self.folder.propose_edit(rel, content, source=winner.source))
                    result.proposed_edits.append(len(self.edits) - 1)
                except NeedsApproval as e:
                    result.pending_approvals.append(e.request_id)
                except HarnessDenied as e:
                    reasons.append(f"edición de {rel} rechazada: {e}")
        if result.differ and winner is not None and winner.verified:
            self._save_episode(result)
            result.episode_saved = True
        self._learn(result)
        self.pesam.record(
            "harness.tarea",
            EpistemicStatus.RESPALDADO if winner is not None and winner.verified else EpistemicStatus.INFERIDO,
            tarea=tid, pedido=request, tema=topic, archivos_contexto=files, comando_pruebas=command,
            esperado=expected, hope=asdict(hope), qwen=asdict(qwen) if qwen else None, elegida=chosen,
            coinciden=agree, difieren=result.differ, motivos=reasons, episodio_guardado=result.episode_saved,
            ediciones_propuestas=[self.edits[i].rel for i in result.proposed_edits],
            aprobaciones_pendientes=result.pending_approvals)
        self.history.append(result)
        return result

    def _save_episode(self, r: TaskResult) -> None:
        w, loser = r.winner, (r.qwen if r.chosen == "hope" else r.hope)
        text = f"Tarea: {r.request}\n{w.text.strip()}"
        if w.answer and not RESULT_RE.search(w.text):
            text += f"\nResultado: {w.answer}"
        ep = {"id": f"ep_{r.id}", "ts": time.time(), "tarea": r.id, "tema": r.topic, "problema": r.request,
              "archivos_contexto": r.context_files, "fuente": w.source, "respuesta": w.text,
              "resultado": w.answer, "ediciones": w.edits,
              "descartada": {"fuente": loser.source, "respuesta": loser.text, "resultado": loser.answer,
                             "errores": loser.issues, "veredicto": loser.verdict},
              "verificacion": {"veredicto": w.verdict, "pruebas": w.tests, "correcta": w.correct,
                               "episodio": w.episode_status, "estado": EpistemicStatus.RESPALDADO.value},
              "texto": text}
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
        self.pesam.record("harness.episodio", EpistemicStatus.RESPALDADO, tarea=r.id, episodio=ep["id"],
                          fuente=w.source, archivo=str(self.episodes_path))

    def _learn(self, r: TaskResult) -> None:
        """Lo verificado de Qwen entra a la base de conocimiento; los errores de Hope, como lecciones."""
        if self.knowledge is None or r.winner is None or not r.winner.verified or r.chosen != "qwen":
            return
        self.knowledge.learn_solution(r.topic, r.request, r.qwen.text, r.qwen.answer, verified=True,
                                      source="qwen")
        if r.hope.answer and r.hope.issues:
            self.knowledge.learn_lesson(r.topic, r.request, r.hope.answer, r.hope.issues, source="harness")

    # --- acciones de la persona ---------------------------------------------------------------
    def apply(self, index: int, *, approval_id: str | None = None) -> Edit:
        edit = self.edits[index] if 0 <= index < len(self.edits) else None
        if edit is None:
            raise IndexError(f"no hay un cambio propuesto con el número {index}")
        self.folder.apply_edit(edit, approval_id=approval_id)
        self.edits[index] = None
        self.pesam.record("harness.edicion", EpistemicStatus.OBSERVADO, archivo=edit.rel, fuente=edit.source,
                          diff=edit.diff, aplicada=True)
        return edit

    def apply_verified(self, indices: list[int], test_command: str | None = None, *,
                       approval_id: str | None = None) -> dict:
        """Aplica varios cambios propuestos, corre las pruebas y **vuelve atrás** si fallan."""
        edits = [self.edits[i] for i in indices if 0 <= i < len(self.edits) and self.edits[i] is not None]
        if not edits:
            raise IndexError("no hay cambios propuestos con esos números")
        r = self.folder.apply_edits_checked(edits, test_command or self.test_command, approval_id=approval_id)
        if r["aplicado"]:
            for i in indices:
                self.edits[i] = None
        self.pesam.record("harness.edicion", EpistemicStatus.RESPALDADO if r["aplicado"] else EpistemicStatus.REFUTADO,
                          archivos=r["archivos"], aplicada=r["aplicado"], revertida=r["revertido"],
                          pruebas=r["pruebas"])
        return r

    def autodebug(self, request: str, path: str, coder, *, test_command: str | None = None,
                  rounds: int = 3) -> dict:
        """Bucle de autodepuración sobre ``path`` (``hope/runtime/autodepurar.py``): el ``coder`` escribe
        bloques buscar/reemplazar, se prueban en el sandbox sobre una copia y se corrigen leyendo la
        salida de las pruebas. Las ediciones de la ronda que pasa quedan **propuestas** (no escritas)."""
        from hope.runtime.autodepurar import autodepurar_carpeta
        command = test_command or self.test_command
        if not command:
            raise ValueError("la autodepuración necesita un comando de pruebas")
        rep = autodepurar_carpeta(self.folder, coder, request, path, command, rounds=rounds)
        idx = []
        for e in rep.edits:
            self.edits.append(e)
            idx.append(len(self.edits) - 1)
        self.pesam.record("harness.autodepuracion", EpistemicStatus.RESPALDADO if rep.ok else EpistemicStatus.INFERIDO,
                          pedido=request, archivo=path, comando_pruebas=command, rondas=rep.rounds,
                          recuperado=[f"{r['ruta']}:{r['linea']}" for r in rep.retrieved], propuestas=idx)
        return {"ok": rep.ok, "propuestas": idx, "rondas": rep.rounds}

    def discard(self, index: int) -> Edit:
        edit = self.edits[index]
        self.edits[index] = None
        self.pesam.record("harness.edicion", EpistemicStatus.OBSERVADO, archivo=edit.rel, fuente=edit.source,
                          aplicada=False)
        return edit

    def approve(self, request_id: str, approver: str, note: str = "") -> None:
        """Solo desde la consola, por la persona (nunca lo llama el modelo)."""
        self.folder.guardian.approve(request_id, approver, note)
        self.folder.grant(request_id)
        self.pesam.record("harness.aprobacion", EpistemicStatus.OBSERVADO, solicitud=request_id,
                          aprobador=approver, decision="aprobada")

    def deny(self, request_id: str, approver: str, note: str = "") -> None:
        self.folder.guardian.deny(request_id, approver, note)
        self.folder.requests.pop(request_id, None)
        self.pesam.record("harness.aprobacion", EpistemicStatus.OBSERVADO, solicitud=request_id,
                          aprobador=approver, decision="denegada")
