"""Consola del harness F9: el usuario escribe un pedido y Hope trabaja sobre una carpeta local fija.

    python -m hope.harness --carpeta ~/proyecto [--maestro configs/teacher.yaml] [--tokens-por-s 200]
                           [--modelo runs/hn1_200m] [--pruebas "python -m pytest -q"]

Cada pedido lo responden **Hope y Qwen en paralelo**; se muestran las dos respuestas, cuál se eligió
y por qué. Los cambios a archivos quedan propuestos con su diff hasta que la persona los acepta.

Comandos:
  /ayuda                  esta ayuda
  /notas [concepto]       cuaderno de Hope: resumen, repasos pendientes o una nota
  /aprobaciones           solicitudes del guardián pendientes de aprobación humana
  /aprobar <id>           aprobar una solicitud (lo hace la persona, no el modelo)
  /denegar <id>           rechazar una solicitud
  /cambios                cambios propuestos con su diff
  /aplicar <n>            escribir el cambio n en la carpeta (pasa por el guardián)
  /descartar <n>          descartar el cambio n
  /aplicar-probado <n…>   escribir los cambios, correr las pruebas y volver atrás si fallan
  /autodepurar <ruta> <pedido>  escribir → probar → leer el error → corregir (queda propuesto)
  /pruebas [comando]      ver o fijar el comando de pruebas (se corre en el sandbox, sobre una copia)
  /salir                  terminar (también Ctrl-D)

Seguridad: el directorio de trabajo del guardián pasa a ser la carpeta elegida. Leer o escribir
fuera de ella requiere aprobación; la configuración, el guardián y el ``.git`` de la carpeta están
protegidos; los comandos corren solo en el sandbox. Todo queda en PESAM y en la auditoría.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from guardian.core import Guardian, GuardianConfig
from hope.harness.session import HarnessSession, TaskResult
from hope.harness.tools import FolderTools, HarnessDenied, NeedsApproval
from hope.memory.pesam import ExperienceLog
from hope.runtime.budget import SpeedTarget

COMMANDS = __doc__.split("Comandos:\n", 1)[1].split("\n\nSeguridad", 1)[0]


def harness_guardian(config_path: str | Path, folder: Path) -> Guardian:
    """Guardián con la carpeta como directorio de trabajo; protege además el ``.git`` de la carpeta."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    base = Path(config_path).resolve().parent.parent              # rutas del YAML: relativas al repo
    absolute = lambda p: str(p if Path(p).is_absolute() else (base / p))  # noqa: E731
    cfg = GuardianConfig(**raw)
    cfg = replace(cfg, workdir=str(folder), budget_path=absolute(cfg.budget_path),
                  audit_path=absolute(cfg.audit_path), approvals_path=absolute(cfg.approvals_path),
                  protected_paths=[absolute(p) for p in cfg.protected_paths] + [str(folder / ".git")])
    return Guardian(cfg)


def _short(text: str, n: int = 1500) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f"\n… ({len(text) - n} caracteres más)"


def format_result(r: TaskResult, edits) -> str:
    out = []
    for side, name in ((r.hope, "Hope"), (r.qwen, "Qwen")):
        if side is None:
            continue
        mark = "✔ verificada" if side.verified else "✘ sin verificar"
        head = f"── {name} ({side.seconds:.2f} s, {side.verdict}, {mark}"
        if side.meta.get("agente"):
            head += f", agente {side.meta['agente']}, {side.meta.get('tokens_por_s', 0):.0f} tok/s"
        out.append(head + ") ──")
        out.append(side.error or _short(side.text))
        if side.issues:
            out.append("  crítica: " + "; ".join(side.issues[:3]))
        if side.tests is not None:
            out.append(f"  pruebas: {'pasan' if side.tests['ok'] else 'fallan'} (código {side.tests['codigo']})")
    who = {"hope": "Hope", "qwen": "Qwen"}.get(r.chosen or "", "ninguna")
    out.append(f"══ Elegida: {who}. " + " ".join(r.reasons))
    if r.episode_saved:
        out.append("   Diferencia guardada como episodio de entrenamiento (verificado).")
    for i in r.proposed_edits:
        out.append(f"   Cambio propuesto [{i}] {edits[i].rel}  → /aplicar {i}  o  /descartar {i}")
    for rid in r.pending_approvals:
        out.append(f"   Pendiente de aprobación humana: {rid}  → /aprobar {rid}")
    return "\n".join(out)


class Console:
    def __init__(self, session: HarnessSession, *, approver: str | None = None, input_fn=None, print_fn=print):
        self.s, self.print = session, print_fn
        self.input = input_fn or (lambda prompt: input(prompt))
        self.approver = approver or getpass.getuser()

    def handle(self, line: str) -> bool:
        """Procesa una línea. Devuelve False para terminar."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            r = self.s.task(line)
            self.print(format_result(r, self.s.edits))
            return True
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        try:
            return self._command(cmd.lower(), arg)
        except (NeedsApproval, HarnessDenied, KeyError, IndexError, ValueError, RuntimeError) as e:
            self.print(f"⚠ {e}")
            return True

    def _command(self, cmd: str, arg: str) -> bool:
        s = self.s
        if cmd in ("/salir", "/exit", "/quit"):
            return False
        if cmd == "/ayuda":
            self.print(COMMANDS)
        elif cmd == "/notas":
            nb = s.notebook
            if nb is None:
                self.print("sin cuaderno")
            elif arg:
                n = nb.lookup(arg)
                self.print(f"{n.concept}: {n.text}" if n else "no hay nota sobre eso")
            else:
                st = nb.stats()
                self.print(f"{st['notas']} notas, {st['automatizadas']} automatizadas, {st['consultas']} consultas")
                for n in nb.due(10):
                    self.print(f"  · para repasar: {n.concept} (consultada {n.lookups} veces)")
        elif cmd == "/aprobaciones":
            pend = s.folder.guardian.pending()
            if not pend:
                self.print("no hay solicitudes pendientes")
            for rid, req in pend.items():
                a = req["action"]
                self.print(f"{rid}  {a['kind']}  {a['params'].get('path') or a['params'].get('argv', '')}  "
                           f"— {'; '.join(req['reasons'])}")
        elif cmd == "/aprobar":
            s.approve(arg, self.approver)
            self.print(f"aprobada {arg}: la próxima vez que se pida exactamente esa acción, se ejecuta una vez")
        elif cmd == "/denegar":
            s.deny(arg, self.approver)
            self.print(f"denegada {arg}")
        elif cmd == "/cambios":
            live = [(i, e) for i, e in enumerate(s.edits) if e is not None]
            if not live:
                self.print("no hay cambios propuestos")
            for i, e in live:
                self.print(f"[{i}] {e.rel} (de {e.source})\n{e.diff or '(sin diferencias)'}")
        elif cmd == "/aplicar":
            e = s.apply(int(arg))
            self.print(f"escrito {e.rel}")
        elif cmd == "/aplicar-probado":
            r = s.apply_verified([int(x) for x in arg.split()])
            t = r["pruebas"] or {}
            self.print(("escritos " + ", ".join(r["archivos"]) + " (pruebas OK)") if r["aplicado"] else
                       f"las pruebas fallaron: se volvió atrás\n{_short(t.get('salida', ''), 800)}")
        elif cmd == "/autodepurar":
            path, _, req = arg.partition(" ")
            coder = getattr(s, "coder", None)
            if coder is None:
                self.print("sin programador: hace falta --modelo o un maestro")
            else:
                r = s.autodebug(req, path, coder)
                self.print(("pasa las pruebas; " if r["ok"] else "no logró pasar las pruebas; ")
                           + (f"cambios propuestos {r['propuestas']} (/cambios, /aplicar-probado)" if r["propuestas"]
                              else f"{len(r['rondas'])} rondas"))
        elif cmd == "/descartar":
            e = s.discard(int(arg))
            self.print(f"descartado {e.rel}")
        elif cmd == "/pruebas":
            if arg:
                s.test_command = arg
            self.print(f"comando de pruebas: {s.test_command or '(ninguno)'}")
        else:
            self.print(f"comando desconocido {cmd}; /ayuda para ver la lista")
        return True

    def loop(self) -> None:
        self.print(f"Hope — harness sobre {self.s.folder.root}  (/ayuda para ver los comandos)")
        while True:
            try:
                line = self.input("› ")
            except (EOFError, KeyboardInterrupt):
                self.print()
                break
            if not self.handle(line):
                break


def build_session(args) -> HarnessSession:
    from hope.memory.knowledge import KnowledgeBase, Notebook
    from hope.tools.builtin import default_catalog

    folder = Path(args.carpeta).expanduser().resolve()
    guardian = harness_guardian(args.guardian, folder)
    out = Path(args.registro)
    out.mkdir(parents=True, exist_ok=True)
    pesam = ExperienceLog(out / "pesam.jsonl")
    kb = KnowledgeBase(path=str(out / "conocimiento.sqlite"))
    nb = Notebook(kb)
    catalog = default_catalog(guardian)
    teacher = None
    if args.maestro and args.maestro.lower() != "ninguno":
        from hope.teacher.client import load_teacher
        teacher = load_teacher(args.maestro, guardian)
    tools = FolderTools(folder, guardian, pesam)
    session = HarnessSession(tools, None, teacher, pesam=pesam, catalog=catalog, knowledge=kb, notebook=nb,
                             target=SpeedTarget(tokens_per_s=args.tokens_por_s),
                             episodes_path=out / "episodios.jsonl", test_command=args.pruebas)
    if args.modelo or args.preset:
        from hope.council.agents import build_local_council
        from hope.runtime.budget import BudgetedResponder
        model = load_model(args.modelo, args.preset, args.device)
        agents = build_local_council(model, catalog, nb, runner_factory=session.runner_factory)
        session.responder = BudgetedResponder(agents, knowledge=kb)
        from hope.runtime.autodepurar import ModelCoder
        session.coder = ModelCoder(model, grammar=True)
    elif teacher is not None:
        from hope.runtime.autodepurar import TeacherCoder
        session.coder = TeacherCoder(teacher)
    pesam.record("harness.sesion", carpeta=str(folder), maestro=args.maestro, modelo=args.modelo,
                 tokens_por_s=args.tokens_por_s, pruebas=args.pruebas)
    return session


def load_model(run: str | None, preset_name: str | None, device: str | None = None):
    import torch
    from hope.core.hn1 import HN1Config, HN1Model, preset
    if run:
        ckpt = torch.load(Path(run) / "model.pt", map_location="cpu", weights_only=False)
        model = HN1Model(HN1Config.from_dict(ckpt["summary"]["config"]))
        model.load_state_dict(ckpt["state_dict"])
    else:
        print(f"aviso: sin --modelo; Hope usa un {preset_name} SIN entrenar (solo para probar el harness)",
              file=sys.stderr)
        model = HN1Model(preset(preset_name))
    if device:
        model.to(device)
    return model.eval()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hope.harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carpeta", required=True, help="carpeta local fija sobre la que trabaja Hope")
    ap.add_argument("--maestro", default="configs/teacher.yaml", help="config del maestro (o «ninguno»)")
    ap.add_argument("--tokens-por-s", type=float, default=200.0, help="velocidad objetivo de Hope")
    ap.add_argument("--modelo", default=None, help="corrida con model.pt (p. ej. runs/hn1_200m)")
    ap.add_argument("--preset", default=None, help="sin --modelo: preset sin entrenar (p. ej. hn1-tiny)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--guardian", default="configs/guardian.yaml")
    ap.add_argument("--registro", default="runs/harness", help="PESAM, base de conocimiento y episodios")
    ap.add_argument("--pruebas", default=None, help="comando de pruebas, p. ej. «python -m pytest -q»")
    args = ap.parse_args(argv)
    Console(build_session(args)).loop()


if __name__ == "__main__":
    main()
