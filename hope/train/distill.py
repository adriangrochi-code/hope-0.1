"""Destilación de probabilidades del maestro con parches = tokens de Qwen (pedido del usuario).

Con ``patching="tokens"`` (``hope/core/tokpatch.py``) cada parche de Hope termina donde termina
un token de Qwen. La salida global de un parche ve exactamente los bytes que vio Qwen al predecir
el **próximo token**, así que se le puede pedir lo mismo: una cabeza liviana
(``HN1Model.distill_head``) proyecta la salida global al vocabulario de Qwen y se entrena con KL
contra la distribución del maestro. La pérdida total es la de siempre (bytes) más ``λ·KL``.

**Maestro elegible** (``--teacher``): Qwen2.5 (por defecto) o NVIDIA Nemotron 3 Nano 4B
(``nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16``, híbrido Mamba–transformer: ``--trust-remote-code``).
El parcheo del alumno tiene que usar **el mismo tokenizador** que el maestro
(``HN1Config.tokenizer``); ``hope.bench.tokens_vs_bytes`` lo toma de ``meta.json``.

**Costo de la cabeza.** Proyectar al vocabulario completo del maestro cuesta ``d × V`` parámetros:
Qwen2.5 (V = 151.936 en la salida del modelo) 38,9M con d = 256 (5 veces hn1-cpu) y 156M con
d = 1024 (hn1-200m); Nemotron 3 Nano (V = 131.072) 33,6M y 134M.
Por eso se usa un **vocabulario reducido**: los K tokens más frecuentes del corpus según el
maestro, más una clase «otro» que junta el resto. Con K = 4096: 1,05M parámetros con d = 256 y
4,2M con d = 1024, y cubre ~90 % de la masa del maestro en el corpus de prueba (medido por
``vocab_coverage``). La cabeza **solo se usa al entrenar**: no participa en la generación y se puede
descartar al exportar, así que no cuenta en el tamaño del modelo que corre.

**Logits offline.** El maestro corre una vez sobre el corpus (CPU, ``transformers``) y se
guardan solo los top-k de cada token, alineados al corpus por posición de byte:

    carpeta/
      meta.json     modelo, k, corpus (tamaño y sha256), tokens, bytes por token
      ends.npy      int64 [N]      posición (byte, absoluta en el .bin) del último byte de cada token
      ids.npy       uint32 [N, k]  ids de Qwen del top-k del token siguiente
      logp.npy      float16 [N, k] log-probabilidades (sin renormalizar) de esos ids
      freq.npy      float64 [V]    masa total que el maestro dio a cada id (para elegir el vocabulario)

    8 + 6·k bytes por token: 56 bytes con k = 8 (~15 bytes por byte de texto en español).

Una entrada en ``ends[i] = e`` es la distribución de Qwen para el token que empieza en ``e + 1``
dado todo el documento hasta ``e``. Un parche de Hope que termina en ``e`` usa esa entrada; si un
parche no coincide con un final de token del maestro (raro: la regla causal coincide en >99,9 %)
o es el último de un documento, no recibe KL.

    python -m hope.train.distill logits --data data/tvb/mix/train.bin --out data/tvb/maestro --k 8
    python -m hope.train.distill vocab --logits data/tvb/maestro --k 4096

**llama.cpp.** Con ``llama-cpp-python`` (``Llama(..., logits_all=True)``) se obtiene lo mismo de
un GGUF cuantizado (más rápido en CPU). No se implementó acá: ``transformers`` en bf16 da ~1.200
tokens/s con Qwen2.5-0.5B en 4 núcleos, suficiente para un corpus de experimento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TEACHER = "Qwen/Qwen2.5-0.5B"


# ---------------------------------------------------------------------------
# Generación de logits offline
# ---------------------------------------------------------------------------
def _token_ends(patcher, doc: bytes) -> tuple[list[int], list[int]] | None:
    """Ids y finales (byte, relativos al documento) de la tokenización completa de Qwen."""
    ids, ends, pos = [], [], 0
    text = doc.decode("utf-8", "surrogateescape")
    for piece in patcher.pieces(text):
        lengths, toks = patcher.piece_tokens(piece.encode("utf-8", "surrogateescape"))
        for n, i in zip(lengths, toks):
            pos += n
            ends.append(pos - 1)
            ids.append(i)
    return (ids, ends) if pos == len(doc) else None


@torch.no_grad()
def build_teacher_logits(data: str | Path, out: str | Path, *, teacher: str = DEFAULT_TEACHER, k: int = 8,
                         ctx: int = 512, stride: int = 384, batch: int = 4, max_bytes: int | None = None,
                         dtype: str = "bfloat16", threads: int | None = None, trust_remote_code: bool = False,
                         log=print) -> dict:
    """Corre el maestro sobre ``data`` (un ``.bin`` con documentos separados por 0x00) y guarda el
    top-k del próximo token en cada token. Documentos largos: ventanas de ``ctx`` tokens que avanzan
    ``stride`` (cada token se emite una sola vez, con al menos ``ctx − stride`` tokens de contexto
    salvo al principio del documento). Se puede retomar: guarda por partes."""
    from transformers import AutoModelForCausalLM
    from hope.core.tokpatch import get_patcher
    if threads:
        torch.set_num_threads(threads)
    raw = Path(data).read_bytes()
    if max_bytes:
        raw = raw[:max_bytes]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    patcher = get_patcher(teacher)
    model = AutoModelForCausalLM.from_pretrained(teacher, dtype=getattr(torch, dtype),
                                                 trust_remote_code=trust_remote_code).eval()
    vocab = model.config.vocab_size
    # (tokens de la ventana, desde qué índice se emiten, finales absolutos de esos tokens)
    jobs: list[tuple[list[int], int, list[int]]] = []
    start = 0
    skipped = 0
    n_tokens = 0
    for doc in raw.split(b"\x00"):
        if doc:
            te = _token_ends(patcher, doc)
            if te is None:
                skipped += 1
            else:
                ids, ends = te
                n_tokens += len(ids)
                abs_ends = [start + e for e in ends]
                # el último token de un documento no recibe entrada (después viene el separador)
                s = 0
                while True:
                    window = ids[s:s + ctx]
                    first = 0 if s == 0 else ctx - stride
                    last = min(len(window), len(ids) - 1 - s)
                    if last > first:
                        jobs.append((window, first, abs_ends[s + first:s + last]))
                    if s + ctx >= len(ids):
                        break
                    s += stride
        start += len(doc) + 1
    log(f"{len(jobs)} ventanas, {n_tokens:,} tokens, {skipped} documentos omitidos (no coinciden en bytes)")
    jobs.sort(key=lambda j: len(j[0]))                  # lotes de largo parecido: menos relleno
    parts = sorted(out.glob("parte_*.npz"))
    done = len(parts) * 16 * batch                     # cada parte guarda 16 lotes
    freq = np.zeros(vocab, dtype=np.float64)
    for p in parts:
        z = np.load(p)
        np.add.at(freq, z["ids"].reshape(-1), np.exp(z["logp"].astype(np.float64)).reshape(-1))
    t0 = time.time()
    buf_e, buf_i, buf_l = [], [], []
    for bi in range(done, len(jobs), batch):
        group = jobs[bi:bi + batch]
        width = max(len(j[0]) for j in group)
        x = torch.zeros(len(group), width, dtype=torch.long)
        mask = torch.zeros(len(group), width, dtype=torch.long)
        for r, (w, _, _) in enumerate(group):
            x[r, :len(w)] = torch.tensor(w)
            mask[r, :len(w)] = 1
        logits = model(input_ids=x, attention_mask=mask).logits
        for r, (w, first, ends) in enumerate(group):
            lp = F.log_softmax(logits[r, first:first + len(ends)].float(), dim=-1)
            val, idx = lp.topk(k, dim=-1)
            buf_e.append(np.asarray(ends, dtype=np.int64))
            buf_i.append(idx.numpy().astype(np.uint32))
            buf_l.append(val.numpy().astype(np.float16))
            np.add.at(freq, idx.numpy().reshape(-1), val.exp().numpy().astype(np.float64).reshape(-1))
        if (bi // batch + 1) % 16 == 0 or bi + batch >= len(jobs):
            np.savez(out / f"parte_{len(list(out.glob('parte_*.npz'))):05d}.npz", ends=np.concatenate(buf_e),
                     ids=np.concatenate(buf_i), logp=np.concatenate(buf_l))
            buf_e, buf_i, buf_l = [], [], []
            el = time.time() - t0
            log(f"  {min(bi + batch, len(jobs))}/{len(jobs)} ventanas  {el / 60:.1f} min")
    # Unir las partes, ordenadas por posición.
    zs = [np.load(p) for p in sorted(out.glob("parte_*.npz"))]
    ends = np.concatenate([z["ends"] for z in zs])
    order = np.argsort(ends, kind="stable")
    np.save(out / "ends.npy", ends[order])
    np.save(out / "ids.npy", np.concatenate([z["ids"] for z in zs])[order])
    np.save(out / "logp.npy", np.concatenate([z["logp"] for z in zs])[order])
    np.save(out / "freq.npy", freq)
    meta = {"teacher": teacher, "k": k, "ctx": ctx, "stride": stride, "corpus": str(data), "corpus_bytes": len(raw),
            "corpus_sha256": hashlib.sha256(raw).hexdigest(), "entries": int(len(ends)), "tokens": n_tokens,
            "bytes_per_token": len(raw) / max(n_tokens, 1), "vocab": vocab, "skipped_docs": skipped,
            "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for p in out.glob("parte_*.npz"):
        p.unlink()
    return meta


# ---------------------------------------------------------------------------
# Lectura y vocabulario reducido
# ---------------------------------------------------------------------------
class TeacherLogits:
    """Logits guardados del maestro, alineados al corpus por posición de byte."""

    def __init__(self, folder: str | Path, vocab_k: int | None = None):
        f = Path(folder)
        self.meta = json.loads((f / "meta.json").read_text(encoding="utf-8"))
        self.ends = np.load(f / "ends.npy", mmap_mode="r")
        self.ids = np.load(f / "ids.npy", mmap_mode="r")
        self.logp = np.load(f / "logp.npy", mmap_mode="r")
        self.freq = np.load(f / "freq.npy") if (f / "freq.npy").exists() else None
        self.map: np.ndarray | None = None
        if vocab_k:
            self.set_vocab(vocab_k)

    def set_vocab(self, k: int) -> np.ndarray:
        """Vocabulario reducido: los ``k`` ids con más masa del maestro; el resto va a la clase ``k``."""
        top = np.argsort(-self.freq, kind="stable")[:k]
        self.map = np.full(len(self.freq), k, dtype=np.int64)
        self.map[top] = np.arange(k)
        self.vocab_k = k
        return top

    def vocab_coverage(self, k: int) -> float:
        """Fracción de la masa (top-k guardado) que cae en los ``k`` ids más frecuentes."""
        f = np.sort(self.freq)[::-1]
        return float(f[:k].sum() / max(f.sum(), 1e-12))

    def lookup(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Para posiciones absolutas de byte: (máscara de las que tienen entrada, índice de la entrada)."""
        idx = np.searchsorted(self.ends, positions)
        idx_c = np.minimum(idx, len(self.ends) - 1)
        found = (idx < len(self.ends)) & (np.asarray(self.ends[idx_c]) == positions)
        return found, idx_c

    def sample_starts(self, rng: np.random.Generator, batch: int, length: int, corpus_len: int) -> np.ndarray:
        return rng.integers(0, max(corpus_len - length, 1), size=batch)


class Distiller:
    """Término de pérdida ``λ·KL(maestro ‖ cabeza)`` para ``HN1Model.loss(..., aux=...)``.

    ``aux = distiller.for_batch(starts)`` con ``starts`` = posición absoluta (en el corpus) del
    primer byte de cada fila del lote. KL promedio por parche alineado (en nats)."""

    def __init__(self, teacher: TeacherLogits, weight: float = 0.25):
        if teacher.map is None:
            raise ValueError("elegir el vocabulario reducido antes (TeacherLogits(..., vocab_k=K))")
        self.t, self.weight = teacher, weight
        self.map_t = torch.from_numpy(teacher.map)

    def targets(self, end: torch.Tensor, starts: np.ndarray):
        """Filas (b, t) de los parches con entrada del maestro y sus objetivos [N, K + 1]."""
        b_idx, t_idx = end.nonzero(as_tuple=True)
        b_np, t_np = b_idx.cpu().numpy(), t_idx.cpu().numpy()
        pos = np.asarray(starts, dtype=np.int64)[b_np] + t_np
        found, idx = self.t.lookup(pos)
        sel = np.nonzero(found)[0]
        k = self.t.vocab_k
        if len(sel) == 0:
            return b_idx[:0], t_idx[:0], torch.zeros(0, k + 1), end.sum().item()
        rows = idx[sel]
        ids = torch.from_numpy(np.asarray(self.t.ids[rows]).astype(np.int64))
        p = torch.from_numpy(np.asarray(self.t.logp[rows]).astype(np.float32)).exp()
        p = p / p.sum(-1, keepdim=True)                  # top-k renormalizado
        target = torch.zeros(len(sel), k + 1).scatter_add_(1, self.map_t[ids], p)
        sel_t = torch.from_numpy(sel).to(end.device)
        return b_idx[sel_t], t_idx[sel_t], target, end.sum().item()

    def for_batch(self, starts: np.ndarray):
        def aux(model, out):
            end, pid, g = out["patch_end"], out["patch_id"], out["global"]
            b, t, target, n_patches = self.targets(end, starts)
            if len(b) == 0:
                z = g.sum() * 0.0
                return z, {"kl": 0.0, "parches_alineados": 0.0}
            gs = g[b, pid[b, t]]
            logq = F.log_softmax(model.distill_head(gs).float(), dim=-1)
            target = target.to(logq.device)
            kl = (target * (target.clamp_min(1e-12).log() - logq)).sum(-1).mean()
            return self.weight * kl, {"kl": kl.item(), "parches_alineados": len(b) / max(n_patches, 1)}
        return aux


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("logits", help="corre el maestro sobre el corpus y guarda el top-k por token")
    a.add_argument("--data", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--teacher", default=DEFAULT_TEACHER)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--ctx", type=int, default=512)
    a.add_argument("--stride", type=int, default=384)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--max-bytes", type=int, default=None)
    a.add_argument("--dtype", default="bfloat16")
    a.add_argument("--threads", type=int, default=None)
    a.add_argument("--trust-remote-code", action="store_true", help="modelos con código propio (Nemotron-H)")
    v = sub.add_parser("vocab", help="cobertura del vocabulario reducido para varios K")
    v.add_argument("--logits", required=True)
    v.add_argument("--k", default="1024,2048,4096,8192,16384")
    args = ap.parse_args(argv)
    if args.cmd == "logits":
        meta = build_teacher_logits(args.data, args.out, teacher=args.teacher, k=args.k, ctx=args.ctx,
                                    stride=args.stride, batch=args.batch, max_bytes=args.max_bytes,
                                    dtype=args.dtype, threads=args.threads,
                                    trust_remote_code=args.trust_remote_code)
        print(json.dumps(meta, indent=2))
    else:
        t = TeacherLogits(args.logits)
        for k in map(int, args.k.split(",")):
            print(f"K = {k:6d}: cobertura {t.vocab_coverage(k):.3f}")


if __name__ == "__main__":
    main()
