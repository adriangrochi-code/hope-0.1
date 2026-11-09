"""kNN-LM por bytes (Khandelwal et al. 2020, «Generalization through Memorization»; técnica 5).

Un almacén (*datastore*) guarda, para cada posición de un corpus local, el **estado del
decodificador** de Hope (la entrada de la cabeza de salida, ``out["hidden"]``) y el **byte que
siguió**. Al generar, se buscan los k estados más parecidos al actual y sus bytes siguientes
forman una segunda distribución, que se mezcla con la del modelo::

    p(x) = (1 − λ) · p_modelo(x) + λ · p_knn(x),   p_knn(x) ∝ Σ_{vecinos con byte x} exp(−d / T)

No agrega parámetros: agrega **memoria** (el almacén) y tiempo de búsqueda, que se miden.

Detalles
--------
* **Reducción de dimensión** (``dim``): PCA sobre una muestra de claves. Con ``dim=64`` las claves
  ocupan 4·64 bytes cada una en float32 (en lugar de 4·d_local) y la búsqueda es ~3–4× más rápida.
* **Búsqueda**: exacta (producto de matrices en bloques con PyTorch) o aproximada con un índice IVF
  propio en numpy/torch (k-medias, se revisan ``nprobe`` listas). Si FAISS está instalado y se pide
  ``indice="faiss"``, se usa ``IndexIVFFlat``.
* **Temperatura** ``T``: si no se da, la mediana de la distancia al vecino más cercano en una
  muestra (así ``exp(−d/T)`` no se satura ni se aplana con cualquier escala de estados).
* **Contexto de cada clave**: el corpus se lee en ventanas de ``ventana`` bytes que se solapan a
  la mitad; de cada ventana (salvo la primera) se guardan solo las posiciones de la segunda mitad,
  que tienen al menos ``ventana/2`` bytes de contexto.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from hope.core.hn1 import HN1Model


@dataclass
class InfoAlmacen:
    claves: int = 0
    dim: int = 0
    segundos_construccion: float = 0.0
    indice: str = "exacto"
    extra: dict = field(default_factory=dict)


class AlmacenKNN:
    def __init__(self, keys: torch.Tensor, vals: torch.Tensor, *, mean: torch.Tensor | None = None,
                 proj: torch.Tensor | None = None, temp: float | None = None):
        self.keys = keys.float().contiguous()                 # [N, D]
        self.vals = vals.to(torch.uint8)                      # [N]
        self.mean, self.proj = mean, proj
        self.knorm = (self.keys * self.keys).sum(-1)          # ‖k‖² para la distancia L2
        self.temp = temp if temp is not None else self._temp_auto()
        self.ivf: dict | None = None
        self.faiss = None
        self.info = InfoAlmacen(claves=len(self.keys), dim=self.keys.shape[1])

    # --- construcción ----------------------------------------------------------------
    @classmethod
    @torch.no_grad()
    def construir(cls, model: HN1Model, textos: list[bytes | str], *, max_claves: int = 300_000, dim: int | None = 64,
                  ventana: int = 512, lote: int = 8, n_iters: int | None = None) -> "AlmacenKNN":
        t0 = time.perf_counter()
        model.eval()
        dev = model.head.weight.device
        ks, vs, total = [], [], 0
        mitad = ventana // 2
        wins: list[tuple[list[int], int]] = []               # (bytes, desde qué posición se guardan)
        for t in textos:
            b = list(t.encode("utf-8") if isinstance(t, str) else t)
            if len(b) < 2:
                continue
            start = 0
            while True:
                w = b[start:start + ventana + 1]
                wins.append((w, 0 if start == 0 else mitad))
                if start + ventana + 1 >= len(b):
                    break
                start += mitad
        for i in range(0, len(wins), lote):
            grp = wins[i:i + lote]
            width = max(len(w) for w, _ in grp) - 1
            x = torch.zeros(len(grp), width, dtype=torch.long, device=dev)
            for j, (w, _) in enumerate(grp):
                x[j, :len(w) - 1] = torch.tensor(w[:-1])
            hid = model(x, n_iters=n_iters)["hidden"].float()
            for j, (w, keep) in enumerate(grp):
                n = len(w) - 1
                ks.append(hid[j, keep:n].cpu())
                vs.append(torch.tensor(w[keep + 1:n + 1], dtype=torch.uint8))
                total += n - keep
            if total >= max_claves:
                break
        keys = torch.cat(ks)[:max_claves]
        vals = torch.cat(vs)[:max_claves]
        mean = proj = None
        if dim is not None and dim < keys.shape[1]:
            g = torch.Generator().manual_seed(0)
            sample = keys[torch.randperm(len(keys), generator=g)[:20_000]]
            mean = sample.mean(0)
            _u, _s, v = torch.linalg.svd(sample - mean, full_matrices=False)
            proj = v[:dim].T.contiguous()                     # [dl, dim]
            keys = (keys - mean) @ proj
        st = cls(keys, vals, mean=mean, proj=proj)
        st.info.segundos_construccion = time.perf_counter() - t0
        return st

    def _temp_auto(self, n: int = 512) -> float:
        if len(self.keys) < 4:
            return 1.0
        g = torch.Generator().manual_seed(1)
        idx = torch.randperm(len(self.keys), generator=g)[:n]
        d, _ = self._exacto(self.keys[idx], 9)
        # [:, 0] es la propia clave (d = 0); se usa la distancia media a los 8 siguientes, sin contar
        # duplicados exactos (textos repetidos en el corpus).
        d = d[:, 1:]
        pos = d[d > 1e-3]
        return max(float(pos.median()) if pos.numel() else 1.0, 1e-3)

    def proyectar(self, q: torch.Tensor) -> torch.Tensor:
        q = q.float().reshape(-1, q.shape[-1])
        if self.proj is not None:
            q = (q - self.mean) @ self.proj
        return q

    # --- índice aproximado --------------------------------------------------------------------
    def indexar(self, nlist: int | None = None, iters: int = 8, *, usar_faiss: bool = False) -> "AlmacenKNN":
        """Índice IVF: k-medias en ``nlist`` grupos; al buscar se revisan las ``nprobe`` listas más cercanas."""
        n = len(self.keys)
        nlist = nlist or max(16, int(4 * math.sqrt(n)))
        if usar_faiss:
            import faiss                                       # noqa: PLC0415 — opcional
            q = faiss.IndexFlatL2(self.keys.shape[1])
            idx = faiss.IndexIVFFlat(q, self.keys.shape[1], nlist)
            arr = self.keys.numpy().astype(np.float32)
            idx.train(arr)
            idx.add(arr)
            self.faiss = idx
            self.info.indice = f"faiss-ivf{nlist}"
            return self
        g = torch.Generator().manual_seed(0)
        sample = self.keys[torch.randperm(n, generator=g)[:max(50 * nlist, 20_000)]]
        cent = sample[torch.randperm(len(sample), generator=g)[:nlist]].clone()
        for _ in range(iters):
            a = self._asignar(sample, cent)
            sums = torch.zeros_like(cent).index_add_(0, a, sample)
            cnt = torch.bincount(a, minlength=nlist).clamp_min(1).unsqueeze(1).float()
            moved = sums / cnt
            empty = torch.bincount(a, minlength=nlist) == 0
            cent = torch.where(empty.unsqueeze(1), cent, moved)
        a = self._asignar(self.keys, cent)
        order = torch.argsort(a)
        counts = torch.bincount(a, minlength=nlist)
        offs = torch.zeros(nlist + 1, dtype=torch.long)
        offs[1:] = counts.cumsum(0)
        self.ivf = {"cent": cent, "order": order, "offs": offs, "keys": self.keys[order].contiguous(),
                    "knorm": self.knorm[order].contiguous(), "vals": self.vals[order].contiguous()}
        self.info.indice = f"ivf{nlist}"
        return self

    @staticmethod
    def _asignar(x: torch.Tensor, cent: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        out = []
        cn = (cent * cent).sum(-1)
        for i in range(0, len(x), chunk):
            xb = x[i:i + chunk]
            out.append((cn[None] - 2 * xb @ cent.T).argmin(-1))
        return torch.cat(out)

    # --- búsqueda ------------------------------------------------------------------------------
    def _exacto(self, q: torch.Tensor, k: int, chunk: int = 131072):
        best_d = best_i = None
        qn = (q * q).sum(-1, keepdim=True)
        for s in range(0, len(self.keys), chunk):
            kb = self.keys[s:s + chunk]
            d = qn - 2 * q @ kb.T + self.knorm[s:s + chunk][None]
            kk = min(k, d.shape[1])
            dv, di = d.topk(kk, dim=-1, largest=False)
            di = di + s
            if best_d is None:
                best_d, best_i = dv, di
            else:
                cd, ci = torch.cat([best_d, dv], 1), torch.cat([best_i, di], 1)
                best_d, sel = cd.topk(min(k, cd.shape[1]), dim=-1, largest=False)
                best_i = ci.gather(1, sel)
        return best_d.clamp_min(0), best_i

    def buscar(self, q: torch.Tensor, k: int = 16, nprobe: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve (distancias L2² [M, k], bytes siguientes [M, k]) para consultas ya proyectadas."""
        if self.faiss is not None:
            self.faiss.nprobe = nprobe
            d, i = self.faiss.search(q.numpy().astype(np.float32), k)
            i = torch.from_numpy(i).clamp_min(0)
            return torch.from_numpy(d).clamp_min(0), self.vals[i]
        if self.ivf is None:
            d, i = self._exacto(q, k)
            return d, self.vals[i]
        iv = self.ivf
        cn = (iv["cent"] * iv["cent"]).sum(-1)
        probes = (cn[None] - 2 * q @ iv["cent"].T).topk(nprobe, dim=-1, largest=False).indices
        ds, vs = [], []
        for r in range(len(q)):
            rows = torch.cat([torch.arange(int(iv["offs"][c]), int(iv["offs"][c + 1])) for c in probes[r].tolist()])
            kb = iv["keys"][rows]
            d = (q[r] * q[r]).sum() - 2 * kb @ q[r] + iv["knorm"][rows]
            kk = min(k, len(rows))
            dv, di = d.topk(kk, largest=False)
            if kk < k:                                         # listas casi vacías: se rellena lejos
                dv = torch.cat([dv, torch.full((k - kk,), float("inf"))])
                di = torch.cat([di, torch.zeros(k - kk, dtype=torch.long)])
            ds.append(dv.clamp_min(0))
            vs.append(iv["vals"][rows[di]])
        return torch.stack(ds), torch.stack(vs)

    def logp_knn(self, hidden: torch.Tensor, k: int = 16, nprobe: int = 8, temp: float | None = None) -> torch.Tensor:
        """Log-distribución del byte siguiente según los vecinos [M, 256] (estados sin proyectar)."""
        q = self.proyectar(hidden)
        d, v = self.buscar(q, k, nprobe)
        w = -d / (temp or self.temp)
        w = torch.where(torch.isfinite(w), w, torch.full_like(w, -1e30))
        logw = torch.log_softmax(w, -1)
        out = torch.full((len(q), 256), -math.inf)
        # log Σ por byte: logsumexp de los pesos de los vecinos con ese byte
        for j in range(v.shape[1]):
            col = v[:, j].long()
            cur = out.gather(1, col[:, None])[:, 0]
            out.scatter_(1, col[:, None], torch.logaddexp(cur, logw[:, j])[:, None])
        return out

    # --- costo --------------------------------------------------------------------------------
    def memoria_mb(self) -> float:
        b = self.keys.numel() * 4 + self.vals.numel() + self.knorm.numel() * 4
        if self.ivf is not None:
            b += sum(t.numel() * t.element_size() for t in self.ivf.values())
        if self.proj is not None:
            b += self.proj.numel() * 4 + self.mean.numel() * 4
        return b / 2 ** 20

    def guardar(self, path) -> None:
        torch.save({"keys": self.keys.half(), "vals": self.vals, "mean": self.mean, "proj": self.proj,
                    "temp": self.temp}, path)

    @classmethod
    def cargar(cls, path) -> "AlmacenKNN":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["keys"], d["vals"], mean=d["mean"], proj=d["proj"], temp=d["temp"])


def mezclar(logp_modelo: torch.Tensor, logp_knn: torch.Tensor, lam: float) -> torch.Tensor:
    """log((1 − λ)·p_modelo + λ·p_knn), estable."""
    if lam <= 0:
        return logp_modelo
    a = logp_modelo + math.log1p(-lam) if lam < 1 else torch.full_like(logp_modelo, -math.inf)
    return torch.logaddexp(a, logp_knn + math.log(lam))


class ProcesadorKNN:
    """Procesador de logits para ``hope.core.paso.generar``: mezcla la predicción con el almacén."""

    def __init__(self, almacen: AlmacenKNN, *, lam: float = 0.25, k: int = 16, nprobe: int = 8,
                 temp: float | None = None):
        self.almacen, self.lam, self.k, self.nprobe, self.temp = almacen, lam, k, nprobe, temp

    def iniciar(self, seq: list[int]) -> None:
        pass

    def avanzar(self, byte: int) -> None:
        pass

    @torch.no_grad()
    def __call__(self, paso, logp: torch.Tensor) -> torch.Tensor:
        lk = self.almacen.logp_knn(paso.hidden[None], self.k, self.nprobe, self.temp)[0]
        return mezclar(logp, lk, self.lam)


@torch.no_grad()
def logprobs_knn(model: HN1Model, almacen: AlmacenKNN, contexto: bytes | str, texto: bytes | str, *,
                 lams: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5), k: int = 16, nprobe: int = 8,
                 temp: float | None = None) -> dict[float, list[float]]:
    """Log-probabilidades (nats) de ``texto`` dado ``contexto`` con varias λ, en **una** pasada del
    modelo (para medir BPB condicional sin generar)."""
    c = list(contexto.encode("utf-8") if isinstance(contexto, str) else contexto) or [10]
    t = list(texto.encode("utf-8") if isinstance(texto, str) else texto)
    x = torch.tensor([c + t[:-1]], device=model.head.weight.device)
    model.eval()
    out = model(x)
    pos = slice(len(c) - 1, len(c) - 1 + len(t))
    lpm = torch.log_softmax(out["logits"][0, pos].float(), -1)
    lk = almacen.logp_knn(out["hidden"][0, pos], k, nprobe, temp)
    tgt = torch.tensor(t)[:, None]
    return {lam: mezclar(lpm, lk, lam).gather(1, tgt)[:, 0].tolist() for lam in lams}
