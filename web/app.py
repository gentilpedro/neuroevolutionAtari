"""
Painel web do trabalho de neuroevolução em jogos de Atari (Asterix / Freeway).

Reaproveita as funções de ../main.py (criar_populacao, avaliar, nova_geracao...)
e expõe controles de iniciar / pausar / continuar / parar / resetar, o estado
do treino em tempo real, a arena ao vivo (aleatório × melhor evoluído, uma
partida compartilhada por todos os espectadores) e replays MJPEG de agentes salvos.
"""

import asyncio
import copy
import csv
import io
import json
import os
import random
import re
import secrets
import shutil
import sys
import threading
import time
from multiprocessing import get_context
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
import main as neuro  # noqa: E402

DADOS = os.environ.get("DADOS_DIR", os.path.join(RAIZ, "dados"))
EXECUCOES = os.path.join(DADOS, "execucoes")
SENHA = os.environ.get("CONTROLE_SENHA", "")
PROCESSOS = int(os.environ.get("PROCESSOS", "2"))
MAX_ESPECTADORES = int(os.environ.get("MAX_ESPECTADORES", "3"))   # replays: um jogo por pessoa
MAX_ARENA = int(os.environ.get("MAX_ARENA", "60"))                # arena: só lê quadros prontos
ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ETAPAS = ["populacao", "avaliacao", "fitness", "selecao", "mutacao", "nova_geracao"]

os.makedirs(EXECUCOES, exist_ok=True)


class Configuracao(BaseModel):
    jogo: str = Field(neuro.JOGO_PADRAO, pattern="^(" + "|".join(neuro.JOGOS) + ")$")
    experimento: str = Field("B", pattern="^(A|B|C|P|T)$")  # P = personalizado, T = todos (A -> B -> C)
    taxa: float = Field(0.10, ge=0.0, le=1.0)
    intensidade: float = Field(0.20, ge=0.0, le=5.0)
    populacao: int = Field(20, ge=4, le=100)
    geracoes: int = Field(30, ge=1, le=300)
    elite: int = Field(2, ge=1, le=20)
    semente: int = Field(42, ge=0, le=1_000_000)
    episodios_teste: int = Field(5, ge=0, le=30)


class Interrompido(Exception):
    pass


class Treinador:
    def __init__(self):
        self.lock = threading.Lock()
        self.liberado = threading.Event()
        self.thread = None
        self._zerar()

    def _zerar(self):
        self.liberado.set()
        self.parar = False
        self.status = "parado"      # parado | rodando | pausado | concluido | interrompido | erro
        self.etapa = None
        self.config = None
        self.id = None
        self.fila = []
        self.posicao_fila = 0
        self._zerar_execucao()
        self.log = []
        self.erro = None
        self.inicio = None
        self.fim = None
        self.pausado_desde = None
        self.tempo_pausado = 0.0

    def _zerar_execucao(self):
        self.geracao = 0
        self.historico = []
        self.fitness_atual = []
        self.elite_atual = []
        self.baseline = None
        self.teste = None
        self.melhor = None

    # ---------------- controle ----------------

    def iniciar(self, cfg: Configuracao):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise HTTPException(409, "Já existe um treinamento em andamento.")
            if cfg.elite >= cfg.populacao:
                raise HTTPException(422, "A elite precisa ser menor que a população.")
            fila = []
            for nome in (["A", "B", "C"] if cfg.experimento == "T" else [cfg.experimento]):
                item = cfg.model_copy(update={"experimento": nome})
                if nome in neuro.EXPERIMENTOS:
                    item.taxa = neuro.EXPERIMENTOS[nome]["taxa"]
                    item.intensidade = neuro.EXPERIMENTOS[nome]["intensidade"]
                fila.append(item)
            self._zerar()
            self.fila = fila
            self.status = "rodando"
            self.inicio = time.time()
            self.thread = threading.Thread(target=self._executar, daemon=True)
            self.thread.start()

    def pausar(self):
        if self.status != "rodando":
            raise HTTPException(409, "Não há treinamento rodando.")
        self.liberado.clear()
        self.status = "pausado"
        self.pausado_desde = time.time()
        self._registrar("Pausa solicitada (vale ao fim dos episódios em andamento).")

    def continuar(self):
        if self.status != "pausado":
            raise HTTPException(409, "O treinamento não está pausado.")
        self.tempo_pausado += time.time() - (self.pausado_desde or time.time())
        self.pausado_desde = None
        self.status = "rodando"
        self.liberado.set()
        self._registrar("Treinamento retomado.")

    def interromper(self):
        if self.status not in ("rodando", "pausado"):
            raise HTTPException(409, "Não há treinamento em andamento.")
        self.parar = True
        self.liberado.set()
        self._registrar("Parada solicitada.")

    def resetar(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise HTTPException(409, "Pare o treinamento antes de resetar.")
            self._zerar()

    # ---------------- ciclo evolutivo ----------------

    def _registrar(self, msg):
        self.log.append({"t": time.time(), "msg": msg})
        self.log = self.log[-300:]

    def _checar(self):
        while not self.liberado.wait(0.5):
            pass
        if self.parar:
            raise Interrompido()

    def _etapa(self, nome, pausa=0.0):
        self.etapa = nome
        if pausa:
            time.sleep(pausa)  # só para o painel conseguir mostrar a etapa

    def _executar(self):
        try:
            with get_context("spawn").Pool(PROCESSOS) as pool:
                for posicao, cfg in enumerate(self.fila):
                    self.posicao_fila = posicao
                    self.config = cfg
                    self.id = time.strftime("%Y%m%d-%H%M%S") + f"_{cfg.experimento}"
                    self._zerar_execucao()
                    self._registrar(
                        f"=== {neuro.JOGOS[cfg.jogo]['env_id']} - experimento {cfg.experimento}: taxa {cfg.taxa:.0%}, "
                        f"intensidade {cfg.intensidade} ({posicao + 1}/{len(self.fila)}) ==="
                    )
                    self._rodar_experimento(pool, cfg)
                    self._salvar("concluido")
            self.status = "concluido"
            self._registrar("Treinamento concluído.")
        except Interrompido:
            self.status = "interrompido"
            self._registrar("Treinamento interrompido.")
        except Exception as e:  # mostra o erro no painel em vez de matar a thread em silêncio
            self.status = "erro"
            self.erro = repr(e)
            self._registrar(f"Erro: {e!r}")
        finally:
            self.etapa = None
            self.fim = time.time()
            if self.historico and self.status != "concluido":
                self._salvar(self.status)

    def _rodar_experimento(self, pool, cfg):
        sementes_teste = [10_000_000 + i for i in range(cfg.episodios_teste)]

        if sementes_teste:
            self._etapa("baseline")
            self._registrar(f"Baseline: agente aleatório em {len(sementes_teste)} episódios.")
            self._checar()
            pontos = pool.map(neuro._avaliar_aleatorio, [(s, cfg.jogo) for s in sementes_teste])
            self.baseline = {"episodios": pontos, "media": float(np.mean(pontos))}
            self._registrar(f"Agente aleatório: média {self.baseline['media']:.2f}.")

        # Parâmetros do experimento no módulo do trabalho.
        neuro.configurar_jogo(cfg.jogo)
        neuro.TAMANHO_POPULACAO = cfg.populacao
        neuro.NUM_ELITE = cfg.elite
        neuro.NUM_PAIS = max(2, cfg.populacao // 2)
        neuro.TAMANHO_TORNEIO = min(3, cfg.populacao)
        np.random.seed(cfg.semente)

        self._etapa("populacao", 0.6)
        populacao = neuro.criar_populacao()
        self._registrar(f"População inicial: {cfg.populacao} agentes com pesos aleatórios.")

        for g in range(cfg.geracoes):
            self.geracao = g
            self._etapa("avaliacao")
            self.fitness_atual = [None] * len(populacao)
            self.elite_atual = list(range(cfg.elite)) if g > 0 else []
            semente_ambiente = cfg.semente * 1000 + g

            for i in range(0, len(populacao), PROCESSOS):
                self._checar()
                lote = populacao[i:i + PROCESSOS]
                pontos = pool.map(neuro._avaliar_tarefa, [(a, semente_ambiente, cfg.jogo) for a in lote])
                for j, (agente, fit) in enumerate(zip(lote, pontos)):
                    agente.fitness = fit
                    self.fitness_atual[i + j] = fit

            self._etapa("fitness", 0.4)
            valores = [a.fitness for a in populacao]
            linha = {
                "geracao": g,
                "melhor": max(valores),
                "media": float(np.mean(valores)),
                "pior": min(valores),
                "desvio": float(np.std(valores)),
            }
            self.historico.append(linha)
            campeao = max(populacao, key=lambda a: a.fitness)
            if self.melhor is None or campeao.fitness >= self.melhor.fitness:
                self.melhor = copy.deepcopy(campeao)
            self._registrar(
                f"[{cfg.experimento}] Geração {g:03d} | melhor {linha['melhor']:.0f} | "
                f"média {linha['media']:.2f} | pior {linha['pior']:.0f}"
            )
            self._salvar("rodando")

            if g < cfg.geracoes - 1:
                self._checar()
                self._etapa("selecao", 0.4)
                self._etapa("mutacao", 0.4)
                populacao = neuro.nova_geracao(populacao, cfg.taxa, cfg.intensidade)
                self._etapa("nova_geracao", 0.4)

        if sementes_teste:
            self._etapa("teste")
            self._checar()
            pontos = pool.map(neuro._avaliar_tarefa, [(self.melhor, s, cfg.jogo) for s in sementes_teste])
            self.teste = {"episodios": pontos, "media": float(np.mean(pontos))}
            self._registrar(f"[{cfg.experimento}] Melhor agente em episódios novos: média {self.teste['media']:.2f}.")

    # ---------------- persistência ----------------

    def _salvar(self, status):
        pasta = os.path.join(EXECUCOES, self.id)
        os.makedirs(pasta, exist_ok=True)
        dados = {
            "id": self.id,
            "origem": "painel",
            "status": status,
            "config": self.config.model_dump(),
            "historico": self.historico,
            "baseline": self.baseline,
            "teste": self.teste,
            "inicio": self.inicio,
            "fim": self.fim,
        }
        with open(os.path.join(pasta, "execucao.json"), "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False)
        if self.melhor is not None:
            neuro.salvar_agente(self.melhor, os.path.join(pasta, "melhor.npz"))

    def estado(self):
        agora = self.fim if self.status not in ("rodando", "pausado") else time.time()
        pausa = self.tempo_pausado + (time.time() - self.pausado_desde if self.pausado_desde else 0)
        return {
            "status": self.status,
            "etapa": self.etapa,
            "id": self.id,
            "config": self.config.model_dump() if self.config else None,
            "fila": [c.experimento for c in self.fila],
            "posicao_fila": self.posicao_fila,
            "geracao": self.geracao,
            "historico": self.historico,
            "fitness_atual": self.fitness_atual,
            "elite_atual": self.elite_atual,
            "baseline": self.baseline,
            "teste": self.teste,
            "melhor_global": self.melhor.fitness if self.melhor else None,
            "log": self.log[-80:],
            "erro": self.erro,
            "tempo": (agora - self.inicio - pausa) if self.inicio and agora else 0,
            "processos": PROCESSOS,
            "protegido": bool(SENHA),
        }


treinador = Treinador()
espectadores = 0
espectadores_lock = threading.Lock()


def _jpeg(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).resize((320, 420), Image.NEAREST).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class Arena:
    """Partida ao vivo compartilhada: agente aleatório × melhor evoluído.

    Uma thread só joga os dois lados na mesma semente e guarda o último quadro
    de cada um; os espectadores apenas leem esses quadros. O custo de CPU não
    cresce com o público (só a banda), e a thread para sozinha quando ninguém
    está assistindo. O melhor agente é copiado no início de cada rodada, então a
    rodada seguinte já usa o campeão mais recente do treino.
    """

    LADOS = ("aleatorio", "evoluido")
    PASSOS_POR_QUADRO = 2     # o jogo roda a 2x e a tela recebe ~15 quadros/s
    INTERVALO = 4 / 60        # os ambientes v5 repetem cada ação por 4 quadros (60 fps)
    ESPERA_SEM_PUBLICO = 15   # segundos jogando depois que o último espectador sai
    PAUSA_ENTRE_RODADAS = 2.5

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.clientes = {lado: 0 for lado in self.LADOS}
        self.ultimo_cliente = 0.0
        self.quadros = {lado: None for lado in self.LADOS}
        self.seq = {lado: 0 for lado in self.LADOS}
        self._placar = {"ativo": False}

    def entrar(self, lado):
        with self.lock:
            if self.clientes[lado] >= MAX_ARENA:
                raise HTTPException(429, "A arena está cheia agora. Tente em instantes.")
            self.clientes[lado] += 1
            self.ultimo_cliente = time.time()
            if not (self.thread and self.thread.is_alive()):
                self.thread = threading.Thread(target=self._rodar, daemon=True)
                self.thread.start()

    def sair(self, lado):
        with self.lock:
            self.clientes[lado] -= 1
            self.ultimo_cliente = time.time()

    def _com_publico(self):
        with self.lock:
            return sum(self.clientes.values()) > 0 or time.time() - self.ultimo_cliente < self.ESPERA_SEM_PUBLICO

    def placar(self):
        return {**self._placar, "espectadores": max(self.clientes.values())}

    def _publicar(self, lado, quadro):
        self.quadros[lado] = quadro
        self.seq[lado] += 1

    def _rodar(self):
        rodada = 0
        try:
            while self._com_publico():
                rodada += 1
                self._rodada(rodada)
                fim = time.time() + self.PAUSA_ENTRE_RODADAS  # placar final na tela
                while time.time() < fim and self._com_publico():
                    time.sleep(0.2)
        finally:
            self._placar = {"ativo": False}

    def _rodada(self, rodada):
        cfg, melhor = treinador.config, treinador.melhor
        jogo = cfg.jogo if cfg else neuro.JOGO_PADRAO
        info = neuro.JOGOS[jogo]
        semente = random.randrange(1_000_000)
        jogadores = {"aleatorio": neuro.AgenteAleatorio(semente, info["acoes"])}
        if melhor is not None:
            jogadores["evoluido"] = copy.deepcopy(melhor)
        else:
            self._publicar("evoluido", None)

        envs, estados, placares = {}, {}, {}
        try:
            for lado in jogadores:
                env = neuro.gym.make(info["env_id"], obs_type="ram", render_mode="rgb_array",
                                     max_episode_steps=info["max_passos"])
                envs[lado] = env
                estados[lado], _ = env.reset(seed=semente)
                placares[lado] = {"pontos": 0.0, "passos": 0, "fim": False}
                self._publicar(lado, _jpeg(env.render()))
            self._placar = {
                "ativo": True, "rodada": rodada, "jogo": jogo, "unidade": info["unidade"],
                "geracao": treinador.geracao + 1 if melhor is not None else None,
                "experimento": cfg.experimento if cfg and melhor is not None else None,
                **placares,
            }
            while not all(p["fim"] for p in placares.values()) and self._com_publico():
                t0 = time.time()
                for lado, agente in jogadores.items():
                    p = placares[lado]
                    if p["fim"]:
                        continue
                    for _ in range(self.PASSOS_POR_QUADRO):
                        acao = agente.agir(estados[lado])
                        estados[lado], reward, terminated, truncated, _ = envs[lado].step(acao)
                        p["pontos"] += reward
                        p["passos"] += 1
                        if terminated or truncated:
                            p["fim"] = True
                            break
                    self._publicar(lado, _jpeg(envs[lado].render()))
                time.sleep(max(0.0, self.INTERVALO - (time.time() - t0)))
        finally:
            for env in envs.values():
                env.close()


arena = Arena()


app = FastAPI(title="Neuroevolução no Atari", docs_url=None, redoc_url=None)


def exigir_senha(x_senha: Optional[str]):
    if SENHA and not secrets.compare_digest((x_senha or "").encode(), SENHA.encode()):
        raise HTTPException(401, "Senha de controle inválida.")


def ler_execucao(id_execucao):
    if not ID_VALIDO.match(id_execucao):
        raise HTTPException(404, "Execução não encontrada.")
    caminho = os.path.join(EXECUCOES, id_execucao, "execucao.json")
    if not os.path.exists(caminho):
        raise HTTPException(404, "Execução não encontrada.")
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
def pagina():
    # no-cache: depois de um deploy o navegador sempre pega a página nova.
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"),
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/estado")
def api_estado():
    return {**treinador.estado(), "arena": arena.placar()}


@app.post("/api/senha")
def api_senha(x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    return {"ok": True}


@app.post("/api/iniciar")
def api_iniciar(cfg: Configuracao, x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    treinador.iniciar(cfg)
    return treinador.estado()


@app.post("/api/pausar")
def api_pausar(x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    treinador.pausar()
    return treinador.estado()


@app.post("/api/continuar")
def api_continuar(x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    treinador.continuar()
    return treinador.estado()


@app.post("/api/parar")
def api_parar(x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    treinador.interromper()
    return treinador.estado()


@app.post("/api/resetar")
def api_resetar(x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    treinador.resetar()
    return treinador.estado()


@app.get("/api/execucoes")
def api_execucoes():
    itens = []
    for nome in os.listdir(EXECUCOES):
        try:
            d = ler_execucao(nome)
        except (HTTPException, ValueError):
            continue
        h = d["historico"]
        itens.append({
            "id": d["id"],
            "origem": d.get("origem"),
            "status": d["status"],
            "config": {"jogo": "freeway", **d["config"]},  # execuções antigas eram todas Freeway
            "geracoes_rodadas": len(h),
            "melhor": max((l["melhor"] for l in h), default=None),
            "media_final": h[-1]["media"] if h else None,
            "teste": d["teste"]["media"] if d.get("teste") else None,
            "baseline": d["baseline"]["media"] if d.get("baseline") else None,
            "tem_agente": os.path.exists(os.path.join(EXECUCOES, nome, "melhor.npz")),
        })
    itens.sort(key=lambda i: i["id"], reverse=True)
    return itens


@app.get("/api/execucoes/{id_execucao}")
def api_execucao(id_execucao: str):
    d = ler_execucao(id_execucao)
    d["config"] = {"jogo": "freeway", **d["config"]}
    return d


@app.get("/api/execucoes/{id_execucao}/csv")
def api_execucao_csv(id_execucao: str):
    d = ler_execucao(id_execucao)
    buf = io.StringIO()
    escritor = csv.DictWriter(buf, fieldnames=["geracao", "melhor", "media", "pior", "desvio"])
    escritor.writeheader()
    escritor.writerows(d["historico"])
    return PlainTextResponse(
        buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="fitness_{id_execucao}.csv"'},
    )


@app.delete("/api/execucoes/{id_execucao}")
def api_excluir(id_execucao: str, x_senha: Optional[str] = Header(None)):
    exigir_senha(x_senha)
    ler_execucao(id_execucao)
    if id_execucao == treinador.id and treinador.status in ("rodando", "pausado"):
        raise HTTPException(409, "Esta execução ainda está em andamento.")
    shutil.rmtree(os.path.join(EXECUCOES, id_execucao))
    return {"ok": True}


@app.delete("/api/execucoes")
def api_excluir_tudo(x_senha: Optional[str] = Header(None)):
    """Apaga todos os treinos salvos e zera o painel, liberando disco e memória da VPS."""
    exigir_senha(x_senha)
    if treinador.status in ("rodando", "pausado"):
        raise HTTPException(409, "Pare o treinamento antes de excluir tudo.")
    removidas = 0
    for nome in os.listdir(EXECUCOES):
        caminho = os.path.join(EXECUCOES, nome)
        if os.path.isdir(caminho) and ID_VALIDO.match(nome):
            shutil.rmtree(caminho)
            removidas += 1
    treinador.resetar()
    return {"ok": True, "removidas": removidas}


CABECALHO_MJPEG = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def _parte_mjpeg(jpeg):
    return b"--quadro\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.get("/api/arena/{lado}")
async def api_arena(lado: str):
    """MJPEG de um lado da arena. Assíncrono de propósito: cada espectador só
    espera o próximo quadro, sem prender uma thread do servidor."""
    if lado not in Arena.LADOS:
        raise HTTPException(404, "Lado desconhecido.")
    arena.entrar(lado)

    async def gerar():
        visto = -1
        try:
            while True:
                seq = arena.seq[lado]
                if seq != visto:
                    visto = seq
                    quadro = arena.quadros[lado]
                    if quadro:
                        yield _parte_mjpeg(quadro)
                await asyncio.sleep(0.03)
        finally:
            arena.sair(lado)

    return StreamingResponse(gerar(), media_type="multipart/x-mixed-replace; boundary=quadro",
                             headers=CABECALHO_MJPEG)


def _quadro(frame, texto):
    img = Image.fromarray(frame).resize((320, 420), Image.NEAREST)
    desenho = ImageDraw.Draw(img)
    desenho.rectangle([0, 0, 320, 18], fill=(0, 0, 0))
    desenho.text((6, 3), texto, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@app.get("/api/assistir/vagas")
def api_assistir_vagas():
    return {"livres": max(0, MAX_ESPECTADORES - espectadores), "total": MAX_ESPECTADORES}


@app.get("/api/assistir")
def api_assistir(fonte: str = "aleatorio", velocidade: float = 1.0, semente: Optional[int] = None,
                 jogo: str = neuro.JOGO_PADRAO):
    """Replay de um episódio como MJPEG: o agente aleatório ou o melhor de uma
    execução salva. Cada replay roda um jogo próprio, por isso tem limite de vagas."""
    global espectadores
    if fonte == "aleatorio":
        if jogo not in neuro.JOGOS:
            raise HTTPException(404, "Jogo desconhecido.")
        agente = neuro.AgenteAleatorio(semente, neuro.JOGOS[jogo]["acoes"])
        rotulo = "Agente aleatório"
    else:
        d = ler_execucao(fonte)
        caminho = os.path.join(EXECUCOES, fonte, "melhor.npz")
        if not os.path.exists(caminho):
            raise HTTPException(404, "Esta execução não tem agente salvo.")
        agente, rotulo = neuro.carregar_agente(caminho), f"Melhor de {fonte}"
        jogo = d["config"].get("jogo", "freeway")

    with espectadores_lock:
        if espectadores >= MAX_ESPECTADORES:
            raise HTTPException(429, "Muitas pessoas assistindo agora. Tente em instantes.")
        espectadores += 1

    velocidade = min(max(velocidade, 0.25), 8.0)
    intervalo = (4 / 60) / velocidade  # os ambientes v5 repetem cada ação por 4 quadros (60 fps)

    info_jogo = neuro.JOGOS[jogo]
    unidade = info_jogo["unidade"]

    def gerar():
        global espectadores
        env = neuro.gym.make(info_jogo["env_id"], obs_type="ram", render_mode="rgb_array",
                             max_episode_steps=info_jogo["max_passos"])
        nomes_acoes = env.unwrapped.get_action_meanings()
        try:
            estado, _ = env.reset(seed=semente)
            total, passo, fim = 0.0, 0, False
            while not fim:
                t0 = time.time()
                acao = agente.agir(estado)
                estado, reward, terminated, truncated, _ = env.step(acao)
                total += reward
                passo += 1
                fim = terminated or truncated
                texto = f"{rotulo} | {unidade} {total:.0f} | passo {passo} | {nomes_acoes[acao]}"
                if fim:
                    texto = f"FIM | {rotulo} | {total:.0f} {unidade} em {passo} passos"
                yield _parte_mjpeg(_quadro(env.render(), texto))
                time.sleep(max(0.0, intervalo - (time.time() - t0)))
        finally:
            env.close()
            with espectadores_lock:
                espectadores -= 1

    return StreamingResponse(gerar(), media_type="multipart/x-mixed-replace; boundary=quadro",
                             headers=CABECALHO_MJPEG)
