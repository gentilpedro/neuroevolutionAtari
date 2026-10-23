"""
Trabalho Prático - Evolução de um Agente Inteligente no Atari
Neuroevolução aplicada ao Asterix (ALE/Asterix-v5)

Fundamentos de Inteligência Artificial - UniSenac Pelotas - 2026/2
Prof. Pablo De Chiaro Rosa

Fluxo: POPULAÇÃO -> AVALIAÇÃO -> FITNESS -> SELEÇÃO -> MUTAÇÃO -> NOVA GERAÇÃO

Uso (nenhuma alteração manual no código é necessária):
    python main.py --explorar              # mostra como o ambiente funciona
    python main.py                         # baseline aleatório + experimentos A, B e C
    python main.py --experimento B         # roda só um experimento
    python main.py --jogo freeway          # mesmo algoritmo no Freeway (ambiente-base do enunciado)
    python main.py --assistir              # renderiza o melhor agente salvo
    python main.py --help                  # todas as opções
"""

import argparse
import copy
import csv
import json
import os
import time
from multiprocessing import Pool

import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)

# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------

# Jogos suportados. O algoritmo é o mesmo; muda só o ambiente e o número de ações.
JOGOS = {
    "asterix": {
        "env_id": "ALE/Asterix-v5",
        # 0 = NOOP, 1 = UP, 2 = RIGHT, 3 = LEFT, 4 = DOWN, 5 = UPRIGHT, 6 = UPLEFT,
        # 7 = DOWNRIGHT, 8 = DOWNLEFT
        "acoes": 9,
        # Um agente aleatório perde as 3 vidas em ~300 passos; o limite só corta
        # agentes muito bons, para a avaliação de uma geração não demorar demais.
        "max_passos": 5000,
        "unidade": "pontos",
    },
    "freeway": {
        "env_id": "ALE/Freeway-v5",
        "acoes": 3,             # 0 = NOOP, 1 = UP, 2 = DOWN
        "max_passos": None,     # a partida já termina sozinha em ~2min16s (2048 passos)
        "unidade": "travessias",
    },
}
JOGO_PADRAO = "asterix"

JOGO = JOGO_PADRAO
ENV_ID = JOGOS[JOGO]["env_id"]
MAX_PASSOS = JOGOS[JOGO]["max_passos"]

TAMANHO_POPULACAO = 20
NUM_GERACOES = 30
NUM_ENTRADAS = 128   # a RAM do Atari 2600 tem 128 bytes -> 128 entradas
NUM_OCULTOS = 16
NUM_ACOES = JOGOS[JOGO]["acoes"]

TAXA_MUTACAO = 0.10
INTENSIDADE_MUTACAO = 0.20
NUM_ELITE = 2
TAMANHO_TORNEIO = 3      # seleção por torneio (ver selecionar)
NUM_PAIS = 10            # quantos indivíduos podem gerar descendentes

# Os três experimentos obrigatórios (seção 7 do enunciado).
EXPERIMENTOS = {
    "A": {"taxa": 0.01, "intensidade": 0.05, "hipotese": "Pouca alteração entre pais e filhos"},
    "B": {"taxa": 0.10, "intensidade": 0.20, "hipotese": "Configuração intermediária"},
    "C": {"taxa": 0.40, "intensidade": 0.50, "hipotese": "Grande diversidade e maior risco de destruir boas soluções"},
}

PASTA_RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")


def configurar_jogo(nome):
    """Troca o jogo usado por criar_ambiente/Agente (também nos processos paralelos)."""
    global JOGO, ENV_ID, NUM_ACOES, MAX_PASSOS
    JOGO = nome
    ENV_ID = JOGOS[nome]["env_id"]
    NUM_ACOES = JOGOS[nome]["acoes"]
    MAX_PASSOS = JOGOS[nome]["max_passos"]


def pasta_resultados(jogo=None):
    jogo = jogo or JOGO
    if jogo == JOGO_PADRAO:
        return PASTA_RESULTADOS
    return os.path.join(PASTA_RESULTADOS, jogo)


def criar_ambiente(render_mode=None):
    return gym.make(
        ENV_ID,
        obs_type="ram",
        render_mode=render_mode,
        max_episode_steps=MAX_PASSOS,
    )


# ---------------------------------------------------------
# ANTES DE PROGRAMAR: entendendo o ambiente
# ---------------------------------------------------------

def explorar_ambiente():
    """Demonstra criar/reiniciar o ambiente, o formato do estado,
    o espaço de ações, env.step(...) e reward/terminated/truncated."""
    env = criar_ambiente()

    # reset() reinicia o jogo e devolve (estado inicial, info).
    estado, info = env.reset(seed=0)
    print("Ambiente ............:", ENV_ID)
    print("Espaço de observação :", env.observation_space)       # Box(0, 255, (128,), uint8)
    print("Formato do estado ...:", estado.shape, estado.dtype)  # (128,) uint8 -> RAM do Atari
    print("Espaço de ações .....:", env.action_space)
    print("Significado das ações:", env.unwrapped.get_action_meanings())
    print("Vidas no início .....:", info.get("lives"))

    # step(acao) devolve 5 valores:
    #   novo_estado -> RAM depois da ação
    #   reward      -> pontos ganhos naquele passo (Asterix: 50 ou mais por item
    #                  coletado; Freeway: 1 por travessia)
    #   terminated  -> True quando o jogo acaba "de verdade" (Asterix: perdeu as 3 vidas)
    #   truncated   -> True quando um limite externo corta o episódio (max_episode_steps)
    #   info        -> dados extras (vidas, número do frame...)
    rng = np.random.default_rng(0)
    passos, total = 0, 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        acao = int(rng.integers(env.action_space.n))
        novo_estado, reward, terminated, truncated, info = env.step(acao)
        total += reward
        passos += 1
    env.close()
    print(f"Episódio com ações aleatórias: {passos} passos, recompensa acumulada = {total:.0f}")
    print(f"terminated = {terminated}, truncated = {truncated}, vidas no fim = {info.get('lives')}")


# ---------------------------------------------------------
# AGENTE / INDIVÍDUO
# ---------------------------------------------------------

class Agente:
    def __init__(self):
        # Cada indivíduo começa com parâmetros diferentes.
        # W1, b1, W2 e b2 são o "DNA" que será evoluído.
        self.W1 = np.random.randn(NUM_ENTRADAS, NUM_OCULTOS)
        self.b1 = np.random.randn(NUM_OCULTOS)
        self.W2 = np.random.randn(NUM_OCULTOS, NUM_ACOES)
        self.b2 = np.random.randn(NUM_ACOES)
        self.fitness = 0.0

    def agir(self, estado):
        # RAM: valores inteiros entre 0 e 255.
        estado = estado.astype(np.float32) / 255.0

        oculto = np.tanh(estado @ self.W1 + self.b1)
        saida = oculto @ self.W2 + self.b2

        # A maior saída define a ação escolhida.
        return int(np.argmax(saida))

    def parametros(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


class AgenteAleatorio:
    """Baseline: ignora o estado e sorteia uma ação com a mesma chance."""

    def __init__(self, semente=None, num_acoes=None):
        self.rng = np.random.default_rng(semente)
        self.num_acoes = num_acoes or NUM_ACOES

    def agir(self, estado):
        return int(self.rng.integers(self.num_acoes))


# ---------------------------------------------------------
# ETAPAS EVOLUTIVAS
# ---------------------------------------------------------

def avaliar(agente, semente=None, render_mode=None):
    """
    AGENTE + AMBIENTE + AÇÃO + RECOMPENSA + FITNESS

    Executa UM episódio completo e devolve o fitness.
    Fitness = soma das recompensas do episódio (pontos do jogo).
    """
    # 1. Criar/reiniciar o ambiente.
    env = criar_ambiente(render_mode)

    # 2. Receber o estado inicial.  <-- ONDE O ESTADO É RECEBIDO
    estado, _ = env.reset(seed=semente)

    recompensa_total = 0.0
    terminado = False

    # 3. Enquanto o episódio não terminar:
    while not terminado:
        # a) pedir uma ação ao agente.  <-- ONDE A AÇÃO É ESCOLHIDA
        acao = agente.agir(estado)

        # b) executar a ação no ambiente;
        # c) receber novo estado e reward.  <-- ONDE A RECOMPENSA É OBTIDA
        estado, reward, terminated, truncated, _ = env.step(acao)

        # d) acumular a recompensa.
        recompensa_total += reward

        # ONDE O EPISÓDIO É ENCERRADO: fim de jogo (terminated) ou corte pelo
        # limite de passos (truncated).
        terminado = terminated or truncated

    # 4. Fechar o ambiente.
    env.close()

    # 5. Retornar o fitness do indivíduo.
    #    Aqui recompensa e fitness têm o mesmo valor, mas são conceitos
    #    diferentes: a recompensa chega passo a passo do ambiente; o fitness
    #    é a nota única que o algoritmo evolutivo usa para comparar agentes.
    return float(recompensa_total)


def _avaliar_tarefa(args):
    """Adaptador para avaliar vários indivíduos em paralelo (multiprocessing).
    Cada processo filho começa com o jogo padrão, por isso o jogo vai junto."""
    agente, semente, jogo = args
    configurar_jogo(jogo)
    return avaliar(agente, semente)


def _avaliar_aleatorio(args):
    semente, jogo = args
    configurar_jogo(jogo)
    return avaliar(AgenteAleatorio(semente), semente)


def criar_populacao():
    """
    INDIVÍDUO + POPULAÇÃO
    Cria TAMANHO_POPULACAO agentes independentes, cada um com pesos
    aleatórios próprios (np.random.randn dentro de Agente.__init__).
    """
    return [Agente() for _ in range(TAMANHO_POPULACAO)]


def selecionar(populacao):
    """
    SELEÇÃO - torneio de tamanho TAMANHO_TORNEIO.

    Para escolher cada pai sorteamos TAMANHO_TORNEIO indivíduos e fica o de
    maior fitness. Repetimos NUM_PAIS vezes.

    Por que torneio: dá preferência aos melhores (o pior da população nunca
    vence um torneio), mas indivíduos medianos ainda têm chance de reproduzir.
    Isso mantém diversidade e evita que um único agente domine a população
    logo nas primeiras gerações. Também funciona com fitness zero ou empatado
    (comum no começo do Freeway), o que não acontece com a roleta proporcional.
    """
    pais = []
    for _ in range(NUM_PAIS):
        competidores = np.random.choice(len(populacao), TAMANHO_TORNEIO, replace=False)
        vencedor = max((populacao[i] for i in competidores), key=lambda a: a.fitness)
        pais.append(vencedor)
    return pais


def mutar(agente, taxa=None, intensidade=None):
    """
    MUTAÇÃO + EXPLORAÇÃO

    Cada peso tem probabilidade TAXA_MUTACAO de ser alterado.
    O peso escolhido recebe um ruído gaussiano N(0, 1) * INTENSIDADE_MUTACAO.
      - taxa        -> QUANTOS pesos mudam
      - intensidade -> QUANTO cada peso muda
    """
    taxa = TAXA_MUTACAO if taxa is None else taxa
    intensidade = INTENSIDADE_MUTACAO if intensidade is None else intensidade

    for nome in ("W1", "b1", "W2", "b2"):
        pesos = getattr(agente, nome)
        mascara = np.random.rand(*pesos.shape) < taxa
        ruido = np.random.randn(*pesos.shape) * intensidade
        setattr(agente, nome, pesos + mascara * ruido)
    return agente


def nova_geracao(populacao, taxa=None, intensidade=None):
    """
    ELITISMO + REPRODUÇÃO + MUTAÇÃO
    """
    ordenada = sorted(populacao, key=lambda a: a.fitness, reverse=True)

    # 1. Selecionar os pais.
    pais = selecionar(populacao)

    # 2. Copiar NUM_ELITE melhores sem alteração (elitismo).
    nova = [copy.deepcopy(a) for a in ordenada[:NUM_ELITE]]

    # 3. Criar descendentes a partir dos selecionados.
    #    deepcopy cria arrays novos: mutar o filho não altera o pai.
    while len(nova) < TAMANHO_POPULACAO:
        pai = pais[np.random.randint(len(pais))]
        filho = copy.deepcopy(pai)

        # 4. Aplicar mutação nos descendentes.
        mutar(filho, taxa, intensidade)
        filho.fitness = 0.0
        nova.append(filho)

    # 5. Retornar exatamente TAMANHO_POPULACAO agentes.
    assert len(nova) == TAMANHO_POPULACAO
    return nova


# ---------------------------------------------------------
# CICLO EVOLUTIVO
# ---------------------------------------------------------

def evoluir(nome, taxa, intensidade, semente, pool, num_geracoes=NUM_GERACOES):
    """Executa o ciclo completo: população -> avaliação -> ... -> nova geração."""
    np.random.seed(semente)
    populacao = criar_populacao()
    historico = []
    melhor_global = None

    for geracao in range(num_geracoes):
        inicio = time.time()

        # Avaliar TODOS os indivíduos da geração.
        # Todos usam a mesma semente do ambiente nesta geração, para que a
        # comparação entre eles seja justa (mesmas condições de jogo).
        semente_ambiente = semente * 1000 + geracao
        tarefas = [(agente, semente_ambiente, JOGO) for agente in populacao]
        for agente, fit in zip(populacao, pool.map(_avaliar_tarefa, tarefas)):
            agente.fitness = fit

        valores = [agente.fitness for agente in populacao]

        melhor = max(valores)
        media = float(np.mean(valores))
        pior = min(valores)

        campeao = max(populacao, key=lambda a: a.fitness)
        if melhor_global is None or campeao.fitness >= melhor_global.fitness:
            melhor_global = copy.deepcopy(campeao)

        historico.append({
            "experimento": nome,
            "geracao": geracao,
            "melhor": melhor,
            "media": media,
            "pior": pior,
            "desvio": float(np.std(valores)),
        })

        print(
            f"[{nome}] Geração {geracao:03d} | "
            f"Melhor: {melhor:.2f} | "
            f"Média: {media:.2f} | "
            f"Pior: {pior:.2f} | "
            f"{time.time() - inicio:.1f}s"
        )

        # Produzir a próxima geração (a última não precisa).
        if geracao < num_geracoes - 1:
            populacao = nova_geracao(populacao, taxa, intensidade)

    return historico, melhor_global


def avaliar_varias_vezes(agente, sementes, pool):
    """Testa um agente em vários episódios novos (sementes não usadas no treino)."""
    return pool.map(_avaliar_tarefa, [(agente, s, JOGO) for s in sementes])


def avaliar_varias_vezes_aleatorio(sementes, pool):
    return pool.map(_avaliar_aleatorio, [(s, JOGO) for s in sementes])


# ---------------------------------------------------------
# RESULTADOS: CSV, GRÁFICOS E MELHOR AGENTE
# ---------------------------------------------------------

def salvar_csv(caminho, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


def _rotulo_fitness():
    return f"Fitness ({JOGOS[JOGO]['unidade']} no episódio)"


def grafico_evolucao(historico, titulo, caminho, baseline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [h["geracao"] for h in historico]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(g, [h["pior"] for h in historico], [h["melhor"] for h in historico],
                    color="#898781", alpha=0.15, label="Faixa pior–melhor")
    ax.plot(g, [h["melhor"] for h in historico], color="#2a78d6", lw=2, marker="o", ms=4, label="Melhor fitness")
    ax.plot(g, [h["media"] for h in historico], color="#eb6834", lw=2, marker="s", ms=4, label="Fitness médio")
    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="#898781", label=f"Agente aleatório ({baseline:.1f})")
    ax.set_xlabel("Geração")
    ax.set_ylabel(_rotulo_fitness())
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def grafico_comparacao(historicos, caminho, baseline=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for nome, hist in historicos.items():
        cfg = EXPERIMENTOS[nome]
        rotulo = f"{nome} (taxa {cfg['taxa']:.0%}, int. {cfg['intensidade']:.2f})"
        g = [h["geracao"] for h in hist]
        ax1.plot(g, [h["melhor"] for h in hist], color=cores[nome], lw=2, marker="o", ms=3, label=rotulo)
        ax2.plot(g, [h["media"] for h in hist], color=cores[nome], lw=2, marker="s", ms=3, label=rotulo)
    for ax, titulo in ((ax1, "Melhor fitness"), (ax2, "Fitness médio")):
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", color="#898781", label="Agente aleatório")
        ax.set_title(titulo)
        ax.set_xlabel("Geração")
        ax.grid(alpha=0.3)
    ax1.set_ylabel(_rotulo_fitness())
    ax2.legend(loc="upper left")
    fig.suptitle(f"{ENV_ID} - experimentos A, B e C (efeito da mutação)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def salvar_agente(agente, caminho):
    np.savez(caminho, fitness=agente.fitness, **agente.parametros())


def carregar_agente(caminho):
    dados = np.load(caminho)
    agente = Agente()
    for nome in ("W1", "b1", "W2", "b2"):
        setattr(agente, nome, dados[nome])
    agente.fitness = float(dados["fitness"])
    return agente


def assistir(caminho, semente=None):
    """Executa/renderiza o melhor agente encontrado em uma janela."""
    agente = carregar_agente(caminho)
    print(f"Assistindo {caminho} (fitness no treino: {agente.fitness:.0f})")
    pontos = avaliar(agente, semente, render_mode="human")
    print(f"{JOGOS[JOGO]['unidade'].capitalize()} neste episódio: {pontos:.0f}")


# ---------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------

def main():
    global TAMANHO_POPULACAO, NUM_ELITE

    parser = argparse.ArgumentParser(description="Neuroevolução em jogos de Atari (ALE)")
    parser.add_argument("--jogo", choices=sorted(JOGOS), default=JOGO_PADRAO)
    parser.add_argument("--experimento", choices=["A", "B", "C", "todos"], default="todos")
    parser.add_argument("--geracoes", type=int, default=NUM_GERACOES)
    parser.add_argument("--populacao", type=int, default=TAMANHO_POPULACAO)
    parser.add_argument("--elite", type=int, default=NUM_ELITE)
    parser.add_argument("--semente", type=int, default=42)
    parser.add_argument("--episodios-teste", type=int, default=10,
                        help="episódios para comparar aleatório x melhor evoluído")
    parser.add_argument("--processos", type=int, default=os.cpu_count(),
                        help="processos para avaliar a população em paralelo")
    parser.add_argument("--explorar", action="store_true", help="só demonstra o ambiente")
    parser.add_argument("--assistir", nargs="?", const="",
                        help="renderiza um agente salvo (padrão: melhor do experimento B)")
    args = parser.parse_args()

    configurar_jogo(args.jogo)
    pasta = pasta_resultados()

    if args.explorar:
        explorar_ambiente()
        return
    if args.assistir is not None:
        assistir(args.assistir or os.path.join(pasta, "melhor_agente_B.npz"))
        return

    TAMANHO_POPULACAO = args.populacao
    NUM_ELITE = max(1, args.elite)  # pelo menos o melhor é sempre preservado
    os.makedirs(pasta, exist_ok=True)

    nomes = ["A", "B", "C"] if args.experimento == "todos" else [args.experimento]
    # Sementes de teste nunca usadas no treino (treino usa semente*1000 + geração).
    sementes_teste = [10_000_000 + i for i in range(args.episodios_teste)]

    with Pool(args.processos) as pool:
        # Baseline: agente aleatório, ANTES da evolução.
        aleatorio = avaliar_varias_vezes_aleatorio(sementes_teste, pool)
        media_aleatorio = float(np.mean(aleatorio))
        print(f"\n[{ENV_ID}] Agente aleatório em {len(sementes_teste)} episódios: "
              f"média {media_aleatorio:.2f} | min {min(aleatorio):.0f} | max {max(aleatorio):.0f}\n")

        historicos, resumo = {}, {
            "configuracao": {
                "ambiente": ENV_ID, "observacao": "RAM (128 bytes)",
                "rede": f"{NUM_ENTRADAS}-{NUM_OCULTOS}-{NUM_ACOES} (tanh, argmax)",
                "max_passos": MAX_PASSOS,
                "populacao": TAMANHO_POPULACAO, "geracoes": args.geracoes,
                "elite": NUM_ELITE, "selecao": f"torneio k={TAMANHO_TORNEIO}, {NUM_PAIS} pais",
                "semente": args.semente,
            },
            "aleatorio": {"episodios": aleatorio, "media": media_aleatorio},
            "experimentos": {},
        }

        for nome in nomes:
            cfg = EXPERIMENTOS[nome]
            print(f"=== Experimento {nome}: taxa {cfg['taxa']:.0%}, intensidade {cfg['intensidade']} ===")
            inicio = time.time()
            hist, melhor = evoluir(nome, cfg["taxa"], cfg["intensidade"], args.semente, pool, args.geracoes)
            historicos[nome] = hist

            teste = avaliar_varias_vezes(melhor, sementes_teste, pool)
            print(f"[{nome}] Melhor agente em {len(sementes_teste)} episódios novos: "
                  f"média {np.mean(teste):.2f} | min {min(teste):.0f} | max {max(teste):.0f} "
                  f"({time.time() - inicio:.0f}s)\n")

            salvar_csv(os.path.join(pasta, f"fitness_{nome}.csv"), hist)
            grafico_evolucao(hist, f"{ENV_ID} - experimento {nome}: taxa {cfg['taxa']:.0%}, intensidade {cfg['intensidade']}",
                             os.path.join(pasta, f"fitness_{nome}.png"), media_aleatorio)
            salvar_agente(melhor, os.path.join(pasta, f"melhor_agente_{nome}.npz"))

            resumo["experimentos"][nome] = {
                **cfg,
                "melhor_final": hist[-1]["melhor"],
                "media_final": hist[-1]["media"],
                "melhor_de_todas": max(h["melhor"] for h in hist),
                "media_ultimas_5": float(np.mean([h["media"] for h in hist[-5:]])),
                "teste_melhor_agente": teste,
                "teste_media": float(np.mean(teste)),
                "segundos": round(time.time() - inicio),
            }

    # Arquivos principais pedidos na entrega: fitness.csv e fitness.png.
    todas = [linha for nome in nomes for linha in historicos[nome]]
    salvar_csv(os.path.join(pasta, "fitness.csv"), todas)
    if len(nomes) > 1:
        grafico_comparacao(historicos, os.path.join(pasta, "fitness.png"), media_aleatorio)
    else:
        grafico_evolucao(historicos[nomes[0]], f"{ENV_ID} - experimento {nomes[0]}",
                         os.path.join(pasta, "fitness.png"), media_aleatorio)

    with open(os.path.join(pasta, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)

    print("Resumo")
    print(f"  Aleatório ............: média {media_aleatorio:.2f}")
    for nome, r in resumo["experimentos"].items():
        print(f"  Experimento {nome} ......: melhor {r['melhor_de_todas']:.0f} | "
              f"média final {r['media_final']:.2f} | teste {r['teste_media']:.2f}")
    print(f"Arquivos em {pasta}")


if __name__ == "__main__":
    main()
