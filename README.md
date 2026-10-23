# Neuroevolução no Atari

Trabalho prático **"Evolução de um agente inteligente no Atari"**
Fundamentos de Inteligência Artificial · UniSenac Pelotas · 2026/2 · Prof. Pablo De Chiaro Rosa

Uma rede neural aprende a jogar **Asterix** (Atari 2600) sem backpropagation, só por evolução: uma população de agentes joga, os melhores são selecionados, sofrem mutação e formam a geração seguinte. O mesmo algoritmo roda no **Freeway**, o jogo-base do enunciado.

**Painel ao vivo:** https://neuroevolution.aamora.com.br
O treino roda no servidor enquanto você assiste. Na arena, o agente aleatório e o melhor agente evoluído jogam a mesma partida, lado a lado.

```
POPULAÇÃO → AVALIAÇÃO → FITNESS → SELEÇÃO → MUTAÇÃO → NOVA GERAÇÃO
```

## O algoritmo

| Etapa | Como foi feito |
|---|---|
| **Indivíduo** | Rede `128 → 16 → N ações` (tanh na camada oculta, argmax na saída). O "DNA" são os pesos `W1, b1, W2, b2`. |
| **Entrada** | A RAM do Atari: 128 bytes, normalizados para 0–1. |
| **População** | 20 agentes com pesos aleatórios (`np.random.randn`). |
| **Avaliação / fitness** | Pontos de uma partida no Asterix (travessias no Freeway). Partidas paralelas com `multiprocessing`. |
| **Seleção** | Torneio de 3: sorteia 3 agentes e fica o de maior fitness, 10 vezes. Prefere os melhores sem eliminar a diversidade e funciona com fitness zero ou empatado. |
| **Elitismo** | Os 2 melhores passam para a geração seguinte sem alteração. |
| **Mutação** | Cada peso muda com probabilidade `taxa` e recebe ruído gaussiano `N(0,1) × intensidade`. |
| **Comparação** | Agente aleatório (ação sorteada a cada passo) jogando com as mesmas sementes: é a linha de referência. |

### Experimentos

| Experimento | Taxa de mutação | Intensidade | Hipótese |
|---|---|---|---|
| A | 1% | 0,05 | Pouca alteração entre pais e filhos |
| B | 10% | 0,20 | Configuração intermediária |
| C | 40% | 0,50 | Grande diversidade e maior risco de destruir boas soluções |

Padrão: 30 gerações, população 20, elite 2, semente 42. No fim, o melhor agente e o aleatório são testados em partidas novas.

## Rodando o script

Requer Python 3.10+.

```bash
pip install "gymnasium[atari]" numpy matplotlib pillow

python main.py --explorar          # mostra como o ambiente funciona
python main.py                     # agente aleatório + experimentos A, B e C
python main.py --experimento B     # só um experimento
python main.py --jogo freeway      # mesmo algoritmo no Freeway
python main.py --assistir          # renderiza o melhor agente salvo (padrão: B)
python main.py --help              # todas as opções
```

| Opção | Padrão | |
|---|---|---|
| `--jogo` | `asterix` | `asterix` ou `freeway` |
| `--experimento` | `todos` | `A`, `B`, `C` ou `todos` |
| `--geracoes` | 30 | |
| `--populacao` | 20 | |
| `--elite` | 2 | |
| `--semente` | 42 | |
| `--episodios-teste` | 10 | partidas no teste final (aleatório × evoluído) |
| `--processos` | nº de CPUs | avaliação em paralelo |

Tudo fica em `resultados/` (no Freeway, em `resultados/freeway/`):

- `fitness.csv` e `fitness.png`: os arquivos da entrega, com todos os experimentos e o gráfico comparativo
- `fitness_A.csv`, `fitness_A.png` … por experimento
- `melhor_agente_A.npz` …: pesos do melhor agente de cada experimento
- `resumo.json`: configuração, aleatório e teste final

## Painel web

`web/app.py` (FastAPI) roda o mesmo `main.py` e serve uma página única (`web/static/index.html`, HTML/CSS/JS puro com Chart.js).

- **Arena ao vivo:** aleatório × melhor evoluído na mesma partida. Uma única thread joga e todos os espectadores recebem os mesmos quadros (MJPEG).
- **Veredito:** "o evoluído faz N× os pontos do aleatório".
- **Gráfico:** fitness por geração (melhor, média, faixa e linha do aleatório) e a população atual.
- **Resultados:** execuções salvas, comparação de curvas, CSV por execução e replay de agentes salvos.
- **Modo telão** para projetor: `?telao=1` ou tecla <kbd>T</kbd>.
- **Controles** (iniciar, pausar, parar, limpar, excluir) protegidos por senha. Sem senha, só dá para assistir.

### Rodando com Docker

```bash
echo "CONTROLE_SENHA=troque-esta-senha" > .env
mkdir -p dados && sudo chown 10001:10001 dados
docker compose up -d --build
# http://127.0.0.1:5103
```

| Variável | Padrão | |
|---|---|---|
| `CONTROLE_SENHA` | (vazia) | senha dos controles |
| `PROCESSOS` | 2 | processos para avaliar a população |
| `MAX_ARENA` | 60 | espectadores por lado da arena |
| `MAX_ESPECTADORES` | 3 | replays simultâneos (cada um roda um jogo próprio) |
| `DADOS_DIR` | `/data` | onde ficam as execuções salvas |

O estado do treino fica na memória do processo (um único worker). Reiniciar o container (`docker compose up -d --build`) interrompe o treino em andamento.

## Estrutura

```
main.py                 algoritmo evolutivo e linha de comando
web/app.py              API, treino em segundo plano, arena e replays
web/static/index.html   painel
Dockerfile, docker-compose.yml
PRODUCT.md              contexto do produto (público, objetivos, princípios)
```
