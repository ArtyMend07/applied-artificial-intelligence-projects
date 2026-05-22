"""
Contextualização do Problema no Universo de Sekiro:

Em Sekiro Shadows Die Twice os jogadores controlam um shinobi em uma versão
reimaginada do Japão no final do século XVI. O combate é caracterizado pela
tensão extrema e pelo entrechoque de espadas focado na mecânica de Postura.
(Visão geral da obra: https://pt.wikipedia.org/wiki/Sekiro:_Shadows_Die_Twice)

Diferente de RPGs tradicionais onde o foco é zerar a vida do inimigo, aqui
o objetivo primário é encher a barra de postura do oponente através de
ataques e deflexões perfeitas. Quando a barra enche a guarda quebra e
permite um golpe mortal imediato.
(Explicação detalhada da mecânica: https://sekiroshadowsdietwice.wiki.fextralife.com/Posture)

Adaptação do Problema para o Modelo Oculto de Markov:

Este projeto adapta a mecânica de postura para um problema de inferência
baseado em som. Nesta simulação assumimos que o jogador não está olhando
para a interface visual mas sim ouvindo o ritmo da luta.

O estado oculto que o algoritmo tenta rastrear é o nível de integridade da
postura do inimigo que pode estar Intacta, Abalada ou Quebrada. As
observações são os sons produzidos pelo choque das lâminas. Um bloqueio
gera um som abafado, uma deflexão gera um som agudo e a quebra gera um
estrondo distinto.

A solução utiliza um Filtro de Partículas SIR para estimar a distribuição de
crença sobre a postura do inimigo a cada instante, permitindo ao agente
decidir o momento certo de atacar mesmo sem ver a barra de status.
"""
import matplotlib.pyplot as plt #Biblioteca utilizada para gerar gráficos
import numpy as np #Biblioteca usada para operações com matrizes e vetores

# Esta seção define os parâmetros de um Modelo Oculto de Markov 
# conforme a especificação teórica.

# 1. Definição do Espaço de Estados (X_t)
# O estado oculto que o filtro tentará rastrear é a Postura do inimigo.
# X_t = { 0: 'POSTURA_INTACTA', 1: 'POSTURA_ABALADA', 2: 'POSTURA_QUEBRADA' }
#
# Postura Intacta (0) representa a barra de postura baixa (branca) e luta normal.
# Postura Abalada (1) representa a barra alta (amarela/laranja) e o inimigo vacilante.
# Postura Quebrada (2) representa a barra cheia (vermelha), pronta para o Golpe Mortal.
ESTADOS = {'POSTURA_INTACTA': 0, 'POSTURA_ABALADA': 1, 'POSTURA_QUEBRADA': 2}
CONTA_ESTADOS = len(ESTADOS)

# 2. Definição do Espaço de Observação (E_t)
# A evidência observável que o agente recebe, neste caso, o som das espadas.
# E_t = { 0: 'BLOQUEIO', 1: 'DEFLEXAO', 2: 'SOM_DE_QUEBRA' }
#
# Bloqueio (0) é o som surdo de um bloqueio normal, com baixo dano de postura.
# Deflexão (1) é o som agudo de uma deflexão bem-sucedida (como um Mikiri).
# Som de Quebra (2) é o som grave indicando que a postura do inimigo quebrou.
OBSERVACOES = {'BLOQUEIO': 0, 'DEFLEXAO': 1, 'SOM_DE_QUEBRA': 2}
CONTA_OBSERVACOES = len(OBSERVACOES)

# 3. Distribuição Inicial P(X_0)
# A crença a priori sobre a postura do inimigo no tempo t=0.
# Assume-se que o inimigo quase sempre começa com a Postura Intacta.
PROB_INICIAL = np.array([0.9, 0.1, 0.0])
# Verificação de sanidade; deve somar 1
assert np.isclose(np.sum(PROB_INICIAL), 1.0)

# 4. Modelo de Transição P(X_t | X_{t-1})
# Representado como uma matriz T[i, j] = P(X_t = j | X_{t-1} = i).
# Cada linha (do estado i) deve somar 1.
TRANSICAO_MATRIZ = np.array([
    # Transições de POSTURA_INTACTA (0) -> [Int, Aba, Que]
    [0.75, 0.25, 0.00],
    # Transições de POSTURA_ABALADA (1) -> [Int, Aba, Que] (Modela a regeneração)
    [0.40, 0.40, 0.20],
    # Transições de POSTURA_QUEBRADA (2) -> [Int, Aba, Que] (Modela a recuperação)
    [0.70, 0.30, 0.00]
])
# Verificação de sanidade; cada linha deve somar 1
assert TRANSICAO_MATRIZ.shape == (CONTA_ESTADOS,CONTA_ESTADOS)
assert np.all(np.isclose(np.sum(TRANSICAO_MATRIZ, axis=1), 1.0))

# 5. Modelo de Sensor P(E_t | X_t)
# Representado como uma matriz O[i, j] = P(E_t = j | X_t = i).
# Cada linha (dado um estado i) deve somar 1.
OBSERVACAO_MATRIZ = np.array([
    # Probabilidades de observação para POSTURA_INTACTA (0) -> [Blo, Def, Que]
    [0.9, 0.1, 0.0],   # Principalmente Bloqueio ou Deflexão
    # Probabilidades de observação para POSTURA_ABALADA (1) -> [Blo, Def, Que]
    [0.10, 0.80, 0.1],   # Principalmente Deflexão (som agudo)
    # Probabilidades de observação para POSTURA_QUEBRADA (2) -> [Blo, Def, Que]
    [0.0, 0.1, 0.9]    # Alta probabilidade do Som de Quebra
])
# Verificação de sanidade; cada linha deve somar 1
assert OBSERVACAO_MATRIZ.shape == (CONTA_ESTADOS,CONTA_OBSERVACOES)
assert np.all(np.isclose(np.sum(OBSERVACAO_MATRIZ, axis=1), 1.0))


def simular_mundo(p0, T, O, n_passos):
    """
    Gera uma sequência de verdade fundamental a partir
    do modelo markoviano oculto.

    Este processo segue a definição gerativa do modelo markoviano oculto nos seguintes passos
    1. Amostra x_0 de P(X_0)
    2. Para t=1..n_passos:
        a. Amostra x_t de P(X_t | X_{t-1} = x_{t-1}) (usando T)
        b. Amostra e_t de P(E_t | X_t = x_t) (usando O)

    Isso cria um conjunto de dados sintético onde os estados ocultos são
    conhecidos, permitindo testar o desempenho do filtro.

    Args:
        p0 (np.array): Distribuição inicial P(X_0).
        T (np.array): Matriz de transição P(X_t | X_{t-1}).
        O (np.array): Matriz de observação P(E_t | X_t).
        n_passos (int): O número de passos de tempo (t) a simular.

    Returns:
        tuple: (lista de estados_reais, lista de observacoes)
               estados_reais tem comprimento n_passos + 1 (pois inclui t=0)
               observacoes tem comprimento n_passos (indo de t=1 até t=n)
    """
    estados_reais = []
    observacoes = []

    # Amostra do estado inicial (t=0)
    estado_atual = np.random.choice(CONTA_ESTADOS,p=p0)
    estados_reais.append(estado_atual)

    # Simulação para t=1 até n_passos
    for _ in range(n_passos):
        # Etapa (a) Amostrar a próxima transição de estado
        # A distribuição de probabilidade é a linha T[estado_atual]
        prob_transicao = T[estado_atual, :]
        estado_atual = np.random.choice(CONTA_ESTADOS, p=prob_transicao)
        estados_reais.append(estado_atual)

        # Etapa (b) Amostrar a observação do estado atual
        # A distribuição de probabilidade é a linha O[estado_atual]
        prob_observacao = O[estado_atual, :]
        obs_atual = np.random.choice(CONTA_OBSERVACOES, p=prob_observacao)
        observacoes.append(obs_atual)

    return estados_reais, observacoes


class FiltroDeParticulasSIR:
    """
    Implementa um Filtro de Partículas Sequential Importance Resampling (SIR)
    para realizar inferência aproximada em um modelo markoviano oculto. 

    Este filtro rastreia um conjunto de N hipóteses estocásticas (partículas)
    sobre o estado oculto. A coleção de partículas ponderadas forma uma
    aproximação empírica da distribuição de crença P(X_t | e_{1:t}).
    """

    def __init__(self, n_particulas, p0, T, O):
        """
        Inicializa o filtro.

            n_particulas (int): O número de partículas a usar
            p0 (np.array): P(X_0), a distribuição inicial
            T (np.array): P(X_t | X_{t-1}), a matriz de transição
            O (np.array): P(E_t | X_t), a matriz de observação
        """
        self.N = n_particulas
        self.P0 = p0
        self.T = T
        self.O = O

        # Atributos de estado do filtro
        # self.particulas armazena um array de inteiros 
        # representando o estado de cada hipótese.
        self.particulas = np.random.choice(CONTA_ESTADOS, size=self.N, p=self.P0)
        
        # self.pesos armazena um array de floats, representando a
        # crença em cada partícula.
        self.pesos = np.ones(self.N) / self.N


    def predicao(self):
        """
        Etapa de Predição (Propagação).

        Move cada partícula para um novo estado, amostrando do modelo de
        transição P(X_t | X_{t-1}).

        Isto aproxima estocasticamente o termo de predição da filtragem
        Bayesiana, que é
        P(X_t | e_{1:t-1}) = Σ_{x_{t-1}} [ P(X_t|x_{t-1}) * P(x_{t-1}|e_{1:t-1}) ]

        Cada partícula x_{t-1}^{(i)} é uma amostra da crença anterior
        P(x_{t-1}|e_{1:t-1}). Amostrar x_t^{(i)} ~ P(X_t | X_{t-1}^{(i)})
        transforma o conjunto de partículas em uma aproximação de
        P(X_t | e_{1:t-1}), a crença antes de ver a observação e_t.
        """
        novas_particulas = np.zeros_like(self.particulas)
        for i in range(CONTA_ESTADOS):
            #encontrar todas as partículas que estão atualmente no estado i
            indices = (self.particulas == i)
            n_indices = np.sum(indices)
            if n_indices > 0:
                # amostrar novas posições para todas essas partículas
                # de uma vez, usando a linha i da matriz de transição.
                novas_particulas[indices] = np.random.choice(
                    CONTA_ESTADOS,
                    size=n_indices,
                    p=self.T[i, :]
                )
        self.particulas = novas_particulas

    def atualizar(self, obs_idx):
        """
        Etapa de Atualização (Ponderação).

        Dada a nova observação e_t (representada por obs_idx), re-pondera
        cada partícula com base em quão provável essa observação era,
        dado o estado (predito) da partícula.

        Isto aplica o termo de verossimilhança do sensor, dado por
        P(X_t|e_{1:t}) ∝ P(e_t | X_t) * P(X_t | e_{1:t-1})

        O novo peso é w_t^{(i)} ∝ w_{t-1}^{(i)} * P(e_t | X_t=x_t^{(i)}).
        No SIR, w_{t-1}^{(i)} é 1/N após a reamostragem, de modo que
        o peso é efetivamente w_t^{(i)} ∝ P(e_t | X_t=x_t^{(i)}).
        """
        # Obter o vetor de verossimilhança para a observação obs_idx.
        # self.O[:, obs_idx] é um vetor [P(e_t|X_t=0), P(e_t|X_t=1),...]
        verossimilhancas = self.O[:, obs_idx]

        # Aplicar as verossimilhanças a cada partícula com base em seu
        # estado atual.
        self.pesos = verossimilhancas[self.particulas]

        # normalização
        soma_pesos = np.sum(self.pesos)
        if soma_pesos > 1e-12:  # Evitar divisão por zero
            self.pesos /= soma_pesos
        else:
            # Caso catastrófico (todas as partículas têm peso zero),
            # re-inicializar com pesos uniformes para evitar falha.
            self.pesos = np.ones(self.N) / self.N

    def reamostrar_se_necessario(self):
        """
        Esta etapa mitiga a degenerescência das partículas.
        A variância dos pesos aumenta ao longo do tempo, e eventualmente
        quase todo o peso (crença) será concentrado em muito poucas
        partículas.

        A reamostragem resolve isso ao descartar estocasticamente partículas de baixo peso e multiplicar as de alto peso
        o novo conjunto de partículas resultante representa a mesma distribuição, porém agora todas possuem pesos uniformes (1/N)
        """
        # Calcular o Tamanho Efetivo da Amostra 
        tea = 1.0 / np.sum(self.pesos ** 2)

        # Limite de reamostragem 
        limite_tea = self.N / 2.0

        if tea < limite_tea:
            # Realizar reamostragem multinomial
            indices_reamostrados = np.random.choice(
                self.N,
                size=self.N,
                replace=True,
                p=self.pesos
            )
            self.particulas = self.particulas[indices_reamostrados]
            self.pesos = np.ones(self.N) / self.N

    def executar_passo(self, obs_idx):
        """
        Executa um ciclo completo de filtragem (Predizer, Atualizar, Reamostrar)
        para uma nova observação.
        """
        # 1. Etapa de Predição (Propagação)
        self.predicao()

        # 2. Etapa de Atualização (Ponderação)
        self.atualizar(obs_idx)

        # 3. Etapa de Reamostragem (Mitigação de Degenerescência)
        self.reamostrar_se_necessario()

    def obter_crenca(self):
        """
        Extrai a crença de filtragem atual, P(X_t | e_{1:t}).

        A distribuição posterior é aproximada pelo histograma ponderado
        das partículas.

        Retorna:
            np.array: Um vetor de probabilidade que representa a
                      crença posterior aproximada.
        """
        crenca = np.bincount(
            self.particulas,
            weights=self.pesos,
            minlength=CONTA_ESTADOS
        )
        
        soma_crenca = np.sum(crenca)
        if soma_crenca > 0:
            crenca /= soma_crenca
        return crenca

def plotar_resultados_markov(historico_crencas, historico_estados_reais, historico_estados_estimados, idx_para_estado):
    """
    Plota dois gráficos:
    1. A evolução da crença do filtro (Stacked Area Plot).
    2. A comparação entre o estado real e o estado estimado (Step Plot).
    """
    
    n_passos_total = len(historico_crencas)
    vetor_tempo = np.arange(n_passos_total)
    
    # 1. Converter a lista de crenças em um array numpy (Passos, N_Estados)
    crencas_array = np.array(historico_crencas)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # ---- Plot 1: Evolução da Crença  ----
    labels_crenca = [idx_para_estado[i] for i in range(crencas_array.shape[1])]
    # Cores para Intacta (Azul), Abalada (Laranja), Quebrada (Vermelho)
    cores = ['#1f77b4', '#ff7f0e', '#d62728'] 
    
    # stackplot espera dados como (N_Estados, N_Passos), por isso usamos .T
    ax1.stackplot(vetor_tempo, crencas_array.T, labels=labels_crenca, alpha=0.8, colors=cores)
    
    ax1.set_title('Evolução da Crença do Filtro de Partículas', fontsize=14)
    ax1.set_ylabel('Probabilidade (Crença)', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # ---- Plot 2: Comparação Estado Real vs. Estimado ----
    # Usa 'drawstyle="steps-post"' para estados discretos
    ax2.plot(vetor_tempo, historico_estados_reais, label='Estado Real', 
             color='blue', linestyle='--', linewidth=2.5, drawstyle="steps-post")
    ax2.plot(vetor_tempo, historico_estados_estimados, label='Estado Estimado)', 
             color='red', alpha=0.7, linewidth=1.5, drawstyle="steps-post")
    
    ax2.set_title('Desempenho do Rastreamento de Estado', fontsize=14)
    ax2.set_xlabel('Passo de Tempo (t)', fontsize=12)
    ax2.set_ylabel('Estado Oculto (Postura)', fontsize=12)
    
    # Definir os rótulos do eixo Y para os estados discretos
    ax2.set_yticks(list(idx_para_estado.keys()))
    ax2.set_yticklabels(list(idx_para_estado.values()))
    ax2.set_ylim(-0.5, CONTA_ESTADOS - 0.5) # Centralizar os ticks
    
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuração dos Hiperparâmetros da Simulação e do Filtro
    N_PARTICULAS = 50000    # número de amostras para o filtro de partículas
    N_PASSOS_TEMPO = 180   # duração da simulação 

    # 1. Geração dos Dados
    # Um cenário real é gerado usando o modelo markoviano
    # O filtro não terá acesso aos estados reais
    estados_reais, observacoes = simular_mundo(
        PROB_INICIAL,
        TRANSICAO_MATRIZ,
        OBSERVACAO_MATRIZ,
        N_PASSOS_TEMPO
    )

    # 2. Inicialização do Filtro
    # O filtro recebe os parâmetros do modelo (T, O, P0),
    # mas não os dados reais da simulação.
    filtro_fp = FiltroDeParticulasSIR(
        N_PARTICULAS,
        PROB_INICIAL,
        TRANSICAO_MATRIZ,
        OBSERVACAO_MATRIZ
    )

    # Mapeamento reverso para impressão
    idx_para_estado = {i: s for s, i in ESTADOS.items()}
    idx_para_obs = {i: o for o, i in OBSERVACOES.items()}

    print("Iniciando Rastreamento de Postura do boss com Filtro de Partículas...")
    print(f"Configuração: {N_PARTICULAS} partículas, {N_PASSOS_TEMPO} passos de tempo.")
    print("Formato: t | Obs. Real       | Estado Real         | Estado Estimado  | Crença P(X_t|e_1:t)")

    historico_crencas = []
    historico_estados_reais = []
    historico_estados_estimados = []

    # 3. Execução do Loop de Filtragem
    n_corretos = 0
    
    # Crença inicial (t=0)
    crenca_t0 = filtro_fp.obter_crenca()
    

    estado_estimado_t0 = np.argmax(crenca_t0)
    estado_real_t0 = estados_reais[0]
    historico_crencas.append(crenca_t0)
    historico_estados_reais.append(estado_real_t0)
    historico_estados_estimados.append(estado_estimado_t0)
   
    
    # Ajustar ljust para os nomes longos dos estados
    est_t0 = idx_para_estado[estado_estimado_t0].ljust(17) # <-- Alterado de np.argmax(crenca_t0)
    real_t0 = idx_para_estado[estado_real_t0].ljust(17) # <-- Alterado de estados_reais[0]
    
    # Abreviações para a tabela de crença
    crenca_str_t0 = (
        f"[Int: {crenca_t0[0]:.2f}, "  # Intacta
        f"Aba: {crenca_t0[1]:.2f}, "  # Abalada
        f"Que: {crenca_t0[2]:.2f}]"   # Quebrada
    )
    print(
        f"{0:03d} | "
        f"{'---'.ljust(14)} | "
        f"{real_t0} | "
        f"{est_t0} | "
        f"{crenca_str_t0}"
    )

    for t in range(N_PASSOS_TEMPO):
        # A observação no tempo t 
        obs_idx = observacoes[t]
        # O estado real no tempo t+1 
        estado_real_idx = estados_reais[t+1] 

        # Alimentar a observação no filtro e executar um passo
        filtro_fp.executar_passo(obs_idx)

        # Obter a crença posterior P(X_{t+1} | e_{1:t+1}) do filtro
        crenca_posterior = filtro_fp.obter_crenca()

        # O estado estimado é o que tem a maior probabilidade
        estado_estimado_idx = np.argmax(crenca_posterior)

        # Acumular estatísticas
        if estado_real_idx == estado_estimado_idx:
            n_corretos += 1

        
        historico_crencas.append(crenca_posterior)
        historico_estados_reais.append(estado_real_idx)
        historico_estados_estimados.append(estado_estimado_idx)
        

        # Formatar saída para a tabela de resultados
        # Ajustar ljust para os nomes longos
        obs_nome = idx_para_obs[obs_idx].ljust(14)
        real_nome = idx_para_estado[estado_real_idx].ljust(17)
        est_nome = idx_para_estado[estado_estimado_idx].ljust(17)
        crenca_str = (
            f"[Int: {crenca_posterior[0]:.2f}, "
            f"Aba: {crenca_posterior[1]:.2f}, "
            f"Que: {crenca_posterior[2]:.2f}]"
        )

        print(
            f"{t+1:03d} | "
            f"{obs_nome} | "
            f"{real_nome} | "
            f"{est_nome} | "
            f"{crenca_str}"
        )

    # 4. Resultados Finais
    acuracia = (n_corretos / N_PASSOS_TEMPO) * 100
    print(f"--- Resultado da Simulação ---")
    print(f"Acurácia da Estimativa (Estado Real == Estado Estimado): "
          f"{acuracia:.2f}%")
    plotar_resultados_markov(
        historico_crencas, 
        historico_estados_reais, 
        historico_estados_estimados, 
        idx_para_estado
    )