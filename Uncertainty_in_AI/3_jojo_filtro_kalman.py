"""
Contextualização do Problema no Universo de JoJo:

Em JoJo's Bizarre Adventure, os personagens lutam usando Stands,
que são manifestações visuais de sua energia psíquica com
habilidades únicas. O vilão da Parte 4, Kira Yoshikage,
possui o Stand Killer Queen, focado em criar explosões.
(Explicação mais aprofundada sobre a obra: https://jjba.fandom.com/pt-br/wiki/Diamond_is_Unbreakable)

Uma das habilidades do Killer Queen é o Sheer Heart Attack, ou SHA.
Ele é um pequeno tanque autônomo e indestrutível que funciona
como uma bomba teleguiada. Canonicamente, ele rastreia o alvo
pela assinatura de calor mais forte na área e o persegue
implacavelmente até o contato, quando então explode.
O protagonista da Parte 3, Jotaro Kujo, viaja para a cidade
de Morioh para investigar Kira, tornando-se seu principal antagonista.

Para mais explicações sobre a habilidade Sheer Heart Attack: https://jojowiki.com/Sheer_Heart_Attack

Adaptação do Problema para o Filtro de Kalman:

Este projeto adapta a premissa do Sheer Heart Attack para um
problema clássico de rastreamento de perseguição.
Nesta simulação, o alvo, Jotaro, move-se em uma trajetória circular
previsível. O rastreador, Sheer Heart Attack, não busca calor,
mas sim a posição exata do alvo. 

A dinâmica de movimento do rastreador é o que define o problema.
Ele possui uma magnitude de aceleração constante 2.0 m/s^2,
mas o vetor dessa aceleração é sempre recalculado para apontar
diretamente para a posição atual do alvo.

Isso introduz uma forte não-linearidade no sistema, pois a
aceleração é uma função da posição relativa entre o rastreador
e o alvo. Um Filtro de Kalman padrão, ou KF, não pode ser usado,
pois exige um modelo de transição linear.

A solução é o Filtro de Kalman Estendido, ou FKE. O FKE contorna
a não-linearidade ao linearizar a função de transição a cada
passo de tempo, usando uma Matriz Jacobiana. Isso permite
que o filtro estime a trajetória do rastreador com precisão,
mesmo recebendo apenas medições ruidosas de um radar.

Limitação do Modelo e Mecânica do Time Stop:

Uma mecânica adicional que foi considerada, para homenagear
o alvo Jotaro Kujo, era sua habilidade de parar o tempo.
A ideia era que, em um ponto aleatório, Jotaro pararia o tempo
e saltaria instantaneamente para uma nova posição no mapa através da habilidade de seu stand (Star Platinum: The World).

Esta mecânica não foi implementada porque um Filtro de Kalman
Estendido padrão não é suficiente para lidar com ela.
O FKE opera sob a premissa de um processo contínuo, onde o
estado no tempo k é previsto com base no estado no tempo k-1.

Um salto instantâneo quebraria essa premissa fundamental.
A próxima medição do radar estaria drasticamente diferente da
posição que o filtro previu. Essa discrepância, ou inovação,
seria tão massiva que faria o filtro meio que perder o rastro
e divergir, sendo incapaz de fundir a predição e a medição
corretamente.

Sistemas que lidam com mudanças tão abruptas de modelo
geralmente exigem técnicas mais avançadas, como um Filtro
de Múltiplos Modelos por exemplo.

Para mais explicações do Time Stop de Jotaro (Star Platinum: The World): https://jojowiki.com/Star_Platinum#Abilities
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
import matplotlib.animation as animation  # importa o módulo de animação

#
# este script implementa um Filtro de Kalman Estendido (FKE) para um
# problema de rastreamento não-linear.
#
# o Filtro de Kalman padrão, ou KF, é ideal apenas para sistemas lineares,
# descritos por equações de estado como: x_k = A*x_k-1 + B*u_k
#
# neste problema, o objeto rastreador possui uma dinâmica não-linear:
# sua aceleração é uma função de sua própria posição e da posição do alvo:
#
# a = f(pos_rastreador, pos_alvo)
#
# o Filtro de Kalman Estendido (FKE) contorna essa não-linearidade
# ao linearizar a função de transição de estado f(x) a cada passo de tempo.
# isso é feito calculando a Matriz Jacobiana, que é a derivada parcial de f,
# no ponto da estimativa atual, permitindo a propagação da covariância.
#


def obter_trajetoria_alvo_jotaro(passos_totais, dt):
    """
    simula a trajetória de um alvo, a verdade absoluta, em movimento circular.
    """
    t = np.arange(0, passos_totais * dt, dt)

    # parâmetros da trajetória circular
    raio = 150.0
    velocidade_angular = 0.02

    # calcula a posição x e y do alvo
    alvo_x = raio * np.cos(velocidade_angular * t)
    alvo_y = raio * np.sin(velocidade_angular * t)

    # retorna um array de N por 2 com as posições x, y
    return np.stack((alvo_x, alvo_y), axis=1)


def funcao_transicao_estado_f(estado_anterior, posicao_alvo, dt):
    """
    esta é a função de transição de estado não-linear f(x).
    ela calcula o próximo estado x_k com base no estado anterior x_k-1.
    
    o vetor de estado é [x_pos, y_pos, x_vel, y_vel].

    a dinâmica do modelo, ou seja, a física do movimento, é:
    1. calcular o vetor diretor do rastreador para o alvo.
    2. aplicar uma aceleração de magnitude constante nessa direção.
    3. integrar a aceleração e velocidade para encontrar o novo estado
       usando equações de movimento de aceleração constante.
    """
    
    # define a magnitude da aceleração do rastreador, que é constante
    MAGNITUDE_ACELERACAO_RASTREADOR = 2.0  # m/s^2

    # desempacota o vetor de estado anterior
    x, y, vx, vy = estado_anterior
    
    # desempacota a posição do alvo
    alvo_x, alvo_y = posicao_alvo

    # calcula o vetor e a distância até o alvo
    dx = alvo_x - x
    dy = alvo_y - y
    distancia = norm([dx, dy])

    # evita divisão por zero se o rastreador alcançar o alvo
    if distancia < 1e-6:
        ax = 0.0
        ay = 0.0
    else:
        # calcula a aceleração em x e y: vetor unitário multiplicado pela magnitude
        ax = MAGNITUDE_ACELERACAO_RASTREADOR * (dx / distancia)
        ay = MAGNITUDE_ACELERACAO_RASTREADOR * (dy / distancia)

    # aplica as equações de movimento, conforme o modelo de aceleração constante
    
    # v_k = v_k-1 + a * dt
    vx_k = vx + ax * dt
    vy_k = vy + ay * dt
    
    # p_k = p_k-1 + v_k-1*dt + 0.5*a*dt^2
    x_k = x + vx * dt + 0.5 * ax * (dt**2)
    y_k = y + vy * dt + 0.5 * ay * (dt**2)

    return np.array([x_k, y_k, vx_k, vy_k])


def calcular_jacobiano_F(estado_anterior, posicao_alvo, dt):
    """
    calcula a Matriz Jacobiana F_j, a derivada de f(x) em relação a x.
    
    F_j = d(f(x)) / d(x)
    
    aqui é utilizada a diferenciação numérica, especificamente diferenças finitas centradas,
    para aproximar o Jacobiano. este método é robusto e evita os
    erros de uma derivação analítica manual complexa.
    """
    
    # o estado tem 4 dimensões: x, y, vx, vy
    num_dimensoes = estado_anterior.shape[0]  
    F_j = np.zeros((num_dimensoes, num_dimensoes))
    
    # epsilon, ou ε, é um pequeno valor para a perturbação
    epsilon = 1e-6

    for j in range(num_dimensoes):
        # cria vetores de perturbação
        estado_mais = estado_anterior.copy()
        estado_menos = estado_anterior.copy()
        
        # perturba a j-ésima variável de estado
        perturbacao = np.zeros(num_dimensoes)
        perturbacao[j] = epsilon
        
        estado_mais += perturbacao
        estado_menos -= perturbacao

        # calcula f(x + epsilon) e f(x - epsilon)
        f_mais = funcao_transicao_estado_f(estado_mais, posicao_alvo, dt)
        f_menos = funcao_transicao_estado_f(estado_menos, posicao_alvo, dt)
        
        # derivada numérica: (f(x+eps) - f(x-eps)) / (2 * eps)
        coluna_j = (f_mais - f_menos) / (2.0 * epsilon)
        
        # atribui ao Jacobiano
        F_j[:, j] = coluna_j

    return F_j


def simular_trajetoria_real_sha(trajetoria_alvo, estado_inicial, dt):
    """
    simula a trajetória real, a verdade absoluta, do rastreador do Sheer Heart Attack,
    propagando o estado usando a função não-linear f(x) sem ruído de processo.
    """
    
    passos_totais = trajetoria_alvo.shape[0]
    num_dimensoes = estado_inicial.shape[0]
    
    trajetoria_real = np.zeros((passos_totais, num_dimensoes))
    trajetoria_real[0, :] = estado_inicial
    
    estado_atual = estado_inicial.copy()

    for k in range(1, passos_totais):
        posicao_alvo_k = trajetoria_alvo[k - 1]
        
        # propaga o estado real usando a função não-linear
        estado_atual = funcao_transicao_estado_f(estado_atual, posicao_alvo_k, dt)
        trajetoria_real[k, :] = estado_atual
        
    return trajetoria_real


def simular_medicoes_radar(trajetoria_real, matriz_R):
    """
    simula as medições ruidosas do sensor, que neste caso é um radar.
    o sensor mede apenas a posição x, y e adiciona ruído gaussiano R.
    """
    
    passos_totais = trajetoria_real.shape[0]
    
    # extrai o desvio padrão da matriz R
    desvio_padrao_R = np.sqrt(np.diag(matriz_R))
    
    # pega as posições reais x, y da trajetória real
    posicoes_reais = trajetoria_real[:, 0:2]
    
    # gera ruído gaussiano com o desvio padrão especificado
    ruido = np.random.randn(passos_totais, 2) * desvio_padrao_R
    
    # adiciona ruído a posição real para simular a medição
    medicoes = posicoes_reais + ruido
    return medicoes


def executar_filtro_fke(medicoes, trajetoria_alvo, estado_inicial_filtro, P_inicial_filtro, Q, R, H, dt):
    """
    Executa o loop principal do Filtro de Kalman Estendido, o FKE.
    """
    
    passos_totais = medicoes.shape[0]
    num_dimensoes = Q.shape[0]  # dimensão do estado, que é 4
    
    # arrays para armazenar os resultados
    historico_estimativas = np.zeros((passos_totais, num_dimensoes))
    historico_estimativas[0, :] = estado_inicial_filtro
    
    # P_k_atual é a covariância da estimativa, ou seja, a incerteza
    P_k_atual = P_inicial_filtro.copy()

    # loop principal do FKE
    for k in range(1, passos_totais):
        
        # --- Predição ---
        
        # pega o estado anterior e a posição do alvo
        estado_anterior_estimado = historico_estimativas[k - 1, :]
        posicao_alvo_k = trajetoria_alvo[k - 1, :]

        # calcula o Jacobiano F_j no ponto x_k-1, que é a linearização
        F_j = calcular_jacobiano_F(estado_anterior_estimado, posicao_alvo_k, dt)

        # 1.1: predição do estado, usando a função não-linear f
        # estado_predito = f(estado_anterior_estimado)
        estado_predito = funcao_transicao_estado_f(estado_anterior_estimado, posicao_alvo_k, dt)

        # 1.2: predição da covariância, usando o Jacobiano F_j
        # P_predito = F_j * P_k-1 * F_j.T + Q
        P_predito = F_j @ P_k_atual @ F_j.T + Q

        
        # ---- Atualização----
        
        # 2.1: cálculo do Ganho de Kalman K
        # S_inovacao = H * P_predito * H.T + R
        S_inovacao = H @ P_predito @ H.T + R
        K_ganho = P_predito @ H.T @ inv(S_inovacao)

        # 2.2: atualização da estimativa do estado com a inovação
        # y_inovacao = z_k - h(estado_predito)
        # como h, a observação, é linear, temos h(x) = H * x
        medicao_k = medicoes[k, :]
        y_inovacao = medicao_k - H @ estado_predito
        
        # estado_atualizado = estado_predito + K_ganho * y_inovacao
        estado_atualizado = estado_predito + K_ganho @ y_inovacao
        
        # 2.3: atualização da covariância da estimativa
        # P_k_atual = (I - K_ganho * H) * P_predito
        I_identidade = np.eye(num_dimensoes)
        P_k_atual = (I_identidade - K_ganho @ H) @ P_predito

        # armazena os resultados
        historico_estimativas[k, :] = estado_atualizado
        
    return historico_estimativas


def plotar_resultados(trajetoria_alvo_plot, trajetoria_real_sha_plot, medicoes_plot, estimativas_fke_plot, vetor_tempo):
    """
    plota os resultados estáticos da simulação.
    """
    
    plt.figure(figsize=(12, 9))
    
    plt.plot(trajetoria_alvo_plot[:, 0], trajetoria_alvo_plot[:, 1], 'g-',
             label='Trajetória do Jotaro', linewidth=2)
    
    plt.plot(trajetoria_real_sha_plot[:, 0], trajetoria_real_sha_plot[:, 1], 'b-',
             label='Trajetória do Sheer Heart Attack', linewidth=2)
    
    # plota as medições ruidosas como pontos
    plt.plot(medicoes_plot[:, 0], medicoes_plot[:, 1], 'r.',
             label='Medições Radar (Ruidoso)', alpha=0.5, markersize=4)
    
    plt.plot(estimativas_fke_plot[:, 0], estimativas_fke_plot[:, 1], 'k--',
             label='Estimativa FKE', linewidth=2.5)
    
    # plota os pontos iniciais
    plt.plot(trajetoria_alvo_plot[0, 0], trajetoria_alvo_plot[0, 1], 'go', markersize=10, label='Jotaro')
    plt.plot(trajetoria_real_sha_plot[0, 0], trajetoria_real_sha_plot[0, 1], 'bo', markersize=10, label='Sheer Heart Attack')

    plt.title('FKE Rastreando Sheer Heart Attack', fontsize=16)
    plt.xlabel('Posição X (m)', fontsize=12)
    plt.ylabel('Posição Y (m)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def animar_resultados(trajetoria_alvo, trajetoria_real_sha, estimativas_fke, medicoes, dt):
    """
    gera uma animação de toda a simulação.
    """
    passos_totais = trajetoria_alvo.shape[0]

    # configura a figura e os eixos
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # define os limites do gráfico com base em todas as trajetórias
    min_x = min(trajetoria_alvo[:, 0].min(), trajetoria_real_sha[:, 0].min()) - 20
    max_x = max(trajetoria_alvo[:, 0].max(), trajetoria_real_sha[:, 0].max()) + 20
    min_y = min(trajetoria_alvo[:, 1].min(), trajetoria_real_sha[:, 1].min()) - 20
    max_y = max(trajetoria_alvo[:, 1].max(), trajetoria_real_sha[:, 1].max()) + 20
    ax.axis('equal')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_title('Sheer Heart Attack perseguindo Jotaro', fontsize=16)
    ax.set_xlabel('Posição X (m)')
    ax.set_ylabel('Posição Y (m)')
    ax.grid(True)
    
    # pontos de início
    ax.plot(trajetoria_alvo[0, 0], trajetoria_alvo[0, 1], 'go', markersize=10, label='Jotaro')
    ax.plot(trajetoria_real_sha[0, 0], trajetoria_real_sha[0, 1], 'bo', markersize=10, label='Sheer Heart Attack')

    # rastros, que são as linhas que serão desenhadas
    rastro_alvo, = ax.plot([], [], 'g-', label='Trajetória do Jotaro', linewidth=2)
    rastro_sha, = ax.plot([], [], 'b-', label='Trajetória do Sheer Heart Attack', linewidth=2)
    rastro_ekf, = ax.plot([], [], 'k--', label='Estimativa FKE', linewidth=2.5)
    
    # pontos atuais, que são os marcadores que se movem
    ponto_alvo, = ax.plot([], [], 'go', markersize=12)
    ponto_sha, = ax.plot([], [], 'bo', markersize=12)
    ponto_ekf, = ax.plot([], [], 'kx', markersize=10, markeredgewidth=3)

    medicoes_pontos, = ax.plot([], [], 'r.', alpha=0.3, markersize=3, label='Medições ruidosas do radar')

    # texto do tempo
    texto_tempo = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    ax.legend(loc='upper right')

    def init():
        """função de inicialização para a animação."""
        rastro_alvo.set_data([], [])
        rastro_sha.set_data([], [])
        rastro_ekf.set_data([], [])
        ponto_alvo.set_data([], [])
        ponto_sha.set_data([], [])
        ponto_ekf.set_data([], [])
        medicoes_pontos.set_data([], [])
        texto_tempo.set_text('')
        return (rastro_alvo, rastro_sha, rastro_ekf, ponto_alvo, ponto_sha, 
                ponto_ekf, medicoes_pontos, texto_tempo)

    def animate(k):
        """função de animação, executada em cada quadro 'k'."""
        # desenha os rastros até o quadro k
        rastro_alvo.set_data(trajetoria_alvo[:k, 0], trajetoria_alvo[:k, 1])
        rastro_sha.set_data(trajetoria_real_sha[:k, 0], trajetoria_real_sha[:k, 1])
        rastro_ekf.set_data(estimativas_fke[:k, 0], estimativas_fke[:k, 1])

        # move os pontos para a posição k
        ponto_alvo.set_data([trajetoria_alvo[k, 0]], [trajetoria_alvo[k, 1]])
        ponto_sha.set_data([trajetoria_real_sha[k, 0]], [trajetoria_real_sha[k, 1]])
        ponto_ekf.set_data([estimativas_fke[k, 0]], [estimativas_fke[k, 1]])
        
        # acumula as medições ruidosas
        medicoes_pontos.set_data(medicoes[:k, 0], medicoes[:k, 1])

        # atualiza o texto do tempo
        texto_tempo.set_text(f'Tempo: {k * dt:.1f} s')
        
        return (rastro_alvo, rastro_sha, rastro_ekf, ponto_alvo, ponto_sha, 
                ponto_ekf, medicoes_pontos, texto_tempo)

    # cria e executa a animação
    ani = animation.FuncAnimation(fig, animate, frames=passos_totais,
                                  init_func=init, blit=True, interval=100, repeat=False)
    
    plt.show()  # mostra a animação


def main():
    
    # parâmetros da simulação
    DT = 0.5            # passo de tempo em segundos
    TEMPO_TOTAL = 100.0   # tempo total da simulação em segundos
    PASSOS_TOTAIS = int(TEMPO_TOTAL / DT)
    vetor_tempo = np.arange(0, TEMPO_TOTAL, DT)

    # definição das matrizes de ruído, Processo Q e Medição R
    
    # Q: ruído do processo, reflete a incerteza do modelo
    # reflete a incerteza em nosso modelo de movimento, a função f.
    # assumimos que a aceleração real pode variar um pouco.
    desvio_padrao_processo = 0.5  # m/s^2
    var_processo = desvio_padrao_processo**2
    
    # modelo de ruído de aceleração constante, ou Discrete White Noise Acceleration
    Q_base = np.array([
        [0.25*(DT**4), 0, 0.5*(DT**3), 0],
        [0, 0.25*(DT**4), 0, 0.5*(DT**3)],
        [0.5*(DT**3), 0, DT**2, 0],
        [0, 0.5*(DT**3), 0, DT**2]
    ])
    Q_ruido_processo = Q_base * var_processo

    # R: ruído da medição, reflete a incerteza do sensor
    # reflete a precisão do radar.
    desvio_padrao_medicao = 15.0  # m
    R_ruido_medicao = np.diag([desvio_padrao_medicao**2, desvio_padrao_medicao**2])

    # H: matriz de observação, que é linear
    # mapeia o vetor de estado [x, y, vx, vy] para o espaço de medição [x, y]
    H_matriz_observacao = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # condições iniciais, do Estado Real e da Estimativa do Filtro
    
    # estado inicial real, a verdade absoluta, do rastreador
    estado_inicial_real = np.array([-200.0, 100.0, 5.0, -2.0])  # [x, y, vx, vy]
    
    # estimativa inicial, ou chute inicial, para o filtro
    # simulamos um erro na estimativa inicial de posição e velocidade
    estado_inicial_estimado = np.array([-180.0, 80.0, 0.0, 0.0])
    
    # P: covariância inicial, a incerteza da estimativa inicial
    # começamos com alta incerteza sobre o estado inicial
    P_incerteza_inicial = np.diag([500.0, 500.0, 100.0, 100.0])
    
    
    # geração dos dados da simulação
    
    # trajetória do Jotaro
    trajetoria_alvo_sim = obter_trajetoria_alvo_jotaro(PASSOS_TOTAIS, DT)
    
    # trajetória real do rastreador do sheer heart attack
    trajetoria_real_sim = simular_trajetoria_real_sha(trajetoria_alvo_sim, estado_inicial_real, DT)
    
    # medições ruidosas do radar
    medicoes_simuladas = simular_medicoes_radar(trajetoria_real_sim, R_ruido_medicao)
    
    
    # execução do Filtro de Kalman Estendido
    
    estimativas_finais = executar_filtro_fke(
        medicoes_simuladas,
        trajetoria_alvo_sim,
        estado_inicial_estimado,
        P_incerteza_inicial,
        Q_ruido_processo,
        R_ruido_medicao,
        H_matriz_observacao,
        DT
    )
    
    
    # visualização dos resultados em animação

    animar_resultados(
        trajetoria_alvo_sim,
        trajetoria_real_sim,
        estimativas_finais,
        medicoes_simuladas,
        DT  # passa o passo de tempo para a exibição
    )


if __name__ == "__main__":
    main()