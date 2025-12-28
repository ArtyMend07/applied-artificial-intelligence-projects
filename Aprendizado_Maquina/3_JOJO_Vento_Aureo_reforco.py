import torch # núcleo de processamento tensorial essencial para construir as redes e calcular gradientes automaticamente
import torch.nn as nn # traz os blocos de construção de redes neurais como camadas lineares e ativações
import torch.nn.functional as F # funções puras para ativação e cálculos de erro sem manter estado
import torch.optim as optim # contém os otimizadores como o Adam que ajustam os pesos da rede
from torch.distributions import Normal # distribuição normal fundamental para a política estocástica do SAC
import numpy as np # manipulação vetorial para a física do ambiente
import matplotlib.pyplot as plt # plotagem de gráficos
import seaborn as sns # deixa os gráficos mais legíveis
import random # gerenciamento de aleatoriedade para o buffer de memória


""" 
Tema e Contextualização:

JoJo Bizarre Adventure é uma saga famosa que narra batalhas sobrenaturais,
e este projeto foca especificamente na quinta parte da história, conhecida
como Vento Aureo. A trama é ambientada na Itália e segue a jornada de
mafiosos tentando reformar o crime organizado.

Nesse universo, os personagens utilizam poderes chamados Stands. Eles são
a manifestação física da energia vital e da alma do usuário, agindo como
entidades guardiãs com habilidades especiais únicas para cada pessoa.

O sistema modela o comportamento do Stand Sex Pistols (referência à uma banda britânica de punk rock), pertencente a Guido
Mista. Ele é um atirador de elite que usa pequenas criaturas espirituais
para chutar projéteis de revólver em pleno ar, alterando sua trajetória
para atingir alvos difíceis. O cenário consiste em uma simulação física
onde a bala precisa ajustar seu ângulo de voo continuamente e com precisão
milimétrica, para interceptar um alvo.

Para mais informações sobre o universo, recomenda-se as leituras dos seguintes links:

JoJo's Bizarre Adventure Vento Aureo: https://jjba.fandom.com/pt-br/wiki/Vento_Aureo

Sex Pistols (stand do mista): https://jjba.fandom.com/pt-br/wiki/Sex_Pistols


Justificativa da escolha do algoritmo:

É fundamental compreender que o DQN funciona escolhendo entre opções
discretas, como pular, correr ou atirar. No entanto, o problema do Sex
Pistols exige uma precisão decimal no ângulo da bala, configurando um
espaço de ação contínuo. Se utilizássemos o DQN, seria necessário dividir
os 360 graus em milhares de pequenos pedaços, o que tornaria o processo
computacionalmente inviável e ineficiente.

Por isso, o algoritmo escolhido foi o SAC, ou Soft Actor-Critic. Ele é uma
abordagem moderna do tipo Actor-Critic, que une as vantagens de dois mundos.
Primeiro, ele herda a capacidade do Q-Learning de estimar o valor de uma
ação através de uma rede Crítica, que julga se o movimento foi bom ou ruim.
Segundo, ele utiliza a filosofia do Gradiente de Política através de uma
rede Atora, que aprende a gerar ações contínuas diretamente, sem precisar
de uma tabela de opções fixas.

O grande diferencial do SAC é a maximização da entropia. Enquanto algoritmos
tradicionais buscam apenas a maior pontuação, o SAC busca a maior pontuação
mantendo a maior aleatoriedade possível. Isso cria um agente criativo que
explora trajetórias variadas e refina sua precisão milimetricamente, evitando
ficar viciado em um único caminho óbvio que poderia ser bloqueado por um
obstáculo.

Fontes:

HAARNOJA, Tuomas et al. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. International Conference on Machine Learning (ICML), 2018: https://arxiv.org/abs/1801.01290

SUTTON, Richard S.; BARTO, Andrew G. Reinforcement Learning: An Introduction. 2nd ed. Cambridge: MIT Press, 2018: http://incompleteideas.net/book/the-book-2nd.html

BISHOP, Christopher M. Pattern Recognition and Machine Learning. New York: Springer, 2006.

KINGMA, Diederik P.; BA, Jimmy. Adam: A Method for Stochastic Optimization. International Conference on Learning Representations (ICLR), 2015.
"""

torch.manual_seed(37)
np.random.seed(37)
random.seed(37)

# classe que simula a física da bala e do alvo
class AmbienteSexPistols:
    def __init__(self):
        # o estado observa 4 valores sendo posição X e Y da bala e posição X e Y do alvo
        self.dim_estado = 4 
        # a ação é apenas 1 valor que representa o ângulo de ajuste em radianos
        self.dim_acao = 1 
        self.maximos_passos = 80 # limite estendido para permitir alcance de alvos distantes

    def resetar(self):
        # a bala sempre sai da arma na origem 0 0
        self.pos_bala = np.array([0.0, 0.0])
        # o alvo aparece aleatoriamente em uma área distante
        self.pos_alvo = np.random.uniform(2.0, 10.0, size=2)
        # a bala começa apontando vagamente para o alvo com erro para forçar correção
        # o que simula justamente a habilidade do stand de Mista
        vetor_alvo = self.pos_alvo - self.pos_bala
        # foi usado a função arctan2 ao invés de arctan, pois a propriedade matemática faz você perder informação 
        # Para demonstrar isso, a função arctan recebe o resultado da fórmula do arcotangente
        # sendo ela: y/x
        # o problema é que só com arcotangente padrão, numa situação onde y e x são ambos -1,
        # a resposta será: (-1)/(-1) = 1, onde arctan(1) = 45 graus, que está totalmente errado para nosso problema,
        # uma vez que isso indica que a bala está indo pro nordeste, sendo que era para ir pro caminho contrario (-135°)
        # ou seja, essa função não é boa para usar com trajetórias que podem ir para trás, que seria o lado esquerdo do círculo, justamente por calcular a divisão primeiro
        # Aí a função arctan2 vê a divisão e os sinais separadamente
        # onde 1 e -1 pra y é cima e baixo respectivamente, e pra x é direita e esquerda (respesctivamente também)
        # por exemplo, a função vê arctang2(-1,-1) = -135°
        angulo_inicial = np.arctan2(vetor_alvo[1], vetor_alvo[0]) 
        erro_mira = np.random.uniform(-0.3, 0.3) # erro de mira do mista
        self.angulo_atual = angulo_inicial + erro_mira
        self.passos_atuais = 0
        return self._obter_observacao()

    def _obter_observacao(self):
        # junta as coordenadas da bala e do alvo num vetor só
        # nesse caso, a direção horizontal é o cosseno do ângulo
        # e a direção vertical é o seno, ambas importantes pra noção de inércia
        vetor_do_alvo = self.pos_alvo - self.pos_bala
        direcao_x = np.cos(self.angulo_atual)
        direcao_y = np.sin(self.angulo_atual)
        return np.concatenate([vetor_do_alvo, [direcao_x, direcao_y]])

    def passo(self, acao):
        # rede neural entrega valor entre -1 e 1 que é convertido para ângulo real
        intensidade_do_chute = np.clip(acao[0], -1.0, 1.0) # é a simulação do chute que o stand do mista dá na bala, onde 0 significa a ausência do chute (inércia)
        ajuste = intensidade_do_chute * (np.pi / 8)  # π / 8 é o efeito do chute instantâneo do stand de 22,5 graus na direção
        self.angulo_atual+= ajuste# soma o ajuste com o ângulo já existente pra dar ideia do chute na bala que o stand faz para mudar a trajetória
        
        # física vetorial com movimentação de 0.6 unidade na direção do ângulo escolhido
        # velocidade reduzida auxilia na precisão do cálculo de impacto
        movimento = np.array([np.cos(self.angulo_atual), np.sin(self.angulo_atual)])
        self.pos_bala += movimento * 0.6
        
        # vetor ideal que aponta diretamente pro alvo
        vetor_ideal = self.pos_alvo-self.pos_bala
        # cálculo da distância euclidiana para mensurar desempenho do agente
        distancia = np.linalg.norm(self.pos_bala - self.pos_alvo)

        # normaliza o vetor ideal (transforma em tamanho 1 para comparar direção)
        if distancia > 0:
            vetor_ideal_norm = vetor_ideal / distancia
        else:
            vetor_ideal_norm = vetor_ideal
        
        alinhamento = np.dot(movimento, vetor_ideal_norm)
        
        # recompensa que foca na direção da bala, ao invés de somente da distância
        recompensa = alinhamento * 2.0
        # penalidade por demorar, é pra acertar o alvo o mais rápido possível
        recompensa -= 0.1
        # penalidade por estar longe
        recompensa -= distancia * 0.1
        
        
        concluido = False
        
        # considera acerto se acertar a hitbox do alvo
        if (distancia < 0.4): 
            recompensa +=100 # bônus alto pelo sucesso
            concluido = True
        elif (self.passos_atuais>= self.maximos_passos):
            concluido = True # acabou a energia
            # penalidade extra por acabar a energia longe do alvo força eficiência
            recompensa -= 10
            
        self.passos_atuais += 1
        return self._obter_observacao(), recompensa, concluido

# sistema de memória para armazenar o que aconteceu Experience Replay
# quebra a correlação temporal dos dados permitindo aprendizado com situações variadas
class MemoriaDeExperiencia:
    def __init__(self, capacidade):
        self.capacidade = capacidade
        self.buffer = []
        self.posicao = 0

    def guardar(self, estado, acao, recompensa, proximo_estado, feito):
        if len(self.buffer) < self.capacidade:
            self.buffer.append(None)
        self.buffer[self.posicao] = (estado, acao, recompensa, proximo_estado, feito)
        self.posicao = (self.posicao + 1) % self.capacidade

    def amostrar(self, tamanho_lote):
        lote = random.sample(self.buffer, tamanho_lote)
        # desempacota os dados e organiza em arrays numpy
        estado, acao, recompensa, proximo_estado, feito = map(np.stack, zip(*lote))
        return estado, acao, recompensa, proximo_estado, feito

    def __len__(self):
        return len(self.buffer)

# rede Crítica ou instinto do Mista
# papel de julgar se uma ação foi boa naquele estado
# arquitetura com duas redes idênticas Q1 e Q2 em paralelo
# mitiga viés de superestimação ao considerar a avaliação mais pessimista
class RedeCritica(nn.Module):
    def __init__(self, dim_estado, dim_acao):
        super().__init__()
        
        # estrutura da primeira rede de julgamento Q1
        self.l1 = nn.Linear(dim_estado + dim_acao, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1) # saída única nota da ação

        # estrutura da segunda rede de julgamento Q2
        self.l4 = nn.Linear(dim_estado + dim_acao, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, estado, acao):
        # concatena estado e ação pois o crítico avalia o par situação e atitude
        xu = torch.cat([estado, acao], 1)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

# rede Ator ou Stand 
# o objetivo é dar parâmetros de uma curva gaussiana média e desvio padrão
# permite criatividade onde desvio alto tenta coisas novas e baixo foca na precisão
class RedeAtor(nn.Module):
    def __init__(self, dim_estado, dim_acao):
        super().__init__()
        self.l1 = nn.Linear(dim_estado, 256)
        self.l2 = nn.Linear(256, 256)
        
        self.camada_media = nn.Linear(256, dim_acao)
        self.camada_log_std = nn.Linear(256, dim_acao)

    def forward(self, estado):
        x = F.relu(self.l1(estado))
        x = F.relu(self.l2(x))
        media = self.camada_media(x)
        log_std = self.camada_log_std(x)
        
        # fixa o desvio padrão num intervalo seguro para estabilidade matemática
        log_std = torch.clamp(log_std, min=-20, max=2)
        return media, log_std

    def amostrar(self, estado):
        media, log_std = self.forward(estado)
        std = log_std.exp()
        normal = Normal(media, std)
        
        # aplica o truque de reparametrização rsample
        # permite que o erro da rede flua através do sorteio aleatório 
        # rede aprende a ajustar sua incerteza no futuro
        x_t = normal.rsample()
        
        # tangente hiperbólica tanh comprime resultado entre -1 e 1
        y_t = torch.tanh(x_t) 
        acao = y_t
        
        # cálculo de ajuste da probabilidade devido à transformação não linear
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - acao.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        media = torch.tanh(media)
        return acao, log_prob, media

# instanciamento do ambiente e memória
ambiente = AmbienteSexPistols()
dim_estado = ambiente.dim_estado
dim_acao = ambiente.dim_acao
memoria = MemoriaDeExperiencia(100000)

# instanciando os modelos
# esses modelos representam a política (que é o ator) e a função de valor (critico) do agente
ator = RedeAtor(dim_estado, dim_acao)
critico = RedeCritica(dim_estado, dim_acao)
critico_alvo = RedeCritica(dim_estado, dim_acao) # cópia estável para calcular metas
critico_alvo.load_state_dict(critico.state_dict())

# otimizadores Adam de referência para ajuste de pesos
otimizador_ator = optim.Adam(ator.parameters(), lr=3e-4)
otimizador_critico = optim.Adam(critico.parameters(), lr=3e-4)

# hiperparâmetros
gamma = 0.90 # fator de desconto valoriza visão de futuro
tau = 0.005 # taxa de atualização lenta da rede alvo para estabilidade
alpha = 0.05 # peso da entropia define nível de aventura do agente
tamanho_lote = 256 # lotes para a ia aprender com exemplos simultâneos

# função que executa um passo de aprendizado nos pesos neurais
def atualizar_parametros():
    if len(memoria) < tamanho_lote:
        return 0, 0

    # pega um punhado de memórias aleatórias
    estado, acao, recompensa, proximo_estado, feito = memoria.amostrar(tamanho_lote)
    
    # converte tudo para tensores do PyTorch
    estado = torch.FloatTensor(estado)
    acao = torch.FloatTensor(acao)
    recompensa = torch.FloatTensor(recompensa).unsqueeze(1)
    proximo_estado = torch.FloatTensor(proximo_estado)
    feito = torch.FloatTensor(feito).unsqueeze(1)

    # 1 atualização do Crítico Instinto
    with torch.no_grad():
        # o ator imagina o que faria no próximo estado
        proxima_acao, proxima_log_prob, _ = ator.amostrar(proximo_estado)
        
        # as redes alvo avaliam essa ação imaginada
        alvo_Q1, alvo_Q2 = critico_alvo(proximo_estado, proxima_acao)
        
        # pega a menor avaliação pessimismo e subtrai a entropia recompensa por explorar
        alvo_Q = torch.min(alvo_Q1, alvo_Q2) - alpha * proxima_log_prob
        
        # equação de Bellman calcula Valor Real somando Recompensa Imediata e Valor Futuro
        q_esperado = recompensa + (1 - feito) * gamma * alvo_Q
    
    atual_Q1, atual_Q2 = critico(estado, acao)
    perda_critica = F.mse_loss(atual_Q1, q_esperado) + F.mse_loss(atual_Q2, q_esperado)
    
    otimizador_critico.zero_grad()
    perda_critica.backward()
    otimizador_critico.step()
    
    # 2 atualização do Ator Sex Pistols
    # ator quer maximizar a nota do crítico mantendo a entropia alta
    # fonte: https://arxiv.org/abs/1801.01290
    acao_nova, log_prob, _ = ator.amostrar(estado)
    Q1_novo, Q2_novo = critico(estado, acao_nova)
    Q_novo = torch.min(Q1_novo, Q2_novo)
    
    # sinal negativo pois otimizadores minimizam valor e objetivo é aumentar
    perda_ator = (alpha * log_prob - Q_novo).mean()
    
    otimizador_ator.zero_grad()
    perda_ator.backward()
    otimizador_ator.step()
    
    # atualização suave da rede alvo Polyak averaging
    for param, param_alvo in zip(critico.parameters(), critico_alvo.parameters()):
        param_alvo.data.copy_(tau * param.data + (1 - tau) * param_alvo.data)
        
    return perda_critica.item(), perda_ator.item()

# loop principal de simulação
# gerações configuradas para garantir convergência em alvos distantes
geracoes = 800 
historico_recompensas = []

for gen in range(geracoes):
    estado = ambiente.resetar()
    recompensa_episodio = 0
    feito = False
    
    while not feito:
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        
        # o ator decide a ação baseada no estado atual
        acao, _, _ = ator.amostrar(estado_tensor)
        acao = acao.detach().numpy()[0]
        
        # o ambiente reage
        proximo_estado, recompensa, feito = ambiente.passo(acao)
        
        # guarda na memória
        memoria.guardar(estado, acao, recompensa, proximo_estado, feito)
        
        estado = proximo_estado
        recompensa_episodio += recompensa
        
        # aprende um pouquinho a cada passo
        atualizar_parametros()
        
    historico_recompensas.append(recompensa_episodio)
    
    if gen % 50 == 0:
        print(f"Geração {gen} Performance {recompensa_episodio:.2f}")

def realizar_teste_tiro(nome_cenario, x_alvo, y_alvo):
    # força posições fixas pra teste determinístico
    ambiente.pos_bala = np.array([0.0, 0.0])
    ambiente.pos_alvo = np.array([float(x_alvo), float(y_alvo)])
    vetor = ambiente.pos_alvo - ambiente.pos_bala # aponta a bala pro alvo no início do teste
    ambiente.angulo_atual=np.arctan2(vetor[1], vetor[0]) # usado arctan2 pelo mesmo motivo discutido na função resetar
    ambiente.passos_atuais = 0
    
    estado = ambiente._obter_observacao()
    feito = False
    trajetoria_x, trajetoria_y = [0.0], [0.0]
    
    print(f"Testando Cenario {nome_cenario}")
    
    while not feito:
        estado_tensor = torch.FloatTensor(estado).unsqueeze(0)
        # no teste utiliza-se a média da gaussiana para tiro preciso sem aleatoriedade
        _, _, acao_media = ator.amostrar(estado_tensor)
        acao = acao_media.detach().numpy()[0]
        
        estado, _, feito = ambiente.passo(acao)
        trajetoria_x.append(ambiente.pos_bala[0])
        trajetoria_y.append(ambiente.pos_bala[1])
        
        # verificação extra para parar de desenhar assim que acertar
        dist_atual = np.linalg.norm(ambiente.pos_bala - ambiente.pos_alvo)
        if dist_atual < 0.4:
            break
    
    distancia_final = np.linalg.norm(ambiente.pos_bala - ambiente.pos_alvo)
    if (distancia_final < 0.4):
        status = "acertou"
    else:
        status = "errou"
    print(f"Resultado {status} Distância {distancia_final:.2f}")
    print("-" * 30)
    return trajetoria_x, trajetoria_y

print("-" * 30)
# cenários baseados nos vilões da Parte 5
t1 = realizar_teste_tiro("Kraft Work Perto e Fixo", 3.0, 3.0)
t2 = realizar_teste_tiro("White Album Longe", 8.0, -2.0)
t3 = realizar_teste_tiro("Green Day Diagonal Alta", 5.0, 5.0)

# gráfico de evolução do aprendizado
plt.figure(figsize=(10, 5))
sns.lineplot(data=historico_recompensas)
plt.title("Evolução da Precisão Recompensa por Geração")
plt.xlabel("Gerações")
plt.ylabel("Pontuação")
plt.grid(True, alpha=0.3)
plt.show()

# gráfico das trajetórias finais realizadas
plt.figure(figsize=(8, 8))
plt.plot(t1[0], t1[1], label='Tiro Curto', marker='o', markersize=3)
plt.plot(t2[0], t2[1], label='Tiro Longo', marker='x', markersize=3)
plt.plot(t3[0], t3[1], label='Tiro Diagonal', marker='^', markersize=3)

# Raio da hitbox
for alvo in [(3,3), (8,-2), (5,5)]:
    circulo = plt.Circle(alvo, 0.4, color='r', fill=False, linestyle='--')
    plt.gca().add_patch(circulo)

# marcadores visuais
plt.scatter([3, 8, 5], [3, -2, 5], color='red', s=100, label='Alvos', zorder=5)
plt.scatter([0], [0], color='black', s=100, label='Mista', zorder=5)

plt.title("Resultado dos projeteis otimizados")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()