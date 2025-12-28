import random 
import math 
# a biblioteca random biblioteca é usada para gerar números aleatórios essenciais ao funcionamento do algoritmo genético (ag), como mutação e seleção.
# já a biblioteca math é usada para cálculos matemáticos.


# CONSTANTES DO PROBLEMA E PARÂMETROS DO AG
NUM_REGRAS = 5             # o dna de cada indivíduo (árvore) é um conjunto de 5 parâmetros numéricos que definem seu crescimento.
TAMANHO_POPULACAO = 100    # define o número total de indivíduos (conjuntos de regras) que competirão e evoluirão em cada geração.
TAXA_MUTACAO = 0.08        # probabilidade (8%) de que uma das regras (gene) de um indivíduo sofra uma pequena alteração aleatória.
NUM_GERACOES = 300         # o algoritmo será executado por 300 ciclos de evolução antes de parar.
TAMANHO_TORNEIO = 5        # em cada seleção, 5 indivíduos são escolhidos aleatoriamente, e o melhor entre eles é selecionado para reprodução.

#1. FUNÇÕES DE SIMULAÇÃO DE CRESCIMENTO E CÁLCULO DE FITNESS


def simular_crescimento_arvore(regras, profundidade_max=5):
    """
    simula de forma simplificada o crescimento da árvore usando um modelo baseado em l-systems (sistemas-l).
    o resultado desta simulação (complexidade e altura) é usado para calcular o fitness (beleza).
    """
    
    # o dna da árvore é mapeado para as variáveis que governam o crescimento:
    # regras: (angulo_base, taxa_encurtamento, prob_ramificacao, fator_gravidade, penalidade_angulo)
    angulo_base = regras[0]  # regra 1: influencia o quão "abertos" são os ramos.
    taxa_encurtamento = regras[1] # regra 2: define o quanto o comprimento de um novo galho diminui em relação ao anterior (ex: 0.8 encurta 20%).
    prob_ramificacao = max(0.1, min(0.9, regras[2])) # regra 3: probabilidade de um galho se dividir em dois novos galhos.
    
    # simulação recursiva: a função crescer é o bloco de construção de um ramo.
    def crescer(comprimento, profundidade):
        # condição de parada: se a profundidade for 0, o ramo para de crescer.
        if profundidade == 0:
            return 1, 0 # retorna complexidade 1 (o próprio ramo final) e altura 0.
        
        # inicialização: o ramo atual já contribui para a complexidade e altura.
        complexidade = 1 
        altura = comprimento # a altura inicial é o comprimento deste tronco/ramo.
        
        # regra de ramificação: a árvore se divide com uma probabilidade definida pela regra 3.
        if random.random() < prob_ramificacao:
            
            novo_comprimento = comprimento * taxa_encurtamento
            
            # 1. ramo esquerdo: chamada recursiva para o novo galho.
            c1, h1 = crescer(novo_comprimento, profundidade - 1)
            complexidade += c1 # soma a complexidade dos sub-ramos.
            
            # 2. ramo direito: chamada recursiva para o segundo novo galho.
            c2, h2 = crescer(novo_comprimento, profundidade - 1)
            complexidade += c2 # soma a complexidade do segundo sub-ramo.
            
            # a altura total é o comprimento do ramo atual + a altura do ramo mais alto que nasceu dele.
            altura += max(h1, h2)
        
        return complexidade, altura # retorna a complexidade (número total de segmentos) e a altura total.

    # inicia o crescimento a partir da base, que foi definido como, inicialmente, 10
    complexidade_final, altura_final = crescer(comprimento=10.0, profundidade=profundidade_max)
    
    return complexidade_final, altura_final

def calcular_beleza_arvore(regras):
    """
    função fitness: avalia a qualidade (beleza) da árvore gerada pelas 'regras'. 
    o objetivo é maximizar este valor, que combina complexidade, altura e penalidades.
    """
    
    fator_gravidade = regras[3]    # regra 4: fator que modula a contribuição da altura.
    penalidade_angulo = regras[4]  # regra 5: penaliza ângulos muito pequenos ou muito grandes.
    
    # a simulação é repetida 5 vezes e a média é tirada para reduzir o efeito da aleatoriedade (prob_ramificacao).
    resultados = [simular_crescimento_arvore(regras) for _ in range(5)]
    complexidade_media = sum(c for c, h in resultados) / 5
    altura_media = sum(h for c, h in resultados) / 5
    
    # heurística: busca árvores complexas (muitos galhos) e com boa altura, mas evita ser excessivamente alta.
    # penalidade_altura: evita árvores muito altas (espinhas).
    penalidade_altura = max(0, altura_media - 100) * 0.5 
    
    # penalidade_angulo_calc: utiliza a regra 5 para penalizar ângulos extremos.
    # max(0, ...): garante que o valor seja sempre não-negativo.
    penalidade_angulo_calc = penalidade_angulo * max(0, 0.5 - abs(regras[0] - 0.5)) * 10 
    
    # fitness final: maximiza complexidade e altura e subtrai as penalidades.
    # math.sqrt(altura_media) * fator_gravidade: usa a raiz quadrada para dar mais valor ao início do crescimento em altura.
    fitness = complexidade_media + (math.sqrt(altura_media) * fator_gravidade) - penalidade_altura - penalidade_angulo_calc
    
    return fitness


# 2. OPERAÇÕES GENÉTICAS

def gerar_individuo_inicial():
    """
    cria um novo indivíduo (conjunto de 5 regras) com valores aleatórios dentro de faixas razoáveis.
    este é o ponto de partida para a evolução.
    """
    # regras iniciais (dna): (angulo, taxa_encurtamento, prob_ramificacao, fator_gravidade, penalidade_angulo)
    regras_iniciais = [
        random.uniform(0.1, 1.0),    # regra 1: ângulo base (faixa razoável)
        random.uniform(0.6, 0.9),    # regra 2: taxa de encurtamento (deve ser menor que 1.0 para que a árvore cresça).
        random.uniform(0.3, 0.7),    # regra 3: probabilidade de ramificação.
        random.uniform(0.5, 1.5),    # regra 4: fator gravidade.
        random.uniform(0.0, 1.0)     # regra 5: penalidade de ângulo.
    ]
    
    return tuple(regras_iniciais)


def selecao_torneio(populacao):
    """
    método de seleção: escolhe aleatoriamente 'tamanho_torneio' indivíduos e seleciona o que tiver o melhor fitness (o mais "bonito").
    """
    
    # escolhe aleatoriamente um subconjunto de indivíduos para competir.
    competidores = random.sample(populacao, TAMANHO_TORNEIO)
    
    melhor_competidor = None
    melhor_fitness = -float('inf') # inicializa com o menor valor possível.
    
    # compara o fitness de cada competidor e seleciona o vencedor.
    for individuo in competidores:
        fitness_atual = calcular_beleza_arvore(individuo)
        if fitness_atual> melhor_fitness:
            melhor_fitness = fitness_atual
            melhor_competidor = individuo
            
    return melhor_competidor


def crossover_ponto_unico(pai1, pai2):
    """
    recombinação (crossover): gera dois filhos combinando as regras (dna) dos pais em um único ponto de corte aleatório.
    """
    
    # escolhe um ponto de corte aleatório.
    ponto_corte = random.randrange(1, NUM_REGRAS - 1) 
    
    # o filho 1 recebe a primeira parte do pai 1 e a segunda parte do pai 2.
    filho1_regras = list(pai1[:ponto_corte]) + list(pai2[ponto_corte:])
    # o filho 2 recebe a primeira parte do pai 2 e a segunda parte do pai 1.
    filho2_regras = list(pai2[:ponto_corte]) + list(pai1[ponto_corte:])
    
    return tuple(filho1_regras), tuple(filho2_regras)


def mutacao(individuo):
    """
    Mutação é uma definição interessante que introduz diversidade na população, alterando aleatoriamente uma dos genes em pequena magnitude.
    """
    
    regras_mutaveis = list(individuo)
    
    # chance de mutação, onde se verifica se a mutação deve ocorrer com base na variável global de taxa mutação.
    if random.random() < TAXA_MUTACAO: 
        
        # escolhe uma regra para ser ajustada.
        idx_mutacao = random.randrange(NUM_REGRAS)
        
        # determina a magnitude do ajuste (+/- 0.1), que seria o erro genético.
        ajuste = random.uniform(-0.1, 0.1) 
        
        # aplica o ajuste à regra.
        nova_regra = regras_mutaveis[idx_mutacao] + ajuste
        
        # garante que os valores das regras permaneçam em um intervalo que faz sentido físico.
        nova_regra = max(0.01, min(2.0, nova_regra)) 
        
        regras_mutaveis[idx_mutacao] = nova_regra

        return tuple(regras_mutaveis)
        
    # se não houver mutação, retorna o indivíduo original.
    return individuo 


def executar_algoritmo_genetico():
    """
     orquestra o ciclo de vida do algoritmo genético através das gerações.
    """
    
    # inicialização: cria a primeira geração de indivíduos aleatórios.
    populacao = [gerar_individuo_inicial() for _ in range(TAMANHO_POPULACAO)]
    
    # rastreamento da melhor solução encontrada em todas as gerações.
    melhor_solucao_global = populacao[0]
    melhor_fitness_global = calcular_beleza_arvore(populacao[0])
    
    # loop principal: iteração através do número definido de gerações.
    for geracao in range(NUM_GERACOES):
        
        # avaliação e ordenação: calcula o fitness de todos e ordena do melhor para o pior.
        populacao_avaliada = sorted(
            populacao, 
            key=lambda ind: calcular_beleza_arvore(ind), 

            reverse=True # true: do maior fitness para o menor.
        )
        
        # elitismo (tracking): o melhor da geração atual é o primeiro da lista ordenada.
        melhor_da_geracao = populacao_avaliada[0]
        fitness_da_geracao = calcular_beleza_arvore(melhor_da_geracao)
        
        # atualização da melhor solução geral.
        if fitness_da_geracao >melhor_fitness_global:
            melhor_fitness_global = fitness_da_geracao
            melhor_solucao_global= melhor_da_geracao
            
        # criação da nova população: usa o elitismo, o melhor indivíduo passa para a próxima geração sem modificação.
        nova_populacao = [populacao_avaliada[0]] 
        
        # preenchimento da nova população até atingir o tamanho definido.
        while len(nova_populacao) < TAMANHO_POPULACAO:
            
            # seleção: escolhe dois pais usando o torneio.
            pai1 = selecao_torneio(populacao_avaliada)
            pai2 = selecao_torneio(populacao_avaliada)
            
            # recombinação: gera dois filhos misturando os genes dos pais.
            filho1, filho2 = crossover_ponto_unico(pai1, pai2)
            
            # mutação: introduz pequenas mudanças genéticas nos filhos.
            filho1 = mutacao(filho1)
            filho2 = mutacao(filho2)
            
            # adiciona os novos indivíduos à próxima geração.
            nova_populacao.append(filho1)
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(filho2)
        
        # a nova população se torna a população atual para o próximo ciclo de evolução.
        populacao = nova_populacao
        
        # feedback do progresso a cada 30 gerações.
        if geracao % 30 == 0: 
            print(f"geração {geracao:3d}: melhor fitness = {melhor_fitness_global:.4f}")
            
    # retorna a melhor combinação de regras (dna) encontrada.
    return melhor_solucao_global, melhor_fitness_global

# execução

print("iniciando evolução de regras de crescimento de árvore (algoritmo genético)...")
print(f"o AG buscará as regras que maximizam complexidade e altura. \n")

melhor_regras, melhor_fitness = executar_algoritmo_genetico()

print("\n--- resultado final da evolução ---")
print(f"melhor fitness (beleza/complexidade): {melhor_fitness:.4f}")
print(f"regras ideais (DNA): {[round(r, 4) for r in melhor_regras]}")