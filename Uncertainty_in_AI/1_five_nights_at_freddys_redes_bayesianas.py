"""
Contextualização do Problema no Universo de FNAF:

Em Five Nights at Freddy's o jogador assume o papel de um guarda noturno
que deve sobreviver até as 6 da manhã em uma pizzaria assombrada. A
mecânica central envolve o gerenciamento de recursos escassos de energia
enquanto se monitora a movimentação de robôs animatrônicos hostis através
de câmeras de segurança.
(Wiki sobre os animatronics : https://twinfinite.net/guides/five-nights-at-freddys-1-all-characters)

No jogo original a inteligência artificial dos inimigos segue padrões de
movimento específicos e pseudoaleatórios que podem ser tecnicamente
analisados e previstos pelo jogador experiente.
(Análise técnica da IA original: https://www.youtube.com/watch?v=3H1olkxEL9s)

Adaptação do Problema para Redes Bayesianas:

Este projeto modela o dilema de incerteza quando o jogador não está
olhando para as câmeras para economizar bateria. Nesta situação o estado
dos animatrônicos torna-se uma variável oculta.

A Rede Bayesiana foi construída para calcular a probabilidade de um
Jumpscare, que é a condição de derrota, com base apenas em evidências
auditivas passivas. Sons como passos na esquerda ou barulhos na cozinha
são tratados como nós de evidência que influenciam a crença sobre a
atividade dos animatrônicos e consequentemente o risco de ataque iminente.

O algoritmo de Eliminação de Variáveis é utilizado para processar essas
evidências e fornecer uma probabilidade exata de risco permitindo uma
tomada de decisão racional sob incerteza.
"""

# Cronologia da Execução do Programa
#
# O programa executa uma inferência em uma Rede Bayesiana estática e usa o algoritmo de Eliminação de Variáveis.
# A sequência de operações é a seguinte.
#
# 1. Definição da Rede ou Setup
#    Primeiro o código define todas as peças do problema.
#    DOMINIOS é um dicionário criado para estabelecer quais valores cada variável pode assumir por exemplo Hora pode ser Inicio Meio ou Fim.
#    Fatores ou CPTs são as tabelas de probabilidade e são definidas como um dicionário fator_....
#    Cada fator armazena as probabilidades de uma variável dados seus pais por exemplo fator_bonnie define P(BonnieAtivo | Hora).
#    Rede Completa é uma lista chamada rede_fnaf criada para agrupar todos os fatores definidos e assim representar a rede Bayesiana completa.
#
# 2. Definição da Consulta ou Setup no main
#    Dentro do bloco if __name__ == "__main__": a pergunta a ser respondida é configurada.
#    var_consulta define a variável que se deseja investigar por exemplo Jumpscare.
#    evidencias é um dicionário preenchido com os fatos já conhecidos por exemplo SomEsquerda é True.
#
# 3. Chamada do Algoritmo ou Execução no main
#    A função principal 'eliminacao_de_variaveis' é chamada e ela recebe a rede completa e a consulta e também as evidências.
#
# 4. Processamento do Algoritmo em 'eliminacao_de_variaveis'
#    Restrição ou Poda: O algoritmo primeiro aplica as evidências. A função 'restringir_fator' é usada em todos os fatores.
#    Ela "corta" e descarta todas as linhas das tabelas que contradizem as evidências e isso torna os fatores menores e os cálculos futuros mais rápidos.
#    Eliminação ou Loop Principal: O algoritmo entra em um loop e segue a 'ordem_otimizada' para eliminar cada variável oculta que não é a consulta nem a evidência.
#    Para cada variável a ser eliminada por exemplo FoxyAtivo o código faz o seguinte.
#    Primeiro a Multiplicação ou Join: A função 'multiplicar_fatores' encontra todos os fatores que contêm FoxyAtivo e os combina em um único fator maior.
#    Segundo a Soma ou Marginalização: A função 'somar_sobre_variavel' é usada nesse fator maior para "espremer" a variável FoxyAtivo e somar todas as suas contribuições.
#    O resultado é um novo fator menor que não contém mais FoxyAtivo mas que preserva sua informação probabilística.
#    Finalização: Após o loop todas as variáveis ocultas foram eliminadas. Sobram apenas fatores que contêm a variável de consulta Jumpscare e eles são todos multiplicados.
#    Normalização: O resultado da multiplicação final é um score. A função 'normalizar' é chamada para converter esses scores em probabilidades finais que somam 1.0.
#
# 5. Impressão ou Final no main
#    O resultado normalizado é recebido e formatado para ser exibido no console e assim mostrar a probabilidade de cada estado da variável de consulta.
#

import matplotlib.pyplot as plt #Biblioteca utilizada para gerar gráficos

# primeiro são definidos os valores possíveis para cada variável
DOMINIOS = {
    'Hora': ['Inicio', 'Meio', 'Fim'], # 1-2h e 3-4h e 5-6h
    'BonnieAtivo': [True, False],
    'ChicaAtiva': [True, False],
    'FoxyAtivo': [True, False],
    'FreddyAtivo': [True, False],
    'GoldenFreddyAparece': [True, False],
    'SomEsquerda': [True, False], # evidência do foxy
    'SomCozinha': [True, False], # evidência da chica
    'RisadaFreddy': [True, False],   # evidência do freddy
    'Jumpscare': [True, False] # a consulta
}

# agora vêm as CPTs que aqui são os fatores
# um fator é a estrutura que guarda as variáveis e também suas probabilidades
# as chaves do cpt são tuplas que representam os valores das variáveis

# P(Hora)
# a noite começa mais calma e fica mais difícil
fator_hora = {
    'vars': ['Hora'],
    'cpt': {
        ('Inicio',): 0.4,
        ('Meio',): 0.4,
        ('Fim',): 0.2
    }
}

# P(BonnieAtivo | Hora)
fator_bonnie = {
    'vars': ['BonnieAtivo', 'Hora'],
    'cpt': {
        (True, 'Inicio'): 0.2,
        (False, 'Inicio'): 0.8,
        (True, 'Meio'): 0.5,
        (False, 'Meio'): 0.5,
        (True, 'Fim'): 0.8,
        (False, 'Fim'): 0.2
    }
}

# P(ChicaAtiva | Hora)
fator_chica = {
    'vars': ['ChicaAtiva', 'Hora'],
    'cpt': {
        (True, 'Inicio'): 0.2,
        (False, 'Inicio'): 0.8,
        (True, 'Meio'): 0.4,
        (False, 'Meio'): 0.6,
        (True, 'Fim'): 0.7,
        (False, 'Fim'): 0.3
    }
}

# P(FoxyAtivo | Hora)
# o foxy corre muito rápido quando ataca, sobrando pouca janela de tempo para reagir se não estiver preparado
fator_foxy = {
    'vars': ['FoxyAtivo', 'Hora'],
    'cpt': {
        (True, 'Inicio'): 0.1,
        (False, 'Inicio'): 0.9,
        (True, 'Meio'): 0.3,
        (False, 'Meio'): 0.7,
        (True, 'Fim'): 0.9,
        (False, 'Fim'): 0.1
    }
}

# P(FreddyAtivo | Hora)
# freddy fica mais ativo com o passar da noite e especialmente no escuro
fator_freddy = {
    'vars': ['FreddyAtivo', 'Hora'],
    'cpt': {
        (True, 'Inicio'): 0.1,
        (False, 'Inicio'): 0.9,
        (True, 'Meio'): 0.3,
        (False, 'Meio'): 0.7,
        (True, 'Fim'): 0.8, # ele é perigoso no final
        (False, 'Fim'): 0.2
    }
}

# P(GoldenFreddyAparece)
# golden freddy é uma alucinação rara e não depende da hora
# ele é um nó raiz ou seja sem pais
fator_golden_freddy = {
    'vars': ['GoldenFreddyAparece'],
    'cpt': {
        (True,): 0.005, # 0.5% de chance de aparecer
        (False,): 0.995
    }
}


# P(SomEsquerda | FoxyAtivo)
# ouvir passos na esquerda é uma evidência de que o foxy saiu
fator_som_esquerda = {
    'vars': ['SomEsquerda', 'FoxyAtivo'],
    'cpt': {
        (True, True): 0.7, # ele faz barulho quando está ativo
        (False, True): 0.3,
        (True, False): 0.1, # às vezes faz cantos raros durante a noite
        (False, False): 0.9
    }
}

# P(SomCozinha | ChicaAtiva)
# a chica pode ir para a cozinha derrubar as panelas
fator_som_cozinha = {
    'vars': ['SomCozinha', 'ChicaAtiva'],
    'cpt': {
        (True, True): 0.8,
        (False, True): 0.2,
        (True, False): 0.05, # barulhos aleatórios
        (False, False): 0.95
    }
}

# P(RisadaFreddy | FreddyAtivo)
# a risada dele indica que ele está se movendo
fator_risada_freddy = {
    'vars': ['RisadaFreddy', 'FreddyAtivo'],
    'cpt': {
        (True, True): 0.6, # ele ri quando está ativo
        (False, True): 0.4,
        (True, False): 0.01, # alucinação
        (False, False): 0.99
    }
}

# P(Jumpscare | B, C, F, Fr, GF)
# esta é a CPT ou Tabela de Probabilidade Condicional para o Jumpscare
# ela tem 5 pais que são Bonnie e Chica e Foxy e Freddy e GoldenFreddy
# e 2^5 ou 32 combinações de pais
# resultando em 64 linhas totais pois são 32 para Jumpscare=True e 32 para Jumpscare=False
fator_jumpscare = {
    'vars': ['Jumpscare', 'BonnieAtivo', 'ChicaAtiva', 'FoxyAtivo', 'FreddyAtivo', 'GoldenFreddyAparece'],
    'cpt': {
        # A ordem das chaves é
        # Jumpscare e Bonnie e Chica e Foxy e Freddy e GoldenFreddy
        
        # casos onde golden freddy aparece e GF=True
        # se Golden Freddy aparece o Jumpscare é 100% garantido e não importa os outros
        (True, True, True, True, True, True): 1.0, (False, True, True, True, True, True): 0.0,
        (True, True, True, True, False, True): 1.0, (False, True, True, True, False, True): 0.0,
        (True, True, True, False, True, True): 1.0, (False, True, True, False, True, True): 0.0,
        (True, True, True, False, False, True): 1.0, (False, True, True, False, False, True): 0.0,
        (True, True, False, True, True, True): 1.0, (False, True, False, True, True, True): 0.0,
        (True, True, False, True, False, True): 1.0, (False, True, False, True, False, True): 0.0,
        (True, True, False, False, True, True): 1.0, (False, True, False, False, True, True): 0.0,
        (True, True, False, False, False, True): 1.0, (False, True, False, False, False, True): 0.0,
        (True, False, True, True, True, True): 1.0, (False, False, True, True, True, True): 0.0,
        (True, False, True, True, False, True): 1.0, (False, False, True, True, False, True): 0.0,
        (True, False, True, False, True, True): 1.0, (False, False, True, False, True, True): 0.0,
        (True, False, True, False, False, True): 1.0, (False, False, True, False, False, True): 0.0,
        (True, False, False, True, True, True): 1.0, (False, False, False, True, True, True): 0.0,
        (True, False, False, True, False, True): 1.0, (False, False, False, True, False, True): 0.0,
        (True, False, False, False, True, True): 1.0, (False, False, False, False, True, True): 0.0,
        (True, False, False, False, False, True): 1.0, (False, False, False, False, False, True): 0.0,

        # casos onde golden freddy n aparece e GF=False
        # a chance de jumpscare depende do número de animatronics B e C e F e Fr ativos
        
        # 4 ATIVOS B e C e F e Fr
        (True, True, True, True, True, False): 0.99, (False, True, True, True, True, False): 0.01,
        
        # 3 ATIVOS
        (True, True, True, True, False, False): 0.95, (False, True, True, True, False, False): 0.05,
        (True, True, True, False, True, False): 0.95, (False, True, True, False, True, False): 0.05,
        (True, True, False, True, True, False): 0.95, (False, True, False, True, True, False): 0.05,
        (True, False, True, True, True, False): 0.95, (False, False, True, True, True, False): 0.05,
        
        # 2 ATIVOS
        (True, True, True, False, False, False): 0.85, (False, True, True, False, False, False): 0.15,
        (True, True, False, True, False, False): 0.85, (False, True, False, True, False, False): 0.15,
        (True, True, False, False, True, False): 0.85, (False, True, False, False, True, False): 0.15,
        (True, False, True, True, False, False): 0.85, (False, False, True, True, False, False): 0.15,
        (True, False, True, False, True, False): 0.85, (False, False, True, False, True, False): 0.15,
        (True, False, False, True, True, False): 0.85, (False, False, False, True, True, False): 0.15,

        # 1 ATIVO
        (True, True, False, False, False, False): 0.60, (False, True, False, False, False, False): 0.40,
        (True, False, True, False, False, False): 0.60, (False, False, True, False, False, False): 0.40,
        (True, False, False, True, False, False): 0.60, (False, False, False, True, False, False): 0.40,
        (True, False, False, False, True, False): 0.60, (False, False, False, False, True, False): 0.40,
        
        # 0 ATIVOS
        (True, False, False, False, False, False): 0.01, (False, False, False, False, False, False): 0.99,
    }
}

# funções do algoritmo de eliminação

# função de normalização que garante que a distribuição final some 1
# ela pega scores brutos por exemplo {T: 0.5, F: 0.25} e os converte
# em probabilidades por exemplo {T: 0.66, F: 0.33}
def normalizar(fator):
    # soma todos os valores ou scores no fator
    soma_total = sum(fator['cpt'].values())
    
    # se a soma for zero não há o que fazer e isso evita divisão por zero
    if soma_total == 0:
        return fator
    
    cpt_normalizada = {}
    # divide cada valor pela soma total para obter a proporção
    for chave, valor in fator['cpt'].items():
        cpt_normalizada[chave] = valor / soma_total
        
    return {'vars': fator['vars'], 'cpt': cpt_normalizada}

# esta função aplica uma evidência por exemplo 'SomEsquerda' = True a um fator
# ela filtra a tabela e mantém só as linhas que batem com a evidência
# e remove a variável de evidência das 'vars' do fator
def restringir_fator(fator, var_evidencia, valor_evidencia):
    
    # define quais variáveis vão sobrar no fator
    vars_novas = []
    for v in fator['vars']:
        if v != var_evidencia:
            vars_novas.append(v)
            
    # encontra o índice ou posição da variável de evidência na lista de 'vars'
    idx_evidencia = fator['vars'].index(var_evidencia)
    
    cpt_novo = {}
    # itera sobre cada linha ou chave da tabela antiga
    for chave_antiga, prob in fator['cpt'].items():
        
        # mantendo apenas as linhas onde o valor é igual à evidência
        if chave_antiga[idx_evidencia] == valor_evidencia:
            
            # criando a nova chave menor sem o valor da evidência
            chave_nova_lista = list(chave_antiga)
            chave_nova_lista.pop(idx_evidencia)
            chave_nova = tuple(chave_nova_lista)
            
            cpt_novo[chave_nova] = prob
            
    # retorna o fator podado
    return {'vars': vars_novas, 'cpt': cpt_novo}

# esta função auxiliar implementa o produto cartesiano 
# ela é um combinador necessário para a multiplicação de fatores
# por exemplo se dominios_list = [[1, 2], ['a']] ela retorna [(1, 'a'), (2, 'a')]
def obter_produto_cartesiano(dominios_list):
    # caso base da recursão
    if not dominios_list:
        return [()]
    
    combinacoes_finais = []
    
    # pega o primeiro domínio por exemplo [1, 2]
    primeiro_dominio = dominios_list[0]
    # e o resto por exemplo [['a']]
    resto_dominios = dominios_list[1:]
    
    # recursão para obter as combinações do resto
    combinacoes_resto = obter_produto_cartesiano(resto_dominios)
    
    # combina o primeiro domínio com o resto
    for valor in primeiro_dominio:
        for combinacao in combinacoes_resto:
            combinacoes_finais.append((valor,) + combinacao)
            
    return combinacoes_finais

# esta é a operação de junção ou multiplicação de fatores
# ela combina dois fatores ou tabelas em um fator maior
# multiplicando as probabilidades das linhas que correspondem
def multiplicar_fatores(fator1, fator2):
    
    # descobre quais variáveis os dois fatores têm
    vars1 = set(fator1['vars'])
    vars2 = set(fator2['vars'])
    # a união é a lista de variáveis do novo fator
    vars_todas = list(vars1.union(vars2))
    
    # iniciando o novo CPT vazio
    cpt_novo = {}
    
    # mapeamento dos índices das variáveis para facilitar a busca
    indices_novo = {}
    for var in vars_todas:
        indices_novo[var] = vars_todas.index(var)

    # domínios das variáveis no novo fator
    dominios_novo = []
    for var in vars_todas:
        dominios_novo.append(DOMINIOS[var])
    
    # iteração sobre todas as combinações de valores do novo fator
    for valores_combinados in obter_produto_cartesiano(dominios_novo):
        
        chave_nova = valores_combinados
        
        # extração das chaves correspondentes para os fatores antigos
        chave1_lista = []
        for var in fator1['vars']:
            chave1_lista.append(valores_combinados[indices_novo[var]])
        chave1 = tuple(chave1_lista)
        
        chave2_lista = []
        for var in fator2['vars']:
            chave2_lista.append(valores_combinados[indices_novo[var]])
        chave2 = tuple(chave2_lista)

        # multiplicação das probabilidades das entradas correspondentes
        p1 = fator1['cpt'][chave1]
        p2 = fator2['cpt'][chave2]
        
        cpt_novo[chave_nova] = p1 * p2
            
    # retorna o novo fator "agrupado"
    return {'vars': vars_todas, 'cpt': cpt_novo}

# esta é a operação de somar colunas ou marginalização
# ela elimina uma variável de um fator somando sobre ela
# por exemplo dado P(A, B) ela pode eliminar B para resultar em P(A)
def somar_sobre_variavel(fator, var_para_eliminar):
    
    # encontra as variáveis que vão sobrar
    vars_novas = []
    for var in fator['vars']:
        if var != var_para_eliminar:
            vars_novas.append(var)
    
    # e o índice da variável a ser eliminada
    idx_eliminar = fator['vars'].index(var_para_eliminar)
    
    cpt_novo = {}
    
    # iteração sobre todas as linhas do CPT antigo
    for chave_antiga, prob in fator['cpt'].items():
        
        # criação da nova chave menor removendo o valor da variável eliminada
        chave_nova_lista = list(chave_antiga)
        chave_nova_lista.pop(idx_eliminar)
        chave_nova = tuple(chave_nova_lista)
        
        # aqui a soma se acumula
        # se a chave já existe então a probabilidade é somada
        if chave_nova in cpt_novo:
            cpt_novo[chave_nova] += prob
        else:
            # se não então ela é iniciada
            cpt_novo[chave_nova] = prob
                
    # retorna o fator espremido
    return {'vars': vars_novas, 'cpt': cpt_novo}


# esta função organiza todo o processo de eliminação de variáveis
def eliminacao_de_variaveis(var_consulta, evidencias, rede, ordem_eliminacao):
    
    # copiando os fatores para não modificar a rede original
    fatores_restringidos = []
    
    # passo 1 é aplicar todas as evidências ou seja restringir e podar
    # itera sobre todos os fatores da rede
    for f in rede:
        fator_novo = f.copy()
        # checa se alguma evidência se aplica a este fator
        for var_evidencia, valor_evidencia in evidencias.items():
            if var_evidencia in fator_novo['vars']:
                # se sim então chama a função "podadora"
                fator_novo = restringir_fator(fator_novo, var_evidencia, valor_evidencia)
        fatores_restringidos.append(fator_novo)
    
    # a partir de agora a lista de fatores é a lista de fatores podados
    fatores = fatores_restringidos

    # passo 2 é eliminar as variáveis ocultas
    # o código itera sobre cada variável oculta uma por uma
    for var in ordem_eliminacao:
        
        # encontra todos os fatores que contêm a variável
        fatores_para_multiplicar = []
        for f in fatores:
            if var in f['vars']:
                fatores_para_multiplicar.append(f)
        
        # remove eles da lista principal
        fatores_temp = []
        for f in fatores:
            if var not in f['vars']:
                fatores_temp.append(f)
        fatores = fatores_temp
        
        # multiplica todos eles para formar um fator grande
        fator_gigante = fatores_para_multiplicar[0]
        for i in range(1, len(fatores_para_multiplicar)):
            fator_gigante = multiplicar_fatores(fator_gigante, fatores_para_multiplicar[i])
        
        # agora a variável é eliminada por somatório ou marginalização
        fator_marginalizado = somar_sobre_variavel(fator_gigante, var)
        
        # e o novo fator menor é colocado de volta na lista
        fatores.append(fator_marginalizado)

    # passo 3 é multiplicar os fatores restantes
    # no final devem sobrar apenas fatores que contêm a variável de consulta
    fator_final = fatores[0]
    for i in range(1, len(fatores)):
        fator_final = multiplicar_fatores(fator_final, fatores[i])
        
    # passo 4 é normalizar para obter a resposta final
    return normalizar(fator_final)

def plotar_resultado_bayesiano(resultado, consulta, evidencias):
    if not resultado['cpt']:
        print("Não é possível plotar: evidência impossível.")
        return

    labels = []
    probabilidades = []
    
    chaves_ordenadas = sorted(resultado['cpt'].keys(), key=lambda k: k[0])

    for chave in chaves_ordenadas:
        labels.append(str(chave[0]))
        probabilidades.append(resultado['cpt'][chave])

    fig, ax = plt.subplots(figsize=(9, 7), dpi=100) # Tamanho da figura e resolução aprimorados
    
    cores = ['#4CAF50' if label == 'False' else '#FF6347' for label in labels] # Cores mais vibrantes
    
    barras = ax.bar(labels, probabilidades, color=cores, width=0.6, edgecolor='none', zorder=2) # Barras mais finas e sem borda
    
    ax.set_ylabel('Probabilidade', fontsize=13, color='#333333', labelpad=10) # Rótulo do eixo Y mais escuro
    ax.set_title(f'Distribuição de P({consulta} | Evidência)', fontsize=16, color='#333333', pad=20) # Título maior e mais escuro
    ax.set_ylim(0, 1.05) # Pequeno espaço acima da barra mais alta
    ax.tick_params(axis='x', labelsize=12, colors='#555555') # Rótulos do eixo X maiores
    ax.tick_params(axis='y', labelsize=12, colors='#555555') # Rótulos do eixo Y maiores

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1) # Grade mais sutil no eixo Y
    ax.set_axisbelow(True) # Grade atrás das barras

    if evidencias:
        ev_str = ", ".join([f"{k}={v}" for k, v in evidencias.items()])
    else:
        ev_str = "Nenhuma"
    fig.text(0.5, 0.02, f"Evidência: {ev_str}", ha='center', fontsize=11, color='#666666', style='italic')
    
    for i, barra in enumerate(barras):
        yval = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2.0, yval + 0.02, f'{yval*100:.2f}%', 
                ha='center', va='bottom', fontsize=12, color='#333333')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

# execução da consulta
if __name__ == "__main__":
    
    # Chance de tomar jumpscare de um animatronic sabendo que tem som tanto na esquerda quanto na direita
    # e que o freddy deu risada
    # P(Jumpscare | SomEsquerda=True, SomCozinha=True, RisadaFreddy=True)
    
    consulta = 'Jumpscare'
    
    evidencias = {
        'SomEsquerda': True,
        'SomCozinha': True,
        'RisadaFreddy': True
    }
    
    # a rede completa é a lista de todos os fatores
    rede_fnaf = [
        fator_hora,
        fator_bonnie,
        fator_chica,
        fator_foxy,
        fator_freddy,
        fator_golden_freddy,
        fator_som_esquerda,
        fator_som_cozinha,
        fator_risada_freddy,
        fator_jumpscare 
    ]
    
    # as variáveis ocultas são todas que não são a consulta nem a evidência
    vars_rede = set(DOMINIOS.keys())
    vars_consulta_evidencia = set([consulta])
    for k in evidencias.keys():
        vars_consulta_evidencia.add(k)
        
    vars_ocultas_set = vars_rede - vars_consulta_evidencia
    
    # a ordem de eliminação é crucial para a eficiência
    # esta ordem é uma heurística e elimina B e C e F e Fr e GF e por último Hora
    ordem_otimizada = [
        'FoxyAtivo', 
        'BonnieAtivo', 
        'ChicaAtiva', 
        'FreddyAtivo', 
        'GoldenFreddyAparece', 
        'Hora'
    ]

    print("cenário fnaf 1 - com 5 animatronics")
    print(f"evidências som esquerda {evidencias['SomEsquerda']}, som cozinha {evidencias['SomCozinha']}, risada {evidencias['RisadaFreddy']}")
    print(f"variáveis ocultas a eliminar {ordem_otimizada}")
    print("calculando P(Jumpscare) usando eliminação de variáveis...")
    
    # executando o algoritmo
    resultado_final = eliminacao_de_variaveis(
        consulta, 
        evidencias, 
        rede_fnaf, 
        ordem_otimizada
    )
    
    # print da saída 
    print("\nresultado")
    if not resultado_final['cpt']:
        # se por algum motivo não fosse possível caulcular
        print("não foi possível calcular pois a evidência é impossível")
    else:
        for chave, prob in resultado_final['cpt'].items():
            # a chave é uma tupla e basta pegar o primeiro valor
            estado_jumpscare = chave[0]
            probabilidade_percentual = prob * 100
            print(f"  P(Jumpscare={estado_jumpscare}) = {probabilidade_percentual:.2f}%")
        plotar_resultado_bayesiano(resultado_final, consulta, evidencias)