from collections import deque

# Importa o módulo deque da biblioteca collections.
# O deque é usado para implementar a fila
# de forma eficiente, fazendo com que aoperação de remoção do primeiro elemento seja constante.

# Estrutura do grafo da dungeon
# O dicionário MASMORRA_MULTICHAVE define o ESPAÇO DE ESTADOS do problema.
# Cada chave do dicionário é um NÓ/estado, representando uma Sala.
# O VALOR associado a cada chave é uma lista de transições (arestas/portas).
# cada transição é uma tupla: ('Sala Vizinha', 'Chave Necessária').
# a rota mais curta leva a uma porta trancada, forçando a BFS a encontrar o caminho mais longo até a chave.
# o Tesouro requer a chave c, que, por sua vez, exige a chave b, que exige a chave a.
MASMORRA_MULTICHAVE = {
    'Sala_Inicial': [('Corredor_1', 'Nenhuma')], # A partir da Sala Inicial, pode-se ir para o Corredor_1 sem chave.
    
    # rota que leva à chave a
    'Corredor_1': [('Sala_Inicial', 'Nenhuma'), ('Corredor_2', 'Nenhuma')],
    'Corredor_2': [('Corredor_1', 'Nenhuma'), ('Sala_Chave_A', 'Nenhuma')],
    'Sala_Chave_A': [('Corredor_2', 'Nenhuma'), ('Corredor_3', 'Nenhuma')],  # a chave A é adquirida aqui (Nó onde o Agente coleta o item)
    
    # caminho que exige ChaveA para liberar o caminho para chave b
    'Corredor_3': [('Sala_Chave_A', 'Nenhuma'), ('Porta_ChaveB_Acesso', 'ChaveA')], # Transição para Porta_ChaveB_Acesso exige chave A
    'Porta_ChaveB_Acesso': [('Corredor_3', 'ChaveA'), ('Corredor_ChaveB_Final', 'Nenhuma')],
    'Corredor_ChaveB_Final': [('Porta_ChaveB_Acesso', 'Nenhuma'), ('Sala_Chave_B', 'Nenhuma')],
    'Sala_Chave_B': [('Corredor_ChaveB_Final', 'Nenhuma'), ('Corredor_Lateral_1', 'Nenhuma')],  # a chave B é adquirida aqui
    
    # rota que exige ChaveB para liberar o caminho para chave c
    'Corredor_Lateral_1': [('Sala_Chave_B', 'Nenhuma'), ('Porta_ChaveC_Acesso', 'ChaveB')], # Transição para Porta_ChaveC_Acesso exige ChaveB
    'Porta_ChaveC_Acesso': [('Corredor_Lateral_1', 'ChaveB'), ('Corredor_ChaveC_Final', 'Nenhuma')],
    'Corredor_ChaveC_Final': [('Porta_ChaveC_Acesso', 'Nenhuma'), ('Sala_Chave_C', 'Nenhuma')],
    'Sala_Chave_C': [('Corredor_ChaveC_Final', 'Nenhuma'), ('Porta_Tesouro', 'Nenhuma')],  # a chave C é adquirida aqui
    
    # tesouro final que precisa da chave c
    'Porta_Tesouro': [('Sala_Chave_C', 'Nenhuma'), ('Sala_Tesouro', 'ChaveC')], # A transição final exige chave c
    'Sala_Tesouro': [('Porta_Tesouro', 'ChaveC')] # caminho de volta do objetivo
}

# Funções auxiliares para o metroidvania
def obter_chave_na_sala(sala):
    """Simula a aquisição de chaves em certas salas."""
    # mapeia qual chave está em qual sala, simulando o CONHECIMENTO ESPECÍFICO do problema.
    if sala == 'Sala_Chave_A':
        return 'ChaveA'
    if sala == 'Sala_Chave_B':
        return 'ChaveB'
    if sala == 'Sala_Chave_C':
        return 'ChaveC'
    return None

def esta_destrancada(chave_requerida, chaves_atuais):
    """Verifica se a porta está destrancada ou se a chave está disponível."""
    # Regra Lógica: É uma restrição de movimento.
    if chave_requerida == 'Nenhuma':
        return True # Se não exige chave, a porta está sempre destrancada.
    return chave_requerida in chaves_atuais # Verifica se a chave necessária está no inventário do Agente.

def busca_largura_masmorra(masmorra, no_inicio, no_objetivo):
    """
    Função principal do algoritmo de Busca em Largura (BFS).
    
    Busca em Largura para encontrar o caminho mais curto (em número de passos)
    considerando a necessidade de chaves (conceito de Masmorra/Metroidvania).

    O conceito nãocitado usado é a Garantia de Caminho Mínimo em Passos,
    inerente à BFS, que explora exaustivamente por nível antes de aprofundar.
    """
    
    # A fila (estrutura FIFO da BFS) armazena o estado completo do agente.
    # O estado é: (sala atual, caminho percorrido, chaves coletadas)
    # A fila armazena sala atual, caminho ate agora, e chaves coletadas)
    fila = deque([
        (no_inicio, [no_inicio], set()) # estado inicial começa na Sala_Inicial, caminho com 1 nó, 0 chaves.
    ])
    
    # Conjunto de estados visitados para evitar ciclos e redundância.
    # O estado aqui é mais complexo que apenas a sala, para lidar com as chaves.
    # Conjunto de estados visitados para evitar ciclos e redundância.
    # o estado é definido pela combinação sala e chaves coletadas
    # pois chegar na mesma sala com um conjunto diferente de chaves é um novo estado útil.
    estados_visitados = set([
        (no_inicio, tuple(sorted(list(set())))) #Estado inicial: (Sala_Inicial, ()) - Tupla vazia de chaves.
    ])

    while fila:
        # o Loop principal da BFS continua enquanto houver estados para explorar.
        
        # Pega o primeiro nó (sala) da fila (BFS)
        # o popleft() garante a ordem FIFO (primeiro a entrar, primeiro a sair)
        sala_atual, caminho, chaves_atuais = fila.popleft()

        # 1. Verifica a condição de parada
        if sala_atual == no_objetivo:
            # Se o nó removido da fila é o objetivo, o caminho encontrado é o mais curto em passos.
            # a impressão do caminho foi movida para fora da função para melhor formatação
            return caminho
        # 2. Coleta a chave na sala, caso tenha
        chave_nova = obter_chave_na_sala(sala_atual)
        # O algoritmo cria o novo estado de chaves que o Agente terá se entrar nas próximas salas.
        # cria uma cópia das chaves atuais, incluindo a nova chave, para o proximo estado
        chaves_proximo_estado = chaves_atuais.copy()
        if chave_nova and chave_nova not in chaves_atuais:
            chaves_proximo_estado.add(chave_nova)
        # O set não é hashable, então é convertido para uma tupla ordenada.
        # para garantir que a lista de chaves use estrutura hash para os estados_visitados
        tupla_chaves_proximo = tuple(sorted(list(chaves_proximo_estado)))
        # 3. Explora os vizinhos
        # Itera sobre todas as transições (arestas/portas) a partir da sala atual.
        for sala_vizinha, chave_requerida in masmorra.get(sala_atual, []):
            # Aplica a restrição de movimento (checa se a porta pode ser aberta).
            # verifica se a porta está destrancada usando o conjunto de chaves atualizadas
            if esta_destrancada(chave_requerida, chaves_proximo_estado):
                # Checa se este novo estado (sala + chaves) já foi visitado.
                # o estado a ser verificado no conjunto de visitados é (vizinho, chaves que teremos lá)
                if (sala_vizinha, tupla_chaves_proximo) not in estados_visitados:
                    estados_visitados.add((sala_vizinha, tupla_chaves_proximo))
                    # Adiciona o novo estado à fila para ser explorado futuramente.
                    # adiciona à fila o novo estado
                    fila.append((sala_vizinha, caminho + [sala_vizinha], chaves_proximo_estado.copy()))

    # Se a fila esvaziar e o objetivo não foi encontrado.
    return "Caminho não encontrado."


# Configuração e início da busca
NOME_INICIO = 'Sala_Inicial'
NOME_OBJETIVO = 'Sala_Tesouro'

# chama a função principal.
solucao = busca_largura_masmorra(MASMORRA_MULTICHAVE, NOME_INICIO, NOME_OBJETIVO)

# formatação da saida
print("\n" + "=" * 80)
print(" " * 28 + "RESULTADO FINAL DA BUSCA")
print("=" * 80)

if isinstance(solucao, list):
    print(f"\nCaminho encontrado com {len(solucao) - 1} passos:\n")
    # imprime o caminho com setas, quebrando a linha se necessário
    linha_formatada = ""
    limite_linha = 75 # define um limite de caracteres por linha para a impressão do caminho
    for i, sala in enumerate(solucao):
        sala_str = sala + (" -> " if i < len(solucao) - 1 else "")
        if len(linha_formatada) + len(sala_str) > limite_linha:
            print(linha_formatada)
            linha_formatada = "  " + sala_str # começa nova linha indentada
        else:
            linha_formatada += sala_str
    print(linha_formatada) # imprime a última linha
    print("\n" + "=" * 80)
else:
    # se não encontrou, imprime a mensagem de erro
    print(f"\n{solucao}\n")
    print("=" * 80)