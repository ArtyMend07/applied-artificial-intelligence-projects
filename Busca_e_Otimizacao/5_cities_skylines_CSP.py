# 1.

# Etapa onde se define a estrutura do planejamento urbano
# utilizando variáveis, domínios, e restrições

class CidadeCSP:
    """
    classe que representa planejamento urbano
    Ela encapsula todas as informações necessárias para o solver:
    - As variáveis (lotes no grid).
    - Os domínios (tipos de zona que cada lote pode ter).
    - As restrições (as regras de urbanismo).
    """

    def __init__(self, tamanho_grid, cotas):
        # define cenário inicial da cidade
        self.tamanho_grid = tamanho_grid
        self.cotas = cotas # Restrição global: quantas zonas de cada tipo queremos.

        # as variáveis do nosso csp são as coordenadas de cada lote no grid
        # por exemplo, (0, 0), (0, 1), (1, 0), etc
        self.variaveis = [(linha, col) for linha in range(tamanho_grid) for col in range(tamanho_grid)]

        # O domínio inicial para cada variável é o mesmo: todos os tipos de zona possíveis.
        # lista de opções de construções de lote
        zonas_permitidas = [zona for zona, cota in self.cotas.items() if cota > 0]
        self.dominios = {variavel: list(zonas_permitidas) for variavel in self.variaveis}

        # calcula vizinhos de cada lote para facilitar a verificação das restrições.
        # um vizinho seria um lote adjascente ao atual tanto horizontal quanto vertical
        self.vizinhos = self._calcular_vizinhos()

    def _calcular_vizinhos(self):
        # função que encontra todos os vizinhos de cada lote.
        vizinhos = {var: [] for var in self.variaveis}
        for linha, col in self.variaveis:
            # Movimentos possíveis: direita, esquerda, baixo, cima
            for dl, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nova_linha, nova_col = linha + dl, col + dc
                if 0 <= nova_linha < self.tamanho_grid and 0 <= nova_col < self.tamanho_grid:
                    vizinhos[(linha, col)].append((nova_linha, nova_col))
        return vizinhos

    def e_consistente(self, variavel, valor, atribuicao, contagens):
        # Essa função é totalmente relacionada com as restrições
        # pois verifica se uma atribuição (por exemplo: lote (0,0) = industrial)
        # viola alguma regra em relação aos vizinhos já estabelecidos.

        # Impede que o número de uma zona ultrapasse a cota definida.
        if contagens[valor] + 1 > self.cotas[valor]:
            return False # violação de cota

        # primeira regra
        # ZONAS INDUSTRIAIS (I) NÃO PODEM SER VIZINHAS DE ZONAS RESIDENCIAIS (R).
        # O motivo dessa restrição é similar com o que acontece tanto no jogo (cities: skylines) quanto na vida real,
        # criar zonas industriais perto de setores residenciais geraria muito barulho, poluição, e queda 
        # da qualidade de vida da população local
        if valor == 'I':
            for vizinho in self.vizinhos[variavel]:
                if atribuicao.get(vizinho) == 'R':
                    return False # violação
                # regra 2: Proíbe Parques ('P') de serem vizinhos de Indústrias ('I').
                if atribuicao.get(vizinho) == 'P':
                    return False # violação

        # terceira 3:
        # ZONAS RESIDENCIAIS (R) TAMBÉM NÃO PODEM SER VIZINHAS DE INDUSTRIAIS (I).
        # a lógica é a mesma de antes, porém, aqui é para garantir que o contrário não seja permitido também
        if valor == 'R':
            for vizinho in self.vizinhos[variavel]:
                if atribuicao.get(vizinho) == 'I':
                    return False # Violação!

        # quarta regra:
        # Uma zona Comercial ('C') deve ter pelo menos um vizinho Residencial ('R') para ter clientes.
        # Esta regra só pode ser conclusivamente falha se todos os vizinhos já estiverem definidos.
        if valor == 'C':
            tem_vizinho_residencial = False
            for vizinho in self.vizinhos[variavel]:
                if atribuicao.get(vizinho) == 'R':
                    tem_vizinho_residencial = True
                    break
            # Se todos os vizinhos estão preenchidos e nenhum é 'R', a regra falha.
            todos_vizinhos_definidos = all(v in atribuicao for v in self.vizinhos[variavel])
            if todos_vizinhos_definidos and not tem_vizinho_residencial:
                return False # Violação!
        
        # REGRA 5
        # Um Parque ('P') deve atender a população, logo, precisa de um vizinho Residencial ('R').
        if valor == 'P':
            tem_vizinho_residencial = False
            for vizinho in self.vizinhos[variavel]:
                if atribuicao.get(vizinho) == 'R':
                    tem_vizinho_residencial = True
                    break
            todos_vizinhos_definidos = all(v in atribuicao for v in self.vizinhos[variavel])
            if todos_vizinhos_definidos and not tem_vizinho_residencial:
                return False # violação
            # Parques não podem ser vizinhos de Indústrias.
            for vizinho in self.vizinhos[variavel]:
                if atribuicao.get(vizinho) == 'I':
                    return False # Violação

        return True # Nenhuma regra foi violada, a atribuição é consistente.


# 2. Implementação do algoritmo de resolução

# Será utilizado o algoritmo de backtracking com otimização de foward check, que vai ser o motor para construir a cidade

# função para selecionar a próxima variável a ser preenchida.
def selecionar_variavel_nao_atribuida_mrv(atribuicao, csp, dominios):
    
    # Será usado a heurística MRV.
    # ela escolhe a variável que tem o menor número de opções legais restantes em seu domínio.
    variaveis_nao_atribuidas = [v for v in csp.variaveis if v not in atribuicao]
    # Retorna a variável com o menor domínio.
    return min(variaveis_nao_atribuidas, key=lambda var: len(dominios[var]))

def solver_backtracking(csp):
    """
    Função principal que inicia o processo de resolução do CSP.
    Ela chama a função recursiva de backtracking com os valores iniciais.
    """
    # A primeira atribuição é um dicionário vazio.
    contagens_iniciais = {zona: 0 for zona in csp.cotas}
    return backtracking_recursivo({}, csp, csp.dominios, contagens_iniciais)

#função recursiva do backtracking
def backtracking_recursivo(atribuicao, csp, dominios, contagens):
    
    # PASSO 1: CONDIÇÃO DE PARADA (SUCESSO).
    # SE TODAS AS VARIÁVEIS FORAM ATRIBUÍDAS, encontramos uma potencial solução.
    if len(atribuicao) == len(csp.variaveis):
        return atribuicao # SUCESSO! ENCONTRAMOS UMA SOLUÇÃO.

    # PASSO 2: SELEÇÃO DA PRÓXIMA VARIÁVEL.
    # utiliza heurística mrv
    variavel = selecionar_variavel_nao_atribuida_mrv(atribuicao, csp, dominios)

    # PASSO 3: ITERAÇÃO SOBRE OS VALORES POSSÍVEIS.
    # tentamos cada zona para o lote escolhido
    for valor in dominios[variavel]:

        # PASSO 4: VERIFICAÇÃO DE CONSISTÊNCIA LOCAL.
        # verificamos se a escolha atual não viola as Regras com vizinhos que já foram definidos.
        if csp.e_consistente(variavel, valor, atribuicao, contagens):
            atribuicao[variavel] = valor # faz a atribuição temporariamente.
            contagens[valor] += 1

            # PASSO 5: OTIMIZAÇÃO COM FORWARD CHECKING.
            # criando uma cópia dos domínios
            dominios_copia = {var: list(vals) for var, vals in dominios.items()}

            # A lógica de poda é que, se o valor é I, removemos R dos domínios dos vizinhos
            # se o valor é R, removemos 'I' dos domínios dos vizinhos.
            mapa_poda = {'I': 'R', 'R': 'I'}
            falha = False
            if valor in mapa_poda:
                valor_a_podar = mapa_poda[valor]
                for vizinho in csp.vizinhos[variavel]:
                    if vizinho not in atribuicao: # apenas para vizinhos não atribuídos
                        if valor_a_podar in dominios_copia[vizinho]:
                            dominios_copia[vizinho].remove(valor_a_podar)
                        if not dominios_copia[vizinho]: # caso um vizinho fique sem opções
                            falha = True # esse caminho levou a um beco sem saída.
                            break
            
            if not falha:
                # PASSO 6: CHAMADA RECURSIVA.
                # se o caminho ainda é promissor, avança para o próximo lote.
                resultado = backtracking_recursivo(atribuicao, csp, dominios_copia, contagens)
                if resultado is not None:
                    return resultado # Propaga a solução encontrada para cima.

            # PASSO 7: BACKTRACKING.
            # SE A CHAMADA RECURSIVA FALHOU (retornou None) OU SEGUIMOS PARA O PRÓXIMO VALOR,
            # DEVEMOS DESFAZER A ATRIBUIÇÃO ATUAL.
            del atribuicao[variavel]
            contagens[valor] -= 1

    return None # nenhuma solução foi encontrada a partir deste ponto da busca


# 3. Interface no terminal


# função para imprimir o mapa da cidade.
def imprimir_solucao(solucao, tamanho_grid):
    if solucao is None:
        print("NENHUMA SOLUÇÃO FOI ENCONTRADA para este planejamento.")
        return
    print("PLANEJAMENTO DA CIDADE CONCLUÍDO:")
    for linha in range(tamanho_grid):
        # Monta a string da linha do grid, usando '.' para lotes não preenchidos.
        linha_str = [solucao.get((linha, col), '.') for col in range(tamanho_grid)]
        print(" ".join(linha_str))


if __name__ == "__main__":
    # definindo o problema
    TAMANHO_GRID = 5

    # definindo as restrições globais
    # o restante do grid será considerado vazio 
    COTAS = {
        'R': 10,  # Residencial
        'C': 5,   # Comercial
        'I': 9,   # Industrial
        'P': 1,   # Parque
    }
    
    total_lotes = TAMANHO_GRID * TAMANHO_GRID
    soma_cotas = sum(COTAS.values())
    
    # Adiciona a cota para os lotes vazios, que preencherão o restante do grid
    cotas_vazias = total_lotes - soma_cotas
    COTAS['_'] = cotas_vazias

    if soma_cotas > total_lotes:
        print("ERRO DE PLANEJAMENTO: A soma das cotas excede o número de lotes disponíveis no grid.")
    else:
        # Se as cotas forem viáveis, o programa prossegue normalmente
        print("Iniciando o planejador urbano...")
        print(f"Grid: {TAMANHO_GRID}x{TAMANHO_GRID}, Cotas exatas: {COTAS}")
        print("-" * 30)

        # instância do problema
        problema_cidade = CidadeCSP(TAMANHO_GRID, COTAS)

        # invoca solução do problema
        solucao = solver_backtracking(problema_cidade)

        # IMPRIMINDO O RESULTADO.
        print("\n--- RESULTADO ---")
        imprimir_solucao(solucao, TAMANHO_GRID)