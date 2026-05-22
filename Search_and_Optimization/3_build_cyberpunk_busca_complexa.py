import math
import random

# --- 1. DEFINIÇÃO DO PROBLEMA  ---

# primeiro, modelamos os dados do jogo. um personagem tem status base e um
# banco de dados de cyberware que pode ser instalado.

# status base de um personagem.
PERSONAGEM_BASE_STATS = {
    'dano_base_arma': 150,
    'taxa_crit': 0.15,  # 15% base
    'dano_crit': 0.50   # 50% base
}

# banco de dados simulado de cyberware disponível.
# cada peça é um dicionário com um id, um slot corporal e os bônus que concede.
CYBERWARE_DB = [
    # --- sistema operacional --- (10 opções)
    {'id': 0, 'slot': 'os', 'nome': 'sandevistan "militech falcon"', 'bonuses': {'taxa_crit': 0.10, 'dano_crit': 0.15}},
    {'id': 1, 'slot': 'os', 'nome': 'ciberdeque "tetratronic rippler"', 'bonuses': {}}, # focado em hacking
    {'id': 8, 'slot': 'os', 'nome': 'berserk "biodyne"', 'bonuses': {'dano_arma': 0.20}}, # focado em dano bruto
    {'id': 9, 'slot': 'os', 'nome': 'sandevistan "zetatech warp dancer"', 'bonuses': {'taxa_crit': 0.18}}, # focado em taxa crítica
    {'id': 23, 'slot': 'os', 'nome': 'berserk "arasaka rampage"', 'bonuses': {'dano_arma': 0.15, 'taxa_crit': 0.05}},
    {'id': 24, 'slot': 'os', 'nome': 'ciberdeque "raven microcyber"', 'bonuses': {'dano_crit': 0.10}}, # deck com bônus passivo
    {'id': 40, 'slot': 'os', 'nome': 'sandevistan "dynalar mk.3"', 'bonuses': {'dano_crit': 0.22}},
    {'id': 41, 'slot': 'os', 'nome': 'berserk "atomic усиление"', 'bonuses': {'dano_arma': 0.18, 'dano_crit': 0.05}},
    {'id': 42, 'slot': 'os', 'nome': 'ciberdeque "netwatch netdriver"', 'bonuses': {'taxa_crit': 0.05}}, # deck com bônus passivo
    {'id': 43, 'slot': 'os', 'nome': 'sandevistan "qian-t mk.4"', 'bonuses': {'taxa_crit': 0.08, 'dano_crit': 0.12}},

    # --- córtex frontal --- (10 opções)
    {'id': 2, 'slot': 'frontal_cortex', 'nome': 'ex-disk', 'bonuses': {'dano_crit': 0.20}},
    {'id': 3, 'slot': 'frontal_cortex', 'nome': 'memória mecatrônica', 'bonuses': {'dano_arma': 0.10}},
    {'id': 10, 'slot': 'frontal_cortex', 'nome': 'limiter remoção', 'bonuses': {'taxa_crit': 0.05, 'dano_crit': 0.10}},
    {'id': 11, 'slot': 'frontal_cortex', 'nome': 'processador visual', 'bonuses': {'dano_crit': 0.18}},
    {'id': 25, 'slot': 'frontal_cortex', 'nome': 'simulador de combate', 'bonuses': {'taxa_crit': 0.12}},
    {'id': 26, 'slot': 'frontal_cortex', 'nome': 'otimizador de kerenzikov', 'bonuses': {'dano_arma': 0.08}},
    {'id': 44, 'slot': 'frontal_cortex', 'nome': 'amplificador de memória ram', 'bonuses': {}}, # hacking
    {'id': 45, 'slot': 'frontal_cortex', 'nome': 'coprocessador de ameaças', 'bonuses': {'dano_crit': 0.15}},
    {'id': 46, 'slot': 'frontal_cortex', 'nome': 'sensor de precisão', 'bonuses': {'taxa_crit': 0.10}},
    {'id': 47, 'slot': 'frontal_cortex', 'nome': 'regulador de adrenalina frontal', 'bonuses': {'dano_arma': 0.05, 'taxa_crit': 0.03}},

    # --- esqueleto --- (10 opções)
    {'id': 4, 'slot': 'esqueleto', 'nome': 'medula densa', 'bonuses': {'dano_arma': 0.15}}, # focado em melee
    {'id': 5, 'slot': 'esqueleto', 'nome': 'ligamentos sinápticos', 'bonuses': {'taxa_crit': 0.08}},
    {'id': 12, 'slot': 'esqueleto', 'nome': 'esqueleto de titânio', 'bonuses': {'dano_arma': 0.10}},
    {'id': 16, 'slot': 'esqueleto', 'nome': 'reforço universal', 'bonuses': {'taxa_crit': 0.06, 'dano_crit': 0.06}},
    {'id': 27, 'slot': 'esqueleto', 'nome': 'microvibradores', 'bonuses': {'dano_crit': 0.14}},
    {'id': 28, 'slot': 'esqueleto', 'nome': 'amortecedores', 'bonuses': {}}, # puramente defensivo
    {'id': 48, 'slot': 'esqueleto', 'nome': 'placas ósseas reforçadas', 'bonuses': {}}, # defesa
    {'id': 49, 'slot': 'esqueleto', 'nome': 'juntas cinéticas', 'bonuses': {'dano_arma': 0.08}}, # melee
    {'id': 50, 'slot': 'esqueleto', 'nome': 'sistema de suporte vital', 'bonuses': {'taxa_crit': 0.04}}, # bônus pequeno
    {'id': 51, 'slot': 'esqueleto', 'nome': 'distribuidor de impacto', 'bonuses': {'dano_crit': 0.10}},

    # --- sistema circulatório --- (10 opções)
    {'id': 6, 'slot': 'circulatorio', 'nome': 'acelerador de sinapses', 'bonuses': {'dano_crit': 0.25}},
    {'id': 7, 'slot': 'circulatorio', 'nome': 'bomba de sangue', 'bonuses': {'dano_arma': 0.12}},
    {'id': 14, 'slot': 'circulatorio', 'nome': 'segundo coração', 'bonuses': {}}, # focado em sobrevivência
    {'id': 15, 'slot': 'circulatorio', 'nome': 'biomonitor', 'bonuses': {'taxa_crit': 0.07, 'dano_crit': 0.07}},
    {'id': 29, 'slot': 'circulatorio', 'nome': 'regulador de adrenalina', 'bonuses': {'dano_arma': 0.10}},
    {'id': 30, 'slot': 'circulatorio', 'nome': 'distribuidor sanguíneo', 'bonuses': {'taxa_crit': 0.09}},
    {'id': 52, 'slot': 'circulatorio', 'nome': 'microgeradores', 'bonuses': {}}, # focado em tech
    {'id': 53, 'slot': 'circulatorio', 'nome': 'bomba de adrenalina', 'bonuses': {'dano_arma': 0.08, 'taxa_crit': 0.04}},
    {'id': 54, 'slot': 'circulatorio', 'nome': 'feedback circulatório', 'bonuses': {'dano_crit': 0.15}},
    {'id': 55, 'slot': 'circulatorio', 'nome': 'nanorrelés', 'bonuses': {'taxa_crit': 0.11}},

    # --- mãos --- (8 opções)
    {'id': 17, 'slot': 'maos', 'nome': 'smart link', 'bonuses': {'taxa_crit': 0.05, 'dano_crit': 0.10}}, # para armas smart
    {'id': 18, 'slot': 'maos', 'nome': 'manopla balística', 'bonuses': {'dano_arma': 0.08}}, # para armas de fogo
    {'id': 19, 'slot': 'maos', 'nome': 'punhos de gorila (passivo)', 'bonuses': {'dano_arma': 0.06}}, # bônus passivo mesmo sem usar
    {'id': 31, 'slot': 'maos', 'nome': 'processador de tato', 'bonuses': {'taxa_crit': 0.07}},
    {'id': 32, 'slot': 'maos', 'nome': 'shock absorber', 'bonuses': {}}, # focado em recuo
    {'id': 56, 'slot': 'maos', 'nome': 'estabilizador de mira', 'bonuses': {'dano_crit': 0.12}},
    {'id': 57, 'slot': 'maos', 'nome': 'sistema de recarga rápida', 'bonuses': {}}, # utilidade
    {'id': 58, 'slot': 'maos', 'nome': 'garras (passivo)', 'bonuses': {'dano_arma': 0.05}}, # melee

    # --- sistema nervoso --- (8 opções)
    {'id': 20, 'slot': 'nervoso', 'nome': 'reforço de reflexos', 'bonuses': {'taxa_crit': 0.10}},
    {'id': 21, 'slot': 'nervoso', 'nome': 'kerenzikov', 'bonuses': {'dano_crit': 0.12}},
    {'id': 22, 'slot': 'nervoso', 'nome': 'neofibra', 'bonuses': {'dano_arma': 0.07}},
    {'id': 33, 'slot': 'nervoso', 'nome': 'condutor sináptico', 'bonuses': {'dano_crit': 0.10}},
    {'id': 34, 'slot': 'nervoso', 'nome': 'acelerador neuronal', 'bonuses': {'taxa_crit': 0.08, 'dano_arma': 0.05}},
    {'id': 59, 'slot': 'nervoso', 'nome': 'modulador de kerenzikov', 'bonuses': {'dano_crit': 0.14}},
    {'id': 60, 'slot': 'nervoso', 'nome': 'rede neural otimizada', 'bonuses': {'taxa_crit': 0.09}},
    {'id': 61, 'slot': 'nervoso', 'nome': 'sistema de alerta', 'bonuses': {}}, # defesa

    # --- sistema tegumentar --- (8 opções)
    {'id': 35, 'slot': 'integumentario', 'nome': 'armadura subdérmica', 'bonuses': {}}, # defesa
    {'id': 36, 'slot': 'integumentario', 'nome': 'camuflagem óptica', 'bonuses': {'taxa_crit': 0.05}}, # bônus passivo
    {'id': 37, 'slot': 'integumentario', 'nome': 'dispersor de calor', 'bonuses': {'dano_arma': 0.04}},
    {'id': 38, 'slot': 'integumentario', 'nome': 'regulador de dor', 'bonuses': {'dano_crit': 0.08}},
    {'id': 39, 'slot': 'integumentario', 'nome': 'pigmento cromático', 'bonuses': {'taxa_crit': 0.03, 'dano_crit': 0.03}},
    {'id': 62, 'slot': 'integumentario', 'nome': 'regenerador celular', 'bonuses': {}}, # cura
    {'id': 63, 'slot': 'integumentario', 'nome': 'adaptador dérmico', 'bonuses': {'dano_arma': 0.03}},
    {'id': 64, 'slot': 'integumentario', 'nome': 'inibidor de dor', 'bonuses': {'dano_crit': 0.06}},
]

# --- 2. A FUNÇÃO DE FITNESS (CÁLCULO DE DPS) ---

# esta é a função objetivo. ela mede a "qualidade" de uma solução (uma build).
# o algoritmo de busca por cucos tentará encontrar a combinação de cyberware
# que maximiza o valor retornado por esta função, que no caso é o dps (dano por segundo).

def calcular_dps_build(build, personagem_stats, cyberware_db):
    """
    função de fitness: calcula o dps (dano por segundo) médio de uma build.
    uma 'build' é uma lista de ids de peças de cyberware.
    """
    # começa com os status base do personagem.
    stats_total = {
        'dano_arma_bonus': 0,
        'taxa_crit': personagem_stats['taxa_crit'],
        'dano_crit': personagem_stats['dano_crit']
    }

    # acumula os bônus de todas as peças de cyberware na build.
    for cyberware_id in build:
        peca = cyberware_db[cyberware_id]
        if 'dano_arma' in peca['bonuses']:
            stats_total['dano_arma_bonus'] += peca['bonuses']['dano_arma']
        if 'taxa_crit' in peca['bonuses']:
            stats_total['taxa_crit'] += peca['bonuses']['taxa_crit']
        if 'dano_crit' in peca['bonuses']:
            stats_total['dano_crit'] += peca['bonuses']['dano_crit']

    # garante que a taxa crítica não passe de 100% (é desperdício de valor crítico ter mais que 100% de taxa).
    taxa_crit_final = min(1.0, stats_total['taxa_crit'])

    # fórmula do dps médio considera a chance de um ataque ser crítico.
    # dps = dano_base * (1 + bonus_dano) * (1 + taxa_crit_final * dano_crit_total)
    dps_medio = personagem_stats['dano_base_arma'] * (1 + stats_total['dano_arma_bonus']) * (1 + taxa_crit_final * stats_total['dano_crit'])
    
    return dps_medio

# --- 3. IMPLEMENTAÇÃO DO ALGORITMO CUCKOO SEARCH ---

# divide o banco de dados de cyberware por slot para gerar builds válidas.
cyberware_por_slot = {
    'os': [p['id'] for p in CYBERWARE_DB if p['slot'] == 'os'],
    'frontal_cortex': [p['id'] for p in CYBERWARE_DB if p['slot'] == 'frontal_cortex'],
    'esqueleto': [p['id'] for p in CYBERWARE_DB if p['slot'] == 'esqueleto'],
    'circulatorio': [p['id'] for p in CYBERWARE_DB if p['slot'] == 'circulatorio'],
}
slots_disponiveis = list(cyberware_por_slot.keys())

def gerar_build_aleatoria():
    """
    gera uma solução candidata (um ninho) completamente aleatória,
    garantindo que seja uma build válida (uma peça por slot).
    """
    build = []
    for slot in slots_disponiveis:
        peca_aleatoria = random.choice(cyberware_por_slot[slot])
        build.append(peca_aleatoria)
    return build

def gerar_solucao_via_levy_flight(ninho_atual):
    """
    esta função simula o "voo de lévy" do cuco.
    ela pega uma solução existente e faz uma alteração aleatória, mas significativa.
    isso representa a busca global do algoritmo.
    """
    novo_ninho = list(ninho_atual)
    
    # escolhe um slot aleatório para mudar a peça de cyberware.
    # um "salto longo" pode envolver mudar mais de um slot, mas vamos simplificar para um.
    num_mudancas = 1 # define o "tamanho" do salto.
    
    for _ in range(num_mudancas):
        slot_index_para_mudar = random.randrange(len(slots_disponiveis))
        slot_a_mudar = slots_disponiveis[slot_index_para_mudar]
        
        # escolhe uma nova peça para aquele slot, garantindo que seja diferente da atual.
        peca_atual = novo_ninho[slot_index_para_mudar]
        opcoes_disponiveis = [p_id for p_id in cyberware_por_slot[slot_a_mudar] if p_id != peca_atual]
        
        if opcoes_disponiveis:
            nova_peca = random.choice(opcoes_disponiveis)
            novo_ninho[slot_index_para_mudar] = nova_peca
            
    return novo_ninho

def otimizador_cuckoo(personagem_stats, cyberware_db, n_ninhos, n_iteracoes, pa_fraccao_abandono):
    """
    algoritmo de busca por cucos (cuckoo search) para maximização.
    """
    # --- inicialização ---
    # cria a população inicial de 'ninhos' (soluções/builds).
    ninhos = [gerar_build_aleatoria() for _ in range(n_ninhos)]
    fitness = [calcular_dps_build(n, personagem_stats, cyberware_db) for n in ninhos]
    
    # encontra o melhor ninho inicial.
    melhor_ninho_global = ninhos[max(range(n_ninhos), key=lambda i: fitness[i])]
    melhor_fitness_global = calcular_dps_build(melhor_ninho_global, personagem_stats, cyberware_db)

    # loop principal de iterações 
    for i in range(n_iteracoes):
        # --- fase 1: voos de lévy  ---
        # para cada ninho, um "cuco" pega um ovo (uma nova solução) e o coloca em outro ninho.
        
        # pega um cuco (seleciona um ninho aleatório para gerar uma nova solução a partir dele).
        ninho_cuco = random.choice(ninhos)
        nova_solucao = gerar_solucao_via_levy_flight(ninho_cuco)
        fitness_nova_solucao = calcular_dps_build(nova_solucao, personagem_stats, cyberware_db)
        
        # escolhe um ninho hospedeiro aleatório para comparar.
        hospedeiro_index = random.randrange(n_ninhos)
        
        # se o ovo do cuco (nova solução) for melhor, ele substitui o ovo do hospedeiro.
        if fitness_nova_solucao > fitness[hospedeiro_index]:
            ninhos[hospedeiro_index] = nova_solucao
            fitness[hospedeiro_index] = fitness_nova_solucao

        # --- fase 2: abandono de ninhos ---
        # uma fração dos piores ninhos é abandonada e novos são construídos.
        # isso evita que o algoritmo fique preso em soluções ruins.
        
        # ordena os ninhos pelo fitness (do pior para o melhor).
        indices_ordenados = sorted(range(n_ninhos), key=lambda k: fitness[k])
        
        # determina quantos ninhos serão abandonados.
        num_abandonados = int(pa_fraccao_abandono * n_ninhos)
        
        for j in range(num_abandonados):
            pior_ninho_index = indices_ordenados[j]
            # substitui o pior ninho por um completamente novo e aleatório.
            ninhos[pior_ninho_index] = gerar_build_aleatoria()
            fitness[pior_ninho_index] = calcular_dps_build(ninhos[pior_ninho_index], personagem_stats, cyberware_db)
            
        # --- atualização ---
        # encontra o melhor ninho da geração atual.
        melhor_ninho_atual_index = max(range(n_ninhos), key=lambda k: fitness[k])
        melhor_fitness_atual = fitness[melhor_ninho_atual_index]
        
        # se o melhor desta geração for melhor que o melhor global já registrado, atualiza.
        if melhor_fitness_atual > melhor_fitness_global:
            melhor_fitness_global = melhor_fitness_atual
            melhor_ninho_global = ninhos[melhor_ninho_atual_index]
            
        if i % 10 == 0:
            print(f"Iteração: {i:3d} | Melhor DPS Global: {melhor_fitness_global:.2f}")

    return melhor_ninho_global, melhor_fitness_global

# --- 4. EXECUÇÃO DO OTIMIZADOR ---

if __name__ == "__main__":
    
    # parâmetros do algoritmo cuckoo search.
    NUM_NINHOS = 25              # tamanho da população de soluções (análogo ao tamanho da população no ag).
    NUM_ITERACOES = 100          # número de gerações que o algoritmo vai rodar.
    PA_FRACCAO_ABANDONO = 0.25   # fração dos piores ninhos que são descartados em cada geração.

    print("Iniciando o otimizador de cyberware com Cuckoo Search...")
    
    melhor_build_encontrada, melhor_dps = otimizador_cuckoo(
        PERSONAGEM_BASE_STATS,
        CYBERWARE_DB,
        n_ninhos=NUM_NINHOS,
        n_iteracoes=NUM_ITERACOES,
        pa_fraccao_abandono=PA_FRACCAO_ABANDONO
    )

    print("\n--- Otimização Concluída ---")
    print(f"Melhor DPS Médio Encontrado: {melhor_dps:.2f}")
    print("Com a seguinte combinação de Cyberware:")
    
    for cyberware_id in melhor_build_encontrada:
        peca = CYBERWARE_DB[cyberware_id]
        print(f"  - Slot: {peca['slot']:<16} | Nome: {peca['nome']:<35} | Bônus: {peca['bonuses']}")