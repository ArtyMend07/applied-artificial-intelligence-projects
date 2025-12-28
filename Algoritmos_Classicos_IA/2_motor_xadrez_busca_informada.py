"""
Este script utiliza o algoritmo Minimax para duas finalidades:
1. Resolver quebra-cabeças de xadrez com passo a passo definidos.
2. Jogar uma partida de xadrez (IA vs. IA).
"""

# aqui importamos as ferramentas necessárias para o projeto.

import chess
# esta é a biblioteca fundamental que entende as regras do xadrez.
# ela nos dá o objeto de tabuleiro, sabe quais lances são legais,
# e pode dizer se o jogo acabou (xeque-mate, empate, etc.).

import chess.svg
# este submódulo da biblioteca chess é usado para a visualização.
# ele tem a capacidade de converter o estado atual do tabuleiro em uma
# imagem no formato svg (gráfico vetorial escalável).

import time
# este módulo nos permite adicionar pausas no código, o que é útil
# para tornar a partida ia vs ia observável para um humano.

import webbrowser
# este módulo nos permite controlar o navegador web do sistema,
# sendo usado aqui para abrir o arquivo de imagem do tabuleiro.

import os
# este módulo interage com o sistema operacional. nós o usamos para
# criar um caminho de arquivo que funcione em qualquer computador (windows, mac, linux).


# --- FUNÇÕES AUXILIARES DE INTERFACE ---
# esta seção cuida de como o usuário vê o que está acontecendo no jogo.

"""
O sistema de visualização utiliza a biblioteca chess.svg para gerar representações
visuais do tabuleiro em formato SVG. O tabuleiro é exibido no navegador padrão
do sistema, permitindo acompanhar visualmente o progresso das jogadas.
Os parâmetros permitem destacar o último movimento e inverter o tabuleiro
quando necessário (ex: jogando com as peças pretas).
"""

def mostrar_tabuleiro(tabuleiro, ultimo_lance=None, invertido=False):
    # esta função cria uma representação visual do tabuleiro e a exibe.
    
    # a função chess.svg.board gera o código svg da imagem do tabuleiro.
    # - board=tabuleiro: diz qual posição do tabuleiro deve ser desenhada.
    # - lastmove=ultimo_lance: destaca a última jogada feita, facilitando o acompanhamento.
    # - flipped=invertido: inverte o tabuleiro, útil quando é a vez das pretas.
    # - size=400: define o tamanho da imagem em pixels.
    svg_dados = chess.svg.board(
        board=tabuleiro,
        lastmove=ultimo_lance,
        flipped=invertido,
        size=600
    )
    # cria um caminho de arquivo absoluto para tabuleiro.svg. o que garante
    # que o arquivo seja encontrado, não importa de onde o script é executado.
    caminho_arquivo = os.path.abspath("tabuleiro.svg")
    # abre o arquivo no modo de escrita e salva os dados svg nele.
    with open(caminho_arquivo, "w") as f:
        f.write(svg_dados)
    # usa o módulo webbrowser para abrir o arquivo svg recém-criado no navegador padrão.
    webbrowser.open("file://" + caminho_arquivo)


# --- FUNÇÃO DE AVALIAÇÃO HEURÍSTICA ---
# a informação na busca informada vem daqui. esta seção define como a ia
# avalia o quão boa uma posição é sem ter que explorar o jogo até o final.

# Valores das peças para a heurística
# primeiro, definimos o valor material de cada peça.
pontuacao_peca = {
    chess.PAWN: 100,      # peão
    chess.KNIGHT: 320,    # cavalo
    chess.BISHOP: 330,    # bispo
    chess.ROOK: 500,      # torre
    chess.QUEEN: 900,     # rainha
    chess.KING: 20000     # rei (valor altíssimo para priorizar sua segurança)
}

# (Tabelas de Peça-Casa, ou Piece-Square Tables - PSTs)
# Adaptado de https://www.chessprogramming.org/Simplified_Evaluation_Function
# em seguida, definimos o valor posicional de cada peça.

"""
As tabelas de Peça-Casa (PSTs) são matrizes 8x8 (representadas como arrays de 64 posições) 
que atribuem bônus ou penalidades para cada tipo de peça dependendo de sua posição no tabuleiro.
Valores positivos indicam posições favoráveis e negativos indicam posições desfavoráveis.
Por exemplo, para peões é favorável avançar em direção à promoção e cavalos são mais
efetivos no centro do tabuleiro.
"""

# esta tabela incentiva os peões a avançarem.
pawn_pst = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0]
# esta tabela incentiva os cavalos a ficarem no centro do tabuleiro.
knight_pst = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50]
# esta tabela incentiva os bispos a ocuparem as longas diagonais.
bishop_pst = [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20]
# esta tabela incentiva as torres a ocuparem colunas abertas e a sétima fileira.
rook_pst = [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0]
# esta tabela incentiva a rainha a ter mobilidade, geralmente no centro.
queen_pst = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20]
# esta tabela incentiva o rei a ficar seguro no início do jogo e a se tornar ativo no final.
king_pst = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20]

# este dicionário mapeia o tipo de peça à sua respectiva tabela pst.
mapa_pst = {
    chess.PAWN: pawn_pst, chess.KNIGHT: knight_pst, chess.BISHOP: bishop_pst,
    chess.ROOK: rook_pst, chess.QUEEN: queen_pst, chess.KING: king_pst
}

def avaliar_tabuleiro(tabuleiro):
    """Calcula a pontuação heurística do tabuleiro."""
    # esta função combina valor material e posicional
    # para gerar uma pontuação numérica final para qualquer posição do tabuleiro.
    
    #primeiro, verifica as condições de fim de jogo, que são as mais importantes.
    if tabuleiro.is_checkmate(): # cheque mate ou qualquer visão dele vira prioridade máxima para a IA
        # se é xeque-mate, o resultado é o melhor (ou pior) possível.
        # usamos -99999 se for a nossa vez de jogar (nós levamos mate) e 99999 se for a vez do outro.
        return -99999 if tabuleiro.turn else 99999
    if tabuleiro.is_stalemate() or tabuleiro.is_insufficient_material():
        # se o jogo for um empate (afogado ou material insuficiente), a pontuação é 0.
        return 0
    
    # inicializa a pontuação total.
    pontuacao = 0
    
    # percorre cada uma das 64 casas do tabuleiro usando o iterador da biblioteca.
    for casa in chess.SQUARES:
        # verifica se há uma peça na casa atual.
        peca = tabuleiro.piece_at(casa)
        if peca:
            # se houver uma peça, calcula seu valor.
            # busca o valor material da peça no dicionário.
            valor_material = pontuacao_peca[peca.piece_type]
            # soma à pontuação se a peça for branca, subtrai se for preta.
            pontuacao += valor_material if peca.color == chess.WHITE else -valor_material
            
            # agora, adiciona o bônus posicional.
            # busca a tabela pst correspondente à peça.
            pst = mapa_pst[peca.piece_type]
            # esta linha é crucial. as psts são escritas da perspectiva das brancas.
            # para usá-las para as pretas, o tabuleiro precisa ser espelhado verticalmente.
            # a função `chess.square_mirror` faz exatamente isso.
            indice_casa = casa if peca.color == chess.WHITE else chess.square_mirror(casa)
            # soma o bônus da pst se a peça for branca, subtrai se for preta.
            pontuacao += pst[indice_casa] if peca.color == chess.WHITE else -pst[indice_casa]
            
    # retorna a pontuação final da posição.
    return pontuacao

# análise e implicações da profundidade de busca 
"""
A profundidade de busca (depth) é o parâmetro mais crucial em algoritmos
de busca informada como o Minimax, pois controla a extensão da árvore
de jogadas explorada. As implicações são:

1. Profundidades Baixas (Ex: depth <= 4):
    - Tendência Posicional: O motor baseia-se fortemente na função de avaliação
      (PSTs e valores materiais) nos nós folha da árvore.
    - Limitação Tática: Há pouca ou nenhuma capacidade de "enxergar" sequências
      táticas longas. Sacrifícios de peça ou Peão, que visam ganho material
      ou posicional a médio/longo prazo, são frequentemente rejeitados, pois a
      IA não consegue ver a compensação antes que o limite de profundidade
      seja atingido.
    - Mate Forçado: Sequências de xeque-mate que exijam mais do que 2 lances
      (depth > 4) não serão encontradas.

2. Profundidades Altas (Ex: depth >= 8):
    - Ganho Tático: Aumenta a capacidade de identificar e executar sequências
      de ataque e defesa mais complexas, incluindo sacrifícios forçados e
      ganhos materiais a médio prazo.
    - Mate Forçado: É essencial para resolver quebra-cabeças com Mate em 4 (depth=8)
      ou Mate em 5 (depth=10), pois a IA pode ver a linha forçada até o fim.
    - Custo Computacional: O tempo de cálculo aumenta exponencialmente
      (explosão combinatória), exigindo um bom desempenho da Poda Alpha-Beta, podendo ultrapassar facilmente
      vários minutos, horas, e até dias dependendo da profundidade.
"""

# --- O ALGORITMO DE BUSCA MINIMAX COM PODA ALPHA-BETA ---
# algoritmo que explora as jogadas futuras.

def buscar_sequencia(tabuleiro, profundidade, alfa, beta, eh_jogador_maximizador):
    """Implementa o Minimax com Poda Alpha-Beta."""
    # esta é a função recursiva que implementa o minimax.
    # ela explora a árvore de jogadas futuras para encontrar o melhor lance.
    
    # condição de parada da recursão: se a profundidade máxima foi atingida
    # ou o jogo acabou, para de se aprofundar e retorna a avaliação da posição.
    if profundidade == 0 or tabuleiro.is_game_over():
        return avaliar_tabuleiro(tabuleiro), []

    # inicializa a melhor sequência de lances encontrada até agora.
    melhor_sequencia = []
    # obtém todos os movimentos legais possíveis a partir da posição atual.
    lances_legais = list(tabuleiro.legal_moves)

    # este bloco é executado se for a vez do jogador que quer maximizar a pontuação (brancas).
    if eh_jogador_maximizador:
        # começa assumindo o pior cenário possível e tenta encontrar um melhor.
        avaliacao_max = -float('inf')
        # testa cada movimento legal.
        for lance in lances_legais:
            # faz o lance em uma cópia temporária do tabuleiro.
            tabuleiro.push(lance)
            # chamada recursiva: a ia chama a si mesma para a próxima camada,
            # diminuindo a profundidade e trocando o jogador (agora o minimizador joga).
            avaliacao, sequencia_subsequente = buscar_sequencia(tabuleiro, profundidade - 1, alfa, beta, False)
            # desfaz o lance para poder explorar outras possibilidades.
            tabuleiro.pop()
            
            # se a avaliação deste lance for melhor que a melhor encontrada até agora, atualiza.
            if avaliacao > avaliacao_max:
                avaliacao_max = avaliacao
                melhor_sequencia = [lance] + sequencia_subsequente
            
            # --- parte da poda alpha-beta ---
            # atualiza o alfa, que é o melhor resultado que o maximizador pode garantir.
            alfa = max(alfa, avaliacao)
            # a poda acontece aqui. se o beta (melhor do minimizador) for menor ou igual ao alfa,
            # significa que o minimizador já tem uma resposta melhor em outro ramo da árvore,
            # então não há necessidade de continuar explorando este ramo.
            if beta <= alfa:
                break
        return avaliacao_max, melhor_sequencia
    # este bloco é executado se for a vez do jogador que quer minimizar a pontuação (pretas).
    else:
        # a lógica é a imagem espelhada do maximizador, mas tentando minimizar a pontuação.
        avaliacao_min = float('inf')
        for lance in lances_legais:
            tabuleiro.push(lance)
            avaliacao, sequencia_subsequente = buscar_sequencia(tabuleiro, profundidade - 1, alfa, beta, True)
            tabuleiro.pop()
            
            if avaliacao < avaliacao_min:
                avaliacao_min = avaliacao
                melhor_sequencia = [lance] + sequencia_subsequente
            
            # atualiza o beta, que é o melhor resultado que o minimizador pode garantir.
            beta = min(beta, avaliacao)
            # a mesma lógica de poda, mas da perspectiva do minimizador.
            if beta <= alfa:
                break
        return avaliacao_min, melhor_sequencia

def encontrar_melhor_lance(tabuleiro, profundidade):
    """Inicia a busca Minimax e retorna o melhor lance."""
    # esta é uma função de conveniência que inicia todo o processo de busca.
    # ela determina se o jogador atual é o maximizador ou minimizador.
    eh_maximizador = tabuleiro.turn == chess.WHITE
    # chama a função de busca e retorna apenas o primeiro lance da melhor sequência encontrada.
    _, melhor_sequencia = buscar_sequencia(tabuleiro, profundidade, -float('inf'), float('inf'), eh_maximizador)
    return melhor_sequencia[0] if melhor_sequencia else None

# --- ORQUESTRAÇÃO E EXECUÇÃO ---
# este é o ponto de entrada do programa que controla a interação com o usuário.
if __name__ == "__main__":
    
    # Lista de quebra-cabeças pré-definidos
    # Para a geração dos códigos fen's, foi utilizado o modo de edição de tabuleiro do lichess.
    # Link para o modo edição de tabuleiro do lichess: https://lichess.org/editor.

    """
Sistema interativo que permite testar o motor de xadrez em dois modos:
1. Resolução de quebra-cabeças pré-definidos: Verifica se a IA encontra
   a sequência correta de lances para posições táticas específicas
2. Partida completa IA vs IA: Permite que duas instâncias do mesmo algoritmo
   joguem uma contra a outra, demonstrando a capacidade de jogo em situações
   gerais

Cada quebra-cabeça é definido por:
- FEN: Notação que descreve a posição inicial
- Descrição: Explicação do objetivo tático
- Profundidade de busca: Número de lances que a IA deve calcular
- Sequência de solução: Lista de lances em notação UCI que resolvem o problema
"""
    # estrutura de dados que armazena os quebra-cabeças
    quebra_cabecas = [
        {
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1",
            "description": "Mate do Pastor (Scholar's Mate).",
            "search_depth": 4,
            "solution_moves_uci": ["f1c4","g8f6","h5f7"]
        },
        {
            "fen": "3q1rk1/p4ppp/1p2p3/2n5/3N4/1P2P3/P4PPP/2RQ2K1 w - - 3 20",
            "description": "Sacrifício de cavalo para dar mate forçado.",
            "search_depth": 4,
            "solution_moves_uci": ["d4c6", "d8d1","c1d1","f8a8", "c6a7", "a8a7", "d1d8"]
        },
    ]
    # Interface do terminal para escolher como testar o algoritmo
    # este é o loop principal do menu interativo.
    while True:
        print("\n--- Solucionador de Problemas e Motor de Xadrez com IA ---")
        print("Escolha um modo:")
        
        # exibe as opções de quebra-cabeças.
        for i, quebra in enumerate(quebra_cabecas):
            print(f"  {i+1}: Resolver Quebra-Cabeça - {quebra['description']}")
        
        #opção de jogar uma partida completa.
        numero_modo_jogo = len(quebra_cabecas) + 1
        print(f"  {numero_modo_jogo}: Jogar uma partida completa (IA vs. IA)")
        print("  0: Sair")

        try:
            # obtém a escolha do usuário.
            escolha_str = input("\nDigite o número da sua escolha: ")
            escolha = int(escolha_str)

            if escolha == 0:
                print("Encerrando o programa. Até a próxima!")
                break # sai do loop se o usuário digitar 0.
            
            # Lógica da I.a contra i.a
            if escolha == numero_modo_jogo:
                tabuleiro = chess.Board() # começa um novo jogo a partir da posição inicial.
                profundidade_jogo = 4 # profundidade fixa da partida
                
                print("\n" + "="*60)
                print(f"--- Iniciando de IA (Brancas) vs. IA (Pretas) ---")
                print(f"Profundidade de busca: {profundidade_jogo}")
                print("="*60)
                
                mostrar_tabuleiro(tabuleiro)
                
                # continua jogando enquanto o jogo não acabar.
                while not tabuleiro.is_game_over():
                    cor_jogador = "Brancas" if tabuleiro.turn == chess.WHITE else "Pretas"
                    print(f"\nTurno das {cor_jogador}. IA pensando...")
                    
                    # encontra o melhor lance para o jogador atual.
                    lance_ia = encontrar_melhor_lance(tabuleiro, profundidade_jogo)
                    
                    if lance_ia:
                        # se um lance for encontrado, executa-o.
                        print(f"IA joga: {tabuleiro.san(lance_ia)}")
                        tabuleiro.push(lance_ia)
                        mostrar_tabuleiro(tabuleiro, ultimo_lance=lance_ia, invertido=False)
                        time.sleep(2) # pausa para visualização.
                    else:
                        print("Fim de jogo: IA não encontrou lances.")
                        break
                
                print("\n" + "="*60)
                print(f"FIM DE JOGO! Resultado: {tabuleiro.result()}")
                print("="*60)
                input("\nPressione Enter para voltar ao menu...")
            
            # Resolução das quests
            # lógica para resolver os quebra-cabeças.
            elif 1 <= escolha <= len(quebra_cabecas):
                quebra = quebra_cabecas[escolha - 1]
                lances_solucao_uci = quebra["solution_moves_uci"]
                
                # carrega o tabuleiro a partir da string fen do quebra-cabeça.
                tabuleiro = chess.Board(quebra["fen"])
                profundidade = quebra["search_depth"]
                
                print(f"\n" + "="*60)
                print(f"--- Resolvendo: {quebra['description']} ---")
                
                # inverte o tabuleiro se for a vez das pretas jogarem.
                deve_inverter = (tabuleiro.turn == chess.BLACK)
                mostrar_tabuleiro(tabuleiro, invertido=deve_inverter)
                print("Tabuleiro inicial aberto no navegador.")

                puzzle_resolvido = True
                indice_lance_atual = 0

                # loop que executa a sequência de lances do quebra-cabeça.
                while not tabuleiro.is_game_over() and indice_lance_atual < len(lances_solucao_uci):
                    cor_ia_str = "Brancas" if tabuleiro.turn == chess.WHITE else "Pretas"
                    # converte a string do lance da solução (formato uci) para um objeto de lance.
                    lance_correto = chess.Move.from_uci(lances_solucao_uci[indice_lance_atual])
                    
                    # Checa se é o turno da IA (que inicia o quebra-cabeça)
                    #a string fen contém um w ou b para indicar de quem é a vez.
                    eh_turno_ia = (tabuleiro.turn == chess.WHITE and quebra['fen'].split()[1] == 'w') or \
                                   (tabuleiro.turn == chess.BLACK and quebra['fen'].split()[1] == 'b')
                    
                    if eh_turno_ia:
                        # se for o turno da ia, pede para ela encontrar o melhor lance.
                        print(f"\nTurno da IA ({cor_ia_str}). Pensando...")
                        tempo_inicio = time.time()
                        lance_ia = encontrar_melhor_lance(tabuleiro, profundidade)
                        tempo_fim = time.time()
                        print(f"Análise concluída em {tempo_fim - tempo_inicio:.2f} segundos.")

                        # compara o lance da ia com a solução correta.
                        if lance_ia == lance_correto:
                            print(f"IA encontrou o lance correto: {tabuleiro.san(lance_ia)}")
                            tabuleiro.push(lance_ia)
                            mostrar_tabuleiro(tabuleiro, ultimo_lance=lance_ia, invertido=deve_inverter)
                        else:
                            # se a ia errar, o quebra-cabeça falhou.
                            print(f"IA falhou! Lance esperado: {tabuleiro.san(lance_correto)}, Lance da IA: {tabuleiro.san(lance_ia) if lance_ia else 'Nenhum'}")
                            puzzle_resolvido = False
                            break
                    else: # Turno do oponente (resposta pré-setada)
                        # se não for o turno da ia, o lance é feito automaticamente a partir da solução.
                        print(f"\nOponente responde com: {tabuleiro.san(lance_correto)}")
                        tabuleiro.push(lance_correto)
                        time.sleep(1.5)
                        mostrar_tabuleiro(tabuleiro, ultimo_lance=lance_correto, invertido=deve_inverter)

                    # avança para o próximo lance na sequência da solução.
                    indice_lance_atual += 1

                print("\n" + "-"*20)
                if puzzle_resolvido:
                    print("Quebra-cabeça resolvido com sucesso!")
                else:
                    print("A IA não conseguiu resolver o quebra-cabeça.")
                
                input("\nPressione Enter para voltar ao menu...")

            else:
                print(f"Opção inválida.")

        except (ValueError, IndexError):
            print("Entrada inválida. Por favor, digite um número válido.")