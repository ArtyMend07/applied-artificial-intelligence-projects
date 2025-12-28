# ETAPA 1: DEFINIÇÃO DAS ESTRUTURAS LÓGICAS

# Para implementar a Lógica de Primeira Ordem em Python, primeiro definimos
# estruturas de dados para representar nossos fatos, regras e variáveis.

class Predicado:
    """ Representa uma afirmação lógica, como por exemplo: viu(ciano, corpo, eletrica) """
    def __init__(self, nome, args):
        self.nome =nome
        self.args =tuple(args)

    def __repr__(self):
        return f"{self.nome}{self.args}"

class Regra:
    """ Representa uma regra de inferência, como 'impostor(X) :- suspeito(X)' """
    def __init__( self, consequente,antecedentes):
        self.consequente = consequente #a cabeça da regra (o que se conclui)

        self.antecedentes = antecedentes #o corpo da regra (as condições)

class Negado:
    """
    Esta classe é uma forma de representar a negação de um predicado.
    É a nossa maneira de dizer "não isto", obrigatório para regras
    que dependem da ausência de uma evidência.
    """
    def __init__(self, predicado):
        self.predicado = predicado
    
    def __repr__(self):
        return f"nao({self.predicado})"

# ETAPA 2: BANCO DE CONHECIMENTOS 

# Esta classe irá armazenar os fatos e as regras, e também conterá
# o mecanismo de inferência que raciocina sobre eles.

class BaseDeConhecimento:
    """
    Armazena os fatos e regras, e implementa o mecanismo de inferência.
    """
    def __init__(self):
        # aqui guardamos tudo que o agente sabe. os fatos são as evidências diretas,
        # enquanto as regras são o conhecimento geral sobre como o mundo funciona.
        self.fatos= []
        self.regras= []
        # este contador é um detalhe técnico para garantir que as variáveis de uma regra
        # não se confundam quando a usamos várias vezes na mesma cadeia de raciocínio.
        self.contador_regras = 0

    def tell(self, item):
        # A operação tell adiciona um novo fato ou regra à base de conhecimento.
        if isinstance(item, Regra):
            self.regras.append(item)
        else: # é um predicado
            self.fatos.append(item)

    def ask(self, consulta):
        # A operação ask inicia o processo de inferência para responder a uma consulta
        # ela retorna todas as substituições de variáveis que tornam a consulta verdadeira.
        
        # O algoritmo de inferência utilizado é o backward chaining.
        # ele começa com o objetivo (a consulta) e trabalha para trás,
        # procurando por fatos e regras que possam provar esse objetivo.
        
        gerador_solucoes = self._encadeamento_para_tras(consulta, {})
        
        # após rodar o algoritmo, as respostas podem conter variáveis internas
        # (exemplo: X1, LOCAL2) e duplicatas. esta parte do código limpa as respostas,
        # mostrando apenas as variáveis que o usuário pediu na consulta original.
        solucoes_finais = []
        for solucao in gerador_solucoes:
            solucao_limpa = self._limpar_solucao(consulta, solucao)
            if solucao_limpa and solucao_limpa not in solucoes_finais:
                 solucoes_finais.append(solucao_limpa)
        return solucoes_finais

    def _limpar_solucao(self, consulta, substituicoes):
        # esta função pega o dicionário de solução completo, que pode conter muitas
        # variáveis internas, e o filtra, retornando apenas os valores para as
        # variáveis que estavam na pergunta original do usuário.
        solucao_final = {}
        for var in consulta.args:
            if isinstance(var, str) and var.isupper():
                valor_final = self._substituir(var, substituicoes)
                solucao_final[var] = valor_final
        return solucao_final

    def _unificar(self, p1, p2, substituicoes):
        # Esta função compara dois predicados para ver se eles podem ser unificados
        #   com as substituições de variáveis atuais.
        
        if p1.nome!= p2.nome or len(p1.args) != len(p2.args):
            return None # Não unificam se os nomes ou o número de argumentos forem diferentes.

        novas_substituicoes = substituicoes.copy()
        for arg1, arg2 in zip(p1.args, p2.args):
            # Percorre os argumentos para encontrar substituições de variáveis.
            arg1 = self._substituir(arg1, novas_substituicoes)
            arg2 = self._substituir(arg2, novas_substituicoes)

            if arg1 =='_' or arg2 == '_':
                continue # O underscore '_' é um coringa, então ele unifica com qualquer coisa.

            if isinstance(arg1, str) and arg1.isupper(): # arg1 é uma variável
                novas_substituicoes[arg1] = arg2
            elif isinstance(arg2, str ) and arg2.isupper(): # arg2 é uma variável
                novas_substituicoes[arg2] = arg1
            elif arg1 != arg2:
                return None # em caso de dois valores concretos diferentes.
        
        return novas_substituicoes

    def _substituir( self, termo, substituicoes):
        # Aplica as substituições de variáveis a um termo.
        # se o termo for um predicado inteiro, ele entra recursivamente e substitui
        # as variáveis dentro dos argumentos.
        if isinstance(termo, (Predicado, Negado)):
            if isinstance(termo, Negado):
                predicado_substituido = self._substituir(termo.predicado, substituicoes)

                return Negado(predicado_substituido)
            else: # é um Predicado
                novos_args = [self._substituir(arg, substituicoes) for arg in termo.args]
                return Predicado(termo.nome, novos_args)
        
        # se o termo for uma variável, busca seu valor final no dicionário de substituições.
        while isinstance(termo, str) and termo.isupper() and termo in substituicoes:
            termo = substituicoes[termo]
        return termo
    
    def _renomear_variaveis(self, termo, sufixo):
        # esta função é um mecanismo de segurança essencial. para evitar que a variável X de uma
        # regra qualquer se confunda com a X de outra regra na mesma busca, nós damos a elas um
        # sobrenome único. assim, X vira X1, X2, e assim sucessivamente...
        if isinstance(termo, (Predicado, Negado)):
            if isinstance(termo, Negado):
                return Negado(self._renomear_variaveis(termo.predicado, sufixo))
            else:
                novos_args = [self._renomear_variaveis(arg, sufixo) for arg in termo.args]
                return Predicado(termo.nome, novos_args)
        elif isinstance(termo, str) and termo.isupper():
            return termo + str(sufixo)
        else:
            return termo

    def _encadeamento_para_tras(self, consulta, substituicoes):
        # Esta é a função recursiva que implementa o algoritmo de encadeamento para trás.
        
        # aqui tratamos a negação como Falha. para provar nao(P), o motor tenta
        # provar P. se ele não encontrar qualquer evidência,
        # ele conclui que nao(P) é verdadeiro.

        if isinstance(consulta, Negado):
            solucoes_internas = self._encadeamento_para_tras(consulta.predicado, substituicoes)
            if not any(solucoes_internas):
                yield substituicoes
            return

        # tenta provar a consulta usando os fatos existentes
        for fato in self.fatos:
            subs_unificadas = self._unificar(consulta, fato, substituicoes)
            if subs_unificadas is not None:
                yield subs_unificadas # retorna a solução

        # tenta provar a consulta usando as regras
        for regra in self.regras:
            self.contador_regras += 1
            regra_renomeada_consequente = self._renomear_variaveis(regra.consequente, self.contador_regras)
            regra_renomeada_antecedentes = [self._renomear_variaveis(a, self.contador_regras) for a in regra.antecedentes]

            subs_unificadas = self._unificar(consulta, regra_renomeada_consequente, substituicoes)

            if subs_unificadas is not None:
                # se uma regra corresponde, tentamos provar cada uma de suas condições (antecedentes).
                solucoes_antecedentes = self._provar_antecedentes(regra_renomeada_antecedentes, subs_unificadas)
                for solucao in solucoes_antecedentes:
                    yield solucao

    def _provar_antecedentes(self, antecedentes, substituicoes):
        # função auxiliar que prova uma lista de condições em sequência.

        if not antecedentes:
            yield substituicoes # se não há mais condições para provar, tivemos sucesso.
            return

        primeiro_antecedente = antecedentes[0]
        restante = antecedentes[1:]
        
        # prova a primeira condição...
        solucoes_primeiro = self._encadeamento_para_tras(primeiro_antecedente, substituicoes)
        for solucao in solucoes_primeiro:
            # ... e para cada solução encontrada, aplica o conhecimento adquirido
            # no restante das condições antes de continuar. é aqui que o
            # conhecimento (exemplo: LOCAL=cafeteria) é propagado para o próximo passo.
            restante_substituido = [self._substituir(p, solucao) for p in restante]
            
            solucoes_restante = self._provar_antecedentes(restante_substituido, solucao)
            for solucao_final in solucoes_restante:
                yield solucao_final

# ETAPA 3: O AGENTE DE IA E A SIMULAÇÃO

class AgenteDetetive:
    """ Representa o agente que interage com a Base de Conhecimento. """
    def __init__(self):
        self.kb = BaseDeConhecimento()

    def tell(self, item):
        print(f"[AGENTE TELL]: Adicionando à KB: {item}")
        self.kb.tell(item)

    def ask(self, consulta):
        print(f"[AGENTE ASK]: Consultando a KB: {consulta}")
        respostas = self.kb.ask(consulta)
        print(f"[KB RESPONDE]: {respostas if respostas else 'Nenhuma conclusão.'}")
        return respostas

if __name__ == "__main__":
    
    print("-" * 30)
    
    agente = AgenteDetetive()
    
    # 1. tell: O agente aprende as regras do jogo.
    # Variáveis são representadas por letras maiúsculas (X, Y, LOCAL).
    agente.tell(Regra(
        Predicado('inocente', ['X']), 
        [Predicado('fez_task_visual', ['X', '_'])]
    ))
    agente.tell(Regra(
        Predicado('suspeito', ['X']),
        [Predicado('viu', ['X', 'corpo', 'LOCAL']), 
         Predicado('corpo_em', ['_', 'LOCAL']),
         # a regra agora inclui a condição crucial de que o jogador não pode ter reportado.
         Negado(Predicado('reportou', ['X', 'corpo', 'LOCAL']))
        ]
    ))
    agente.tell(Regra(
        Predicado('impostor', ['Y']), 
        [Predicado('suspeito', ['Y']), Negado(Predicado('inocente', ['Y']))]
    ))

    # 2. INÍCIO DA RODADA - PERCEPÇÕES 
    print("\n--- A rodada começa. O agente observa os eventos. ---\n")
    agente.tell(Predicado('fez_task_visual', ['amarelo', 'medbay_scan']))
    
    # 3. PRIMEIRO RACIOCÍNIO
    print("\n--- O agente processa suas observações. ---\n")
    agente.ask(Predicado('inocente', ['amarelo']))
    
    # 4. UM CORPO É ENCONTRADO
    print("\n--- Alerta! Um corpo foi encontrado na cafeteria. ---\n")
    agente.tell(Predicado('corpo_em', ['azul', 'cafeteria']))
    agente.tell(Predicado('viu', ['vermelho', 'corpo', 'cafeteria']))
    # Ciano reportou, então adicionamos esse fato para que ele não se torne suspeito.
    agente.tell(Predicado('reportou', ['ciano', 'corpo', 'cafeteria']))
    # A ausência do fato 'reportou(vermelho,...)' é a chave para a inferência.

    # 5. RACIOCÍNIO FINAL
    print("\n--- Reunião de emergência. O agente analisa as evidências. ---\n")
    
    # Pergunta 1: Quem é suspeito?
    suspeitos = agente.ask(Predicado('suspeito', ['QUEM']))

    # Pergunta 2: Quem é o impostor?
    impostores = agente.ask(Predicado('impostor', ['ASSASSINO']))

    print("-" * 30)
    if impostores:
        # a resposta vem como um dicionário de substituições, como {'ASSASSINO': 'vermelho'}.
        # extraímos o valor da variável que pedimos.
        impostor_nome = impostores[0]['ASSASSINO']
        print(f"Conclusão final do agente: O impostor é {impostor_nome}.")
    else:
        print("Conclusão final do agente: Não há evidências suficientes para apontar um impostor.")