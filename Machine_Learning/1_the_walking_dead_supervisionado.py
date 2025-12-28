import pandas as pd # importante para organizar os dados em dataframe ao invés de usar listas complexas, evita uso de loops para acesso e manipulação de dados
import numpy as np # manipulação de vetores
from sklearn.ensemble import GradientBoostingClassifier #importa algoritmo de aprendizado do gradientboosting para classificação
from sklearn.model_selection import train_test_split # Função que separa o dataset em dois subconjuntos, um de treino outro de teste, pra não usar os mesmos dados em ambos
from sklearn.metrics import accuracy_score # importa métrica de avaliação para saber a acurácia do modelo ao comparar com os dados de teste
import matplotlib.pyplot as plt # função gráfica
import seaborn as sns # função gráfica auxiliar ao matplot
# classification_report gera métricas detalhadas como precisão, recall, F1-score
# para avaliar a qualidade do modelo além da acurácia.

# confusion_matrix mostra a quantidade de acertos e erros do modelo divididos
# em categorias, o que permite analisar
# como o modelo confunde as classes.

# roc_auc_score calcula a AUC (Área sob a curva ROC), que mede a capacidade
# do modelo distinguir corretamente entre as classes. Quanto mais próximo de 1,
# melhor o modelo separa aceitos e rejeitados.

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# permutation_importance é um método mais confiável de explicar o modelo,
# pois mede o impacto real de cada variável no desempenho.
# Ele embaralha cada coluna individualmente no conjunto de teste
# e verifica quanto a performance do modelo cai.
# Quanto maior a queda, mais importante era aquela variável.
from sklearn.inspection import permutation_importance

""" 
Tema e Contextualização:

O projeto simula decisões de sobrevivência inspiradas no universo de The Walking Dead.
A história se passa após um colapso causado por um vírus que transforma humanos em zumbis.
Os sobreviventes precisam avaliar rapidamente quem pode entrar no grupo, já que recursos são escassos
e pessoas perigosas representam risco. Profissões, sanidade e lealdade tornam-se fatores essenciais
para decidir se alguém contribui para o grupo ou ameaça sua segurança.
Para mais informações da temática: https://walkingdead.fandom.com/wiki/The_Walking_Dead_(TV_Series)

Diferença Fundamental do GradientBoosting para o Random Forest:

É fundamental explicar a diferença entre gradientboost e random forest, enquanto o random forest cria árvores
paralelamente, o gradientboost cria essas árvores de forma sequencial, ou seja, ambos utilizam árvores de decisão,
e o que muda na prática é o seu comportamento de gerar as árvores.

"""

# simulação de dados de 2000 sobreviventes 
np.random.seed(37) #só remover o parâmetro se quiser deixar estocástico a cada execução 
n_amostras = 2000
sanidade = np.random.randint(0,11,n_amostras)


# Coluna das profissões:

profissoes = np.random.choice( ['Medico', 'Enfermeiro','Policial', 'Mecanico', 'Professor', 'Cacador', 'Fazendeiro', 'Jardineiro', 'Engenheiro', 'Carpinteiro', 'Desempregado', 'Militar', 'Cozinheiro', 'Quimico', 'Farmaceutico'], n_amostras)

data = {
    # quantos zumbis a pessoa eliminou
    'zumbis_mortos': np.random.randint(0, 800, n_amostras),
    
    # quantos humanos vivos a pessoa eliminou
    'humanos_mortos': np.random.randint(0, 40, n_amostras),
    
    # lealdade ao grupo anterior, indo de 0 à 100
    'nivel_de_lealdade': np.random.randint(0, 101, n_amostras),
    
    # habilidade de busca de suprimentos, vai de 0 à 10
    'sabe_buscar': np.random.randint(0, 11, n_amostras),
    
    # sanidade mental, que vai de 0 à 10
    'nivel_de_sanidade': np.random.randint(0, 11, n_amostras),

    'profissao': profissoes

}

# criação do dataframe com informações do dicionário data
df = pd.DataFrame(data)

# Definindo o target pra treinar o modelo
# A lógica se baseia em o grupo aceitar quem é útil e possui sanidade, 
# mas rejeita quem matou muitos humanos,
# a menos que a lealdade seja extrema.
# a função define como cada sobrevivente deve ser avaliado no universo de The Walking Dead,
# considerando profissão, sanidade, lealdade, habilidades e histórico de violência.
# O objetivo é transformar esses atributos em uma lógica de aceitação ou rejeição,
# criando um valor que o modelo supervisionado irá aprender a reproduzir.

def classificacao_rick(linha):
    score = (linha['sabe_buscar'] * 1.5)
    prof = linha['profissao']
    sanidade = linha['nivel_de_sanidade']
    lealdade = linha['nivel_de_lealdade']

    
    # Lógica das profissões da saúde 
    # em geral, não se procuram médicos loucos, mas sim os que estão são
    if prof in ['Medico', 'Enfermeiro', 'Farmaceutico', 'Quimico']:
        if sanidade >= 7:
            score+=25
        elif sanidade <4:
            score -=50
        else:
            score +=5
    #Lógica das profissões de combate
    # é mais aturável estarem malucos do que os da área da saúde
    elif prof in ['Militar', 'Policial', 'Cacador']:
        score += 15
        if sanidade < 3:
            score -= 10
        else:
            score += sanidade * 0.5
    # Lógica dos técnicos
    # a sanidade média ou alta também é primordial pra essa área, e no geral, fazem um péssimo trabalho caso contrário
    elif prof in ['Engenheiro', 'Mecanico', 'Carpinteiro']:
        score +=10
        if sanidade > 5:
            score += sanidade * 1.5
        else:
            score -= 5
    elif prof in ['Fazendeiro', 'Jardineiro', 'Cozinheiro']:
        score += 12 # Base boa, comida é essencial
        
        # Lógica do jardineiro, que pode ajudar o químico ou o farmacêutico
        if prof == 'Jardineiro':
            # Se sabe buscar bem, acha ervas medicinais 
            if linha['sabe_buscar'] > 7:
                score += 10 
            # Se a sanidade for baixa, cuida mal das plantas e pode até destruir a plantação
            if sanidade < 4:
                score -= 10
        
        # Lógica do cozinheiro 
        elif prof == 'Cozinheiro':
            # cozinheiro precisa ser leal, um cozinheiro desleal poderia efetuar um envenenamento coletivo
            if lealdade > 80:
                score += 15
            elif lealdade < 40:
                score -= 20 
        
        # Lógica do fazendeiro
        elif prof == 'Fazendeiro':
            # Precisa de sanidade para colheitas de longo prazo
            if sanidade > 6:
                score += 10
            else:
                score -= 5
    # Professor ajuda a manter a sanidade do grupo se ele próprio for são
    elif prof == 'Professor':
        if sanidade > 8:
            score += 15 
        else:
            score += 2
    #Lógica do desempregado
    # o desempregado deve provar que é valioso, já que n possui vantagens imediatas, portanto depende dos atributos
    elif prof == 'Desempregado':
        score -= 5 
    

    # penalidades gerais
    if linha['humanos_mortos'] > 5:
        if prof in ['Militar', 'Policial', 'Cacador']:
            score -= 5 
        else:
            score -= 20 
            
    # Bônus Geral de Lealdade
    if linha['nivel_de_lealdade'] > 90:
        score += 10
        
    return 1 if score > 20 else 0


df['aceito'] = df.apply(classificacao_rick, axis=1)
print("Distribuição de classes (aceito):")
print(df['aceito'].value_counts())
print(df['aceito'].value_counts(normalize=True))
df = pd.get_dummies(df, columns=['profissao'])

# preparação / treinamento

X = df.drop('aceito', axis=1)
y = df['aceito']

# Separar dados de treino para o modelo aprender, e teste para validar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37, stratify=y)  # test_size =0.2 significa 20% dedicado à testes, portanto 80% é pra treino
# stratify=y garante que a proporção das classes seja a mesma
# tanto no conjunto de treino quanto no conjunto de teste.
# isso evita que o modelo aprenda com dados desbalanceados sem perceber
# e impede um cenário onde o treino teria, por exemplo, muito mais aceitos que o teste.


# Em vez de utilizar floresta aleatória, que é citado no slide 39 da aula 18, é usado o GradientBoosting
# learning_rate=0.1: o modelo aprende devagar com os erros pra não decorar os dados, já que se o overfitting for muito alto, o sistema será enviesado facilmente
# n_estimators=100: faz 100 rodadas de correção de erros
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=37) #max_depth = 3 define a profundidade máxima de 3 níves por árvore

model.fit(X_train, y_train)

#validação e interpretação

# Testar a precisão
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print(f"Precisão do Protocolo de Segurança: {acc*100:.2f}%")
print("-" * 30)
print("Relatório de Classificação:")
print(classification_report(y_test, predictions))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, predictions))

if hasattr(model, "predict_proba"):
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"AUC: {auc:.4f}")

print("-" * 30)


# mostrar o que o modelo considerou mais importante
importancia_da_variavel = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Fatores decisivos para a sobrevivência:")
print(importancia_da_variavel)
print("-" * 30)
resultado = permutation_importance(
    model, X_test, y_test,
    n_repeats=20,
    random_state=37,
    n_jobs=-1
)

importancia_da_permutacao = pd.Series(resultado.importances_mean, index=X.columns).sort_values(ascending=False)
print("Média das permutações:")
print(importancia_da_permutacao)
print("-" * 30)

# função para testar os sobreviventes
def avaliar_sobrevivente(nome, profissao, zumbis, humanos, lealdade, busca, sanidade):
    # cria dicionário com dados numéricos
    dados_input = {
        'zumbis_mortos': [zumbis],
        'humanos_mortos': [humanos],
        'nivel_de_lealdade': [lealdade],
        'sabe_buscar': [busca],
        'nivel_de_sanidade': [sanidade]
    }
    
    # criação de um novo dataframe para garantir que a entrada do teste tenha o mesmo shape do treino
    novo_df = pd.DataFrame(dados_input)
    
    # essa parte irá adicionar as colunas de profissão zeradas
    # o modelo espera colunas como 'profissao_Medico', por exemplo
    # é pego todas as colunas que o modelo conhece através do X.columns
    # cria uma lista vazia para guardar as colunas que faltam
    colunas_modelo = []

# passa coluna por coluna que o modelo conhece 
    for col in X.columns:
    # se a coluna não estiver nos dados do sobrevivente
        if col not in dados_input:
            colunas_modelo.append(col) # adiciona na lista para preencher com 0 depois
    
    for col in colunas_modelo:
        novo_df[col] = 0 # Começa tudo com 0 
        
    # marca 1 apenas na profissão certa
    coluna_alvo = f"profissao_{profissao}"
    if coluna_alvo in novo_df.columns:
        novo_df[coluna_alvo] = 1
    else:
        print(f"AVISO: A profissão '{profissao}' não existe no treinamento. Será tratada como 'Outros'.")
    
    # garante a ordem exata das colunas 
    novo_df = novo_df[X.columns]
    
    # Previsão
    resultado = model.predict(novo_df)[0]
    prob = model.predict_proba(novo_df)[0][1]
    
    status = "APROVADO" if resultado == 1 else "REJEITADO"
    print(f"Sobrevivente: {nome} ({profissao})")
    print(f"Decisão: {status} (Confiança: {prob:.1%})")
    print("-" * 20)

# Casos de teste 
avaliar_sobrevivente("Carol Peletier", "Cozinheiro", zumbis=500, humanos=15, lealdade=100, busca=10, sanidade=7)

avaliar_sobrevivente("O Governador", "Policial", zumbis=200, humanos=30, lealdade=10, busca=8, sanidade=4)

avaliar_sobrevivente("Eugene (S4)", "Desempregado", zumbis=0, humanos=0, lealdade=50, busca=2, sanidade=5)

avaliar_sobrevivente("Sheyla", "Jardineiro", zumbis=7, humanos=0, lealdade=80, busca=9, sanidade=8)


avaliar_sobrevivente("Cristina", "Jardineiro", zumbis=10, humanos=0, lealdade=80, busca=3, sanidade=8)


avaliar_sobrevivente("Walter White", "Quimico", zumbis=5, humanos=5, lealdade=50, busca=10, sanidade=2) # referência à breaking bad

# Verificando se houve overfitting (alta variância)
score_treino = model.score(X_train, y_train)
score_teste = model.score(X_test, y_test)

print(f"Acerto no Treino: {score_treino:.2%}")
print(f"Acerto no Teste: {score_teste:.2%}")
# Se os números forem próximos, quer dizer que a variância está controlada

# plota gráfico das variáveis que mais influenciaram o modelo
grafico = pd.Series(model.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(10, 8))
sns.barplot(x=grafico, y=grafico.index)
plt.title("Importância das variáveis do Gradient Boosting")
plt.xlabel("Importância")
plt.ylabel("Atributo")
plt.tight_layout()
plt.show()
grafico_da_permutacao = importancia_da_permutacao.sort_values()

plt.figure(figsize=(10, 8))
sns.barplot(x=grafico_da_permutacao, y=grafico_da_permutacao.index)
plt.title("Importância das permutações do Gradient Boosting")
plt.xlabel("Importância média da permutação")
plt.ylabel("Atributo")
plt.tight_layout()
plt.show()