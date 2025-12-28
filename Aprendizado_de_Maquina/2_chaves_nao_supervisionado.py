import pandas as pd # importante para organizar os dados em dataframe ao invés de usar listas complexas, evita uso de loops para acesso e manipulação de dados
import matplotlib.pyplot as plt # função gráfica
import seaborn as sns #função gráfica auxiliar ao matplot, pois permite criação de cores e tamanhos variáveis com poucos comandos
from sklearn.preprocessing import MinMaxScaler #Importantíssima para colocar variáveis de grandezas diferentes em uma mesma escala de 0 à 1, como por exemplo idade e renda 
from sklearn.mixture import GaussianMixture # Classe que contém o modelo da mistura gaussiana, será usada pra treinar

""" 
Tema e Contextualização:

O projeto utiliza elementos narrativos do universo de Chaves para construir um cenário onde os personagens apresentam características variadas
que influenciam sua organização social. A série se passa em uma vila simples, marcada por fortes diferenças de idade, 
renda e comportamento entre os moradores. Apesar do tom humorístico da obra original, 
suas relações apresentam estruturas reconhecíveis do ponto de vista analítico, como agrupamentos sociais formados por afinidades,
condições econômicas, rotinas e dinâmicas familiares.

Entre os personagens centrais, observa-se uma diversidade que contribui diretamente para a formação desses agrupamentos.
Chaves representa o extremo da vulnerabilidade econômica, vivendo com recursos praticamente inexistentes. 
Quico e Dona Florinda compõem um núcleo com melhores condições financeiras, caracterizado por comportamentos que reforçam distinções sociais dentro da vila.
Seu Madruga e Chiquinha formam outro agrupamento, marcado pela limitação econômica combinada a laços sociais fortes e presença constante no cotidiano da vizinhança.
Personagens como Nhonho e o Sr. Barriga adicionam uma camada de contraste, representando maior poder aquisitivo e influência sobre o funcionamento do local.
Outros, como Popis, Paty, Jaiminho ou Dona Clotilde, ampliam a diversidade etária e comportamental, introduzindo nuances que enriquecem o conjunto de possíveis relações.

Nesse contexto, cada personagem representa um conjunto distinto de atributos que podem ser formalizados em variáveis numéricas,
permitindo a observação de padrões de semelhança entre indivíduos. A Vila funciona como um microambiente onde diferenças de renda,
idade e hábitos moldam subconjuntos sociais, ainda que esses grupos não sejam explicitamente declarados na narrativa.
A adaptação para o aprendizado não supervisionado consiste em transformar essas características em dimensões mensuráveis que
tornam possível a identificação automática de grupos internos, preservando a essência do universo enquanto direciona o foco para a análise estatística.

Para mais informações da temática: https://pt.wikipedia.org/wiki/El_Chavo_del_Ocho

"""

# Definição do dataset com variáveis de Renda e Peso para criar complexidade tridimensional
# A renda possui alta variância e o peso auxilia na distinção de clusters específicos
dados = {
    'Personagem': [
        'Chaves', 'Quico', 'Chiquinha', 'Nhonho', 'Godinez', 'Popis', 'Paty',
        'Seu Madruga', 'Dona Florinda', 'Prof. Girafales', 'Dona Clotilde', 
        'Sr. Barriga', 'Jaiminho', 'Dona Neves', 'Glória', 'Seu Furtado'
    ],
    'Idade':    [8, 9, 8, 8, 8, 8, 8, 52, 39, 45, 71, 48, 75, 70, 30, 40],
    'Renda':    [0, 500, 50, 2000, 50, 400, 300, 20, 400, 800, 100, 5000, 200, 20, 300, 10],
    'Peso':  [45, 50, 40, 90, 45, 42, 40, 55, 60, 80, 50, 110, 75, 50, 58, 65]
}

# criação do dataframe com o dicionário de dados
df = pd.DataFrame(dados)

# Seleção das variáveis numéricas que servirão de entrada para o modelo estatístico
cols = ['Idade', 'Renda', 'Peso']

# Instanciação do normalizador MinMaxScaler
# A normalização é obrigatória pois o algoritmo gaussianomixture utiliza cálculos de variância e distância
# Variáveis com escalas grandes como renda dominariam o cálculo sobre variáveis menores como idade
scaler = MinMaxScaler()
# essa função fit_transform é responsável por transformar valores em uma escala de 0 à 1
X = scaler.fit_transform(df[cols])

# Configuração do modelo gaussian mixture
# Define-se a busca por 4 componentes subjacentes aos dados
# O algoritmo assume que os dados são gerados por uma mistura de distribuições gaussianas finitas
gmm = GaussianMixture(n_components=4, random_state=11) #defini como 11 a semente do random, mas caso queira deixar estocástico para cada execução, é so remover o random state

# Execução do algoritmo Expectation-Maximization
# O método ajusta iterativamente as médias e covariâncias das gaussianas aos dados fornecidos
# o algoritmo calcula internamente a probabilidade de cada ponto pertencer a cada grupo, introduzindo ao conceito de soft clustering
# Enquanto os demais algoritmos fazem clustering binário (ou pertence à um grupo ou pertence ao outro), o soft clustering calcula a curva de distribuição a qual o grupo pertence
# por exemplo, nhonho é uma demonstração do conceito de soft clustering, uma vez que ele gera confusão no modelo, ele é 
# atribuido como criança, mas tem renda de adulto, e peso que também poderia ser de um
# portanto, ele tem chance variada de estar em diferentes clusters
# O objetivo é maximizar a verossimilhança dos dados pertencerem às distribuições estimadas (basicamente, verossimilhança diz o quão certa a IA estava em julgar à qual cluster o personagem pertencia)

gmm.fit(X) # a função fit pegará o array bidimensional x pra aprender os clusters

# Atribuição dos clusters baseada na maior probabilidade calculada para cada ponto
labels = gmm.predict(X)
df['Cluster'] = labels

# Visualização da distribuição dos clusters
from matplotlib.ticker import ScalarFormatter

plt.figure(figsize=(14, 9))
sns.set_theme(style="whitegrid")

# usa os dados reais, se houver sobreposição, é porque
# os personagens são matematicamente idênticos nessas características.
sns.scatterplot(
    data=df, 
    x='Renda', 
    y='Idade', 
    hue='Cluster', 
    style='Cluster',      # Formas diferentes ajudam a distinguir sobreposições
    size='Peso', 
    sizes=(100, 500),     
    palette='bright',     
    markers=['o', 's', '^', 'X'], 
    alpha=0.7,            # Transparência: essencial para ver se tem um ponto atrás do outro
    edgecolor="black"     
)

# Usa symlog para permitir o 0 e os valores altos
plt.xscale('symlog', linthresh=100)

plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.xticks([0, 50, 100, 500, 1000, 5000])

# Limites para garantir visualização
plt.xlim(left=-5) 
plt.ylim(0, 85)

for i in range(df.shape[0]):
    # Se a renda for 0, ajusta o texto um pouco mais pra direita pra não cortar
    if (df['Renda'][i] < 10):
        ajuste_x = 5
    else:
        ajuste_x = 0
    
    # Alternância vertical para os nomes não se atropelarem
    if (i % 2):
        offset_y = 3
    else:
        offset_y = -4

    
    plt.text(
        df['Renda'][i] + ajuste_x, 
        df['Idade'][i] + offset_y, 
        df['Personagem'][i], 
        fontsize=10, 
        fontweight='bold', 
        color='#333333', 
        ha='center'
    )

plt.title('Agrupamento real da Vila', fontsize=16)
plt.xlabel('Renda mensal', fontsize=12)
plt.ylabel('Idade', fontsize=12)

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Grupos e Peso")
plt.tight_layout()
# parte que printa todos os clusters
colisoes = df.groupby(['Renda', 'Idade'])['Personagem'].apply(list)
sobreposicoes = colisoes[colisoes.apply(len) > 1]

# parte que printa as sobreposições
if not sobreposicoes.empty:
    print("\nSobreposições:")
    for (renda, idade), nomes in sobreposicoes.items():
        print(f"Renda {renda}, Idade {idade}: {', '.join(nomes)}")
plt.show()