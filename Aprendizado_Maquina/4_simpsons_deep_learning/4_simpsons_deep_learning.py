import torch # gerencia os tensores 
import torch.nn as nn # contém as camadas necessárias pra montar a rede (nn é neural network)
import torch.optim as optim # contém os algoritmos matemáticos que ajustam os pesos no aprendizado
import torchvision # pacote auxiliar que tem datasets e ferramentas de visão computacional
import torchvision.transforms as transforms # ferramentas pra editar ou transformar as imagens antes de entrarem na rede
from torch.utils.data import DataLoader, Subset # utilitários para carregar os dados em lotes
import matplotlib.pyplot as plt # biblioteca padrão para gerar os gráficos visuais
import numpy as np # biblioteca matemática para lidar com arrays numéricos

# Nome: Artur Mendonça Arruda
# Matrícula: 231033737
#
# =================================================================================
# PROJETO 4: DEEP LEARNING COM CLASSIFICAÇÃO BIOMÉTRICA (CNN): OS SIMPSONS
# =================================================================================

"""
Tema e Contextualização:

Este projeto utiliza o universo de Os Simpsons, uma aclamada sitcom animada que
satiriza o estilo de vida americano através de uma família disfuncional na cidade
fictícia de Springfield. O cenário hipotético foca na Usina Nuclear de Springfield,
propriedade do Sr. Burns, especificamente no Setor 7-G onde trabalha o protagonista,
Homer Simpson. O objetivo é criar um sistema de segurança automatizado para a porta
de entrada deste setor, onde só os trabalhadores daquela usina teriam acesso à entrada,
tendo que barrar quem não for um deles.

Justificativa da Arquitetura:

É crucial explicar por que não usamos a mesma rede densa (MLP) do projeto anterior.
Uma rede densa comum trata cada pixel como um valor isolado na entrada. Se o rosto do
Homer estiver um pouco para a esquerda ou para a direita, a rede densa se perde, pois
os valores dos pixels mudaram de posição.

A Rede Neural Convolucional (CNN) resolve isso preservando a estrutura espacial.
Ela utiliza kernels (kernels são filtros aprendidos) que deslizam pela imagem procurando padrões visuais
(como o formato dos olhos, a cor da pele, curvas do cabelo), independentemente de onde
eles estejam na imagem. Isso é fundamental para tarefas de visão computacional modernas.

Dataset utilizado: 
The Simpsons Characters Data (Kaggle): https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset

Fontes: 

(Utilizado como base matemática para os princípios de reconhecimento de padrões e a extração de características em dados visuais)
BISHOP, Christopher M. Pattern Recognition and Machine Learning. New York: Springer, 2006.

(Referência para justificar a escolha de Redes Neurais Convolucionais em detrimento de redes densas para o processamento de imagens)
Buduma, N. (2017). Fundamentals of Deep Learning: Designing Next-Generation Machine Intelligence.

(Base técnica para a implementação do código, estrutura das classes e pipeline de dados na biblioteca PyTorch)
Stevens, E., Antiga, L., & Viehmann, T. (2020). Deep Learning with PyTorch. Manning Publications.

(Fonte teórica para as técnicas de regularização via Dropout, funções de ativação e o funcionamento do otimizador Adam)
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
""" 

# semente freezada para reproduzir o mesmo resultado, é só comentar ou tirar essa seção para rodar aleatoriamente
torch.manual_seed(12)
np.random.seed(12)

# hiperparâmetros 
# a rede processa 32 imagens de uma vez, calcula a média do erro delas e só então ajusta os pesos
tamanho_lote = 32
# define o tamanho do passo que o otimizador dá na direção da solução correta
taxa_aprendizado = 0.001
# quantas vezes a IA vai ver o álbum de fotos inteiro para aprender
geracoes = 15 
tamanho_imagem = 64 # tamanho da imagem em escala  T x T, onde T = tamanho_imagem

# verifica se tem GPU para acelerar, senão usa o processador
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executando o treinamento no dispositivo: {dispositivo}")

# pipeline de transformação da imagem bruta para números que a rede entende
transformacao = transforms.Compose([
    # redimensiona todas as imagens para ficarem iguais (padronização é vital, visto que a rede tem entrada fixa)
    transforms.Resize((tamanho_imagem, tamanho_imagem)),
    
    # transforma a imagem (que é lida como altura x largura x cor) para Tensor (cor x altura x largura)
    # e escala os valores de pixel de 0-255 para 0-1
    transforms.ToTensor(),
    
    # passo de normalização: subtrai a média (0.5) e divide pelo desvio padrão (0.5) em cada canal RGB
    # isso centraliza os dados no zero (-1 a 1), evitando que a matemática da rede "exploda" ou fique lenta
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# caminho do dataset
diretorio_dados = "./simpsons_dataset" 

# Classe auxiliar para corrigir o problema dos labels
# O ImageFolder dá labels baseados em todas as pastas (por exemplo, homer vira 15), mas filtra apenas alguns.
# Se a rede tem 8 saídas (0 a 7) e recebe o rótulo 15, dá erro, então essa classe traduz o 15 para o novo índice correto.
class DatasetRemodelado(torch.utils.data.Dataset):
    def __init__(self, dataset_original, indices, mapa_labels):
        self.dataset = dataset_original
        self.indices = indices
        self.mapa_labels = mapa_labels

    def __getitem__(self, idx):
        # busca o dado original usando o índice filtrado
        idx_real = self.indices[idx]
        img, label_original = self.dataset[idx_real]
        # retorna a imagem e o rótulo traduzido
        return img, self.mapa_labels[label_original]

    def __len__(self):
        return len(self.indices)

try:
    # carrega a estrutura de pastas
    dataset_completo = torchvision.datasets.ImageFolder(root=diretorio_dados, transform=transformacao)
    
    # lista dos personagens que serão usados no modelo
    classes_alvo = [
        'homer_simpson', 'lenny_leonard', 'carl_carlson', 'charles_montgomery_burns', # Trabalhadores
        'marge_simpson', 'lisa_simpson', 'bart_simpson', 'maggie_simpson'             # Família/Intrusos
    ]
    
    # lógica de mapeamento dos IDs
    mapa_traducao = {}
    novo_id = 0
    
    # dicionário reverso para buscar o ID pelo nome (será usado na simulação final para printar o nome)
    idx_to_nome_final = {}
    
    # dicionário auxiliar do PyTorch que já mapeia nome_da_pasta -> id_original
    nome_to_idx = dataset_completo.class_to_idx
    
    classes_encontradas = []
    
    for nome in classes_alvo:
        if nome in nome_to_idx:
            id_original = nome_to_idx[nome]
            mapa_traducao[id_original] = novo_id
            idx_to_nome_final[novo_id] = nome # guarda quem é quem para o print final
            classes_encontradas.append(nome)
            novo_id += 1
            
    # varre o dataset e pega apenas os índices das imagens que pertencem às classes que escolhemos
    indices_filtrados = [i for i, (path, label) in enumerate(dataset_completo.samples) 
                         if label in mapa_traducao]
    
    # cria o dataset remodelado que entrega os labels corretos (que vai de 0 a 7)
    dataset_final = DatasetRemodelado(dataset_completo, indices_filtrados, mapa_traducao)
    
    # separação: 80% para treino, 20% para teste
    tamanho_treino = int(0.8 * len(dataset_final))
    tamanho_teste = len(dataset_final) - tamanho_treino
    dataset_treino, dataset_teste = torch.utils.data.random_split(dataset_final, [tamanho_treino, tamanho_teste])

    # Os DataLoaders entregam os lotes de dados mastigados para a rede
    # shuffle=True no treino é vital para a rede não decorar a ordem das fotos
    carregador_treino = DataLoader(dataset_treino, batch_size=tamanho_lote, shuffle=True)
    carregador_teste = DataLoader(dataset_teste, batch_size=tamanho_lote, shuffle=False)
    
    num_classes = len(classes_encontradas)
    print(f"Classes carregadas para o sistema de segurança: {classes_encontradas}")
    print(f"Total de imagens filtradas: {len(dataset_final)}")

except Exception as e:
    print(f"ERRO: {e}")
    print("AVISO: Dataset não encontrado ou estrutura incorreta.")
    num_classes = 8
    carregador_treino = None
    carregador_teste = None

# arquitetura da CNN Rede Neural Convolucional
class RedeNeuralSpringfield(nn.Module):
    def __init__(self, num_classes):
        super(RedeNeuralSpringfield, self).__init__()
        
        # CAMADA 1: Extração de traços grossos
        # nn.Conv2d: Cria filtros que aprendem a ver bordas e cores
        # in_channels=3 (RGB), out_channels=32 (cria 32 mapas de características diferentes)
        self.camada_convolucao1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(), # Função de ativação: zera valores negativos (traz não-linearidade, essencial para deep learning)
            nn.MaxPool2d(kernel_size=2, stride=2) # Pooling: reduz a imagem pela metade (64 vira 32), mantendo apenas o pixel mais "forte" da região
        )
        
        # CAMADA 2: Extração de formas
        # Aumentamos para 64 canais para capturar padrões mais complexos (olhos, orelhas)
        self.camada_convolucao2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # imagem agora vira 16x16
        )
        
        # CAMADA 3: Detalhes específicos
        # Aumentamos para 128 canais (identifica rostos inteiros, texturas específicas)
        self.camada_convolucao3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # imagem agora vira 8x8
        )
        
        # Achatamento (Flatten)
        # A saída da conv3 é um cubo de dados (128 canais x 8 altura x 8 largura)
        # Para classificar, precisamos "esticar" isso num vetor linguiça (uma linha só)
        self.achatamento = nn.Flatten()
        
        # Classificador de rede densa
        # Entrada: 128 * 8 * 8 = 8192 neurônios que vieram das imagens processadas
        self.camadas_densas = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), # camada densa que cruza todas as informações
            nn.ReLU(),
            nn.Dropout(0.5), # desliga 50% dos neurônios aleatoriamente no treino, pois evita que tenha overfitting da imagem na rede
            nn.Linear(512, num_classes) # emite 8 notas, uma para cada personagem
        )

    def forward(self, x):
        # define o fluxo dos dados dentro da rede (forward pass)
        x = self.camada_convolucao1(x) # passa na primeira convolução
        x = self.camada_convolucao2(x) # passa na segunda
        x = self.camada_convolucao3(x) # passa na terceira
        x = self.achatamento(x) # achata o cubo num vetor
        x = self.camadas_densas(x) # classifica e dá a resposta
        return x

# instancia o modelo e joga para a GPU se tiver
modelo = RedeNeuralSpringfield(num_classes=num_classes).to(dispositivo)

# função de perda: CrossEntropyLoss é a padrão para classificação multiclasse
# ela calcula a diferença entre a resposta da rede e a resposta certa
criterio = nn.CrossEntropyLoss()

# otimizador: Adam é uma evolução moderna do SGD (Stochastic Gradient Descent)
# ele ajusta a taxa de aprendizado individualmente para cada peso para que possa aprender mais rápido
#
otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizado)

# loop de treinamento
if carregador_treino:
    historico_erro = []
    historico_acuracia = []

    print("Iniciando o treinamento da rede neural...")

    for geracao in range(geracoes):
        modelo.train() # coloca a rede em modo de treino (ativa o Dropout)
        erro_acumulado = 0.0
        acertos = 0
        total_imagens = 0
        
        for entradas, etiquetas in carregador_treino:
            # move os dados para a GPU se disponível
            entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
            
            # 1. zera os gradientes antigos: o PyTorch acumula gradientes por padrão, precisamos limpar antes de começar
            otimizador.zero_grad()
            
            # 2. forward: passa a imagem pela rede e obtém a previsão (chute da IA)
            saidas = modelo(entradas)
            
            # 3. calcula o erro (Loss): quão longe o chute passou da realidade?
            erro = criterio(saidas, etiquetas)
            
            # 4. backward: o passo mágico do Backpropagation. Calcula a direção para corrigir cada peso da rede
            erro.backward()
            
            # 5. step: o otimizador atualiza os pesos da rede na direção calculada
            otimizador.step()
            
            # coleta estatísticas para monitorar o progresso
            erro_acumulado += erro.item()
            _, predito = torch.max(saidas.data, 1) # pega qual classe teve a maior nota
            total_imagens += etiquetas.size(0)
            acertos += (predito == etiquetas).sum().item()
            
        # médias da geração atual
        media_erro = erro_acumulado / len(carregador_treino)
        acuracia_epoca = 100 * acertos / total_imagens
        historico_erro.append(media_erro)
        historico_acuracia.append(acuracia_epoca)
        
        print(f"Geração [{geracao+1}/{geracoes}] - Erro: {media_erro:.4f} - Acurácia Treino: {acuracia_epoca:.2f}%")

    # teste final em todo o conjunto de validação
    print("\n" + "-"*30)
    print("Avaliação final no conjunto de teste...")
    modelo.eval() # coloca em modo de avaliação 
    acertos_teste = 0
    total_teste = 0
    
    with torch.no_grad(): # desliga o cálculo de gradiente pra economizar memória, já que não vai ter mais treino
        for entradas, etiquetas in carregador_teste:
            entradas, etiquetas = entradas.to(dispositivo), etiquetas.to(dispositivo)
            saidas = modelo(entradas)
            _, predito = torch.max(saidas.data, 1)
            total_teste += etiquetas.size(0)
            acertos_teste += (predito == etiquetas).sum().item()
            
    acuracia_final = 100 * acertos_teste / total_teste
    print(f"Acurácia Final no Teste: {acuracia_final:.2f}%")
    print("-"*30)

    # gráficos de resultado
    plt.figure(figsize=(10, 5))
    plt.plot(historico_erro, label='Erro (Loss)', color='red')
    # normalizando a acurácia para ficar na mesma escala do erro (0 a 1 aprox)
    plt.plot([x/100 for x in historico_acuracia], label='Acurácia Normalizada', color='blue')
    plt.title("Evolução do Aprendizado: Setor 7-G")
    plt.xlabel("Gerações")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # simulação com 5 imagens aleatórias para demonstrar visualmente a regra de negócio
    # é um teste a parte que visa mostrar a aplicação na "catraca" da usina
    def simular_catraca(qtd=5):
        print("\n" + "="*40)
        print("INICIANDO PROTOCOLO DE SEGURANÇA SETOR 7-G")
        print("="*40)
        
        modelo.eval() # Garante que o dropout está desligado
        indices = torch.randperm(len(dataset_teste))[:qtd] # Pega indices aleatórios
        
        plt.figure(figsize=(15, 6))
        
        # Lista de funcionários autorizados (Regra de Negócio)
        trabalhadores_usina = ['homer_simpson', 'lenny_leonard', 'carl_carlson', 'charles_montgomery_burns']

        with torch.no_grad():
            for i, idx in enumerate(indices):
                img_tensor, label_real = dataset_teste[idx]
                img_tensor = img_tensor.unsqueeze(0).to(dispositivo) # Adiciona dimensão do lote (1, 3, 64, 64)
                
                saidas = modelo(img_tensor)
                # Softmax transforma os números brutos em porcentagens de probabilidade
                probabilidade = torch.nn.functional.softmax(saidas, dim=1)
                confianca, predito_idx = torch.max(probabilidade, 1)
                
                nome_predito = idx_to_nome_final[predito_idx.item()]
                confianca_perc = confianca.item() * 100
                
                # Regra de Negócio: Verifica se está na lista de autorizados
                if nome_predito in trabalhadores_usina:
                    status = "ACESSO AUTORIZADO"
                    cor_texto = 'green'
                else:
                    status = "ACESSO NEGADO"
                    cor_texto = 'red'
                
                # Visualização
                ax = plt.subplot(1, qtd, i + 1)
                # Desnormaliza a imagem para mostrar as cores originais 
                img_plot = img_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img_plot = img_plot * 0.5 + 0.5
                plt.imshow(img_plot)
                plt.title(f"{nome_predito}\n{status}\nConfiança: {confianca_perc:.1f}%", 
                          color=cor_texto, fontsize=9, fontweight='bold')
                plt.axis('off')
                
                print(f"Identificado: {nome_predito:<25} | Status: {status}")

        plt.tight_layout()
        plt.show()

    simular_catraca()