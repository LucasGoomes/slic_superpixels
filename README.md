# Segmentação de Imagem com SLIC usando Python

Este repositório contém um aplicativo Python para realizar segmentação de imagem utilizando o algoritmo SLIC (Simple Linear Iterative Clustering). A segmentação de imagens é um processo fundamental em visão computacional e o SLIC é uma técnica popular para agrupar pixels em regiões semelhantes.

## Setup

1. Faça o download do repositório usando:
```
    git clone https://github.com/LucasGoomes/slic_superpixels.git

```
2. Dentro da raiz do projeto, instale as dependências executando o código abaixo no CMD:

```
    pip install -r requirements.txt
```

## Execução
1. Ainda na raiz do projeto, execute o código abaixo no CMD:
```
    streamlit run main.py
```

2. Na interface web, selecione sua imagem, número de clusters, número de iterações do algoritmo e clique em 'Processar Imagem'.