import datetime
import sys
import time

import base64
import csv
import httpx
import math
import os
import random
from pathlib import Path
from PIL import Image

PATH = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
REPORTED = []

def saveImg(filepath, b64_content):
    with open(filepath, 'wb') as file:
        file.write(base64.b64decode(b64_content))

def getSignalGain(g, model): # algoritmo para o ganho de sinal
    n = 64
    if model == "2": s = 436
    else: s = 794

    for c in range(n):
        for l in range(s):
            y = 100 + (1 / 20) * l * math.sqrt(l)
            g[l + c * s] = g[l + c * s] * y
    return g

def getSignal(model: str): # pega o sinal de acordo com o modelo e retorna o signal gain
    if model == "2": # 30
        sinal = random.choice(["g-30x30-1","g-30x30-2","A-30x30-1"])
    
        with open(PATH / "signals" / f"{sinal}.csv", "r") as file:
            reader = csv.reader(file)
            array = list(map(lambda x: float(x[0]), reader))
            return getSignalGain(array, model)
    if model == "1": # 60x60
        sinal = random.choice(["G-1","G-2","A-60x60-1"])
    
        with open(PATH / "signals" / f"{sinal}.csv", "r") as file:
            reader = csv.reader(file)
            array = list(map(lambda x: float(x[0]), reader))
            return getSignalGain(array, model)
        
def buildReport(nome):
    # abre o report ou cria se nao tiver 
    with open(PATH / "users" / nome / "report.txt", 'a', encoding='utf-8') as file: 
        REPORTED.append('report.txt')
        # abre os arquivos na pasta
        for root, dirs, files in os.walk(PATH / "users" / nome, topdown=False):
            for image in files: # vai nos arquivos
                if image not in REPORTED: # abre ele se nao tiver no report
                    with open(PATH / "users" / nome / image, 'rb') as imgFile: 
                        with Image.open(imgFile) as img: # abre pra pegar o metadata
                            # le o metadata dela e coloca as info no report -> metadata eh pra ta no img.txt
                            titulo = img.info['Title']
                            autor = img.info['Author']
                            username = img.info['Username']
                            alg = img.info['Algorithm'] 
                            comeco = img.info['Start']
                            dateComeco = datetime.datetime.strptime(comeco, "%Y-%m-%d %H:%M:%S")
                            fim = img.info['End']
                            dateFim = datetime.datetime.strptime(fim, "%Y-%m-%d %H:%M:%S")
                            tamanho = img.info['Size']
                            iteracoes = img.info['Iterations']
                            tempo = str(dateFim - dateComeco)
                            linha = titulo + ', ' + username + ', ' + iteracoes + ', ' + tempo + ', ' + alg + '\n'
                            file.write(linha)
                            REPORTED.append(image)
    file.close()           
    return
        
def makeDir(nome: str):
    User = PATH / "users" / nome
    if not User.exists():
        os.mkdir(User)

if __name__ == "__main__":
    nome = input(str(('digite seu nome:\n')))
    makeDir(nome)
    i = 0

    while i < 10: #True:
        print(f"Requisição {i}")
        espera = random.randint(1, 15) # tempo em segundos

        modelo = random.choice(["1", "2"]) # escolhe um modelo
        modelo = "1"
        algoritmo = random.choice(["cgne", "cgnr"]) # escolhe um algoritmo
        sinal = getSignal(modelo) # escolhe o sinal pelo modelo
        json = {'algoritmo': algoritmo, 'model': modelo, 'sinal': sinal} # arruma os dados

        httpx.post(f'http://127.0.0.1:12000/user/{nome}/criar', json=json) # usa o post
        print(f"Esperando {espera} segundos para a proxima requisicao.")
        time.sleep(espera)
        i = i + 1

        req = httpx.get(f'http://127.0.0.1:12000/user/{nome}/imagenscriadas') # usa o get 
        json = req.json()
        for key, value in json.items():
            saveImg(PATH / "users" / nome / key, value)
            buildReport(nome)
