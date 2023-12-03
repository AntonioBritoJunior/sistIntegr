import sys
from multiprocessing import Queue
from time import time, sleep
import base64
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import uvicorn
from datetime import datetime, timezone
from fastapi import FastAPI
from numpy.linalg import norm
from numpy.typing import NDArray
from pathlib import Path
from pydantic import BaseModel
from scipy.linalg.blas import sgemm
from threading import Thread
from typing import Callable

# Tipos e classes
Img = NDArray[np.float32]
Modelo = NDArray[np.float32]
Sinal = NDArray[np.float32]
Modelos = dict[str, Modelo]

CGNR_func = Callable[[Modelo, Sinal], Img]
CGNR_funcs = list[CGNR_func]

# Parametros e ctes
ERRO = .0001
MAX_WORKERS = 4
PATH = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
IMAGEM_TAMANHO = {'1': (3600, 1), '2': (900, 1)}
MODELO_TAMANHO = {'1': (50816, 3600), '2': (27904, 900)}
TAMANHO_FINAL = {'1': (60, 60), '2': (30, 30)}

usuarios = []
img64 = {}

class InputPost(BaseModel):
    algoritmo: str
    model: str
    sinal: list[float]

# Leitor de modelos para ler os arq de modelo
def lerModelo(model: str) -> Modelo:
    with open(PATH / "models" / f"H-{model}.csv", "r") as file:
        reader = csv.reader(file, delimiter=',')
        res = np.empty(MODELO_TAMANHO[model], dtype=np.float32)
        i = 0
        for line in reader:
            res[i] = np.array(line, np.float32)
            i += 1
        return res

# Algoritmos
def getError(b: NDArray[np.float32], a: NDArray[np.float32]):
    return norm(b, 2) - norm(a, 2)

# com base no do prof
def cgne(h, g, imgTamanhoModelo, tamanhoFinal):
    f0 = np.zeros(imgTamanhoModelo, np.float32)
    r0 = g - sgemm(1.0, h, f0)
    p0 = sgemm(1.0, h, r0, trans_a=True) #transposta aqui

    iteratTot = 0

    while iteratTot < 30: #True: # ate alcancar o erro
        iteratTot = iteratTot + 1

        a0 = sgemm(1.0, r0, r0, trans_a=True) / sgemm(1.0, p0, p0, trans_a=True)

        f0 = f0 + a0 * p0
        r1 = r0 - a0 * sgemm(1.0, h, p0)

        error = abs(getError(r1, r0))
        if error < ERRO:
            break

        beta = sgemm(1.0, r1, r1, trans_a=True) / sgemm(1.0, r0, r0, trans_a=True)
        p0 = sgemm(1.0, h, r0, trans_a=True) + beta * p0
        r0 = r1
    f0 = f0.reshape(tamanhoFinal)
    #f0 = f0.transpose(f0).reshape(tamanhoFinal)

    return f0, iteratTot

# com base no do prof
def cgnr(h: Modelo, g: Sinal, imgTamanhoModelo, tamanhoFinal) -> tuple[Img, int]:
    f0 = np.zeros(imgTamanhoModelo, np.float32)
    r0 = g - sgemm(1.0, h, f0)
    z0 = sgemm(1.0, h, r0, trans_a=True)
    p0 = np.copy(z0)

    iteratTot = 0

    while iteratTot < 30: #True: # ate alcancar o erro
        iteratTot = iteratTot + 1

        w = sgemm(1.0, h, p0)
        norm_z = norm(z0, 2) ** 2
        a = norm_z / norm(w) ** 2
        f0 = f0 + a * p0
        r1 = r0 - a * w

        error = abs(getError(r1, r0))
        if error < ERRO:
            break

        z0 = sgemm(1.0, h, r1, trans_a=True)
        b = norm(z0, 2) ** 2 / norm_z
        p0 = z0 + b * p0
        r0 = r1
    f0 = f0.reshape(tamanhoFinal)
    # f0 = f0.transpose(f0).reshape(tamanhoFinal)

    return f0, iteratTot

ALG = {'cgne': cgne,'cgnr': cgnr}

# Worker -> thread
class Worker:
    def __init__(self, id: int, queue: Queue):
        self.id = id
        self.queue = queue

# Guarda info das requisicoes -> mais facil de pegar
class RequestInfo:
    def __init__(self, nome: str, algoritmo: str, model: str, sinal: Sinal) -> None:
        self.nome = nome
        self.algoritmo = algoritmo
        self.model = model
        self.sinal = sinal

def makeDir(name: str):
    imgPathUser = PATH / "images" / name
    if not imgPathUser.exists():
        os.mkdir(imgPathUser)

def runWorker(worker: Worker, models, infoReq: RequestInfo):
    print(f"worker - {worker.id} iniciou.")

    # pega as info
    nome = infoReq.nome
    modelo = infoReq.model

    # Parametros para o CGNR
    model = models[modelo]
    sinal = infoReq.sinal
    imgTamanhoModelo = IMAGEM_TAMANHO[modelo]
    imgTamanho_Final = TAMANHO_FINAL[modelo]

    # usa o alg para fazer a img
    startTime = time()
    img, iterations = ALG[infoReq.algoritmo](model, sinal, imgTamanhoModelo, imgTamanho_Final)
    endTime = time()

    # arruma as info
    arqNome = f"{nome}-{modelo}-{endTime}.png" 
    arqPath = PATH / "images" / nome / arqNome
    start = datetime.fromtimestamp(startTime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end = datetime.fromtimestamp(endTime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Salvando a img com o metadata
    meta = {
        'Title': arqNome.replace(".png", ""),
        'Author': f"CGNR Processor",
        'Username': str(nome),
        'Algorithm': str(infoReq.algoritmo.upper()), 
        'Start': str(start),
        'End': str(end),
        'Size': str(TAMANHO_FINAL[modelo]), 
        'Iterations': str(iterations)
    }

    # qnd acabar ele salva
    plt.imsave(arqPath, img, cmap='gray', metadata=meta)
    with open(arqPath, 'rb') as f:
        conv = base64.b64encode(f.read())
        img64[arqNome] = conv

    print(f"worker - {worker.id} acabou, voltando para a fila.")
    worker.queue.put(worker.id) # trabalhador volta pra fila

def getResServer(): # retorna cpu e memoria do server
    process = psutil.Process(os.getpid())
    return process.cpu_percent(1), process.memory_info().rss # cpu, memor

def resourceReport():
    # cria um dir e arquivo de report
    reportPath = PATH / "report"
    if not reportPath.exists():
        os.mkdir(reportPath)

    while True:
        with open(reportPath / "report.txt", 'a', encoding='utf-8') as file:
            sleep(2)
            tempo = datetime.fromtimestamp(time(), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            cpu, memor = getResServer()
            memor = memor / 1_000_000
            info = str(tempo) + ", " + str(cpu) + ", " + str(memor) + "MB \n"
            file.write(info)
            file.close()
            

def runQWorker(requestQ: Queue, models):
    workerQ = Queue(MAX_WORKERS) 
    for i in range(MAX_WORKERS): # vai colocando na fila
        workerQ.put(i)

    while True:
        workerID = workerQ.get()
        infoReq = requestQ.get()
        worker = Worker(workerID, workerQ)  # id, fila
        worker_thread = Thread(target=runWorker, args=[worker, models, infoReq])
        worker_thread.start()

# Main
def main():
    models = {"1": lerModelo("1"),"2": lerModelo("2")} # 60x60 e 30x30

    app = FastAPI()
    requestQ = Queue() # fila de requisicoes

    # worker para o relatorio dos recursos do servidor
    resourceWorker = Thread(target=resourceReport)
    resourceWorker.start()

    # fila dos workers para trabalhar
    qWorker = Thread(target=runQWorker, args=[requestQ, models])
    qWorker.start()

    # endpoints:
    # get -> download das imagens criadas para o cliente
    @app.get("/user/{nome}/imagenscriadas")
    async def imagenscriadas(nome: str):
        usuario = {}
        for key, value in img64.items(): #pega as imgs e coloca pro usuario
            if nome in key:
                usuario[key] = value
        return usuario

    # post -> faz o server comecar a criar as img
    @app.post("/user/{nome}/criar")
    async def criar(nome: str, input: InputPost):
        if not nome in usuarios:
            usuarios.append(nome)
            makeDir(nome)

        sinal = np.array(input.sinal, np.float32).reshape((-1, 1))
        infoReq = RequestInfo(nome, input.algoritmo, input.model, sinal)
        requestQ.put(infoReq)

    # fecha tudo e tchau brigado
    @app.on_event("shutdown")
    def shutdown_event():
        img64.clear()

    uvicorn.run(app, host="0.0.0.0", port=12000)

if __name__ == "__main__":
    main()
