import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys


m=5
nome_da_pasta = 'data'
#nome_do_arquivo = 'resultados/resultado_simulacao_numero_alternativas_%d'%m
#nome_do_arquivo = 'resultados/resultado_simulacao_numero_alternativas_10'
#nome_do_arquivo = 'resultados/resultado_simulacao_numero_alternativas_5'

nome_do_arquivo = 'simulation_result_for_%d_alternatives'%m


fp = open("%s/%s" % (nome_da_pasta, nome_do_arquivo), "r")
for instancia in fp.readlines():
    dict_dados = json.loads(instancia)
    tau_pred = dict_dados["tau_medio_predicao_ideal"]
    tau_current = dict_dados["tau_medio_current_ideal"]
    desvio_erro = dict_dados["desvio_medio"]
    #pontos_dentro = dict_dados["pontos_dentro_intervalo"]

print(desvio_erro)
SNR = np.arange(0,51)

linewidth1 = 1
linewidth2 = 1
min = np.min(tau_pred)
max = np.max(tau_pred)
print(min)
print(max)

plt.figure(figsize=(10, 10))
plt.title(r"$m=%d$"%m, fontsize=40)
plt.plot(SNR, tau_current, '--', label=r'$\tau^c_{r^* \times  r^c}$',   color='black', linewidth=linewidth1)
plt.plot(SNR, tau_pred, label=r'$\hat{\tau}_{r^* \times  \hat{r}}$', color ='red', linewidth=linewidth2)
plt.xlabel(r"SNR", fontsize=35)
plt.ylabel(r"$\tau$" , fontsize=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlim([-1, 51])
plt.ylim([min, max])
plt.legend(fontsize=40)
plt.savefig('tauxsnr_%dalter'%(m))
#plt.savefig('../../doutorado/A2 - predicao/21_05 terceira versao/figuras/tauxsnr_%dalter'%(m))
plt.show()


'''
plt.figure(figsize=(50, 40))
#plt.title(r"$Gráfico \ 1: \tau_{g^* \times  \hat{g}} \times SNR $", fontsize=50)
plt.plot(SNR, desvio_erro, '--', label=r'desvio',   color='black')
plt.xlabel(r"SNR", fontsize=35)
plt.ylabel(r"$\tau$" , fontsize=50)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=40)
plt.show()

plt.figure(figsize=(30, 20))
#plt.title(r"$Gráfico \ 1: \tau_{g^* \times  \hat{g}} \times SNR $", fontsize=50)
#plt.plot(SNR, pontos_dentro,  'o', color='black')
plt.xlabel(r"SNR", fontsize=35)
plt.ylabel(r"PIC (%) of $p_{ij(T+\lambda)}$", fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=50)
plt.show()
'''