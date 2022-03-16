import numpy as np
import sys
import math
import json
from kendal_tau import Kendal
from rls import FilterRLS_beta
from promethee import Promethee
#from lms_normalizado import LMS_n
from numpy import sqrt
import matplotlib.pylab as plt
#from promethee_dominance_degrees import Promethee_do_de
import random
import pandas as pd
np.set_printoptions(suppress=True)

# Important: the results shown in matrix M are stochastic, and may vary a little each time this code is executed

 #choose the pattern (as in the article) for which you want to see the results. The patterns are 1 (results show in Table I and II), 2 (results show in Table III and IV), 3 (results show in Table V and VI) or 4 (results show in Table VII and VIII)
pattern = 4


gerar_sinais = False # True if the signals have to be gerenerate or False if just repeat the alredy gernerate signal



n_iteracao_smaa = 10000

# <editor-fold desc="parametros">
nome_da_pasta = 'data'
nome_arquivo = 'signals_for_pattern%d'%pattern
if pattern == 1 or pattern == 2:
    SNR = 30
elif pattern == 3:
    SNR = 10
elif pattern == 4:
    SNR = 0
else:
    print("A wrong value was chosen for the pattern")
n_iterations = 1
# Signal
N = 200  # Number of sample, it means the size of the signal
variancia_ruido = 1 # White noise variance

# Adaptative filter
filt_paramet = 2  # Quantity of adaptative filtering parameters
passo = 1
# RLS:
lamb = 0.9 # The forgetting factor of the RLS
gama = 0.1 # To initialize the RLS R-matrix


# Parameters for the MCDA method
n_a = 5 # Number of alternatives
v_p = [0.25, 0.25, 0.25, 0.25] # Weight vector
crit_max = [True] * n_a
alt = ['a%d' % (d + 1) for d in range(n_a)] # This vector is just to name the variables as a_1...a_n

# Promethee
funçao_de_pref = [('usual', 0), ('usual', 0),  ('usual', 0),  ('usual', 0)]
# </editor-fold>


def gerar_sinal():

    # Signals criterium 1
    t = np.arange(1, N + 1)
    a = np.random.uniform(2, 20, n_a)
    b = np.random.uniform(0.5, 0.8, n_a)
    d_1 = []
    for i in range(len(a)):
        d = a[i] + b[i] * t
        d_1.append((d).tolist())



    n = np.arange(1, N + 1)
    a = np.random.uniform(2, 20, n_a)
    freq = np.random.uniform(0.7, 0.75, n_a) * math.pi
    d_2 = []
    for i in range(len(freq)):
        d = a[i] + np.sin(freq[i] * n)
        d_2.append((d).tolist())


    # Sinais para critério 3
    d_3 = []
    a = np.random.uniform(2, 20, n_a)
    for i in range(n_a):
        d = a[i] + (-1) ** t
        d_3.append((d).tolist())



    # Sinais para critério 4
    r_i = np.random.uniform(1, 20, n_a)
    d_4 = []
    for i in range(len(r_i)):
        d = (r_i[i]) + t ** 0.075
        d_4.append((d).tolist())


    # The white noise vector:
    r = np.random.normal(0, math.sqrt(variancia_ruido), N)


    fp = open("%s/%s" % (nome_da_pasta, nome_arquivo), "w")
    dict_dados = {
        'd_1': d_1,
        'd_2': d_2,
        'd_3': d_3,
        'd_4': d_4,
        'r': r.tolist(),
    }

    fp.write(json.dumps(dict_dados) + "\n")



def calc_ord(matriz_decisao, promethee = 0):

    # Ordenamento PROMETHEE, o promethee = 0 é o clássico, promethee = 1 é com dominancia na func de prob como no artigo e o promethee = 2 é dominancia como propus
    if promethee == 0:
        R_promethee = Promethee()
        v_ordenado = R_promethee.run(matriz_decisao, crit_max, funçao_de_pref, v_p, alt)
    
    ranking = []
    for i in range(len(v_ordenado)):
        ranking.append(v_ordenado[i][2])

    return ranking



if gerar_sinais:
    gerar_sinal()

# <editor-fold desc="LerDadosSinais">
fp = open("%s/%s" % (nome_da_pasta, nome_arquivo), "r")
for instancia in fp.readlines():
    dict_dados = json.loads(instancia)
    d_1 = dict_dados['d_1']
    d_2 = dict_dados['d_2']
    d_3 = dict_dados['d_3']
    d_4 = dict_dados['d_4']
    r = np.array(dict_dados['r'])
# </editor-fold>



sinais = [d_1, d_2, d_3, d_4]


R_mat_pred = np.zeros((len(sinais), n_a))  # make a transpose matrix to facilitate the way to compute values
R_mat_pred_ideal = np.zeros((len(sinais), n_a))
m_classica = np.zeros((len(sinais), n_a))
mat_media_desvio = np.zeros((len(sinais), n_a), dtype=bytearray)
desvio_medio_total = 0

for c, criterion in enumerate(sinais):

    for a, sig_alt in enumerate(criterion):

        # To calculate the alpha value
        # <editor-fold desc="adicionando_ruido">
        power = (10 * math.log10(np.var(sig_alt)) - SNR) / 20  # Esta é a fórmula para calcular de quanto tem que ser o alfa para cada SNR
        alfa = math.pow(10, power)

        # To add a noise in the signal
        d = sig_alt[:len(sig_alt)-1] + alfa * r[:len(r)-1]

        # </editor-fold>

        # Para gerar a referância
        # <editor-fold desc="To_generate_ref_vector">
        x = [[0] * filt_paramet]
        aux = np.concatenate((np.zeros(filt_paramet), d))
        for k in range(filt_paramet, len(d) + filt_paramet):
            x.append([aux[k - i] for i in range(0, filt_paramet)])
        # </editor-fold>

        # Filtrando com RLS
        Rf = FilterRLS_beta()
        Ry, Re, Rw, Rpredicao = Rf.run(d, x, passo, N, lamb, gama, filt_paramet)
        #plt.hist(Re)
        #plt.show()
        #plt.plot(Ry)
        #plt.plot(d)
        #plt.show()

        # Calculando o desvio do erro
        # <editor-fold desc="desvio_e">

        re_2 = np.square(Re[3:])
        sum_re_2 = np.sum(re_2)
        var_erro = (sum_re_2 / (len(re_2) - filt_paramet))
        desvio_erro = sqrt(var_erro)
        desvio_medio_total += np.mean(desvio_erro)
        # </editor-fold>
     

        # Guardo o valor de predição de cada alternativa em cada critério
        R_mat_pred[c][a] = Rpredicao
        

        # Guardo o valor ideal em cada alternativa em cada critério
        referencia = sig_alt[-1] + alfa * r[-1]
        R_mat_pred_ideal[c][a] = referencia
       

        interval = 1.96 * desvio_erro
        lower, upper = Rpredicao - interval, Rpredicao + interval
        intervalo_de_confianca = np.array([lower, upper])
        

        # Guardo o que seria o valor current data em cada alternativa em cada critério
        m_classica[c][a] = sig_alt[-passo - 1]

        # Em uma matriz, eu guardo a média e o desvio
        mat_media_desvio[c][a] = np.array([Rpredicao, desvio_erro])





# IMPORTANTE: preciso trabalhar com as matrizes transpostas
R_mat_pred = np.transpose(R_mat_pred)
R_mat_pred_ideal = np.transpose(R_mat_pred_ideal)
m_classica = np.transpose(m_classica)
mat_media_desvio = np.transpose(np.array(mat_media_desvio))


# <editor-fold desc="Cálculo dos rankings">
print('ideal')
PR_ideal_ranking = calc_ord(R_mat_pred_ideal)
print('predicao')
PR_ranking = calc_ord(R_mat_pred)
#classico_ranking = calc_ord(m_classica)


# <editor-fold desc="calcular o tau para cada tipo de matriz">
# To compare the ranking with the ideal decision matrix
R_kendal = Kendal()
R_tau = R_kendal.run(PR_ideal_ranking, PR_ranking)
#PR_aux_mean_tau.append(R_tau)

# comparar o ranking ideal com o classico ou current
R2_kendal = Kendal()
#R2_tau = R2_kendal.run(PR_ideal_ranking, classico_ranking)
#PR2_aux_mean_tau.append(R2_tau)


# SMAA
mat_descmesure = np.zeros((n_a, n_a))
cont_ranking_g_chapeu = 0
for itera in range(n_iteracao_smaa):
    # Decision matrix m_d

    d_m = []
    for t, alter in enumerate(mat_media_desvio):
        d_m.append([random.gauss(c[0], c[1]) for c in alter])

    S_promethee = Promethee()
    v_ordenado_smaa = S_promethee.run(np.array(d_m), crit_max, funçao_de_pref, v_p, alt)
    ranking_smaa = []
    for i in range(len(v_ordenado_smaa)):
        ranking_smaa.append(v_ordenado_smaa[i][2])
        r_a = v_ordenado_smaa[i][2]
        mat_descmesure[r_a][i] += 1



PR_ideal_ranking = np.add(PR_ideal_ranking, ([1]*len(PR_ideal_ranking)))
PR_ranking = np.add(PR_ranking, ([1]*len(PR_ranking)))
print('Benchmark ranking', PR_ideal_ranking)
print('Prediction ranking', PR_ranking)

print('matrix M')
matriz_percentual = (mat_descmesure/n_iteracao_smaa)*100

df = pd.DataFrame( matriz_percentual)
print(df)