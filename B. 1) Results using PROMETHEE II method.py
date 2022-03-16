import numpy as np
#import sys
import math
import json
from kendal_tau import Kendal
from rls import FilterRLS_beta
from promethee import Promethee
#from lms_normalizado import LMS_n
from numpy import sqrt
import matplotlib.pylab as plt
#from promethee_dominance_degrees import Promethee_do_de




# Parameters:

n_iterations = 1000 # Number of interations - Monte Carlo simulation - Times to repet the test
gerar_sinais = False # True if the signals have to be gerenerate or False if just repeat the alredy gernerate signal
n_a = 5 # Number of alternatives

# <editor-fold desc="parametros">
nome_da_pasta = 'data' # folder name from the code search the file with the input data to run.



# Signal
N = 200  # Number of sample, it means the size of the signal
variancia_ruido = 1 # White noise variance
n_alfa = 51 # Quantity of diferents alpha, it is a linear factor which multiply the noise to increase it

# Adaptative filter
filt_paramet = 2  # Quantity of adaptative filtering parameters
passo = 1
# RLS:
lamb = 0.9 # The forgetting factor of the RLS
gama = 0.1 # To initialize the RLS R-matrix


# Parameters for the MCDA method

v_p = [1/4, 1/4, 1/4, 1/4] # Weight vector
crit_max = [True] * n_a
alt = ['a%d' % (d + 1) for d in range(n_a)] # This vector is just to name the variables as a_1...a_n

nome_arquivo = 'sinais_simulacao_num_alternativas_%d'%n_a   # File name: the file to run depends on the alternatives' name

# Promethee
funçao_de_pref = [('usual', 0), ('usual', 0),  ('usual', 0),  ('usual', 0)]
# </editor-fold>

# The strcture is that:

# The first for is to make monte carlo simulation
# The second for is to increase the white noise by an alpha factor
# The third for is to take all alternatives of a given criterion
# The fourth for is to take each alternative of the criterion and calculate the signal prediction
# The last for is to the SMAA method, in each iteration, we take a radom number in the interval calculated in the step before


PR_sum_tau = np.zeros(n_alfa)
PR2_sum_tau = np.zeros(n_alfa)
sum_toal_desvio_do_erro = np.zeros(n_alfa)
ponto_dento = np.zeros(n_alfa)


def gerar_sinal():

    m_d_1 = []
    m_d_2 = []
    m_d_3 = []
    m_d_4 = []
    m_r = []

    for iteracao in range(n_iterations):

        # Signals criterion 1
        t = np.arange(1, N + 1)
        a = np.random.uniform(2, 20, n_a)
        b = np.random.uniform(0.5, 0.8, n_a)
        d_1 = []
        for i in range(len(a)):
            d = a[i] + b[i] * t
            d_1.append((d).tolist())
        m_d_1.append(d_1)

        # Signals criterion 2
        n = np.arange(1, N + 1)
        a = np.random.uniform(2, 20, n_a)
        freq = np.random.uniform(0.7, 0.75, n_a) * math.pi
        d_2 = []
        for i in range(len(freq)):
            d = a[i] + np.sin(freq[i] * n)
            d_2.append((d).tolist())
        m_d_2.append(d_2)

        # Signals criterion 3

        d_3 = []
        a = np.random.uniform(2, 20, n_a)
        for i in range(n_a):
            d = a[i] + (-1) ** t
            d_3.append((d).tolist())

        m_d_3.append(d_3)


        # Signals criterion 4
        r_i = np.random.uniform(1, 20, n_a)
        d_4 = []
        for i in range(len(r_i)):
            d = (r_i[i]) + t ** 0.075
            d_4.append((d).tolist())
        m_d_4.append(d_4)

        # The white noise vector:
        r = np.random.normal(0, math.sqrt(variancia_ruido), N)
        m_r.append(r.tolist())


    fp = open("%s/%s" % (nome_da_pasta, nome_arquivo), "w")
    dict_dados = {
        'd_1': m_d_1,
        'd_2': m_d_2,
        'd_3': m_d_3,
        'd_4': m_d_4,
        'r': m_r,
    }

    fp.write(json.dumps(dict_dados) + "\n")



def calc_ord(matriz_decisao, promethee = 0):

    # PROMETHEE ordering, promethee = 0 is the classic one, promethee = 1 is with dominance in the prob function as in the article and promethee = 2 is dominance as I proposed
    if promethee == 0:
        R_promethee = Promethee()
        v_ordenado = R_promethee.run(matriz_decisao, crit_max, funçao_de_pref, v_p, alt)
        ranking = []
    elif promethee == 1:
        #promethee_novo = Promethee_do_de()
        #v_ordenado = promethee_novo.run(matriz_decisao, crit_max, v_p, True)
        pass
    else:
        #promethee_novo = Promethee_do_de()
        #v_ordenado = promethee_novo.run(matriz_decisao, crit_max, v_p, False)
        pass

    ranking = []
    for i in range(len(v_ordenado)):
        ranking.append(v_ordenado[i][2])

    return ranking



def contagem_pontos_dentro_intervalo(ponto_predicao, desvio_erro, valor_ideal):

    interval = 1.96 * desvio_erro
    lower, upper = ponto_predicao - interval, ponto_predicao + interval
    intervalo_de_confianca = np.array([lower, upper])


    if lower <= valor_ideal <= upper:
        return 1
    else:
        return 0




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
    m_r = np.array(dict_dados['r'])
# </editor-fold>



c_tau_final = 0
for interation in range(n_iterations):

    print('inte: ', interation)

    sinais = [d_1[interation], d_2[interation], d_3[interation], d_4[interation]]
    r = m_r[interation]

    PR_aux_mean_tau = []
    PR2_aux_mean_tau = []
    aux_media_desvio_do_erro = []

    for SNR in range(n_alfa):

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

                # Calculando o desvio do erro
                # <editor-fold desc="desvio_e">
                re_2 = np.square(Re[3:])
                sum_re_2 = np.sum(re_2)
                var_erro = (1 / (len(re_2) - filt_paramet)) * sum_re_2
                desvio_erro = sqrt(var_erro)
                desvio_medio_total += np.mean(desvio_erro)
                # </editor-fold>

                # Guardo o valor de predição de cada alternativa em cada critério
                R_mat_pred[c][a] = Rpredicao

                # Guardo o valor ideal em cada alternativa em cada critério
                valor_ideal = sig_alt[-1] + alfa * r[-1]
                R_mat_pred_ideal[c][a] = valor_ideal

                # Guardo o que seria o valor current data em cada alternativa em cada critério
                m_classica[c][a] = sig_alt[-passo - 1]

                # Em uma matriz, eu guardo a média e o desvio
                mat_media_desvio[c][a] = np.array([Rpredicao, desvio_erro])

                ponto_dento[SNR] += contagem_pontos_dentro_intervalo(Rpredicao, desvio_erro, valor_ideal)


        # IMPORTANTE: preciso trabalhar com as matrizes transpostas
        R_mat_pred = np.transpose(R_mat_pred)
        R_mat_pred_ideal = np.transpose(R_mat_pred_ideal)
        m_classica = np.transpose(m_classica)
        mat_media_desvio = np.transpose(np.array(mat_media_desvio))
        aux_media_desvio_do_erro.append(desvio_erro / (len(sinais) * n_a))



        # <editor-fold desc="Cálculo dos rankings">
        PR_ranking = calc_ord(R_mat_pred)
        PR_ideal_ranking = calc_ord(R_mat_pred_ideal)
        classico_ranking = calc_ord(m_classica)



        # <editor-fold desc="calcular o tau para cada tipo de matriz">
        # To compare the ranking with the ideal decision matrix
        R_kendal = Kendal()
        R_tau = R_kendal.run(PR_ideal_ranking, PR_ranking)
        PR_aux_mean_tau.append(R_tau)

        # comparar o ranking ideal com o classico ou current
        R2_kendal = Kendal()
        R2_tau = R2_kendal.run(PR_ideal_ranking, classico_ranking)
        PR2_aux_mean_tau.append(R2_tau)




    PR_sum_tau += PR_aux_mean_tau
    PR2_sum_tau += PR2_aux_mean_tau
    sum_toal_desvio_do_erro += aux_media_desvio_do_erro


tau_medio_predicao_ideal = PR_sum_tau/n_iterations
tau_medio_current_ideal = PR2_sum_tau/n_iterations
desvio_medio = sum_toal_desvio_do_erro/n_iterations

ponto_dento = (ponto_dento/(n_a*len(sinais)*n_iterations))*100







data_r = {
        'tau_medio_predicao_ideal': tau_medio_predicao_ideal.tolist(),
        'tau_medio_current_ideal': tau_medio_current_ideal.tolist(),
        'desvio_medio': desvio_medio.tolist(),
        'pontos_dentro_intervalo': ponto_dento.tolist(),
        }

print('aqui')
fp = open("%s/%s" % (nome_da_pasta, 'simulation_result_for_%d_alternatives'%n_a), "w")
fp.write(json.dumps(data_r) + "\n")



'''
plt.figure(figsize=(10, 10))
plt.plot(desvio_medio, tau_medio_predicao_ideal, label="classico")
plt.plot(desvio_medio, tau_medio_current_ideal, label="current")

plt.xlabel(r"desvio", fontsize=35)
plt.ylabel(r"$\tau_{g^* \times  \hat{g}}$", fontsize=50)
plt.legend(fontsize=5)
plt.show()
'''