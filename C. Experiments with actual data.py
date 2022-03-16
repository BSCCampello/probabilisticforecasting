import pandas as pd
from rls import FilterRLS_beta
import numpy as np
import sys
from promethee import Promethee
from kendal_tau import Kendal
from numpy import sqrt
import random
import matplotlib.pylab as plt
from matplotlib import pyplot
import pandas as pd
import collections


np.set_printoptions(suppress=True)

q = 15
a = 5
#q = 12
#a = 4
c = 3

filt_paramet = 2
# define the step (lambda)
v_passo = [1] # Step = 1
#v_passo = [3] # Step = 3
#v_passo = [5] # Step = 5
n_iteracao_smaa = 10000

lamb = 0.99 # Fator de esquecimento do RLS
gama = 0.1 # Para inicializar a matriz R do RLS

crit_max = [True, False, False]

funçao_de_pref = [('usual', 0), ('usual', 0),  ('usual', 0)]

v_p = [1/3, 1/3, 1/3]
alt = ['a%d' % (d + 1) for d in range(a)] # This vector is just to name the variables as a_1...a_n

nome_da_pasta = 'data'
nome_arquivo = 'actual_data.xlsx'
#nome_arquivo = '4paises_criterios_com_inflacao.xlsx'
df = pd.read_excel("%s/%s" % (nome_da_pasta, nome_arquivo))
matriz_classica = np.zeros(q)
matriz_pred_ideal = np.zeros(q)
matriz_pred_RLS = np.zeros(q)


matriz_media_des_erro = []
matriz_intervalo_predicao = []
ponto_dento = 0
v_d_futuro = []
v_intervalo_de_confianca = []
v_valor_real_dentro_intervalo = []
v_R_predicao = []
v_desvio_padrao = []

for passo in v_passo:

    for i in range(q):

        d_completo = df.loc[i]
        #print(len(d_completo))

        d_completo = np.array(d_completo)

        # Montar as matrizes com valores atuais (current value) e com valores "futuros"
        d_atual = d_completo[-1-passo]
        d_futuro = d_completo[-1]
        v_d_futuro.append(d_futuro)
        matriz_classica[i] = d_atual
        matriz_pred_ideal[i] = d_futuro


        if i == 1 or i == 4 or i == 7 or i == 10 or i == 13 or i == 16:
            lamb = 0.99

        else:
            lamb = 0.95


        #sinal desejado d
        d = d_completo[1:len(d_completo)-passo]


        # <editor-fold desc="obter o x, que é o que considero o sinal de entrada do lmsn e rls">
        N = len(d)
        x = [[0] * filt_paramet]
        aux = np.concatenate((np.zeros(filt_paramet), d))
        for k in range(filt_paramet, len(d) + filt_paramet):
            x.append([aux[k - i] for i in range(0, filt_paramet)])
        # </editor-fold>

        # Filtrando com RLS

        Rf = FilterRLS_beta()
        Ry, Re, Rw, Rpredicao = Rf.run(d, x, passo, N, lamb, gama, filt_paramet)



        """"
        # <editor-fold desc="grafico_sinais">
        if i == 0:
            a_i = 1
            c_j = 1
            p_ij = 'p(1,1,:)'
        elif i == 1:
            a_i = 1
            c_j = 2
            p_ij = 'p(1,2,:)'
        elif i == 2:
            a_i = 1
            c_j = 3
            p_ij = 'p(1,3,:)'
        elif i == 3:
            a_i = 2
            c_j = 1
            p_ij = 'p(2,1,:)'
        elif i == 4:
            a_i = 2
            c_j = 2
            p_ij = 'p(2,2,:)'
        elif i == 5:
            a_i = 2
            c_j = 3
            p_ij = 'p(2,3,:)'
        elif i == 6:
            a_i = 3
            c_j = 1
            p_ij = 'p(3,1,:)'
        elif i == 7:
            a_i = 3
            c_j = 2
            p_ij = 'p(3,2,:)'
        elif i == 8:
            a_i = 3
            c_j = 3
            p_ij = 'p(3,3,:)'
        elif i == 9:
            a_i = 4
            c_j = 1
            p_ij = 'p(4,1,:)'
        elif i == 10:
            a_i = 4
            c_j = 2
            p_ij = 'p(4,2,:)'
        elif i == 11:
            a_i = 4
            c_j = 3
            p_ij = 'p(4,3,:)'
        elif i == 12:
            a_i = 5
            c_j = 1
            p_ij = 'p(5,1,:)'
        elif i == 13:
            a_i = 5
            c_j = 2
            p_ij = 'p(5,2,:)'
        elif i == 14:
            a_i = 5
            c_j = 3
            p_ij = 'p(5,3,:)'
        sinal = np.array(d)
        anos = np.arange(1980, 2018)
        plt.figure(figsize=(10, 6))
        plt.title(r"Signal %s $- a_%d, c_%d$" % (p_ij, a_i, c_j), fontsize=30)
        plt.plot(anos, Ry, '--',  label='Prediction',   color='black')
        plt.plot(anos, sinal, label='Target', color='red')
        plt.xlabel(r"$t$", fontsize=20)
        plt.ylabel(r"Criteria value", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.xlim(1980, 2017)
        plt.legend(fontsize=20)
        plt.show()
        plt.show()

        #plt.hist(Re[3:])
        #plt.show()
        #plt.hist(Re[3:])
        #plt.show()
        # </editor-fold>
        """

        re_2 = np.square(Re[3:])
        sum_re_2 = np.sum(re_2)
        var_erro = (1/(len(re_2)-filt_paramet))*sum_re_2
        desvio_erro = sqrt(var_erro)
        amplitude = 1.96 * desvio_erro
        intervalo_predicao = [Rpredicao - amplitude, Rpredicao + amplitude]

        matriz_media_des_erro.append([Rpredicao, desvio_erro])
        matriz_intervalo_predicao.append([Rpredicao - amplitude, Rpredicao + amplitude])
        v_R_predicao.append(Rpredicao)
        v_desvio_padrao.append(desvio_erro)

        matriz_pred_RLS[i] = Rpredicao

        interval = 1.96 * desvio_erro
        lower, upper = Rpredicao - interval, Rpredicao + interval
        intervalo_de_confianca = np.array([lower, upper])
        v_intervalo_de_confianca.append(intervalo_de_confianca)

        if lower <= d_futuro <= upper:
            
            ponto_dento += 1
            v_valor_real_dentro_intervalo.append(1)
        else:
            print(d_futuro)
            print(intervalo_de_confianca)
            v_valor_real_dentro_intervalo.append(0)
            




    m_c = matriz_classica.reshape((a, c))
    m_p_i = matriz_pred_ideal.reshape((a, c))
    m_p_RLS = matriz_pred_RLS.reshape((a, c))
    matriz_media_des_erro = np.array(matriz_media_des_erro).reshape((a, c,2))
    matriz_intervalo_predicao = np.array(matriz_intervalo_predicao).reshape((a, c, 2))


    # <editor-fold desc="rankin_c">
    C_promethee = Promethee()
    c_v_ordenado = C_promethee.run(m_c, crit_max, funçao_de_pref, v_p, alt)

    c_ranking = []
    for k in range(len(c_v_ordenado)):
        c_ranking.append(c_v_ordenado[k][2])
    # </editor-fold>
    print('Current ranking', np.add(c_ranking, [1]*len(c_ranking)) )



    # <editor-fold desc="ranking_pi">
    P_promethee = Promethee()
    p_v_ordenado = P_promethee.run(m_p_i, crit_max, funçao_de_pref, v_p, alt)

    p_ranking = []
    for k in range(len(p_v_ordenado)):
        p_ranking.append(p_v_ordenado[k][2])
    # </editor-fold>




    print('Benchmark ranking', np.add(p_ranking, [1]*len(p_ranking)))

    # <editor-fold desc="kendal1">
    # To compare the ranking with the ideal decision matrix
    kendal = Kendal()
    tau1 = kendal.run(c_ranking, p_ranking)
    # </editor-fold>


    # <editor-fold desc="ranking_p">
    PR_promethee = Promethee()
    pr_v_ordenado = PR_promethee.run(m_p_RLS, crit_max, funçao_de_pref, v_p, alt)

    pr_ranking = []
    for k in range(len(pr_v_ordenado)):
        pr_ranking.append(pr_v_ordenado[k][2])
    # </editor-fold>

  

    # <editor-fold desc="kendal2">
    # To compare the ranking with the ideal decision matrix
    kendal = Kendal()
    tau2 = kendal.run(p_ranking, pr_ranking)
    # </editor-fold>
    print('Tau value')
    print(tau2)



# SMAA
mat_descmesure = np.zeros((a, a))
cont_ranking_g_chapeu = 0
for itera in range(n_iteracao_smaa):
    # Decision matrix m_d


    d_m = []
    for alter in matriz_media_des_erro:
        d_m.append([random.gauss(c[0], c[1]) for c in alter])


    promethee_smaa = Promethee()
    v_ordenado_smaa = promethee_smaa.run(np.array(d_m), crit_max, funçao_de_pref, v_p, alt)

    ranking_smaa = []
    for l in range(a):
        r_a = v_ordenado_smaa[l][2]
        ranking_smaa.append(v_ordenado_smaa[l][2])
        mat_descmesure[l][r_a] += 1





mat_descmesure = np.transpose(mat_descmesure)
data = {'$\bar{s}_e$': v_desvio_padrao,
        'Point prediction': v_R_predicao,
        'Target': v_d_futuro,
        'Prediction interval':v_intervalo_de_confianca,
        'Target $\in$ PI':v_valor_real_dentro_intervalo,
        }
data = collections.OrderedDict(data)

pd.set_option('display.float_format', lambda x: '%.2f' % x)
s = pd.DataFrame(data)

#print(s.to_latex(index=False))

print('Matrix of probabilities (M)')
mat_percentual = (mat_descmesure/n_iteracao_smaa)*100
print(np.array(mat_percentual))
ponto_dento = (ponto_dento/(a*c))*100
#print('ponto dentro',ponto_dento)

df = pd.DataFrame(mat_percentual)
#print(df.to_latex(index=False))

print("Important: the results shown in matrix M are stochastic, and may vary each time this code is run")





