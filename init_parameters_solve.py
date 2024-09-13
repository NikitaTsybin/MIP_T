import streamlit as st
import numpy as np
import sympy
from sympy import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import factorial
import pandas as pd
import sympy.printing as printing


st.write('Ниже записано общее решение уравнения изгиба балки с применением метода начальных параметров')
st.write('''$EI \\cdot v_i(x) = EI \\cdot v_{i-1}(x) + EI \\cdot \\Delta v_{i-1} + EI \\cdot \\Delta \\varphi \\cdot (x - x_i) - \\dfrac{\\Delta M \\cdot (x - x_i)^2}{2!} - \\dfrac{\\Delta Q \\cdot (x - x_i)^3}{3!} +  \\dfrac{(q_i - q_{i-1}) \\cdot (x - x_i)^4}{4!} $''')

init_data = pd.DataFrame([
        {'xi': 0.0, 'dv*EI': '0',  'dφ*EI': '!',    'dM': '0', 'dQ': '!',   'q': 0.0},
        {'xi': 2.0, 'dv*EI': '0',  'dφ*EI': '0',    'dM': '0', 'dQ': '-6',  'q': 4.0},
        {'xi': 4.0, 'dv*EI': '0',  'dφ*EI': '!',    'dM': '0', 'dQ': '0',   'q': 4.0},
        ])

column_conf = {
        'xi': st.column_config.NumberColumn('xi', help='Координата начала участка', format='%.2f', required=True, default=0.0),
        'dv*EI': st.column_config.TextColumn('dv*EI', help='Дополнительный прогиб в начале участка', required=True, default=0),
        'dφ*EI': st.column_config.TextColumn('dφ*EI', help='Дополнительный угол поворота в начале участка', required=True, default=0),
        'dM': st.column_config.TextColumn('dM', help='Дополнительный момент в начале участка', required=True, default=0),
        'dQ': st.column_config.TextColumn('dQ', help='Дополнительная сосредоточенная сила в начале участка', required=True, default=0),
        'q': st.column_config.NumberColumn('q', help='Интенсивность распределенной нагрузки на участке', format='%.1f', required=True, default=0),
        }
st.write('Для решения и построения графиков введите в таблицу ниже значения начальных параметров для каждого из участков балки, а также общую длину балки в поле под таблицей. На место неизвестных начальных параметров впишите !')
data = st.data_editor(init_data, hide_index=False, use_container_width=True, num_rows='dynamic', column_config=column_conf)
L = st.number_input(label=r'Полная длина балки', value=6.0, min_value=1.0)


x_val = data['xi'].tolist()
num_elems = len(x_val)
x_val.append(L)
EIdv_val = data['dv*EI'].tolist()
EIdf_val = data['dφ*EI'].tolist()
dM_val = data['dM'].tolist()
dQ_val = data['dQ'].tolist()
q_val = data['q'].tolist()

###Известные значения начальных параметров
###Эти данные должны считываться из таблицы
###Рассматривается балка из трех участков. Координаты начала каждого из участков
##x_val = [0, 2, 4]
###Число участков
##num_elems = len(x_val)
###Общая длина
##L_beam = 6
##x_val.append(L_beam)
###В балке нет параллелограмных механизмов, а вначале первого участка шарнирная опора (перемещение равно нулю)
##EIdv_val = [0, 0, 0]
###В начале первого участка шарнир, поэтому неизвестен угол поворота. В начале третьего участка врезанный шарнир, также неизвестен угол
##EIdf_val = ['!', 0, '!']
###В начале балки шарнир (момент равен нулю), сосредоточенных моментов в началах участка 2 и 3 нет
##dM_val = [0, 0, 0]
###В начале первого участка шарнирная опора, реакция неизвестна. В начале второго участка сосредоточенная сила
##dQ_val = ['!', -6, 0]
###Распределенные нагрузки на втором и третьем участке
##q_val = [0, 4, 4]

#Генерация переменных
x = symbols('x', real=True)
xi = symbols(['x_'+str(i) for i in range(num_elems)], real=True)
EIdvi = symbols(['EI\\cdot\\Delta\ v_'+str(i) for i in range(num_elems)], real=True)
EIdfi = symbols(['EI\\cdot\\Delta\\varphi_'+str(i) for i in range(num_elems)], real=True)
dMi = symbols(['\\Delta\ M_'+str(i) for i in range(num_elems)], real=True)
dQi = symbols(['\\Delta\ Q_'+str(i) for i in range(num_elems)], real=True)
qi = symbols(['q_'+str(i) for i in range(num_elems)], real=True)

init_subs = {}
#Заполняем подстановки известными значениями
for i in range(num_elems):
    if x_val[i] != '!':
        init_subs[xi[i]] = x_val[i]
    if EIdv_val[i] != '!':
        init_subs[EIdvi[i]] = EIdv_val[i]
    if EIdf_val[i] != '!':
        init_subs[EIdfi[i]] = EIdf_val[i]
    if dM_val[i] != '!':
        init_subs[dMi[i]] = dM_val[i]
    if dQ_val[i] != '!':
        init_subs[dQi[i]] = dQ_val[i]
    if q_val[i] != '!':
        init_subs[qi[i]] = q_val[i]


#Фунции для нулевого участка
#Перемещения
displacements = [EIdvi[0] + EIdfi[0]*(x-xi[0]) - dMi[0]*(x-xi[0])**2/factorial(2) - dQi[0]*(x-xi[0])**3/factorial(3) + (qi[0]- 0)*(x-xi[0])**4/factorial(4)]
displacements[0] = displacements[0].subs(init_subs)
#Углы поворота
angles = [EIdfi[0] - dMi[0]*(x-xi[0]) - dQi[0]*(x-xi[0])**2/factorial(2) + (qi[0]- 0)*(x-xi[0])**3/factorial(3)]
angles[0] = angles[0].subs(init_subs)
#Моменты
moments = [dMi[0] + dQi[0]*(x-xi[0]) - (qi[0]- 0)*(x-xi[0])**2/factorial(2)]
moments[0] = moments[0].subs(init_subs)
#Поперечные силы
forces = [dQi[0] - (qi[0]- 0)*(x-xi[0])]
forces[0] = forces[0].subs(init_subs)

#Генерируем функции для второго и последующего участков
for i in range(1,num_elems):
    tmpv = displacements[i-1] + EIdvi[i] + EIdfi[i]*(x-xi[i]) - dMi[i]*(x-xi[i])**2/factorial(2) - dQi[i]*(x-xi[i])**3/factorial(3) + (qi[i]- qi[i-1])*(x-xi[i])**4/factorial(4)
    tmpv = tmpv.subs(init_subs)
    displacements.append(tmpv)
    tmpa = angles[i-1] + EIdfi[i] - dMi[i]*(x-xi[i]) - dQi[i]*(x-xi[i])**2/factorial(2) + (qi[i]- qi[i-1])*(x-xi[i])**3/factorial(3)
    tmpa = tmpa.subs(init_subs)
    angles.append(tmpa)
    tmpm = moments[i-1] + dMi[i] + dQi[i]*(x-xi[i]) - (qi[i]- qi[i-1])*(x-xi[i])**2/factorial(2)
    tmpm = tmpm.subs(init_subs)
    moments.append(tmpm)
    tmpf = forces[i-1] + dQi[i] - (qi[i]- qi[i-1])*(x-xi[i])
    tmpf = tmpf.subs(init_subs)
    forces.append(tmpf)

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})

with st.expander('Функции метода начальных параметров'):
    tabs = st.tabs(['Перемещения','Углы поворота','Моменты','Поперечные силы'])
    with tabs[0]:
        st.write('Функции перемещений по участкам')
        for i in range(len(displacements)):
            st.latex('v_' + str(i) + '(x)=' + printing.latex(round_expr(displacements[i],2)))
    with tabs[1]:
        st.write('Функции углов поворота по участкам')
        for i in range(len(angles)):
            st.latex('\\varphi_' + str(i) + '(x)=' + printing.latex(round_expr(angles[i],2)))
    with tabs[2]:
        st.write('Функции моментов по участкам')
        for i in range(len(moments)):
            st.latex('M_' + str(i) + '(x)=' + printing.latex(round_expr(moments[i],2)))
    with tabs[3]:
        st.write('Функции поперечных сил по участкам')
        for i in range(len(forces)):
            st.latex('Q_' + str(i) + '(x)=' + printing.latex(round_expr(forces[i],2)))

#Находим неизвестные
unknows = displacements[num_elems-1].free_symbols - set([x])
#Число неизвестных
num_unknows = len(unknows)
st.subheader('Неизвестные величины')
st.latex(unknows)

bc_data = [[0,0,0,0] for i in range(num_unknows)]

##st.write(bc_data)


cols = st.columns([1, 1, 1, 1])
cols[0].write('Участок')
cols[1].write('Функция')
cols[2].write('Координата')
cols[3].write('Значение')
for i in range(num_unknows):
    cols = st.columns([1, 1, 1, 1])
    bc_data[i][0] = cols[0].selectbox(label='Участок'+str(i), options=[i for i in range(num_elems)], label_visibility='collapsed')
    bc_data[i][1] = cols[1].selectbox(label='Функция'+str(i), options=['Прогиб', 'Угол', 'Момент', 'Сила'], label_visibility='collapsed')
    bc_data[i][2] = cols[2].number_input(label='Координата'+str(i), value=x_val[bc_data[i][0]+1], label_visibility='collapsed')
    bc_data[i][3] = cols[3].number_input(label='Значение'+str(i), value=0.0, label_visibility='collapsed')


#Фукция генерирующая граничное условие
def generate_bc (elem, fun, coord, value):
    if fun == 'Прогиб':
        return Eq(displacements[elem].subs(x,coord), value)
    if fun == 'Угол':
        return Eq(angles[elem].subs(x,coord), value)
    if fun == 'Момент':
        return Eq(moments[elem].subs(x,coord), value)
    if fun == 'Сила':
        return Eq(forces[elem].subs(x,coord), value)

#Массив с граничным условиями
##bc = [(2, 'Прогиб', 6, 0), (2, 'Угол', 6, 0), (1, 'Момент', 4, 0)]
#Генерация системы уравнений на основе г.у.
bc_eqns = []
for i in bc_data:
    bc_eqns.append(generate_bc(i[0], i[1], i[2], i[3]))
#Решение системы уравнений из г.у.
rez_bc = solve(bc_eqns)
print(init_subs)
print(unknows)
st.latex(printing.latex(rez_bc))

def subs_rez_bc():
    rezv = []
    rezf = []
    rezM = []
    rezQ = []
    for i in range(num_elems):
        rezv.append(displacements[i].subs(rez_bc))
        rezf.append(angles[i].subs(rez_bc))
        rezM.append(moments[i].subs(rez_bc))
        rezQ.append(forces[i].subs(rez_bc))
    return rezv, rezf, rezM, rezQ

displacements, angles, moments, forces = subs_rez_bc()


print(displacements)
print(angles)
print(moments)
print(forces)

st.latex(printing.latex(displacements))
        



















####print(printing.latex(init_subs))
####sympy.pprint(printing.latex(EIdfi))
##
##x0, x1, x2, x3, x4, x5 = symbols('x0, x1, x2, x3, x4, x5', real=True)
##x_var = [x0, x1, x2, x3, x4, x5]
##EIdv0, EIdv1, EIdv2, EIdv3, EIdv4 = symbols('EIdv0, EIdv1, EIdv2, EIdv3, EIdv4', real=True)
##EIdv_var = [EIdv0, EIdv1, EIdv2, EIdv3, EIdv4]
##EIdf0, EIdf1, EIdf2, EIdf3, EIdf4 = symbols('EIdf0, EIdf1, EIdf2, EIdf3, EIdf4', real=True)
##EIdf_var = [EIdf0, EIdf1, EIdf2, EIdf3, EIdf4]
##dM0, dM1, dM2, dM3, dM4 = symbols('dM0, dM1, dM2, dM3, dM4', real=True)
##dM_var = [dM0, dM1, dM2, dM3, dM4]
##dQ0, dQ1, dQ2, dQ3, dQ4 = symbols('dQ0, dQ1, dQ2, dQ3, dQ4', real=True)
##dQ_var = [dQ0, dQ1, dQ2, dQ3, dQ4]
##q0, q1, q2, q3, q4 = symbols('q0, q1, q2, q3, q4', real=True)
##q_var = [q0, q1, q2, q3, q4]
##
##
##
##
##
###Известные значения начальных параметров
###Рассматривается балка из трех участков. Координаты начала каждого из участков
##x_val = [0, 2, 4]
###Число участков
##num_elems = len(x_val)
###Общая длина
##L_beam = 6
##x_val.append(L_beam)
###В балке нет параллелограмных механизмов, а вначале первого участка шарнирная опора (перемещение равно нулю)
##EIdv_val = [0, 0, 0]
###В начале первого участка шарнир, поэтому неизвестен угол поворота. В начале третьего участка врезанный шарнир, также неизвестен угол
##EIdf_val = ['!', 0, '!']
###В начале балки шарнир (момент равен нулю), сосредоточенных моментов в началах участка 2 и 3 нет
##dM_val = [0, 0, 0]
###В начале первого участка шарнирная опора, реакция неизвестна. В начале второго участка сосредоточенная сила
##dQ_val = ['!', -6, 0]
###Распределенные нагрузки на втором и третьем участке
##q_val = [0, 4, 4]
##
##
##
##
###Пустые словари для подстановки
##x_subs = {}
##EIdv_subs = {}
##EIdf_subs = {}
##dM_subs = {}
##dQ_subs = {}
##q_subs = {}
##
##init_subs = {}
##
###Заполняем подстановки известными значениями
##for i in range(num_elems):
##    if x_val[i] != '!':
##        init_subs[x_var[i]] = x_val[i]
##    if EIdv_val[i] != '!':
##        init_subs[EIdv_var[i]] = EIdv_val[i]
##    if EIdf_val[i] != '!':
##        init_subs[EIdf_var[i]] = EIdf_val[i]
##    if dM_val[i] != '!':
##        init_subs[dM_var[i]] = dM_val[i]
##    if dQ_val[i] != '!':
##        init_subs[dQ_var[i]] = dQ_val[i]
##    if q_val[i] != '!':
##        init_subs[q_var[i]] = q_val[i]
##
##
###Общее решение для функции перемещений каждого из участков
##EIv0 = 0    + EIdv0 + EIdf0*(x-x0) - dM0*(x-x0)**2/factorial(2) - dQ0*(x-x0)**3/factorial(3) + (q0- 0)*(x-x0)**4/factorial(4)
##EIv0 = EIv0.subs(init_subs)
##EIv1 = EIv0 + EIdv1 + EIdf1*(x-x1) - dM1*(x-x1)**2/factorial(2) - dQ1*(x-x1)**3/factorial(3) + (q1-q0)*(x-x1)**4/factorial(4)
##EIv1 = EIv1.subs(init_subs)
##EIv2 = EIv1 + EIdv2 + EIdf2*(x-x2) - dM2*(x-x2)**2/factorial(2) - dQ2*(x-x2)**3/factorial(3) + (q2-q1)*(x-x2)**4/factorial(4)
##EIv2 = EIv2.subs(init_subs)
##EIv3 = EIv2 + EIdv3 + EIdf3*(x-x3) - dM3*(x-x3)**2/factorial(2) - dQ3*(x-x3)**3/factorial(3) + (q3-q2)*(x-x3)**4/factorial(4)
##EIv3 = EIv3.subs(init_subs)
##EIv4 = EIv3 + EIdv4 + EIdf4*(x-x4) - dM4*(x-x4)**2/factorial(2) - dQ4*(x-x4)**3/factorial(3) + (q4-q3)*(x-x4)**4/factorial(4)
##EIv4 = EIv4.subs(init_subs)
###Собираем перемещения в один список
##displacements = [EIv0, EIv1, EIv2, EIv3, EIv4]
##
###Общее решение для функции углов поворота каждого из участков
##EIf0 = 0    + EIdf0 - dM0*(x-x0) - dQ0*(x-x0)**2/factorial(2) + (q0- 0)*(x-x0)**3/factorial(3)
##EIf0 = EIf0.subs(init_subs)
##EIf1 = EIf0 + EIdf1 - dM1*(x-x1) - dQ1*(x-x1)**2/factorial(2) + (q1-q0)*(x-x1)**3/factorial(3)
##EIf1 = EIf1.subs(init_subs)
##EIf2 = EIf1 + EIdf2 - dM2*(x-x2) - dQ2*(x-x2)**2/factorial(2) + (q2-q1)*(x-x2)**3/factorial(3)
##EIf2 = EIf2.subs(init_subs)
##EIf3 = EIf2 + EIdf3 - dM3*(x-x3) - dQ3*(x-x3)**2/factorial(2) + (q3-q2)*(x-x3)**3/factorial(3)
##EIf3 = EIf3.subs(init_subs)
##EIf4 = EIf3 + EIdf4 - dM4*(x-x4) - dQ4*(x-x4)**2/factorial(2) + (q4-q3)*(x-x4)**3/factorial(3)
##EIf4 = EIf4.subs(init_subs)
###Собираем углы поворота в один список
##angles = [EIf0, EIf1, EIf2, EIf3, EIf4]
##
###Общее решение для функции моментов каждого из участков
##M0 = 0  + dM0 + dQ0*(x-x0) - (q0- 0)*(x-x0)**2/factorial(2)
##M0 = M0.subs(init_subs)
##M1 = M0 + dM1 + dQ1*(x-x1) - (q1-q0)*(x-x1)**2/factorial(2)
##M1 = M1.subs(init_subs)
##M2 = M1 + dM2 + dQ2*(x-x2) - (q2-q1)*(x-x2)**2/factorial(2)
##M2 = M2.subs(init_subs)
##M3 = M2 + dM3 + dQ3*(x-x3) - (q3-q2)*(x-x3)**2/factorial(2)
##M3 = M3.subs(init_subs)
##M4 = M3 + dM4 + dQ4*(x-x4) - (q4-q3)*(x-x4)**2/factorial(2)
##M4 = M4.subs(init_subs)
###Собираем моменты в один список
##moments = [M0, M1, M2, M3, M4]
##
###Общее решение для функции поперечных сил каждого из участков
##Q0 = 0  + dQ0 - (q0- 0)*(x-x0)
##Q0 = Q0.subs(init_subs)
##Q1 = Q0 + dQ1 - (q1-q0)*(x-x1)
##Q1 = Q1.subs(init_subs)
##Q2 = Q1 + dQ2 - (q2-q1)*(x-x2)
##Q2 = Q2.subs(init_subs)
##Q3 = Q2 + dQ3 - (q3-q2)*(x-x3)
##Q3 = Q3.subs(init_subs)
##Q4 = Q3 + dQ4 - (q4-q3)*(x-x4)
##Q4 = Q4.subs(init_subs)
###Собираем поперечные силы в один список
##force = [Q0, Q1, Q2, Q3, Q4]
##
###Находим неизвестные
##unknows = displacements[num_elems-1].free_symbols - set([x])
###Число неизвестных
##num_unknows = len(unknows)
##
##eq1 = Eq(displacements[2].subs(x,6), 0)
##eq2 = Eq(angles[2].subs(x,6), 0)
##eq3 = Eq(moments[1].subs(x,4), 0)
##bc_eqns = [eq1, eq2, eq3]
##rez_bc = solve(bc_eqns)


##EJvi = sp.IndexedBase('EJvi')
##EJdvi = sp.IndexedBase('EJdvi')
##EJdfi = sp.IndexedBase('EJdfi')
##dMi = sp.IndexedBase('dMi')
##dQi = sp.IndexedBase('dQi')
##qi = sp.IndexedBase('qi')
##i = sp.symbols('i', integer=True)
##x = sp.symbols('x', real=True)
##a = sp.Sum(EJvi[i],(i,1,10))
##EJvi[i] = EJdvi[i]

##

##
###Общее решение методом начальных параметров для произвольного участка
##def EJvi(EJdv, EJdf, dM, dQ, qi, qp, xi):
##    return lambda x: EJdv + EJdf*(x-xi) - dM*(x-xi)**2/factorial(2) - dQ*(x-xi)**3/factorial(3) + (qi-qp)*(x-xi)**4/factorial(4)
##
##def EJfi(EJdv, EJdf, dM, dQ, qi, qp, xi):
##    return lambda x: EJdf - dM*(x-xi) - dQ*(x-xi)**2/factorial(2) + (qi-qp)*(x-xi)**3/factorial(3)
##
##def Mi(EJdv, EJdf, dM, dQ, qi, qp, xi):
##    return lambda x: dM + dQ*(x-xi) - (qi-qp)*(x-xi)**2/factorial(2)
##
##def Qi(EJdv, EJdf, dM, dQ, qi, qp, xi):
##    return lambda x: dQ - (qi-qp)*(x-xi)
##
##
##

###Значение распределенной нагрузки на предыдущем участке
##qp_arr = [0]
##for i in range(1, len(qi_arr)):
##    qp_arr.append(qi_arr[i-1])
##
##v_arr = []
##f_arr = []
##M_arr = []
##Q_arr = []
##for k in range(num_of_elements):
##    v_arr.append(EJvi(EJdv_arr[k], EJdf_arr[k], dM_arr[k], dQ_arr[k], qi_arr[k], qp_arr[k], xi_arr[k]))
##    f_arr.append(EJfi(EJdv_arr[k], EJdf_arr[k], dM_arr[k], dQ_arr[k], qi_arr[k], qp_arr[k], xi_arr[k]))
##    M_arr.append(Mi(EJdv_arr[k], EJdf_arr[k], dM_arr[k], dQ_arr[k], qi_arr[k], qp_arr[k], xi_arr[k]))
##    Q_arr.append(Qi(EJdv_arr[k], EJdf_arr[k], dM_arr[k], dQ_arr[k], qi_arr[k], qp_arr[k], xi_arr[k]))
##
##
##num_points = 9
##points = []
##for i in range(num_of_elements):
##    points.append(np.linspace(xi_arr[i], xi_arr[i+1], num_points).tolist())
##
##v = [[0 for k in points[i]] for i in range(num_of_elements)]
##f = [[0 for k in points[i]] for i in range(num_of_elements)]
##M = [[0 for k in points[i]] for i in range(num_of_elements)]
##Q = [[0 for k in points[i]] for i in range(num_of_elements)]
##
##for i in range(num_of_elements):
##    for p in range(len(points[i])):
##        for k in range(0,i+1):
##            v[i][p] = v[i][p] + v_arr[k](points[i][p])
##            f[i][p] = f[i][p] + f_arr[k](points[i][p])
##            M[i][p] = M[i][p] + M_arr[k](points[i][p])
##            Q[i][p] = Q[i][p] + Q_arr[k](points[i][p]) 
##    
##
####def plots():
##fig = make_subplots(rows = 2,cols = 2, subplot_titles=['Перемещения v*EI', 'Углы поворота φ*EI', 'Момент Mz', 'Поперечная сила Qy'])
##for i in range(num_of_elements):
##    fig.add_trace(go.Scatter(x=points[i], y=v[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 1, col = 1)
##    fig.add_trace(go.Scatter(x=points[i], y=f[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 1, col = 2)
##    fig.add_trace(go.Scatter(x=points[i], y=M[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 2, col = 1)
##    fig.add_trace(go.Scatter(x=points[i], y=Q[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 2, col = 2)
##
##fig.update_yaxes(autorange="reversed", row=1, col=1)
##fig.update_yaxes(autorange="reversed", row=1, col=2)
##fig.update_yaxes(autorange="reversed", row=2, col=1)
##fig.update_yaxes(autorange="reversed", row=2, col=2)
##fig.update_layout(height=500)
####    fig.show(config={ 'modeBarButtonsToRemove': ['zoom', 'pan'] })
####    return fig
##
####st.components.v1.html(fig.to_html(include_mathjax='cdn'), height=500)
####st.write(1)
##
##st.plotly_chart(fig)
##
##
##
##
##
##
##
##
##
