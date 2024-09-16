import streamlit as st
import numpy as np
import sympy
from sympy import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import factorial
import pandas as pd
import sympy.printing as printing
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# Specify canvas parameters in application
##drawing_mode = st.sidebar.selectbox(
##    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
##)

##stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
##if drawing_mode == 'point':
##    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
##stroke_color = st.sidebar.color_picker("Stroke color hex: ")
##st.write(stroke_color)
##bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
##bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

##realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color='#000000',
    background_color="#eee",
##    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=False,
    height=200,
    drawing_mode="freedraw",
    point_display_radius=0,
    key="canvas",
)


with st.expander('Общий вид решения'):
    st.write('Ниже записано общее решение уравнения изгиба балки с применением метода начальных параметров')
    st.write('''$EI \\cdot v_i(x) = EI \\cdot v_{i-1}(x) + EI \\cdot \\Delta v_{i-1} +
    EI \\cdot \\Delta \\varphi_i \\cdot (x - x_i) - \\dfrac{\\Delta M_i \\cdot (x - x_i)^2}{2!} -
    \\dfrac{\\Delta Q_i \\cdot (x - x_i)^3}{3!} +  \\dfrac{(q_i - q_{i-1}) \\cdot (x - x_i)^4}{4!}; $''')

    st.write('''$EI \\cdot \\varphi_i(x) = EI \\cdot \\varphi_{i-1}(x) +
    EI \\cdot \\Delta \\varphi_i - \\Delta M_i \\cdot (x - x_i) -
    \\dfrac{\\Delta Q_i \\cdot (x - x_i)^2}{2!} +  \\dfrac{(q_i - q_{i-1}) \\cdot (x - x_i)^3}{3!}; $''')

    st.write('''$M_i(x) = M_{i-1}(x) + \\Delta M_i + \\Delta Q_i \\cdot (x - x_i) - \\dfrac{(q_i - q_{i-1}) \\cdot (x - x_i)^2}{2!}; $''')

    st.write('''$Q_i(x) = Q_{i-1}(x) + \\Delta Q_i - (q_i - q_{i-1}) \\cdot (x - x_i).$''')

init_data = pd.DataFrame([
        {'xi': '0', 'dv*EI': '0',  'dφ*EI': '!',    'dM': '0', 'dQ': '!',   'q': '0'},
        {'xi': '2', 'dv*EI': '0',  'dφ*EI': '0',    'dM': '0', 'dQ': '-6',  'q': '4'},
        {'xi': '4', 'dv*EI': '0',  'dφ*EI': '!',    'dM': '0', 'dQ': '0',   'q': '4'},
        ])

column_conf = {
        'xi': st.column_config.TextColumn('xi', help='Координата начала участка', required=True, default='0'),
        'dv*EI': st.column_config.TextColumn('dvi*EI', help='Дополнительный прогиб в начале участка', required=True, default='0'),
        'dφ*EI': st.column_config.TextColumn('dφi*EI', help='Дополнительный угол поворота в начале участка', required=True, default='0'),
        'dM': st.column_config.TextColumn('dMi', help='Дополнительный момент в начале участка', required=True, default='0'),
        'dQ': st.column_config.TextColumn('dQi', help='Дополнительная сосредоточенная сила в начале участка', required=True, default='0'),
        'q': st.column_config.TextColumn('qi', help='Интенсивность распределенной нагрузки на участке', required=True, default='0'),
        }
st.write('''Для решения и построения графиков введите в таблицу ниже значения начальных параметров
 для каждого из участков балки, а также общую длину балки в поле под таблицей.
 На место неизвестных начальных параметров впишите ! (просто восклицательный знак).
 Для десятичных значений лучше вводить дробь, например, 2.5=25/10''')
data = st.data_editor(init_data, hide_index=False, use_container_width=True, num_rows='dynamic', column_config=column_conf)
L = st.text_input(label=r'Полная длина балки', value='6')


x_val = data['xi'].tolist()
num_elems = len(x_val)
##st.write(num_elems)
x_val.append(L)
EIdv_val = data['dv*EI'].tolist()
EIdf_val = data['dφ*EI'].tolist()
dM_val = data['dM'].tolist()
dQ_val = data['dQ'].tolist()
q_val = data['q'].tolist()

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

with st.expander('Начальные выражения для функции метода начальных параметров'):
    tabs = st.tabs(['Перемещения','Углы поворота','Моменты','Поперечные силы'])
    with tabs[0]:
        st.write('Функции перемещений по участкам')
        for i in range(len(displacements)):
            st.write('$ v_' + str(i) + '(x)=' + printing.latex(displacements[i])+'$')
    with tabs[1]:
        st.write('Функции углов поворота по участкам')
        for i in range(len(angles)):
            st.write('$ \\varphi_' + str(i) + '(x)=' + printing.latex(angles[i])+'$')
    with tabs[2]:
        st.write('Функции моментов по участкам')
        for i in range(len(moments)):
            st.write('$M_' + str(i) + '(x)=' + printing.latex(moments[i])+'$')
    with tabs[3]:
        st.write('Функции поперечных сил по участкам')
        for i in range(len(forces)):
            st.write('$Q_' + str(i) + '(x)=' + printing.latex(forces[i])+'$')

#Находим неизвестные
unknows = displacements[num_elems-1].free_symbols - set([x])
#Число неизвестных
num_unknows = len(unknows)
st.subheader('Неизвестные величины')
st.latex(unknows)

bc_data = [[0,0,0,0] for i in range(num_unknows)]

##st.write(bc_data)


cols = st.columns([1, 1, 1, 1])
init_bc = [[2, 0, 6, 0], [2, 1, 6, 0], [1, 2, 4, 0]]
if num_unknows>3:
    while len(init_bc)<num_unknows:
     init_bc.append(init_bc[-1])

if num_elems<3:
    init_bc[0][0] = 0
    init_bc[1][0] = 0
    init_bc[2][0] = 0

cols[0].write('Участок')
cols[1].write('Функция')
cols[2].write('Координата')
cols[3].write('Значение')
for i in range(num_unknows):
    cols = st.columns([1, 1, 1, 1])
    bc_data[i][0] = cols[0].selectbox(label='Участок'+str(i), options=[i for i in range(num_elems)], label_visibility='collapsed', index = init_bc[i][0])
    bc_data[i][1] = cols[1].selectbox(label='Функция'+str(i), options=['Прогиб', 'Угол', 'Момент', 'Сила'], label_visibility='collapsed', index = init_bc[i][1])
    bc_data[i][2] = cols[2].text_input(label='Координата'+str(i), value=str(x_val[bc_data[i][0]+1]), label_visibility='collapsed')
    bc_data[i][3] = cols[3].text_input(label='Значение'+str(i), value='0', label_visibility='collapsed')


#Фукция генерирующая граничное условие
def generate_bc (elem, fun, coord, value):
    tmp_val = symbols('tmp_val', real=True)
    if fun == 'Прогиб':
        return Eq(displacements[elem].subs(x,coord), tmp_val).subs(tmp_val,value)
    if fun == 'Угол':
        return Eq(angles[elem].subs(x,coord), tmp_val).subs(tmp_val,value)
    if fun == 'Момент':
        return Eq(moments[elem].subs(x,coord), tmp_val).subs(tmp_val,value)
    if fun == 'Сила':
        return Eq(forces[elem].subs(x,coord), tmp_val).subs(tmp_val,value)

#Массив с граничным условиями
##bc = [(2, 'Прогиб', 6, 0), (2, 'Угол', 6, 0), (1, 'Момент', 4, 0)]
#Генерация системы уравнений на основе г.у.
bc_eqns = []
for i in bc_data:
    bc_eqns.append(generate_bc(i[0], i[1], i[2], i[3]))
#Решение системы уравнений из г.у.
rez_bc = solve(bc_eqns)
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



with st.expander('Функции метода начальных параметров с учетом г.у.'):
    tabs = st.tabs(['Перемещения','Углы поворота','Моменты','Поперечные силы'])
    with tabs[0]:
        st.write('Функции перемещений по участкам')
        for i in range(len(displacements)):
            st.write('$ v_' + str(i) + '(x)=' + printing.latex(displacements[i])+'$')
    with tabs[1]:
        st.write('Функции углов поворота по участкам')
        for i in range(len(angles)):
            st.write('$ \\varphi_' + str(i) + '(x)=' + printing.latex(angles[i])+'$')
    with tabs[2]:
        st.write('Функции моментов по участкам')
        for i in range(len(moments)):
            st.write('$M_' + str(i) + '(x)=' + printing.latex(moments[i])+'$')
    with tabs[3]:
        st.write('Функции поперечных сил по участкам')
        for i in range(len(forces)):
            st.write('$Q_' + str(i) + '(x)=' + printing.latex(forces[i])+'$')
        

num_points = 9
points = []
for i in range(num_elems):
    points.append(np.linspace(float((sympify(x_val[i])).evalf()), float((sympify(x_val[i+1])).evalf()), num_points).tolist())

v = [[0 for k in points[i]] for i in range(num_elems)]
f = [[0 for k in points[i]] for i in range(num_elems)]
M = [[0 for k in points[i]] for i in range(num_elems)]
Q = [[0 for k in points[i]] for i in range(num_elems)]

for i in range(num_elems):
    for p in range(len(points[i])):
        v[i][p] = float(displacements[i].subs(x, points[i][p]))
        f[i][p] = float(angles[i].subs(x, points[i][p]))
        M[i][p] = float(moments[i].subs(x, points[i][p]))
        Q[i][p] = float(forces[i].subs(x, points[i][p]))

##v = np.array(v)
##f = np.array(f)
##M = np.array(M)
##Q = np.array(Q)
##st.write(Q)

def draw_plots():
    fig = make_subplots(rows = 2,cols = 2, subplot_titles=['Перемещения v*EI', 'Углы поворота φ*EI', 'Момент Mz', 'Поперечная сила Qy'])
    for i in range(num_elems):
        fig.add_trace(go.Scatter(x=points[i], y=v[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 1, col = 1)
        fig.add_trace(go.Scatter(x=points[i], y=f[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 1, col = 2)
        fig.add_trace(go.Scatter(x=points[i], y=M[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 2, col = 1)
        fig.add_trace(go.Scatter(x=points[i], y=Q[i], showlegend=False, line=dict(color = "LightSkyBlue")), row = 2, col = 2)

        fig.update_yaxes(autorange="reversed", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=2)
        fig.update_layout(height=500)
##    fig.show(config={ 'modeBarButtonsToRemove': ['zoom', 'pan'] })
    return fig


st.plotly_chart(draw_plots())

        







