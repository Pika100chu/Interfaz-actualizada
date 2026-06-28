

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

st.set_page_config(page_title="Simulador Oligopolio", layout="wide")

st.title("Simulador de Oligopolio")
st.caption("Cournot admite múltiples empresas. Bertrand y Stackelberg se mantienen en 2 empresas.")
st.write("Modelos disponibles: Cournot, Bertrand y Stackelberg")

# =====================
# FUNCIONES ECONÓMICAS
# =====================


def cournot_n(a,b,costos,costos_fijos):

    n = len(costos)

    A = np.full((n,n), b)
    np.fill_diagonal(A, 2*b)

    B = np.array([a-c for c in costos])

    cantidades = np.linalg.solve(A,B)
    cantidades = np.maximum(cantidades,0)

    Q_star = np.sum(cantidades)
    P_star = a - b*Q_star
    beneficios = (P_star - np.array(costos))*cantidades - np.array(costos_fijos)

    return cantidades,Q_star,P_star,beneficios

def stackelberg(a,b,c1,c2,F1,F2):

    def BR2(q1):
        return max((a-c2-b*q1)/(2*b),0)

    def beneficio_lider(q1):
        q2 = BR2(q1)
        P = a-b*(q1+q2)
        return (P-c1)*q1

    q1_grid = np.linspace(0,a/b,1000)
    beneficios = [beneficio_lider(q1) for q1 in q1_grid]

    q1_star = q1_grid[np.argmax(beneficios)]
    q2_star = BR2(q1_star)

    Q_star = q1_star+q2_star
    P_star = a-b*Q_star

    pi1 = (P_star-c1)*q1_star - F1
    pi2 = (P_star-c2)*q2_star - F2

    return q1_star,q2_star,Q_star,P_star,pi1,pi2


def bertrand(a,b,c1,c2,F1,F2):

    if c1 == c2:
        P_star = c1
        Q_star = (a-P_star)/b
        q1 = Q_star/2
        q2 = Q_star/2
        pi1 = -F1
        pi2 = -F2

    elif c1 < c2:
        P_star = c2
        Q_star = (a-P_star)/b
        q1 = Q_star
        q2 = 0
        pi1 = (P_star-c1)*q1 - F1
        pi2 = -F2

    else:
        P_star = c1
        Q_star = (a-P_star)/b
        q1 = 0
        q2 = Q_star
        pi1 = -F1
        pi2 = (P_star-c2)*q2 - F2

    return q1,q2,Q_star,P_star,pi1,pi2


# =====================
# GRÁFICOS
# =====================

def grafico_mercado(a,b,Q_star,P_star,titulo):
    
# Cantidad máxima donde la demanda llega a precio 0
    Q_max = a/b
# Genera muchos valores de Q para dibujar la curva
    Q = np.linspace(0,Q_max,300)
# Función de demanda inversa: P = a - bQ
    P_demanda = a-b*Q

# Corta la curva cuando el precio se vuelve negativo
    mask = P_demanda >= 0
    Q = Q[mask]
    P_demanda = P_demanda[mask]
    
# Crea la figura y los ejes
    fig, ax = plt.subplots(figsize=(6,4))
# Grafica la curva de demanda
    ax.plot(Q,P_demanda,label="Demanda")
 # Marca el punto de equilibrio
    ax.scatter(Q_star,P_star,s=250,zorder=3)

# Línea vertical desde el equilibrio hasta el eje X
    ax.plot([Q_star,Q_star],[0,P_star], linestyle="--",color="black")
# Línea horizontal desde el equilibrio hasta el eje Y
    ax.plot([0,Q_star],[P_star,P_star], linestyle="--",color="black")

# Hace que ambos ejes comiencen en 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

# Etiquetas de los ejes
    ax.set_xlabel("Cantidad total Q")
    ax.set_ylabel("Precio P")
# Título del gráfico
    ax.set_title(titulo)
# Agrega cuadrícula y leyenda
    ax.grid()
    ax.legend()

    return fig


def grafico_cournot(a,b,c1,c2,q1_star,q2_star):
# Valores posibles de producción para graficar
    q = np.linspace(0,a,300)

# Curva de reacción de la Empresa 1
    BR1 = (a-c1-q)/(2*b)
# Curva de reacción de la Empresa 2
    BR2 = (a-c2-q)/(2*b)

# Elimina los tramos donde la producción sería negativa
    mask1 = BR1 >= 0
    mask2 = BR2 >= 0

    q_BR1 = q[mask1]
    BR1 = BR1[mask1]

    q_BR2 = q[mask2]
    BR2 = BR2[mask2]
# Crea figura y ejes
    fig, ax = plt.subplots(figsize=(6,4))
    
# Grafica la reacción óptima de la Empresa 1
    ax.plot(q_BR1,BR1,label='Curva reaccion Empresa 1')
# Grafica la reacción óptima de la Empresa 2
    ax.plot(BR2,q_BR2,label='Curva reaccion Empresa 2')

# Marca el equilibrio de Cournot
    ax.scatter(q2_star,q1_star,s=250,zorder=3)

# Líneas punteada hasta los ejes
    ax.plot([q2_star,q2_star],[0,q1_star], linestyle='--',color="black")
    ax.plot([0,q2_star],[q1_star,q1_star], linestyle='--',color="black")

# Ejes comenzando en 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

# Etiquetas
    ax.set_xlabel('q2')
    ax.set_ylabel('q1')
# Título del gráfico
    ax.set_title('Cournot - Curvas de reacción')
# Leyenda y cuadrícula
    ax.legend()
    ax.grid()

    return fig


def grafico_stackelberg(a,b,c2,q1_star,q2_star):

# Posibles cantidades de la empresa líder
    q1 = np.linspace(0,a/b,300)
# Función de reacción de la empresa seguidora
    BR2 = (a-c2-b*q1)/(2*b)

# cortar cuando P llega a 0 (evita cantidades negativas)
    mask = BR2 >= 0
    q1_plot = q1[mask]
    BR2 = BR2[mask]
# Crea figura y ejes
    fig, ax = plt.subplots(figsize=(6,4))
    
# Grafica la función de reacción de la seguidora
    ax.plot(q1_plot,BR2,label='Reaccion de Seguidora')
    ax.scatter(q1_star,q2_star,s=250,zorder=3)

# Líneas auxiliares hasta los ejes
    ax.plot([q1_star,q1_star],[0,q2_star], linestyle='--',color="black")
    ax.plot([0,q1_star],[q2_star,q2_star], linestyle='--',color="black")
# Ejes comenzando en 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

# Etiquetas
    ax.set_xlabel('q1 Empresa líder')
    ax.set_ylabel('q2 Empresa seguidora')
# Título del grafico
    ax.set_title('Stackelberg - Curvas de reaccion')
# Cuadrícula y leyenda
    ax.grid()
    ax.legend()

    return fig


# =====================
# INTERFAZ
# =====================

modelo = st.selectbox(
    "Seleccione modelo",
    ["Cournot","Bertrand","Stackelberg"]
)

col1,col2,col3 = st.columns(3)

with col1:
    a = st.number_input("a (intercepto demanda)", min_value=0.01, value=100.0)
    b = st.number_input("b (pendiente demanda)", min_value=0.01, value=1.0)

with col2:
    if modelo == "Cournot":
        n_empresas = st.number_input("Cantidad de empresas", min_value=2, max_value=10, value=2, step=1)
    else:
        n_empresas = 2
        st.write("Cantidad de empresas: 2 (fijo para este modelo)")


costos = []
costos_fijos = []

with col3:
    st.subheader("Parámetros por empresa")

    for i in range(int(n_empresas)):
        with st.expander(f"Empresa {i+1}", expanded=(i < 2)):
            cm = st.number_input(
                f"Costo marginal empresa {i+1}",
                value=float(10),
                key=f"cm_{i}"
            )

            cf = st.number_input(
                f"Costo fijo empresa {i+1}",
                min_value=0.0,
                value=0.0,
                step=1.0,
                key=f"cf_{i}"
            )

            costos.append(cm)
            costos_fijos.append(cf)

c1 = costos[0]
c2 = costos[1]
F1 = costos_fijos[0]
F2 = costos_fijos[1]


if st.button("Calcular equilibrio"):

    if modelo == "Cournot":
        cantidades,Q_star,P_star,beneficios = cournot_n(a,b,costos,costos_fijos)

    elif modelo == "Bertrand":
        resultado = bertrand(a,b,c1,c2,F1,F2)

    else:
        resultado = stackelberg(a,b,c1,c2,F1,F2)

    if modelo == "Cournot":
        q1_star = cantidades[0]
        q2_star = cantidades[1]
        pi1 = beneficios[0]
        pi2 = beneficios[1]
    else:
        q1_star,q2_star,Q_star,P_star,pi1,pi2 = resultado

    st.subheader(f"Equilibrio {modelo}")

    c1r,c2r,c3r = st.columns(3)

    if modelo == "Cournot":
        for i,q in enumerate(cantidades):
            c1r.metric(f"Cantidad empresa {i+1}", f"{q:.2f}")
    else:
        c1r.metric("Cantidad empresa 1", f"{q1_star:.2f}")
        c1r.metric("Cantidad empresa 2", f"{q2_star:.2f}")

    c2r.metric("Cantidades totales", f"{Q_star:.2f}")
    c2r.metric("P", f"{P_star:.2f}")

    if modelo == "Cournot":
        for i,pi in enumerate(beneficios):
            c3r.metric(f"Beneficio empresa {i+1}", f"{pi:.2f}")
    else:
        c3r.metric("Beneficio empresa 1", f"{pi1:.2f}")
        c3r.metric("Beneficio empresa 2", f"{pi2:.2f}")

    if modelo == "Cournot":
        if n_empresas == 2:
            st.pyplot(grafico_cournot(a,b,c1,c2,q1_star,q2_star))
        else:
            st.info("Las funciones de reacción solo se muestran para 2 empresas.")
        st.pyplot(grafico_mercado(a,b,Q_star,P_star,"Mercado Cournot"))

    elif modelo == "Bertrand":
        st.pyplot(grafico_mercado(a,b,Q_star,P_star,"Mercado Bertrand"))

    else:
        st.pyplot(grafico_stackelberg(a,b,c2,q1_star,q2_star))
        st.pyplot(grafico_mercado(a,b,Q_star,P_star,"Mercado Stackelberg"))

    st.subheader("Comparación de modelos")

    cournot_q,cournot_Q,cournot_P,cournot_pi = cournot_n(a,b,costos,costos_fijos)
    bertrand_res = bertrand(a,b,c1,c2,F1,F2)
    stack_res = stackelberg(a,b,c1,c2,F1,F2)

    comparacion = pd.DataFrame({
        "Modelo":["Cournot","Bertrand","Stackelberg"],
        "Precio":[cournot_P, bertrand_res[3], stack_res[3]],
        "Cantidad":[cournot_Q, bertrand_res[2], stack_res[2]],
        "Beneficio 1":[cournot_pi[0], bertrand_res[4], stack_res[4]],
        "Beneficio 2":[cournot_pi[1] if len(cournot_pi)>1 else 0, bertrand_res[5], stack_res[5]]
    })

    st.dataframe(comparacion, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(comparacion["Modelo"], comparacion["Precio"], color=["royalblue","#E34D4D","seagreen"])
    ax.set_title("Comparación de precios")
    ax.set_ylabel("Precio")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.bar(comparacion["Modelo"], comparacion["Cantidad"], color=["royalblue","#E34D4D","seagreen"])
    ax2.set_title("Producción total")
    ax2.set_ylabel("Cantidad")
    st.pyplot(fig2)
