"""
Herramienta inteligente para análisis cinético y predicción de estabilidad.

Ejecutar:  streamlit run app.py
"""
import io
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ---------------------------------------------------------------------------
# Constantes y datos de ejemplo
# ---------------------------------------------------------------------------
R_GAS = 8.314462618  # J/(mol·K)
FUENTE_GRAFICAS = "Nunito, sans-serif"  # fuente de títulos y ejes de las gráficas

# tX = tiempo para que quede el X % de la concentración inicial
# (t50 = vida media, t90 = vida útil, etc.)
FRACCIONES = {"t50": 0.50, "t90": 0.90, "t95": 0.95, "t98": 0.98}
NOMBRES_ORDEN = {0: "Orden cero", 1: "Orden uno", 2: "Orden dos"}
TRANSFORMACIONES = {
    0: ("[A]", lambda c: c),
    1: ("ln[A]", np.log),
    2: ("1/[A]", lambda c: 1.0 / c),
}

EJEMPLO_CINETICA = pd.DataFrame(
    {
        "Tiempo": [0, 2, 4, 6, 8, 10, 12],
        "Concentración": [100.0, 90.2, 82.1, 74.5, 66.8, 60.9, 54.5],
    }
)
EJEMPLO_ARRHENIUS = pd.DataFrame(
    {"Temperatura": [25, 35, 45, 55], "k": [0.050, 0.110, 0.229, 0.457]}
)


def vacia_cinetica():
    return pd.DataFrame(
        {"Tiempo": pd.Series(dtype=float), "Concentración": pd.Series(dtype=float)}
    )


def vacia_arrhenius():
    return pd.DataFrame(
        {"Temperatura": pd.Series(dtype=float), "k": pd.Series(dtype=float)}
    )


# ---------------------------------------------------------------------------
# Utilidades de datos
# ---------------------------------------------------------------------------
def a_numerico(serie):
    """Convierte una serie a números (acepta coma decimal)."""
    if not pd.api.types.is_numeric_dtype(serie):
        serie = serie.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(serie, errors="coerce")


def leer_archivo(archivo, hoja_preferida):
    """Lee un Excel (hoja preferida o la primera) o un CSV."""
    if archivo.name.lower().endswith(".csv"):
        return pd.read_csv(archivo, sep=None, engine="python")
    xls = pd.ExcelFile(archivo)
    hoja = hoja_preferida if hoja_preferida in xls.sheet_names else xls.sheet_names[0]
    return xls.parse(hoja)


def plantilla_excel():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        EJEMPLO_CINETICA.to_excel(escritor, sheet_name="Datos", index=False)
        EJEMPLO_ARRHENIUS.to_excel(escritor, sheet_name="Arrhenius", index=False)
    return buffer.getvalue()


def parsear_lista(texto):
    valores = []
    for parte in texto.replace(";", ",").split(","):
        try:
            valores.append(float(parte.strip()))
        except ValueError:
            pass
    return valores


def unidad_k(orden, ut, uc):
    return {0: f"{uc}/{ut}", 1: f"1/{ut}", 2: f"1/({uc}·{ut})"}[orden]


# ---------------------------------------------------------------------------
# Cálculos cinéticos
# ---------------------------------------------------------------------------
def ajustar_ordenes(t, c):
    """Método integral: regresión lineal para orden 0, 1 y 2."""
    resultados = {}
    for orden, (etiqueta, f) in TRANSFORMACIONES.items():
        mascara = np.ones(len(c), dtype=bool) if orden == 0 else (c > 0)
        if mascara.sum() < 3 or len(np.unique(t[mascara])) < 2:
            resultados[orden] = None
            continue
        x = t[mascara]
        y = f(c[mascara])
        lr = stats.linregress(x, y)
        r2 = float(lr.rvalue ** 2) if np.isfinite(lr.rvalue) else np.nan
        k = -lr.slope if orden in (0, 1) else lr.slope
        resultados[orden] = {
            "orden": orden,
            "etiqueta": etiqueta,
            "x": x,
            "y": y,
            "pendiente": float(lr.slope),
            "intercepto": float(lr.intercept),
            "r2": r2,
            "k": float(k),
            "omitidos": int((~mascara).sum()),
            "valido": bool(np.isfinite(r2) and k > 0),
        }
    return resultados


def a0_desde_ajuste(res):
    b = res["intercepto"]
    if res["orden"] == 0:
        return b
    if res["orden"] == 1:
        return float(np.exp(b))
    return 1.0 / b if b > 0 else np.nan


def tiempo_fraccion(orden, k, a0, p):
    """Tiempo para que quede la fracción p de la concentración inicial."""
    k = np.asarray(k, dtype=float)
    if orden == 0:
        return (1 - p) * a0 / k
    if orden == 1:
        return -np.log(p) / k
    return (1 / p - 1) / (k * a0)


def ajustar_arrhenius(temp, k, unidad):
    temp = np.asarray(temp, dtype=float)
    k = np.asarray(k, dtype=float)
    tk = temp + 273.15 if unidad == "°C" else temp
    if np.any(tk <= 0):
        return None
    inv_t = 1.0 / tk
    ln_k = np.log(k)
    if len(np.unique(inv_t)) < 2:
        return None
    lr = stats.linregress(inv_t, ln_k)
    return {
        "Tk": tk,
        "inv_T": inv_t,
        "ln_k": ln_k,
        "pendiente": float(lr.slope),
        "intercepto": float(lr.intercept),
        "r2": float(lr.rvalue ** 2),
        "Ea": float(-lr.slope * R_GAS),  # J/mol
        "Ea_kJ": float(-lr.slope * R_GAS / 1000),
        "A": float(np.exp(lr.intercept)),
        "n": int(len(k)),
    }


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def grafico_ajuste(x, y, pendiente, intercepto, r2, titulo, xlab, ylab):
    xs = np.linspace(np.min(x), np.max(x), 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Datos"))
    fig.add_trace(
        go.Scatter(x=xs, y=pendiente * xs + intercepto, mode="lines", name="Ajuste lineal")
    )
    fig.update_layout(
        font=dict(family=FUENTE_GRAFICAS),
        title=f"{titulo}<br><sup>y = {pendiente:.4g}·x + {intercepto:.4g}  |  R² = {r2:.4f}</sup>",
        xaxis_title=xlab,
        yaxis_title=ylab,
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------------
# Conclusiones (reglas + IA)
# ---------------------------------------------------------------------------
def armar_resumen(ctx):
    ut, uc = ctx["unidad_t"], ctx["unidad_c"]
    L = []
    ajustes = ctx.get("ajustes")
    if ajustes:
        L.append("Método integral (ajustes lineales):")
        for o, r in ajustes.items():
            if r:
                extra = "" if r["valido"] else " (no válido: k <= 0)"
                L.append(
                    f"- {NOMBRES_ORDEN[o]}: R² = {r['r2']:.4f}, pendiente = {r['pendiente']:.4g}, "
                    f"intercepto = {r['intercepto']:.4g}, k = {r['k']:.4g} {unidad_k(o, ut, uc)}{extra}"
                )
    if ctx.get("orden_sel") is not None:
        o = ctx["orden_sel"]
        L.append(
            f"Modelo usado: {NOMBRES_ORDEN[o]}; k = {ctx['k_sel']:.4g} {unidad_k(o, ut, uc)}; "
            f"[A]0 = {ctx['a0_sel']:.4g} {uc}"
        )
        for nombre, v in ctx["tiempos"].items():
            L.append(f"- {nombre} = {v:.4g} {ut}")
    arr = ctx.get("arr")
    if arr:
        L.append(
            f"Arrhenius: Ea = {arr['Ea_kJ']:.2f} kJ/mol, A = {arr['A']:.4g}, "
            f"R² = {arr['r2']:.4f}, n = {arr['n']} temperaturas"
        )
    pred = ctx.get("pred")
    if pred is not None:
        L.append(f"Predicción (temperatura en {ctx['unidad_T']}, tiempos en {ut}):")
        L.append(pred.round(5).to_string(index=False))
    return "\n".join(L)


def conclusiones_reglas(ctx):
    ut = ctx["unidad_t"]
    P = []
    ajustes = ctx.get("ajustes")
    mejor = ctx.get("mejor")
    if ajustes and mejor is not None:
        r = ajustes[mejor]
        otros = ", ".join(
            f"{NOMBRES_ORDEN[o]} (R² = {x['r2']:.4f})"
            for o, x in ajustes.items()
            if x and o != mejor
        )
        txt = (
            f"El modelo que mejor describe los datos es **{NOMBRES_ORDEN[mejor].lower()}** "
            f"(R² = {r['r2']:.4f}), frente a {otros}."
        )
        if ctx.get("margen") is not None and ctx["margen"] < 0.01:
            txt += " La diferencia entre modelos es pequeña, así que conviene confirmar con más datos."
        if r["r2"] < 0.95:
            txt += " El R² es bajo: revisa los datos o considera otro mecanismo."
        P.append(txt)
    if ctx.get("orden_sel") is not None:
        t = ctx["tiempos"]
        P.append(
            f"Con k = {ctx['k_sel']:.4g} {unidad_k(ctx['orden_sel'], ut, ctx['unidad_c'])}, "
            f"la vida media (t50) es {t['t50']:.4g} {ut} y la vida útil (t90) es {t['t90']:.4g} {ut}; "
            f"t95 = {t['t95']:.4g} {ut} y t98 = {t['t98']:.4g} {ut}."
        )
    arr = ctx.get("arr")
    if arr:
        q10 = float(np.exp(arr["Ea"] / R_GAS * (1 / 298.15 - 1 / 308.15)))
        txt = (
            f"El módulo de Arrhenius da Ea = {arr['Ea_kJ']:.1f} kJ/mol y A = {arr['A']:.3g} "
            f"(R² = {arr['r2']:.4f}). Entre 25 y 35 °C la constante k se multiplica por ≈ {q10:.2f} (Q10)."
        )
        if arr["n"] < 3:
            txt += " Con menos de 3 temperaturas el ajuste no es confiable."
        if arr["Ea"] < 0:
            txt += " La Ea negativa no es esperable: revisa los datos."
        P.append(txt)
    pred = ctx.get("pred")
    if pred is not None and len(pred) >= 2:
        p = pred.sort_values("Temp")
        lo, hi = p.iloc[0], p.iloc[-1]
        cambio = (hi["t90"] / lo["t90"] - 1) * 100
        P.append(
            f"Al subir la temperatura de {lo['Temp']:g} a {hi['Temp']:g} {ctx['unidad_T']}, "
            f"t90 pasa de {lo['t90']:.4g} a {hi['t90']:.4g} {ut} ({cambio:+.1f} %). "
            "Las predicciones fuera del rango de temperaturas experimentales son extrapolaciones."
        )
    return P


PROMPT_IA = (
    "Eres un experto en cinética química y estudios de estabilidad. A partir de los siguientes "
    "resultados, redacta conclusiones claras en español (máximo 200 palabras): qué orden de "
    "reacción es más probable y por qué, qué significan k, t50 y t90 para la estabilidad, qué "
    "indica la energía de activación sobre la sensibilidad a la temperatura, y recomendaciones de "
    "almacenamiento. Señala limitaciones (pocos datos, extrapolación, R² bajo) si las hay.\n\n"
    "RESULTADOS:\n"
)


def conclusiones_ia(resumen, api_key, modelo):
    import anthropic

    cliente = anthropic.Anthropic(api_key=api_key)
    r = cliente.messages.create(
        model=modelo,
        max_tokens=1000,
        messages=[{"role": "user", "content": PROMPT_IA + resumen}],
    )
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


# ---------------------------------------------------------------------------
# Estilo visual: hover y pestañas (el resto está en .streamlit/config.toml)
# ---------------------------------------------------------------------------
PALETAS_CSS = {
    "light": {  # Neapolitan
        "acento": "#604734",       # chocolate: pestaña activa y subrayado
        "hover_tab": "#f6d8d6",    # fondo al pasar el mouse sobre una pestaña
        "texto_tab": "#604734",
        "hover_boton": "#f0c5d4",  # fondo al pasar el mouse sobre un botón
        "texto_boton": "#604734",
    },
    "dark": {  # Rosa oscuro
        "acento": "#e5a9bf",
        "hover_tab": "#372d36",
        "texto_tab": "#fdd6dd",
        "hover_boton": "#e5a9bf",
        "texto_boton": "#241a22",
    },
}

CSS_ESTILO = Template("""<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@600;700&display=swap');
h1 { font-family: 'Dancing Script', cursive !important; font-size: 3.2rem !important; }
button[data-baseweb="tab"] { border-radius: 8px 8px 0 0; transition: background-color .15s; }
button[data-baseweb="tab"]:hover { background-color: $hover_tab !important; }
button[data-baseweb="tab"]:hover p { color: $texto_tab !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: $acento !important; font-weight: 700; }
div[data-baseweb="tab-highlight"] { background-color: $acento !important; }
[data-testid="stBaseButton-secondary"]:hover { background-color: $hover_boton !important; border-color: $acento !important; }
[data-testid="stBaseButton-secondary"]:hover p { color: $texto_boton !important; }
</style>""")


def aplicar_estilo():
    try:
        tema = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        tema = "light"
    st.markdown(CSS_ESTILO.substitute(PALETAS_CSS[tema]), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Cinética y estabilidad", page_icon="⌬", layout="wide")
    aplicar_estilo()
    st.title("🧪 Análisis cinético y predicción de estabilidad")

    if "base_cin" not in st.session_state:
        st.session_state.base_cin = EJEMPLO_CINETICA.copy()
        st.session_state.ver_cin = 0
        st.session_state.base_arr = EJEMPLO_ARRHENIUS.copy()
        st.session_state.ver_arr = 0

    ctx = {}
    tabs = st.tabs(
        ["1. Datos", "2. Gráficos", "3. Orden y parámetros", "4. Arrhenius", "5. Predicción", "6. Conclusiones"]
    )

    # ------------------------------------------------------------------ DATOS
    with tabs[0]:
        st.subheader("Datos experimentales")
        c1, c2 = st.columns(2)
        unidad_t = c1.text_input("Unidad de tiempo", "h")
        unidad_c = c2.text_input("Unidad de concentración", "mg/mL")
        ctx["unidad_t"], ctx["unidad_c"] = unidad_t, unidad_c

        st.download_button(
            "⇩ Descargar plantilla de Excel",
            plantilla_excel(),
            file_name="plantilla_cinetica.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        modo = st.radio(
            "¿Cómo quieres ingresar los datos?",
            ["Tabla en la app", "Cargar Excel / CSV"],
            horizontal=True,
            key="modo_cin",
        )

        tiempo_serie = conc_serie = None
        if modo == "Tabla en la app":
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("Cargar ejemplo", key="ej_cin"):
                st.session_state.base_cin = EJEMPLO_CINETICA.copy()
                st.session_state.ver_cin += 1
                st.rerun()
            if b2.button("Vaciar tabla", key="vac_cin"):
                st.session_state.base_cin = vacia_cinetica()
                st.session_state.ver_cin += 1
                st.rerun()
            tabla = st.data_editor(
                st.session_state.base_cin,
                num_rows="dynamic",
                key=f"editor_cin_{st.session_state.ver_cin}",
            )
            if tabla.shape[1] >= 2:
                tiempo_serie, conc_serie = tabla.iloc[:, 0], tabla.iloc[:, 1]
        else:
            archivo = st.file_uploader(
                "Archivo con una columna de tiempo y otra de concentración",
                type=["xlsx", "xls", "csv"],
                key="arch_cin",
            )
            if archivo is not None:
                try:
                    crudo = leer_archivo(archivo, "Datos")
                except Exception as e:
                    st.error(f"No pude leer el archivo: {e}")
                else:
                    nombres = [str(c) for c in crudo.columns]
                    if len(nombres) < 2:
                        st.error("El archivo necesita al menos 2 columnas.")
                    else:
                        d1, d2 = st.columns(2)
                        ic_t = d1.selectbox(
                            "Columna de tiempo", range(len(nombres)), 0, format_func=lambda i: nombres[i], key="col_t"
                        )
                        ic_c = d2.selectbox(
                            "Columna de concentración", range(len(nombres)), 1, format_func=lambda i: nombres[i], key="col_c"
                        )
                        tiempo_serie, conc_serie = crudo.iloc[:, ic_t], crudo.iloc[:, ic_c]
                        st.dataframe(crudo.head(20))

        datos_ok = False
        t = c = None
        if tiempo_serie is not None:
            ts, cs = a_numerico(tiempo_serie), a_numerico(conc_serie)
            valido = ts.notna() & cs.notna()
            t, c = ts[valido].to_numpy(float), cs[valido].to_numpy(float)
            orden_idx = np.argsort(t)
            t, c = t[orden_idx], c[orden_idx]
            if len(t) >= 3 and len(np.unique(t)) >= 2:
                datos_ok = True
                st.success(f"{len(t)} puntos válidos. Continúa en la pestaña «Gráficos».")
            else:
                st.info("Ingresa al menos 3 puntos con tiempos distintos.")

    ajustes = mejor = None
    validos = {}
    margen = None
    if datos_ok:
        ajustes = ajustar_ordenes(t, c)
        validos = {o: r for o, r in ajustes.items() if r and r["valido"]}
        if validos:
            mejor = max(validos, key=lambda o: validos[o]["r2"])
            r2s = sorted((r["r2"] for r in validos.values()), reverse=True)
            margen = r2s[0] - r2s[1] if len(r2s) > 1 else None
    ctx.update(ajustes=ajustes, mejor=mejor, margen=margen)

    # --------------------------------------------------------------- GRÁFICOS
    with tabs[1]:
        st.subheader("Método integral: gráficos y regresión lineal")
        if not datos_ok:
            st.info("Primero ingresa los datos en la pestaña «Datos».")
        else:
            columnas = st.columns(3)
            ylabs = {0: f"[A] ({unidad_c})", 1: "ln[A]", 2: f"1/[A] (1/{unidad_c})"}
            for col, orden in zip(columnas, (0, 1, 2)):
                r = ajustes[orden]
                with col:
                    if r is None:
                        st.warning(f"{NOMBRES_ORDEN[orden]}: no hay suficientes puntos válidos.")
                        continue
                    fig = grafico_ajuste(
                        r["x"], r["y"], r["pendiente"], r["intercepto"], r["r2"],
                        f"{r['etiqueta']} vs t", f"Tiempo ({unidad_t})", ylabs[orden],
                    )
                    st.plotly_chart(fig, key=f"graf_{orden}")
                    if r["omitidos"]:
                        st.caption(f"Se omitieron {r['omitidos']} punto(s) con [A] ≤ 0.")
            filas = []
            for o, r in ajustes.items():
                if r is None:
                    continue
                filas.append(
                    {
                        "Modelo": NOMBRES_ORDEN[o],
                        "Gráfico": f"{r['etiqueta']} vs t",
                        "Pendiente": f"{r['pendiente']:.5g}",
                        "Intercepto": f"{r['intercepto']:.5g}",
                        "R²": f"{r['r2']:.5f}",
                        "k": f"{r['k']:.5g} {unidad_k(o, unidad_t, unidad_c)}",
                        "¿k > 0?": "Sí" if r["k"] > 0 else "No",
                    }
                )
            st.dataframe(pd.DataFrame(filas), hide_index=True)

    # ------------------------------------------------------ ORDEN Y PARÁMETROS
    ajuste_ok = False
    orden_sel = k_sel = a0_sel = None
    tiempos = {}
    with tabs[2]:
        st.subheader("Orden de reacción y parámetros cinéticos")
        if not datos_ok:
            st.info("Primero ingresa los datos en la pestaña «Datos».")
        elif mejor is None:
            st.error("Ningún modelo da una constante k positiva. Revisa tus datos.")
        else:
            st.success(
                f"Modelo cinético más probable: **{NOMBRES_ORDEN[mejor]}** "
                f"(R² = {validos[mejor]['r2']:.4f})"
            )
            if margen is not None and margen < 0.01:
                st.warning(
                    "La diferencia de R² entre los mejores modelos es menor a 0.01. "
                    "Conviene tener más puntos o revisar los residuales."
                )
            opciones = list(validos)
            orden_sel = st.selectbox(
                "Orden a usar en los cálculos (por defecto, el de mayor R²)",
                opciones,
                index=opciones.index(mejor),
                format_func=lambda o: NOMBRES_ORDEN[o],
            )
            origen_a0 = st.radio(
                "[A]₀ para los cálculos",
                ["Primer dato experimental", "Intercepto del ajuste"],
                horizontal=True,
            )
            r_sel = validos[orden_sel]
            a0_sel = c[0] if origen_a0 == "Primer dato experimental" else a0_desde_ajuste(r_sel)
            if not np.isfinite(a0_sel) or a0_sel <= 0:
                st.warning("El intercepto no da un [A]₀ válido; se usa el primer dato experimental.")
                a0_sel = c[0]
            k_sel = r_sel["k"]
            tiempos = {n: float(tiempo_fraccion(orden_sel, k_sel, a0_sel, p)) for n, p in FRACCIONES.items()}
            ajuste_ok = True

            m = st.columns(5)
            m[0].metric(f"k ({unidad_k(orden_sel, unidad_t, unidad_c)})", f"{k_sel:.4g}")
            for col, (n, v) in zip(m[1:], tiempos.items()):
                col.metric(f"{n} ({unidad_t})", f"{v:.4g}")
            st.caption(
                "tX = tiempo para que quede el X % de la concentración inicial "
                "(t50 = vida media, t90 = vida útil)."
            )
            with st.expander("Ecuaciones utilizadas"):
                ecs = {
                    0: (r"[A] = [A]_0 - k\,t", r"t_p = \frac{(1-p)\,[A]_0}{k}"),
                    1: (r"\ln[A] = \ln[A]_0 - k\,t", r"t_p = -\frac{\ln p}{k}"),
                    2: (r"\frac{1}{[A]} = \frac{1}{[A]_0} + k\,t", r"t_p = \frac{1/p - 1}{k\,[A]_0}"),
                }
                st.latex(ecs[orden_sel][0])
                st.latex(ecs[orden_sel][1])
                st.write(f"[A]₀ = {a0_sel:.4g} {unidad_c}; p = 0.50, 0.90, 0.95, 0.98.")
    ctx.update(orden_sel=orden_sel, k_sel=k_sel, a0_sel=a0_sel, tiempos=tiempos)

    # -------------------------------------------------------------- ARRHENIUS
    arr = None
    unidad_T = "°C"
    with tabs[3]:
        st.subheader("Módulo Arrhenius")
        unidad_T = st.radio("Unidad de temperatura", ["°C", "K"], horizontal=True, key="uni_T")
        modo_arr = st.radio(
            "¿Cómo quieres ingresar las constantes k?",
            ["Tabla en la app", "Cargar Excel / CSV"],
            horizontal=True,
            key="modo_arr",
        )
        temp_serie = k_serie = None
        if modo_arr == "Tabla en la app":
            b1, b2, _ = st.columns([1, 1, 3])
            if b1.button("Cargar ejemplo", key="ej_arr"):
                st.session_state.base_arr = EJEMPLO_ARRHENIUS.copy()
                st.session_state.ver_arr += 1
                st.rerun()
            if b2.button("Vaciar tabla", key="vac_arr"):
                st.session_state.base_arr = vacia_arrhenius()
                st.session_state.ver_arr += 1
                st.rerun()
            tabla_arr = st.data_editor(
                st.session_state.base_arr,
                num_rows="dynamic",
                key=f"editor_arr_{st.session_state.ver_arr}",
            )
            if tabla_arr.shape[1] >= 2:
                temp_serie, k_serie = tabla_arr.iloc[:, 0], tabla_arr.iloc[:, 1]
            if ajuste_ok:
                st.markdown("**Agregar la k del análisis actual a la tabla**")
                ca, cb = st.columns([2, 1])
                T_exp = ca.number_input(
                    f"Temperatura de este experimento ({unidad_T})", value=25.0, key="T_exp"
                )
                if cb.button("Agregar a la tabla", key="add_k"):
                    nuevo = pd.DataFrame({"Temperatura": [T_exp], "k": [k_sel]})
                    actual = tabla_arr.iloc[:, :2].set_axis(["Temperatura", "k"], axis=1)
                    st.session_state.base_arr = pd.concat([actual, nuevo], ignore_index=True)
                    st.session_state.ver_arr += 1
                    st.rerun()
        else:
            archivo_arr = st.file_uploader(
                "Archivo con temperatura y k", type=["xlsx", "xls", "csv"], key="arch_arr"
            )
            if archivo_arr is not None:
                try:
                    crudo_arr = leer_archivo(archivo_arr, "Arrhenius")
                except Exception as e:
                    st.error(f"No pude leer el archivo: {e}")
                else:
                    nombres = [str(x) for x in crudo_arr.columns]
                    if len(nombres) < 2:
                        st.error("El archivo necesita al menos 2 columnas.")
                    else:
                        d1, d2 = st.columns(2)
                        i_T = d1.selectbox(
                            "Columna de temperatura", range(len(nombres)), 0, format_func=lambda i: nombres[i], key="col_T"
                        )
                        i_k = d2.selectbox(
                            "Columna de k", range(len(nombres)), 1, format_func=lambda i: nombres[i], key="col_k"
                        )
                        temp_serie, k_serie = crudo_arr.iloc[:, i_T], crudo_arr.iloc[:, i_k]

        if temp_serie is not None:
            Ts, ks = a_numerico(temp_serie), a_numerico(k_serie)
            ok = Ts.notna() & ks.notna() & (ks > 0)
            Ts, ks = Ts[ok].to_numpy(float), ks[ok].to_numpy(float)
            if len(Ts) < 2:
                st.info("Ingresa al menos 2 temperaturas con k > 0 (idealmente 3 o más).")
            else:
                arr = ajustar_arrhenius(Ts, ks, unidad_T)
                if arr is None:
                    st.error("Revisa las temperaturas: deben ser distintas y mayores a 0 K.")
                else:
                    if arr["n"] < 3:
                        st.warning("Con solo 2 temperaturas el R² siempre es 1. Usa 3 o más.")
                    tabla_calc = pd.DataFrame(
                        {
                            f"T ({unidad_T})": Ts,
                            "T (K)": arr["Tk"],
                            "1/T (1/K)": arr["inv_T"],
                            "k": ks,
                            "ln(k)": arr["ln_k"],
                        }
                    )
                    st.dataframe(tabla_calc, hide_index=True)
                    m = st.columns(4)
                    m[0].metric("Ea (kJ/mol)", f"{arr['Ea_kJ']:.2f}")
                    m[1].metric("A (factor preexponencial)", f"{arr['A']:.4g}")
                    m[2].metric("R²", f"{arr['r2']:.4f}")
                    m[3].metric("Pendiente (−Ea/R)", f"{arr['pendiente']:.5g}")
                    st.plotly_chart(
                        grafico_ajuste(
                            arr["inv_T"], arr["ln_k"], arr["pendiente"], arr["intercepto"], arr["r2"],
                            "ln(k) vs 1/T", "1/T (1/K)", "ln(k)",
                        ),
                        key="graf_arr",
                    )
                    st.latex(r"\ln k = \ln A - \frac{E_a}{R}\cdot\frac{1}{T}")
                    if arr["Ea"] < 0:
                        st.warning("La Ea salió negativa: revisa los datos.")
    ctx.update(arr=arr, unidad_T=unidad_T)

    # -------------------------------------------------------------- PREDICCIÓN
    pred = None
    with tabs[4]:
        st.subheader("Predicción de estabilidad")
        if not ajuste_ok:
            st.info("Completa el análisis cinético (pestañas 1 a 3).")
        elif arr is None:
            st.info("Completa el módulo Arrhenius (pestaña 4).")
        else:
            st.caption(
                f"Se usa {NOMBRES_ORDEN[orden_sel].lower()} y [A]₀ = {a0_sel:.4g} {unidad_c}. "
                "Asegúrate de que las k del módulo Arrhenius correspondan a ese mismo orden."
            )
            c1, c2 = st.columns(2)
            txt = c1.text_input(
                f"Temperaturas a predecir ({unidad_T}), separadas por coma", "4, 25, 30, 40"
            )
            T_ref = c2.number_input(
                f"Temperatura de referencia ({unidad_T})",
                value=25.0 if unidad_T == "°C" else 298.15,
            )
            lista = parsear_lista(txt)
            if not lista:
                st.warning("Escribe al menos una temperatura válida.")
            else:
                temps = np.array(sorted(set(lista) | {float(T_ref)}))
                tk = temps + 273.15 if unidad_T == "°C" else temps
                if np.any(tk <= 0):
                    st.error("Hay temperaturas menores o iguales a 0 K.")
                else:
                    k_T = arr["A"] * np.exp(-arr["Ea"] / (R_GAS * tk))
                    pred = pd.DataFrame({"Temp": temps, "k": k_T})
                    for n, p in FRACCIONES.items():
                        pred[n] = tiempo_fraccion(orden_sel, k_T, a0_sel, p)

                    ref = pred.loc[np.isclose(pred["Temp"], T_ref)].iloc[0]
                    mostrar = pred.copy()
                    for n in FRACCIONES:
                        mostrar[f"Δ{n} (%)"] = (pred[n] / ref[n] - 1) * 100
                    mostrar = mostrar.rename(columns={"Temp": f"T ({unidad_T})"})
                    st.markdown(f"**Cambios respecto a {T_ref:g} {unidad_T}** (tiempos en {unidad_t})")
                    st.dataframe(mostrar.round(5), hide_index=True)

                    if np.any(temps < Ts.min()) or np.any(temps > Ts.max()):
                        st.caption("⚠️ Hay temperaturas fuera del rango experimental: son extrapolaciones.")

                    g1, g2 = st.columns(2)
                    fig_k = go.Figure(go.Scatter(x=pred["Temp"], y=pred["k"], mode="lines+markers"))
                    fig_k.update_layout(
                        font=dict(family=FUENTE_GRAFICAS),
                        title="k predicha vs temperatura",
                        xaxis_title=f"Temperatura ({unidad_T})",
                        yaxis_title="k",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    g1.plotly_chart(fig_k, key="graf_pred_k")
                    fig_t = go.Figure()
                    for n in FRACCIONES:
                        fig_t.add_trace(go.Scatter(x=pred["Temp"], y=pred[n], mode="lines+markers", name=n))
                    fig_t.update_layout(
                        font=dict(family=FUENTE_GRAFICAS),
                        title="t50, t90, t95 y t98 vs temperatura",
                        xaxis_title=f"Temperatura ({unidad_T})",
                        yaxis_title=f"Tiempo ({unidad_t})",
                        yaxis_type="log",
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    g2.plotly_chart(fig_t, key="graf_pred_t")
    ctx["pred"] = pred

    # ----------------------------------------------------------- CONCLUSIONES
    with tabs[5]:
        st.subheader("Conclusiones:")
        if not datos_ok:
            st.info("Ingresa datos para generar conclusiones.")
        else:
            st.markdown("#### Conclusiones basadas en reglas (siempre disponibles)")
            for parrafo in conclusiones_reglas(ctx):
                st.markdown(parrafo)

if __name__ == "__main__":
    main()
