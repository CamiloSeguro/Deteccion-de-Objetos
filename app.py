# app.py — Detección de Objetos (YOLOv5/v8) con Streamlit
import io
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List

st.set_page_config(page_title="Detección de Objetos", page_icon="🔍", layout="wide")

st.title("🔍 Detección de Objetos en Imágenes")
st.caption("Compatible con **Ultralytics YOLO (v8/v5su)** y **YOLOv5 (torch.hub)** · Streamlit + PyTorch")

# ─────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def pil_bytes_from_bgr(img_bgr) -> bytes:
    _, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes()

def list_to_none_if_empty(lst):
    return None if (not lst or len(lst) == 0) else lst

# ─────────────────────────────────────────────────────────────
# Carga de modelo (robusta y cacheada)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model(
    prefer_ultralytics: bool = True,
    yolo_variant: str = "auto",  # "auto", "ultralytics", "yolov5"
    weights: str = "yolov5s.pt",
):
    """
    Devuelve (backend, model, names, runner_fn)
      backend: "ultralytics" | "yolov5"
      model: objeto del modelo
      names: dict/Lista de nombres de clase
      runner_fn(img_bgr, params) -> (df, img_annotated_bgr)
    """
    # Intentar Ultralytics: YOLOv8 o v5su si existen
    if prefer_ultralytics and (yolo_variant in ("auto", "ultralytics")):
        try:
            from ultralytics import YOLO  # pip install ultralytics

            # Si piden yolov5s, cargamos modelo liviano equivalente v8 (yolov8n) o 'yolov5su.pt' si disponible
            # El usuario puede cambiar weights en la UI si quiere otro.
            m = YOLO(weights)  # puede ser yolov8n.pt, yolov8s.pt, yolov5su.pt, yolov5s.pt (si está soportado)
            names = m.names

            def run_ultralytics(img_bgr, params):
                # Ultralytics usa BGR/NumPy directamente; controlar conf/iou/max_det/classes/agnostic_nms/imgsz
                results = m.predict(
                    img_bgr,
                    conf=params["conf"],
                    iou=params["iou"],
                    max_det=params["max_det"],
                    classes=list_to_none_if_empty(params["classes"]),
                    agnostic_nms=params["agnostic_nms"],
                    imgsz=params["imgsz"],
                    verbose=False,
                    device=0 if torch.cuda.is_available() else "cpu",
                )
                r0 = results[0]
                # DataFrame de detecciones
                # r0.boxes.xyxy (N,4), r0.boxes.conf (N,), r0.boxes.cls (N,)
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    cls = r0.boxes.cls.cpu().numpy().astype(int)
                    rows = []
                    for i in range(len(conf)):
                        x1, y1, x2, y2 = xyxy[i].tolist()
                        c = int(cls[i])
                        rows.append({
                            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                            "confidence": float(conf[i]),
                            "class_id": c, "name": names[c] if c in names else str(c)
                        })
                    df = pd.DataFrame(rows)
                else:
                    df = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "confidence", "class_id", "name"])

                # Imagen anotada (BGR)
                annotated = r0.plot()  # devuelve BGR
                return df, annotated

            return "ultralytics", m, names, run_ultralytics
        except Exception as e:
            st.info(f"Ultralytics no disponible o falló la carga: {e}. Intentando YOLOv5 (torch.hub)…")

    # Fallback: YOLOv5 por torch.hub
    # pesos típicos: yolov5s, yolov5m, yolov5l, yolov5x
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.hub.load("ultralytics/yolov5", "custom", path=weights if os.path.exists(weights) else None, pretrained=not os.path.exists(weights))
        m.to(device)
        names = m.names

        def run_yolov5(img_bgr, params):
            # Ajuste de NMS/params
            m.conf = params["conf"]
            m.iou = params["iou"]
            m.max_det = params["max_det"]
            try:
                m.agnostic = params["agnostic_nms"]
            except Exception:
                pass
            # Filtrado por clases en YOLOv5: via 'classes' en predict
            results = m(img_bgr, size=params["imgsz"], classes=list_to_none_if_empty(params["classes"]))
            # DataFrame
            try:
                df = results.pandas().xyxy[0].rename(columns={"class": "class_id", "name": "name"})
                df = df[["xmin", "ymin", "xmax", "ymax", "confidence", "class_id", "name"]]
            except Exception:
                # Parse manual
                preds = results.pred[0] if hasattr(results, "pred") else None
                rows = []
                if preds is not None and len(preds) > 0:
                    for row in preds:
                        x1, y1, x2, y2, conf, cls = row.tolist()
                        cls = int(cls)
                        rows.append({
                            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                            "confidence": float(conf),
                            "class_id": cls, "name": names[cls] if cls in names else str(cls)
                        })
                df = pd.DataFrame(rows)

            # Imagen anotada: results.render() modifica results.imgs in-place
            results.render()
            annotated = results.imgs[0] if hasattr(results, "imgs") and len(results.imgs) else img_bgr
            return df, annotated

        return "yolov5", m, names, run_yolov5
    except Exception as e:
        st.error(f"❌ No se pudo cargar ningún backend de YOLO: {e}")
        return None, None, {}, None

# ─────────────────────────────────────────────────────────────
# Sidebar — Parámetros
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parámetros de detección")
    backend_pref = st.selectbox("Backend preferido", ["auto", "ultralytics", "yolov5"], index=0)
    weights = st.text_input("Pesos/Modelo", value="yolov8n.pt", help="Ej: yolov8n.pt, yolov8s.pt, yolov5su.pt, yolov5s.pt, ruta a .pt local, etc.")
    imgsz = st.slider("Tamaño de inferencia (imgsz)", 320, 1280, 640, step=32)
    conf = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU NMS", 0.0, 1.0, 0.45, 0.01)
    max_det = st.number_input("Detecciones máximas", 1, 3000, 300, 1)
    agnostic_nms = st.toggle("NMS class-agnostic", value=False)
    classes_str = st.text_input("Filtrar clases (IDs separadas por coma, vacío = todas)", value="")
    classes = [int(x) for x in classes_str.split(",") if x.strip().isdigit()] if classes_str.strip() else []

    st.divider()
    st.caption(f"🖥️ Dispositivo: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    st.caption("💡 Tip: usa `yolov8n.pt` para velocidad o `yolov8s/5s` para mejor precisión.")

# ─────────────────────────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────────────────────────
with st.spinner("Cargando modelo…"):
    backend, model, names, runner = load_model(
        prefer_ultralytics=(backend_pref != "yolov5"),
        yolo_variant=backend_pref,
        weights=weights
    )

if model is None:
    st.stop()

st.success(f"✅ Modelo cargado · Backend: **{backend}**")

# ─────────────────────────────────────────────────────────────
# Entrada de imagen: Cámara o Uploader
# ─────────────────────────────────────────────────────────────
col_in_l, col_in_r = st.columns(2)
with col_in_l:
    st.subheader("📷 Cámara")
    cam_img = st.camera_input("Capturar imagen", key="camera")

with col_in_r:
    st.subheader("📎 Subir imagen")
    up_img = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "webp"])

# Seleccionar fuente
img_bgr = None
source_name = None
if cam_img is not None:
    source_name = "camera.png"
    bytes_data = cam_img.getvalue()
    img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
elif up_img is not None:
    source_name = up_img.name
    bytes_data = up_img.read()
    img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

if img_bgr is None:
    st.info("Captura o sube una imagen para detectar objetos.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Inferencia
# ─────────────────────────────────────────────────────────────
params = {
    "conf": conf,
    "iou": iou,
    "max_det": int(max_det),
    "agnostic_nms": agnostic_nms,
    "classes": classes,
    "imgsz": int(imgsz),
}

with st.spinner("Detectando objetos…"):
    try:
        df, annotated_bgr = runner(img_bgr, params)
    except Exception as e:
        st.error(f"Error durante la detección: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# Resultados
# ─────────────────────────────────────────────────────────────
c1, c2 = st.columns([6, 4])

with c1:
    st.subheader("Imagen anotada")
    st.image(bgr_to_rgb(annotated_bgr), use_container_width=True, caption=source_name)

    # Botón de descarga de la imagen anotada
    st.download_button(
        "⬇️ Descargar imagen anotada (PNG)",
        data=pil_bytes_from_bgr(annotated_bgr),
        file_name=f"{os.path.splitext(source_name)[0]}_detecciones.png",
        mime="image/png",
        use_container_width=True,
    )

with c2:
    st.subheader("Objetos detectados")
    if len(df) == 0:
        st.info("No se detectaron objetos con los parámetros actuales. Prueba bajando la confianza.")
    else:
        # Conteo por clase
        counts = df["name"].value_counts().rename_axis("Categoría").reset_index(name="Cantidad")
        st.dataframe(
            df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]]
              .rename(columns={"name": "Categoría", "confidence": "Confianza"}),
            use_container_width=True, hide_index=True
        )
        st.bar_chart(counts.set_index("Categoría"))

        # Descarga CSV
        st.download_button(
            "⬇️ Descargar resultados (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{os.path.splitext(source_name)[0]}_detecciones.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("YOLOv8/YOLOv5 · Streamlit · Manejo de parámetros, clases y exportaciones.")
