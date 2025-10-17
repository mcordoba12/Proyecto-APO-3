# ---- batch_annotate.py (CSV plano: landmark_i_x / combinado + mapa índices) ----
import sys, os, glob, math
print(">>> START batch_annotate.py", flush=True)
print("cwd:", os.getcwd(), flush=True)

# --- Imports con logs (ayuda a depurar versiones) ---
try:
    import cv2
    print("cv2 version:", cv2.__version__, flush=True)
except Exception as e:
    print("ERROR importando cv2:", repr(e), flush=True); sys.exit(1)

try:
    import mediapipe as mp
    print("mediapipe version:", getattr(mp, "__version__", "unknown"), flush=True)
except Exception as e:
    print("ERROR importando mediapipe:", repr(e), flush=True); sys.exit(1)

import numpy as np
import pandas as pd

# =========================
# Configuración
# =========================
INPUT_DIR  = os.path.join(os.getcwd(), "input")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
VIDEO_EXTS = ("*.mp4","*.MP4","*.mov","*.MOV","*.avi","*.AVI")

# Tamaños de landmarks MediaPipe Holistic
POSE_N, HAND_N, FACE_N = 33, 21, 468

# Estilo de columnas: plano "landmark_i_x" (requerido por el profe)
COLUMN_STYLE = "flat"   # (mantenido por claridad; solo usamos "flat" aquí)

# Acumuladores globales
GLOBAL_ALL_ROWS = []   # todas las filas (todos los videos) en formato plano
ALL_METRICS     = []   # métricas por video

# =========================
# Utilidades
# =========================
def ensure_out():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def list_videos(folder):
    paths = []
    for pat in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(paths)

def _make_index_map():
    """
    Mapa de índices globales (g_idx, part, local_idx).
    Orden: Pose (0-32), Mano Izq (33-53), Mano Der (54-74), Cara (75-542).
    """
    mapping = []
    g = 0
    for i in range(POSE_N): mapping.append((g, "pose", i)); g += 1
    for i in range(HAND_N): mapping.append((g, "lh",   i)); g += 1
    for i in range(HAND_N): mapping.append((g, "rh",   i)); g += 1
    for i in range(FACE_N): mapping.append((g, "face", i)); g += 1
    return mapping

INDEX_MAP = _make_index_map()

def _save_index_map():
    path = os.path.join(OUTPUT_DIR, "landmark_index_map.csv")
    rows = [{"global_index": g, "part": part, "local_index": li} for (g, part, li) in INDEX_MAP]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"📄 Diccionario de índices guardado en: {path}", flush=True)

def _open_writer_with_fallbacks(out_base, fps, width, height):
    """Intenta mp4v (.mp4) → XVID (.avi) → MJPG (.avi). Devuelve (writer, out_path, codec)."""
    # 1) MP4
    out_video = out_base + "_annot.mp4"
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, out_video, "mp4v"

    print("[WARN] mp4v no disponible, probando XVID...", flush=True)
    out_video = out_base + "_annot.avi"
    fourcc    = cv2.VideoWriter_fourcc(*"XVID")
    writer    = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, out_video, "XVID"

    print("[WARN] XVID no disponible, probando MJPG...", flush=True)
    out_video = out_base + "_annot.avi"
    fourcc    = cv2.VideoWriter_fourcc(*"MJPG")
    writer    = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, out_video, "MJPG"

    return None, None, None

def _init_row_with_meta_flat(base, frame_idx, fps):
    """Crea una fila con meta + TODAS las columnas landmark_i_{x,y,z} inicializadas a NaN."""
    time_s = (frame_idx - 1) / fps if fps > 0 else None
    row = {"video": base, "frame": frame_idx, "time_s": time_s,
           "has_pose": 0, "has_lh": 0, "has_rh": 0, "has_face": 0}
    for g_idx, _, _ in INDEX_MAP:
        row[f"landmark_{g_idx}_x"] = np.nan
        row[f"landmark_{g_idx}_y"] = np.nan
        row[f"landmark_{g_idx}_z"] = np.nan
    return row

# =========================
# Procesamiento por video
# =========================
def annotate_and_export(video_path, model_complexity=1):
    print(f"[INFO] Procesando: {video_path}", flush=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_base = os.path.join(OUTPUT_DIR, base)
    out_csv  = out_base + "_coords.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path}", flush=True)
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Propiedades -> {width}x{height} @ {fps:.2f} fps | frames: {total}", flush=True)

    writer, out_video, codec_name = _open_writer_with_fallbacks(out_base, fps, width, height)
    if writer is None:
        print("[ERROR] Ningún codec de escritura funcionó.", flush=True)
        cap.release()
        return None
    print(f"[INFO] Escribiendo con codec: {codec_name} -> {out_video}", flush=True)

    mp_holistic = mp.solutions.holistic
    mp_draw     = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles

    rows_full = []
    frames_total = 0
    pose_ok = lh_ok = rh_ok = face_ok = 0
    brillo_total = 0.0
    mov_total    = 0.0
    mov_count    = 0

    ok, prev = cap.read()
    if not ok:
        print(f"[ERROR] No se pudo leer el primer frame de: {video_path}", flush=True)
        cap.release(); writer.release()
        return None

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_total += 1

                # MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = holistic.process(rgb)
                rgb.flags.writeable = True

                annotated = frame.copy()
                row = _init_row_with_meta_flat(base, frames_total, fps)

                # Pose -> indices 0..32
                if res.pose_landmarks:
                    pose_ok += 1
                    row["has_pose"] = 1
                    mp_draw.draw_landmarks(
                        annotated, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )
                    for i, lm in enumerate(res.pose_landmarks.landmark):
                        if i >= POSE_N: break
                        g = i  # offset 0
                        row[f"landmark_{g}_x"] = lm.x
                        row[f"landmark_{g}_y"] = lm.y
                        row[f"landmark_{g}_z"] = lm.z

                # Mano Izquierda -> indices 33..53
                if res.left_hand_landmarks:
                    lh_ok += 1
                    row["has_lh"] = 1
                    mp_draw.draw_landmarks(
                        annotated, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style()
                    )
                    base_off = POSE_N
                    for i, lm in enumerate(res.left_hand_landmarks.landmark):
                        if i >= HAND_N: break
                        g = base_off + i
                        row[f"landmark_{g}_x"] = lm.x
                        row[f"landmark_{g}_y"] = lm.y
                        row[f"landmark_{g}_z"] = lm.z

                # Mano Derecha -> indices 54..74
                if res.right_hand_landmarks:
                    rh_ok += 1
                    row["has_rh"] = 1
                    mp_draw.draw_landmarks(
                        annotated, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style()
                    )
                    base_off = POSE_N + HAND_N
                    for i, lm in enumerate(res.right_hand_landmarks.landmark):
                        if i >= HAND_N: break
                        g = base_off + i
                        row[f"landmark_{g}_x"] = lm.x
                        row[f"landmark_{g}_y"] = lm.y
                        row[f"landmark_{g}_z"] = lm.z

                # Cara -> indices 75..542
                if res.face_landmarks:
                    face_ok += 1
                    row["has_face"] = 1
                    mp_draw.draw_landmarks(
                        annotated, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    base_off = POSE_N + HAND_N + HAND_N  # 75
                    for i, lm in enumerate(res.face_landmarks.landmark):
                        if i >= FACE_N: break  # algunos backends incluyen iris extra
                        g = base_off + i
                        row[f"landmark_{g}_x"] = lm.x
                        row[f"landmark_{g}_y"] = lm.y
                        row[f"landmark_{g}_z"] = lm.z

                # EDA simple
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brillo_total += float(np.mean(gray))
                diff = cv2.absdiff(prev, frame)
                mov_total += float(np.mean(diff))
                mov_count += 1
                prev = frame

                # guardar fila y video anotado
                rows_full.append(row)
                writer.write(annotated)

                if frames_total % 50 == 0:
                    print(f"[INFO] {base}: {frames_total}/{total if total>0 else '?'} frames", flush=True)

    except Exception as e:
        print("[EXCEPTION] durante procesamiento:", repr(e), flush=True)

    cap.release()
    writer.release()

    # CSV por video (formato plano)
    if rows_full:
        df_full = pd.DataFrame(rows_full)
        df_full.to_csv(out_csv, index=False)
        GLOBAL_ALL_ROWS.extend(rows_full)

    # métricas por video
    dur_s = frames_total / fps if fps > 0 else math.nan
    brillo_prom = (brillo_total / frames_total) if frames_total else math.nan
    mov_prom    = (mov_total / mov_count) if mov_count else math.nan
    metrics = {
        "video": base,
        "frames": frames_total,
        "fps": round(float(fps), 2),
        "duracion_s": round(dur_s, 3) if not math.isnan(dur_s) else None,
        "ancho": width, "alto": height,
        "brillo_promedio": round(brillo_prom, 3) if frames_total else None,
        "movimiento_promedio": round(mov_prom, 3) if mov_count else None,
        "frames_pose_detectados": int(pose_ok),
        "frames_mano_izq": int(lh_ok),
        "frames_mano_der": int(rh_ok),
        "frames_rostro": int(face_ok),
    }
    print(f"[OK] {base}: {frames_total} frames | {width}x{height}@{fps:.1f} | dur={metrics['duracion_s']}s", flush=True)
    return metrics

# =========================
# Main
# =========================
def main():
    print(">>> MAIN INICIO", flush=True)
    print("INPUT_DIR:", INPUT_DIR, flush=True)
    print("OUTPUT_DIR:", OUTPUT_DIR, flush=True)
    ensure_out()

    # guardar mapa de índices (una vez)
    _save_index_map()

    videos = list_videos(INPUT_DIR)
    if not videos:
        print(f"[ERROR] No se encontraron videos en {INPUT_DIR} con extensiones {VIDEO_EXTS}", flush=True)
        sys.exit(1)

    print(f"[INFO] Encontrados {len(videos)} videos:", flush=True)
    for v in videos:
        print("   -", v, flush=True)

    for vp in videos:
        m = annotate_and_export(vp, model_complexity=1)
        if m: ALL_METRICS.append(m)

    # Resumen EDA por video
    if ALL_METRICS:
        dfm = pd.DataFrame(ALL_METRICS)
        out_summary = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
        dfm.to_csv(out_summary, index=False)
        print(f"✅ Resumen EDA guardado en: {out_summary}", flush=True)
    else:
        print("[WARN] No se generaron métricas.", flush=True)

    # CSV combinado de TODOS los videos
    if GLOBAL_ALL_ROWS:
        df_all = pd.DataFrame(GLOBAL_ALL_ROWS)
        out_all = os.path.join(OUTPUT_DIR, "all_coords_full.csv")
        df_all.to_csv(out_all, index=False)
        print(f"✅ CSV combinado (FLAT): {out_all}  | Filas: {len(df_all)}", flush=True)
        # (Opcional) también en Parquet si lo deseas (descomenta):
        # df_all.to_parquet(os.path.join(OUTPUT_DIR, "all_coords_full.parquet"), index=False)
    else:
        print("[WARN] No hubo landmarks para combinar (¿no detectó?).", flush=True)

if __name__ == "__main__":
    main()
# ---- fin ----
