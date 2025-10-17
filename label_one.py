import os, cv2, pandas as pd, argparse

INPUT_DIR  = os.path.join(os.getcwd(), "output")   # carpeta por defecto
OUTPUT_DIR = os.path.join(os.getcwd(), "output")

LABEL_KEYS = {
    ord('1'): "caminar_frente",
    ord('2'): "caminar_espalda",
    ord('3'): "girar",
    ord('4'): "sentado",
    ord('5'): "ponerse_de_pie",
}
LABEL_ORDER = ["caminar_frente","caminar_espalda","girar","sentado","ponerse_de_pie"]

HELP = "I:inicio  O:fin  U:deshacer  C:cancelar  1..5:etiqueta  SPACE:pausa  -/+:vel  N/B:±1f  ←/→:±15f  S:guardar CSV  G:generar clips  Q:salir"
MIN_SEG_SECONDS = 0.7
SEEK_STEP = 15
MIN_RATE, MAX_RATE, INIT_RATE = 0.1, 2.0, 0.25  # arranca muy lento

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds = os.path.join(OUTPUT_DIR, "dataset")
    os.makedirs(ds, exist_ok=True)
    for lab in LABEL_ORDER:
        os.makedirs(os.path.join(ds, lab), exist_ok=True)
    return ds

def human_time(frame, fps):
    t = frame / max(fps, 1.0)
    return f"{int(t//60):02d}:{int(t%60):02d}.{int((t*1000)%1000):03d}"

def draw_osd(frame, cur_label, rate, frame_idx, total, fps, status=""):
    cv2.rectangle(frame, (5,5), (frame.shape[1]-5, 90), (0,0,0), -1)
    color = (0,255,0) if cur_label else (200,200,200)
    labtxt = cur_label if cur_label else "NINGUNO"
    cv2.putText(frame, f"Etiqueta actual: {labtxt}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Velocidad: {rate:.2f}x   |   {HELP}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    tt = f"{human_time(frame_idx,fps)}  ({frame_idx}/{total})   {status}"
    cv2.putText(frame, tt, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    # barra de progreso
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = 10, h-18, w-10, h-10
    cv2.rectangle(frame, (x0,y0), (x1,y1), (80,80,80), 1)
    progress = 0 if total<=0 else min(1.0, max(0.0, frame_idx/total))
    fill = int(x0 + (x1-x0)*progress)
    cv2.rectangle(frame, (x0,y0), (fill,y1), (60,180,255), -1)

def finalize_segment(segments, fps, label, start_f, end_f):
    if label is None or start_f is None or end_f is None: 
        return False
    if end_f <= start_f:
        return False
    dur = (end_f - start_f) / max(fps,1.0)
    if dur < MIN_SEG_SECONDS:
        return False
    segments.append({
        "label": label,
        "start_frame": int(start_f),
        "end_frame": int(end_f),
        "start_time": round(start_f / fps, 3),
        "end_time":   round(end_f   / fps, 3),
        "duration_s": round(dur, 3)
    })
    print(f"[SEG] {label}: {start_f} -> {end_f} ({dur:.2f}s)")
    return True

def cut_segments_to_files(video_path, df_segments, dataset_dir, fps_src, burn_label=True):
    """Recorta segmentos y (opcional) quema el texto de la etiqueta en cada clip."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] (cut) No se pudo abrir: {video_path}")
        return
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for _, r in df_segments.iterrows():
        label = r["label"]; sf = int(r["start_frame"]); ef = int(r["end_frame"])
        if ef <= sf: continue

        out_dir = os.path.join(dataset_dir, label)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{base}_{label}_{sf}-{ef}.mp4"
        out_path = os.path.join(out_dir, out_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        if not writer.isOpened():
            out_path = out_path.replace(".mp4",".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        if not writer.isOpened():
            print(f"[ERROR] (cut) No se pudo crear: {out_path}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
        to_write = ef - sf
        wcount = 0
        while wcount < to_write:
            ok, frame = cap.read()
            if not ok: break
            if burn_label:
                cv2.rectangle(frame, (5,5), (320, 45), (0,0,0), -1)
                cv2.putText(frame, label, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
            writer.write(frame)
            wcount += 1
        writer.release()
        print(f"[CUT] {label}: {out_path}  ({wcount} frames)")

    cap.release()

def pick_video():
    """Si no se pasa --video, lista archivos de input y deja elegir uno."""
    files = [f for f in sorted(os.listdir(INPUT_DIR), key=str.lower)
             if os.path.splitext(f)[1].lower() in (".mp4",".mov",".avi",".mkv")]
    if not files:
        print(f"[ERROR] No hay videos en {INPUT_DIR}")
        return None
    print("Selecciona un video:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")
    while True:
        try:
            sel = int(input("Número: "))
            if 1 <= sel <= len(files):
                return os.path.join(INPUT_DIR, files[sel-1])
        except Exception:
            pass
        print("Opción inválida. Intenta de nuevo.")

def label_one_video(video_path, do_cut_on_exit=True):
    ds = ensure_dirs()
    base = os.path.splitext(os.path.basename(video_path))[0]
    seg_csv_path = os.path.join(OUTPUT_DIR, f"{base}_segments.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {base}: {width}x{height}@{fps:.2f}fps | frames={total}")

    win = "Etiquetar - " + base
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280,width), min(720,height))

    # Estado
    segments = []               # lista de segmentos guardados
    current_label = LABEL_ORDER[0]  # etiqueta actual
    seg_open_start = None       # si hay segmento abierto: frame de inicio
    frame_idx = 0
    paused = False
    rate = INIT_RATE
    status = ""

    # carga previa si existe CSV (para continuar trabajo)
    if os.path.exists(seg_csv_path):
        try:
            prev = pd.read_csv(seg_csv_path)
            if {"label","start_frame","end_frame"}.issubset(prev.columns):
                for _, r in prev.iterrows():
                    segments.append({
                        "label": r["label"],
                        "start_frame": int(r["start_frame"]),
                        "end_frame": int(r["end_frame"]),
                        "start_time": float(r.get("start_time", 0.0)),
                        "end_time": float(r.get("end_time", 0.0)),
                        "duration_s": float(r.get("duration_s", 0.0)),
                    })
                print(f"[INFO] Cargados {len(segments)} segmentos previos desde CSV.")
        except Exception as e:
            print("[WARN] No se pudo leer CSV previo:", e)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print("[ERROR] No se pudo leer el primer frame.")
        cap.release(); cv2.destroyWindow(win); return

    def save_csv():
        if segments:
            df = pd.DataFrame(segments)
            df.to_csv(seg_csv_path, index=False, encoding="utf-8-sig")
            print(f"[OK] CSV guardado: {seg_csv_path}")
        else:
            print("[INFO] No hay segmentos para guardar aún.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # fin del video → cerrar si hay segmento abierto
                if seg_open_start is not None:
                    if finalize_segment(segments, fps, current_label, seg_open_start, frame_idx):
                        save_csv()
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()

        # OSD
        draw_osd(frame, current_label, rate, frame_idx, total, fps, status=status)
        if seg_open_start is not None:
            cv2.putText(frame, "REC", (frame.shape[1]-85, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3, cv2.LINE_AA)
        cv2.imshow(win, frame)
        status = ""

        # delay según velocidad
        delay_ms = 0 if paused else max(1, int(1000/(fps*rate)))
        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord('q'):  # salir
            if seg_open_start is not None:
                finalize_segment(segments, fps, current_label, seg_open_start, frame_idx)
            save_csv()
            break
        elif key == 32:  # SPACE pausa/continúa
            paused = not paused

        # velocidad
        elif key in (ord('+'), ord('=')): rate = min(MAX_RATE, round(rate+0.1,2))
        elif key in (ord('-'), ord('_')): rate = max(MIN_RATE, round(rate-0.1,2))

        # pasos finos
        elif key == ord('n'): paused = True; frame_idx = min(total-1, frame_idx+1)
        elif key == ord('b'): paused = True; frame_idx = max(0, frame_idx-1)

        # saltos
        elif key in (81, ord('a')): paused = True; frame_idx = max(0, frame_idx-SEEK_STEP)   # ←
        elif key in (83, ord('d')): paused = True; frame_idx = min(total-1, frame_idx+SEEK_STEP)  # →

        # control de segmento
        elif key == ord('i'):  # inicio
            seg_open_start = frame_idx
            status = "Inicio de segmento marcado (I)."
        elif key == ord('o'):  # fin
            if seg_open_start is None:
                status = "No hay segmento abierto. Presiona I primero."
            else:
                if finalize_segment(segments, fps, current_label, seg_open_start, frame_idx):
                    save_csv()
                seg_open_start = None
        elif key == ord('u'):  # deshacer último segmento
            if segments:
                last = segments.pop()
                print(f"[UNDO] Eliminado: {last}")
                save_csv()
            else:
                status = "No hay segmentos para deshacer."
        elif key == ord('c'):  # cancelar segmento abierto
            seg_open_start = None
            status = "Segmento abierto cancelado."

        # cambio de etiqueta
        elif key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            status = f"Etiqueta: {current_label}"

        # aplicar saltos/frames cuando está en pausa
        if paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    cv2.destroyWindow(win)

    # Generar clips si se pidió (o si usas tecla G dentro del flujo)
    # Aquí dejamos solo generación al final si do_cut_on_exit=True
    if segments:
        df = pd.DataFrame(segments)
        if do_cut_on_exit:
            cut_segments_to_files(video_path, df, ensure_dirs(), fps_src=fps, burn_label=True)

def main():
    ap = argparse.ArgumentParser(description="Etiquetar UN video (uno por uno) y recortar por segmentos.")
    ap.add_argument("--video", "-v", type=str, help="Ruta al video (si se omite, se muestra selección).")
    ap.add_argument("--nocut", action="store_true", help="No recortar clips al salir (solo CSV).")
    args = ap.parse_args()

    if args.video and not os.path.isabs(args.video):
        # permitir rutas relativas
        vp = os.path.join(os.getcwd(), args.video)
    else:
        vp = args.video

    if vp is None:
        vp = pick_video()
        if vp is None:
            return

    if not os.path.exists(vp):
        print(f"[ERROR] No existe el archivo: {vp}")
        return

    label_one_video(vp, do_cut_on_exit=(not args.nocut))

if __name__ == "__main__":
    main()
