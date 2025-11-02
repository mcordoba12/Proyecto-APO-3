"""
Script para agregar etiquetas al CSV de landmarks extrayendo info desde nombres de clips.

Lee:
  - all_coords_full.csv (landmarks por frame)
  - clips en output/dataset/{label}/*.mp4

Genera:
  - all_coords_full_labeled.csv (con columna 'label' agregada)
  - segments_reconstructed.csv (segmentos reconstruidos desde clips)
"""

import os
import re
import pandas as pd
import glob

# =========================
# Configuración
# =========================
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
COORDS_CSV = os.path.join(OUTPUT_DIR, "all_coords_full.csv")
OUTPUT_LABELED = os.path.join(OUTPUT_DIR, "all_coords_full_labeled.csv")
OUTPUT_LABELED_ONLY = os.path.join(OUTPUT_DIR, "all_coords_labeled_only.csv")  # NUEVO
OUTPUT_SEGMENTS = os.path.join(OUTPUT_DIR, "segments_reconstructed.csv")

# Etiquetas esperadas (carpetas en dataset/)
LABELS = ["caminar_frente", "caminar_espalda", "girar", "sentado", "ponerse_de_pie"]

# Patrón para parsear nombres: {video}_annot_{label}_{start}-{end}.extensión
# Ejemplo: video 1_annot_caminar_espalda_190-268.mp4
# El video puede tener espacios y múltiples palabras
CLIP_PATTERN = re.compile(r"^(.+?)_annot_([a-z_]+)_(\d+)-(\d+)\.(mp4|avi|mov)$", re.IGNORECASE)


# =========================
# Funciones
# =========================
def extract_segments_from_clips():
    """
    Escanea output/dataset/ y extrae segmentos desde nombres de archivos.
    Retorna lista de dicts: {video, label, start_frame, end_frame}
    """
    segments = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] No existe la carpeta: {DATASET_DIR}")
        return segments
    
    # Buscar clips en cada carpeta de etiqueta
    for label in LABELS:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.exists(label_dir):
            continue
        
        clips = glob.glob(os.path.join(label_dir, "*.*"))
        
        for clip_path in clips:
            filename = os.path.basename(clip_path)
            match = CLIP_PATTERN.match(filename)
            
            if match:
                video_base = match.group(1)
                label_extracted = match.group(2)
                start_frame = int(match.group(3))
                end_frame = int(match.group(4))
                
                # Nota: El video_base ya NO incluye "_annot" gracias al patrón regex
                # Si el CSV tiene "video 1" y el clip es "video 1_annot_...", ya coinciden
                
                # Validación: la etiqueta del nombre debe coincidir con la carpeta
                if label_extracted == label:
                    segments.append({
                        "video": video_base,
                        "label": label,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "clip_file": filename
                    })
                    print(f"✓ {filename} → {video_base} | {label} | frames {start_frame}-{end_frame}")
                else:
                    print(f"[WARN] Etiqueta inconsistente: {filename} (carpeta={label}, nombre={label_extracted})")
            else:
                print(f"[WARN] No se pudo parsear: {filename}")
    
    return segments


def map_labels_to_frames(df_coords, segments):
    """
    Agrega columna 'label' al DataFrame de coordenadas.
    Cada frame se mapea según los segmentos.
    """
    # Inicializar columna con NaN (sin etiqueta)
    df_coords["label"] = None
    
    # Normalizar nombres de video a minúsculas para comparación
    df_coords["video_lower"] = df_coords["video"].str.lower()
    
    # DEBUG: Mostrar videos únicos en el CSV
    print("\n DEBUG - Videos en all_coords_full.csv:")
    unique_videos = df_coords["video"].unique()
    for v in unique_videos:
        print(f"   '{v}'")
    
    print("\n DEBUG - Videos en segmentos extraídos:")
    segment_videos = set(seg["video"] for seg in segments)
    for v in segment_videos:
        print(f"   '{v}'")
    
    # Iterar por cada segmento y asignar etiquetas
    matches_found = 0
    no_match_count = 0
    
    print("\n DEBUG DETALLADO - Primeros 5 segmentos:")
    for i, seg in enumerate(segments[:5]):  # Solo primeros 5 para debug
        video = seg["video"].lower()
        label = seg["label"]
        start = seg["start_frame"]
        end = seg["end_frame"]
        
        print(f"\n   Segmento {i+1}: video='{video}', frames {start}-{end}, label={label}")
        
        # Ver si existe el video en el CSV
        video_exists = (df_coords["video_lower"] == video).any()
        print(f"      ¿Video '{video}' existe en CSV? {video_exists}")
        
        if video_exists:
            # Ver qué frames tiene ese video
            video_frames = df_coords[df_coords["video_lower"] == video]["frame"].values
            print(f"      Frames disponibles para '{video}': min={video_frames.min()}, max={video_frames.max()}, total={len(video_frames)}")
            print(f"      ¿Frame {start} existe? {start in video_frames}")
            print(f"      ¿Frame {end} existe? {end in video_frames}")
    
    print("\n Procesando todos los segmentos...")
    for seg in segments:
        video = seg["video"].lower()  # Comparar en minúsculas
        label = seg["label"]
        start = seg["start_frame"]
        end = seg["end_frame"]
        
        # Crear máscara: mismo video Y frame dentro del rango
        mask = (
            (df_coords["video_lower"] == video) &
            (df_coords["frame"] >= start) &
            (df_coords["frame"] <= end)
        )
        
        matches = mask.sum()
        if matches > 0:
            matches_found += matches
            # Asignar etiqueta
            df_coords.loc[mask, "label"] = label
        else:
            no_match_count += 1
            if no_match_count <= 3:  # Solo mostrar primeros 3 fallos
                print(f"   Sin match: video='{video}', frames {start}-{end}, label={label}")
    
    if no_match_count > 3:
        print(f"   ... y {no_match_count - 3} segmentos más sin matches")
    
    # Eliminar columna temporal
    df_coords.drop(columns=["video_lower"], inplace=True)
    
    print(f"\n Total de frames con etiqueta asignada: {matches_found}")
    
    return df_coords


def main():
    print("=" * 60)
    print("AGREGANDO ETIQUETAS AL CSV DE LANDMARKS")
    print("=" * 60)
    
    # 1. Extraer segmentos desde nombres de clips
    print("\n[1/4] Extrayendo segmentos desde nombres de archivos...")
    segments = extract_segments_from_clips()
    
    if not segments:
        print("[ERROR] No se encontraron clips válidos en output/dataset/")
        print("Verifica que:")
        print("  - Existan carpetas: caminar_frente, caminar_espalda, girar, sentado, ponerse_de_pie")
        print("  - Los clips tengan formato: {video}_{label}_{start}-{end}.mp4")
        return
    
    print(f"\n {len(segments)} segmentos extraídos")
    
    # Guardar segmentos reconstruidos (opcional, para verificación)
    df_segments = pd.DataFrame(segments)
    df_segments.to_csv(OUTPUT_SEGMENTS, index=False)
    print(f" Segmentos guardados en: {OUTPUT_SEGMENTS}")
    
    # 2. Leer CSV de coordenadas
    print(f"\n[2/4] Leyendo {COORDS_CSV}...")
    if not os.path.exists(COORDS_CSV):
        print(f"[ERROR] No existe: {COORDS_CSV}")
        print("Primero ejecuta batch_annotate.py para generar este archivo.")
        return
    
    df_coords = pd.read_csv(COORDS_CSV)
    print(f"Cargado: {len(df_coords)} filas, {len(df_coords.columns)} columnas")
    
    # 3. Mapear etiquetas
    print("\n[3/4] Mapeando etiquetas a frames...")
    df_labeled = map_labels_to_frames(df_coords, segments)
    
    # VERIFICACIÓN INMEDIATA
    print("\nVERIFICACIÓN POST-MAPEO:")
    print(f"   ¿Columna 'label' existe? {('label' in df_labeled.columns)}")
    if 'label' in df_labeled.columns:
        print(f"   Valores NO nulos en 'label': {df_labeled['label'].notna().sum()}")
        print(f"   Primeras 10 filas de 'label':")
        print(f"   {df_labeled[['video', 'frame', 'label']].head(10).to_string()}")
    
    # Estadísticas
    total_frames = len(df_labeled)
    labeled_frames = df_labeled["label"].notna().sum()
    unlabeled_frames = total_frames - labeled_frames
    
    print(f"Frames etiquetados: {labeled_frames} / {total_frames} ({100*labeled_frames/total_frames:.1f}%)")
    print(f"   Frames sin etiqueta: {unlabeled_frames}")
    
    # Distribución por etiqueta
    print("\nDistribución de etiquetas:")
    label_counts = df_labeled["label"].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count} frames")
    
    # 4. Guardar CSV etiquetado COMPLETO
    print(f"\n[4/5] Guardando CSV completo (con frames vacíos)...")
    
    # Verificar que la columna label existe
    if "label" not in df_labeled.columns:
        print("[ERROR] La columna 'label' no existe en el DataFrame!")
        return
    
    print(f"   Columnas en DataFrame: {len(df_labeled.columns)}")
    print(f"   ¿Tiene columna 'label'? {('label' in df_labeled.columns)}")
    print(f"   Valores únicos en 'label': {df_labeled['label'].unique()}")
    
    # Reordenar columnas: poner 'label' DESPUÉS de video y frame para facilitar visualización
    # Orden: video, frame, label, time_s, has_*, landmarks...
    meta_cols_before = ["video", "frame", "label"]
    meta_cols_after = ["time_s", "has_pose", "has_lh", "has_rh", "has_face"]
    landmark_cols = [col for col in df_labeled.columns if col.startswith("landmark_")]
    new_order = meta_cols_before + meta_cols_after + landmark_cols
    df_labeled = df_labeled[new_order]
    
    print(f"   Orden de columnas: {', '.join(new_order[:10])}... (+ {len(landmark_cols)} landmarks)")
    
    df_labeled.to_csv(OUTPUT_LABELED, index=False, encoding='utf-8-sig')
    print(f"CSV completo guardado: {OUTPUT_LABELED}")
    
    # 5. Guardar CSV solo con frames ETIQUETADOS (filtrado)
    print(f"\n[5/5] Guardando CSV solo con frames etiquetados...")
    df_only_labeled = df_labeled[df_labeled["label"].notna()].copy()
    
    if len(df_only_labeled) == 0:
        print("No hay frames etiquetados para guardar en el CSV filtrado.")
    else:
        df_only_labeled.to_csv(OUTPUT_LABELED_ONLY, index=False, encoding='utf-8-sig')
        print(f"CSV filtrado guardado: {OUTPUT_LABELED_ONLY}")
        print(f"   Frames en CSV filtrado: {len(df_only_labeled)} / {len(df_labeled)} ({100*len(df_only_labeled)/len(df_labeled):.1f}%)")
        
        # Distribución en CSV filtrado
        print(f"\nDistribución en CSV filtrado:")
        label_counts_filtered = df_only_labeled["label"].value_counts()
        for label, count in label_counts_filtered.items():
            print(f"   {label}: {count} frames")
    
    print("\n" + "=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)
    print(f"\nArchivos generados:")
    print(f"  1. {OUTPUT_LABELED}")
    print(f"     → CSV completo con TODOS los frames (etiquetados y sin etiquetar)")
    print(f"  2. {OUTPUT_LABELED_ONLY}")
    print(f"     → CSV filtrado SOLO con frames etiquetados (recomendado para ML)")
    print(f"  3. {OUTPUT_SEGMENTS}")
    print(f"     → Segmentos reconstruidos desde nombres de clips")
    print(f"\nColumnas en ambos CSVs:")
    print(f"  - video, frame, label, time_s, has_pose, has_lh, has_rh, has_face")
    print(f"  - landmark_0_x ... landmark_542_z (1629 columnas de landmarks)")
    print(f"\n Usa 'all_coords_labeled_only.csv' para entrenar modelos (solo datos etiquetados)")


if __name__ == "__main__":
    main()