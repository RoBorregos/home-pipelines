import subprocess
import sys
import os

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}


def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "fps=10",
        os.path.join(output_dir, "frame_%05d.png"),
    ]

    subprocess.run(cmd, check=True)
    print(f"Frames extraídos en: {output_dir}")


def process_directory(videos_dir, output_dir):
    if not os.path.isdir(videos_dir):
        print(f"Error: el directorio '{videos_dir}' no existe.")
        sys.exit(1)

    videos = [
        f for f in sorted(os.listdir(videos_dir))
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    ]

    if not videos:
        print(f"No se encontraron videos en '{videos_dir}'.")
        sys.exit(1)

    for video in videos:
        name = os.path.splitext(video)[0]
        video_path = os.path.join(videos_dir, video)
        folder_out = os.path.join(output_dir, name)
        print(f"Procesando: {video}")
        extract_frames(video_path, folder_out)

    print(f"\nTodos los videos procesados. Resultados en: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Uso: python {sys.argv[0]} <directorio_videos> <directorio_destino>")
        sys.exit(1)

    process_directory(sys.argv[1], sys.argv[2])
