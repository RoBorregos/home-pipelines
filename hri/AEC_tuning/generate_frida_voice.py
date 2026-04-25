#!/usr/bin/env python3
"""
Generate Frida's robot voice dataset using Kokoro TTS.

Creates a library of utterances that Frida would say, synthesized with
Kokoro TTS using the same voice/settings as the real robot. These are
used as "reference signals" for AEC fine-tuning of DeepFilterNet.

Usage:
    python generate_frida_voice.py --output_dir data/frida_voice --num_utterances 500
"""

import argparse
import json
import os
import wave

import numpy as np
from kokoro import KPipeline
from scipy import signal


# Frida's typical utterances - mix of Spanish and English
FRIDA_UTTERANCES = [
    # Greetings & Social
    "Hola, soy Frida, tu asistente robótica. ¿En qué puedo ayudarte?",
    "Hello, my name is Frida. How can I help you today?",
    "Bienvenido, estoy lista para ayudarte.",
    "Hi there! I'm Frida, your friendly robot assistant.",
    "Mucho gusto en conocerte. ¿Qué necesitas?",
    "Good morning! I'm here to assist you.",
    "Buenos días, ¿cómo estás?",
    "Good afternoon, how may I help?",
    "Buenas tardes, bienvenido.",
    "Hey! I'm Frida. Let me know what you need.",

    # Navigation & Movement
    "Voy a ir a la cocina ahora.",
    "I'm heading to the living room.",
    "Déjame ir al cuarto de estar.",
    "I'll navigate to the kitchen now.",
    "Estoy yendo hacia la puerta principal.",
    "Let me move to the dining area.",
    "Voy a acercarme a ti.",
    "I need to go to the bedroom first.",
    "Permíteme pasar, necesito llegar al otro lado.",
    "I'm going to the entrance now.",
    "Estoy calculando la mejor ruta.",
    "Please step aside, I need to pass through.",
    "Ya llegué al destino.",
    "I have arrived at the location.",
    "Necesito ir al baño para buscar la toalla.",

    # Object Manipulation
    "Voy a tomar el vaso de la mesa.",
    "Let me grab that object for you.",
    "Estoy levantando la botella de agua.",
    "I'll pick up the cup from the table.",
    "Déjame poner esto en la mesa.",
    "I'm placing the item on the counter.",
    "Voy a abrir la puerta.",
    "Let me hand you this object.",
    "Estoy buscando el control remoto.",
    "I found what you were looking for.",
    "Aquí tienes lo que me pediste.",
    "Let me put this away for you.",

    # Task Responses
    "Entendido, voy a hacer eso ahora mismo.",
    "Sure, I'll do that right away.",
    "Dame un momento, estoy procesando tu solicitud.",
    "Working on it, please give me a second.",
    "Ya terminé la tarea que me pediste.",
    "I've completed the task you requested.",
    "Lo siento, no puedo hacer eso en este momento.",
    "I'm sorry, I can't do that right now.",
    "Necesito más información para completar esa tarea.",
    "Could you please repeat that?",
    "No entendí bien, ¿puedes repetirlo?",
    "I didn't quite catch that, could you say it again?",

    # Status Updates
    "Mi batería está al ochenta por ciento.",
    "All systems are functioning normally.",
    "Estoy lista para la siguiente instrucción.",
    "I'm waiting for your next command.",
    "Procesando información, un momento por favor.",
    "Processing your request, please wait.",
    "He detectado un obstáculo en el camino.",
    "I've detected an obstacle ahead.",
    "Todo está en orden.",
    "Everything looks good.",

    # Competition / RoboCup specific
    "Soy Frida, del equipo RoBorregos del Tecnológico de Monterrey.",
    "I am Frida, from team RoBorregos, Tecnológico de Monterrey.",
    "Estoy lista para la prueba.",
    "I'm ready to start the task.",
    "¿Puedes decirme tu nombre?",
    "What is your name?",
    "¿Qué objeto necesitas que te traiga?",
    "Which item would you like me to fetch?",
    "Voy a la cocina a buscar una bebida.",
    "I'll go to the kitchen to get a drink.",
    "¿Hay alguien en la sala?",
    "Is there anyone in the living room?",
    "He encontrado a una persona en la habitación.",
    "I found a person in the room.",
    "La persona está sentada en el sofá.",
    "The person is sitting on the couch.",
    "Voy a guiarte hasta la salida.",
    "Let me guide you to the exit.",
    "Sígueme, por favor.",
    "Please follow me.",

    # Longer utterances (more realistic for echo scenarios)
    "Hola, soy Frida del equipo RoBorregos. Estoy aquí para ayudarte con lo que necesites. ¿Podrías decirme qué tarea debo realizar?",
    "I'm going to navigate to the kitchen, pick up the water bottle, and bring it back to you. Please wait here.",
    "He detectado tres personas en la habitación. Una está sentada en el sofá, otra está de pie cerca de la ventana, y la tercera está en la puerta.",
    "I've completed the navigation task. The route was clear and I arrived at the destination without any obstacles.",
    "Permíteme presentarme. Mi nombre es Frida y soy un robot de servicio doméstico desarrollado por el equipo RoBorregos del Tecnológico de Monterrey.",
    "I need to inform you that I cannot reach that object. It's too high for my arm to reach. Could you please place it lower?",
    "Estoy procesando la imagen para detectar los objetos en la mesa. Un momento por favor mientras analizo la escena.",
    "The object you requested is not in the expected location. Let me search the surrounding area to find it.",
    "Voy a repetir las instrucciones para confirmar. Necesito ir a la cocina, tomar la taza roja, y llevarla a la persona en la sala.",
    "I successfully identified the person you described. They are wearing a blue shirt and are located near the bookshelf.",
]


def generate_more_utterances():
    """Generate variations of base utterances for more diversity."""
    variations = []
    fillers = [
        "Mmm, déjame pensar.",
        "Okay, let me see.",
        "Un momento.",
        "Right away.",
        "Claro que sí.",
        "Of course.",
        "Sí, entiendo.",
        "Yes, I understand.",
        "Perfecto.",
        "Great.",
        "Voy en camino.",
        "On my way.",
        "Listo.",
        "Done.",
        "Aquí estoy.",
        "Here I am.",
        "Disculpa.",
        "Excuse me.",
        "Permiso.",
        "Pardon me.",
    ]

    numbers = [
        "Uno, dos, tres, cuatro, cinco.",
        "The temperature is twenty two degrees.",
        "Son las tres de la tarde.",
        "It's approximately two meters away.",
        "Hay cinco objetos en la mesa.",
        "There are three people in the room.",
        "El objeto pesa aproximadamente medio kilogramo.",
        "The distance is about one point five meters.",
    ]

    variations.extend(fillers)
    variations.extend(numbers)
    return variations


class FridaVoiceGenerator:
    def __init__(self, voice="af_heart", speed=1.0, device="cpu"):
        self.voice = voice
        self.speed = speed
        self.original_sr = 24000
        self.target_sr = 16000  # Match noise_cancellation pipeline

        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

        print(f"Initializing Kokoro TTS on {device}...")
        self.pipeline = KPipeline(lang_code="a", device=device)

        # Warm up
        print("Warming up model...")
        for i, (_, _, audio) in enumerate(self.pipeline("Test.", voice=self.voice)):
            if i > 0:
                break
        print("Kokoro TTS ready.")

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text and return float32 audio at target_sr."""
        chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice, speed=self.speed):
            chunks.append(audio)

        if not chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks)

        # Resample to 16kHz to match the noise cancellation pipeline
        if self.original_sr != self.target_sr:
            audio = signal.resample_poly(
                audio, self.target_sr, self.original_sr
            ).astype(np.float32)

        return audio

    def save_wav(self, audio: np.ndarray, path: str):
        """Save float32 audio as 16-bit WAV."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.target_sr)
            wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Generate Frida's voice dataset using Kokoro TTS"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/frida_voice",
        help="Output directory for generated audio files",
    )
    parser.add_argument(
        "--voice", type=str, default="af_heart",
        help="Kokoro voice to use (must match robot's actual voice)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speech speed (must match robot's actual speed)",
    )
    parser.add_argument(
        "--speeds", type=str, default="0.9,1.0,1.1",
        help="Comma-separated speeds for augmentation",
    )
    parser.add_argument(
        "--extra_utterances_file", type=str, default=None,
        help="Path to a text file with additional utterances (one per line)",
    )
    args = parser.parse_args()

    # Collect all utterances
    utterances = list(FRIDA_UTTERANCES)
    utterances.extend(generate_more_utterances())

    if args.extra_utterances_file and os.path.exists(args.extra_utterances_file):
        with open(args.extra_utterances_file, "r") as f:
            extra = [line.strip() for line in f if line.strip()]
        utterances.extend(extra)
        print(f"Loaded {len(extra)} extra utterances from {args.extra_utterances_file}")

    speeds = [float(s) for s in args.speeds.split(",")]

    print(f"Total base utterances: {len(utterances)}")
    print(f"Speeds: {speeds}")
    print(f"Total files to generate: {len(utterances) * len(speeds)}")

    generator = FridaVoiceGenerator(voice=args.voice, speed=args.speed, device="cpu")

    manifest = []
    file_idx = 0

    for speed in speeds:
        speed_dir = os.path.join(args.output_dir, f"speed_{speed:.1f}")
        generator.speed = speed

        for i, text in enumerate(utterances):
            filename = f"frida_{file_idx:05d}.wav"
            filepath = os.path.join(speed_dir, filename)

            try:
                audio = generator.synthesize(text)
                if len(audio) == 0:
                    print(f"  SKIP (empty): {text[:50]}")
                    continue

                generator.save_wav(audio, filepath)
                duration = len(audio) / generator.target_sr

                manifest.append({
                    "file": filepath,
                    "text": text,
                    "voice": args.voice,
                    "speed": speed,
                    "duration_s": round(duration, 2),
                    "sample_rate": generator.target_sr,
                })

                file_idx += 1
                if file_idx % 20 == 0:
                    print(f"  Generated {file_idx} files...")

            except Exception as e:
                print(f"  ERROR on '{text[:50]}...': {e}")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Generated {file_idx} audio files.")
    print(f"Manifest saved to {manifest_path}")
    print(f"Total audio duration: {sum(m['duration_s'] for m in manifest):.1f}s")


if __name__ == "__main__":
    main()
