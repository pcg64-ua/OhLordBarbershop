#--------------------GOOGLE CALENDAR--------------------------------------
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta, date
import pytz
from pathlib import Path

#--------------Ignorar warnings-----------------------------------
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
#------------------OPEN AI----------------------------------------
import openai
#-------------------ELEVEN LABS-----------------------------------
from playsound import playsound
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
#--------------------------------------------------------------

openai.api_key = os.getenv('OPENAI_API_KEY')
client = ElevenLabs(api_key="ELEVEN_KEY")
#----------------------------------------------------------------
import whisper
global modelo
global idioma_detectado
global idioma_ya_analizado
idioma_ya_analizado = False
idioma_detectado = "es"
modelo = whisper.load_model("small")  # Puedo usar "base" "small", "medium", o "large" para más precisión
conversation_history = [] # Memoria de conversación
with open("contextoInicial.txt",'r',encoding="utf-8") as f:
    conversation_history.append({"role":"assistant", "content": f.read()})

INTENCIONES = [] # lista de intenciones que tiene el usuario para poder identificarlas (reservar, cancelar, salir...)
with open("intenciones.txt",'r',encoding="utf-8") as f:
    INTENCIONES.append({"role": "system", "content": f.read()})
MODELO_OPEN_AI = "gpt-4o"
#MODELO_OPEN_AI = "gpt-3.5-turbo-0125"

# Conocer la intención del input del usuario/cliente
def conocer_intencion(texto: str):
    texto_usuario_con_intenciones = []
    if texto == "" or texto is None:
        texto_usuario_con_intenciones.append({"role": "system", "content":INTENCIONES[0]["content"]})
    else:
        texto_usuario_con_intenciones.append({"role": "system", "content": texto + '\n' + INTENCIONES[0]["content"]})
    
    try:
        response = openai.ChatCompletion.create( 
            model=MODELO_OPEN_AI, 
            messages=texto_usuario_con_intenciones, 
            temperature=0.5
        )
        id_intencion = response['choices'][0]['message']['content'].strip()
        print(id_intencion)
        return id_intencion
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Generar un audio a partir de un texto
def sintetizar_audio(text: str) -> IO[bytes]:
    global idioma_detectado
    
    if idioma_detectado == "en":
        voice_id = "bjWoFm4NbYKFCfqzw9BN"  # voz en inglés
    elif idioma_detectado == "de":
        voice_id = "ulTHPKT60YxEfyrVgZFh"  # voz en alemán
    else:
        voice_id = "kwNLkNjbQHMw9YUFZsHI"  # voz en español kwNLkNjbQHMw9YUFZsHI h2cd3gvcqTp3m65Dysk7

    response = client.text_to_speech.convert(
        voice_id = voice_id,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_flash_v2_5",
         optimize_streaming_latency=0, # eleven_turbo_v2_5
        voice_settings=VoiceSettings(
            stability=0.4,
            similarity_boost=0.6,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    filename = "text_to_speech_output.mp3"
    file_path = os.path.join(os.getcwd(), filename)

    with open(file_path, "wb") as audio_file:
        audio_file.write(audio_stream.read())

    playsound(file_path)

def sintetizar_audio_con_gtts(text: str):
    from gtts import gTTS
    from io import BytesIO
    from pydub import AudioSegment
    from pydub.playback import play
    global idioma_detectado

    tts = gTTS(text=text, lang=idioma_detectado, slow=False)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    play(audio)
    #import requests

    #api_key = "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"
    #headers = {
    #    "Authorization": f"Bearer {api_key}",
    #}

    #data = {
    #    "model": "tts-1",
    #    "input": text,
    #    "voice": "nova",
    #}

    #response = requests.post(
    #    "https://api.openai.com/v1/audio/speech",
    #    headers=headers,
    #    json=data
    #)

    #with open("output.mp3", "wb") as f:
    #    f.write(response.content)
    #playsound("output.mp3")
    #from gtts import gTTS
    #global idioma_detectado

    #tts = gTTS(text=text, lang=idioma_detectado, slow=False)
    #filename = f"audio_{uuid.uuid4()}.mp3"
    #filename = f"audio_con_gtts.mp3"
    #tts.save(filename)
    #playsound(filename)
    #return filename

def grabar_audio():
    FREQ = 16000  # Frecuencia de muestreo
    THRESHOLD = 500  # Umbral para detectar voz
    SILENCE_DURATION = 0.8  # Tiempo en segundos de silencio continuo para detener la grabación
    MAX_DURATION = 60  # Máximo tiempo de grabación en segundos

    print("🎙️ Esperando a que hables...")

    buffer = []  # Almacenar los fragmentos de audio
    silence_counter = 0  # Contador para medir la duración del silencio
    frame_size = int(FREQ * 0.1)  # Tamaño del bloque de audio (100 ms)
    recording_started = False  # Estado de la grabación

    def callback(indata, frames, time, status):
        nonlocal silence_counter, buffer, recording_started
        if status:
            print(f"Error en grabación: {status}")
        
        amplitude = np.abs(indata).mean()
        
        if amplitude > THRESHOLD:
            recording_started = True
            silence_counter = 0  # Restablecer el contador de silencio
        elif recording_started:
            silence_counter += 1
        
        if recording_started:
            buffer.append(indata.copy())

    with sd.InputStream(samplerate=FREQ, channels=1, dtype='int16', blocksize=frame_size, callback=callback, device=1):
        while not recording_started:
            pass  
        
        print("🎙️ Grabando...")
        while silence_counter < (SILENCE_DURATION * (FREQ // frame_size)):
            if len(buffer) * frame_size >= FREQ * MAX_DURATION:
                print("⏳ Tiempo máximo de grabación alcanzado.")
                break

    audio_data = np.concatenate(buffer, axis=0)
    write("grabacion.wav", FREQ, audio_data)
    print(f"✅ Grabación guardada como 'grabacion.wav'")

def detectar_idioma(audio_pcm: bytes, modelo):
    """
    Devuelve el idioma usando sólo los primeros 15 s del audio.
    """
    # Whisper acepta "audio" como ndarray; usamos load_audio + pad/trim
    import tempfile, soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio_pcm, 16_000)
        result = modelo.transcribe(tmp.name, task="transcribe", 
                                   language=None, 
                                   fp16=True, 
                                   no_speech_threshold=0.5,
                                   initial_prompt=None,
                                   verbose=False)
    global idioma_ya_analizado
    global idioma_detectado
    idioma_ya_analizado = True
    idioma_detectado = result["language"]
    return result["language"]

def transcribir_audio() -> str:
    global modelo
    global idioma_ya_analizado
    global idioma_detectado

    ruta = "grabacion.wav"
    audio_pcm, sr = whisper.audio.load_audio(ruta), 16_000

    if not idioma_ya_analizado:
        idioma = detectar_idioma(audio_pcm[:sr*2], modelo)
    else:
        idioma = idioma_detectado

    resultado = modelo.transcribe(
        ruta,
        language=idioma,
        task="transcribe",
        fp16=True,
        verbose=False,
    )
    texto = resultado["text"].strip()
    with open("transcripcion.txt", "w", encoding="utf-8") as f:
         f.write(texto)
    print(f"✅ Transcripción guardada: {texto}")
    return texto

# Transcribir un audio a texto
# def transcribir_audio(): 
#     def detectar_idioma():
#         print("🔍 Detectando idioma...")
#         global modelo
#         global idioma_detectado
#         global idioma_ya_analizado
        
#         # Primero decodificamos parte del audio
#         audio = modelo.transcribe("grabacion.wav", task="transcribe", language=None)
        
#         # Whisper devuelve también 'language' detectado
#         idioma_detectado = audio.get('language', 'unknown')
#         idioma_ya_analizado = True
#         print(f"🌎 Idioma detectado: {idioma_detectado}")
#         return idioma_detectado
    
#     print("📝 Transcribiendo el audio...")
#     global modelo
#     global idioma_ya_analizado
#     global idioma_detectado
    
#     opciones = {}
#     if not idioma_ya_analizado:
#         idioma = detectar_idioma()
#     else:  
#         idioma = idioma_detectado
    
#     opciones['language'] = idioma

#     resultado = modelo.transcribe("grabacion.wav", **opciones)
#     texto = resultado['text']
    
#     with open("transcripcion.txt", "w", encoding="utf-8") as f:
#         f.write(texto)    
    
#     print(f"✅ Transcripción guardada: {texto}")
#     return texto

#     # resultado = modelo.transcribe("grabacion.wav")
#     # texto = resultado['text']
    
#     # with open("transcripcion.txt", "w", encoding="utf-8") as f:
#     #     f.write(texto)    
    
#     # print(f"✅ Transcripción guardada: {texto}")
#     # return texto

def generar_bienvenida():
    global conversation_history
    prompt_bienvenida = conversation_history.copy()
    prompt_bienvenida.append({"role":"system", "content": "Genera un mensaje corto de bienvenida para el asistente de la peluquería \
                              y pregúntale en que le puedes ayudar"})
    try:
        response = openai.ChatCompletion.create( 
            model=MODELO_OPEN_AI,  
            messages=prompt_bienvenida,
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip() 
        conversation_history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

def generar_despedida():
    global conversation_history
    global idioma_detectado
    prompt_despedida = conversation_history.copy()
    prompt_despedida.append({"role":"system", "content": f"Teniendo en cuenta toda la conversación, Genera un mensaje corto de despedida de forma amable y dejar abierta la posibilidad de contacto, como: \
        ¡Esperamos verte pronto en nuestra peluquería! Si tienes más dudas, no dudes en llamarnos. ¡Que tengas un buen día! \
        No hagas preguntas, simplemente despidete de forma educada. Genera el mensaje en el idioma: {idioma_detectado}"})
    try:
        response = openai.ChatCompletion.create( # Generar respuesta utilizando la API de OpenAI
            model=MODELO_OPEN_AI,  # Modelo que deseas usar
            messages=prompt_despedida, # max_tokens=150,
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
        conversation_history.append({"role": "assistant", "content": reply}) # Guardar la respuesta en el historial
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

def generar_idiomas_disponibles():
    global conversation_history
    #idiomas = ["es", "en", "de"]
    idiomas = ["en"]
    for idioma in idiomas:
        if idioma == "es":
            idioma_detectado = "es-es"
            texto = "Podemos atenderte en español, inglés y alemán."
        elif idioma == "en":
            idioma_detectado = "en-us"
            texto = "we can attend you in English, Spanish and German."
        elif idioma == "de":
            idioma_detectado = "de-de"
            texto = "Wir können Sie auf Deutsch, Englisch und Spanisch bedienen."
        sintetizar_audio_con_gtts(texto)

def adecuar_respuesta_en_idioma(respuesta):
    global idioma_detectado
    prompt = []
    prompt.append({"role":"system", "content": f"Teniendo en cuenta el contexto, traduce la siguiente respuesta al idioma: {idioma_detectado}. \
        No hagas preguntas, simplemente traduce la respuesta. \
        Respuesta a traducir: {respuesta}"})
    try:
        response = openai.ChatCompletion.create( # Generar respuesta utilizando la API de OpenAI
            model=MODELO_OPEN_AI,  # Modelo que deseas usar
            messages=prompt, # max_tokens=150,
            temperature=0.5
        )
        reply = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Generar una respuesta de texto del asistente en base a la intencion del usuario
def generar_texto_respuesta_asistente(rol, texto_usuario: str):
    global conversation_history
    global idioma_detectado
    if rol == 'system':
        conversation_history.append({"role": "system", "content": texto_usuario}) # Añadir el mensaje del usuario al historial de la conversación
    else:
        conversation_history.append({"role": "user", "content": texto_usuario})
    try:
        response = openai.ChatCompletion.create( 
            model=MODELO_OPEN_AI,  
            messages=conversation_history, 
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
        conversation_history.append({"role": "assistant", "content": reply}) # Guardar la respuesta en el historial
        if idioma_detectado != "es" or idioma_detectado != "es-es":
            reply = adecuar_respuesta_en_idioma(reply)
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Guardar el historial en un fichero
def guardar_conversacion():
    global conversation_history
    with open("historialConversacion.txt",'w',encoding="utf-8") as f:
        for interaccion in conversation_history:
            for role, content in interaccion.items():
                f.write('{'+ f"{role}" +':'+ f"{content}"+'},')
            f.write('\n')

# Obtener el dia y mes actual, y en palabras (uno,dos,tres.. ; enero, febrero...)
def obtener_dia_mes_actual():
    dia_actual = str(date.today().day).zfill(2)
    mes_actual = str(date.today().month).zfill(2)
    ano_actual = date.today().year
    #fecha_actual = date.today().strftime("%d-%m-%Y")

    dias_en_palabras = [
        "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
        "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve", "veinte",
        "veintiuno", "veintidós", "veintitrés", "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve", "treinta", "treinta y uno"
    ]
    meses_en_palabras = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
    ]
    dia_con_palabra = dias_en_palabras[int(dia_actual)-1]
    mes_con_palabra = meses_en_palabras[int(mes_actual)-1]
    fecha_actual_en_palabra = f"{dia_con_palabra} de {mes_con_palabra} de dos mil venticinco"
    return mes_actual,fecha_actual_en_palabra

# Obtener el dia de la peticion del usuario
def obtener_dia(texto_usuario: str):
    hoy = date.today()
    hoy_xx = hoy.strftime("%d")

    # Calcular el número de días hasta el lunes de la próxima semana
    dias_hasta_lunes = (0 - hoy.weekday() + 7) % 7  # 0 es lunes

    # Si hoy es lunes, se suma 7 días para obtener el próximo lunes
    #if dias_hasta_lunes == 0:
     #   dias_hasta_lunes = 7

    # Obtener el lunes de la próxima semana
    lunes_proxima = hoy + timedelta(days=dias_hasta_lunes)

    # Obtener los días de lunes a viernes de la próxima semana
    dias_semana_proxima_calculo = [lunes_proxima + timedelta(days=i) for i in range(5)] 

    # Obtener el lunes de la siguiente semana
    lunes_siguiente = lunes_proxima + timedelta(weeks=1)

    # Obtener los días de lunes a viernes de la semana siguiente
    dias_semana_siguiente_calculo = [lunes_siguiente + timedelta(days=i) for i in range(5)]  

    # Formatear los días en formato "dd"
    dias_semana_proxima = [dia.strftime("%d") for dia in dias_semana_proxima_calculo]
    dias_semana_siguiente = [dia.strftime("%d") for dia in dias_semana_siguiente_calculo]

    _, fecha_actual_con_palabras = obtener_dia_mes_actual()
    
    prompt_instructions = f"Extrae el día del siguiente texto. Te lo puede especificar como formato número o como día de la semana (lunes, martes,\
        miércoles, jueves, viernes, sábado o domingo). Ten en cuenta que la fecha de hoy es: {fecha_actual_con_palabras}. \
        Devuelveme el número de día que corresponda en cada caso. Si no se especifica un día o el día es mayor que 31, devuelve el número 00. \
        Devuelve sólo el número. Si el día se corresponde del 0 al 9, devuelvelo en el formato 00,01,02,03,04,05,06,07,08 o 09. '\n' \
        Por ejemplo: '\n' \
        Posibles expresiones del usuario: \
        Hola quiero reservar el próximo lunes, el lunes de la semana que viene, \
        el siguiente lunes, el lunes de la semana siguiente, el lunes de la semana entrante. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_proxima[0]} '\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Hola quiero reservar el próximo martes, el martes de la semana que viene, \
        el siguiente martes, el martes de la semana siguiente, el martes de la semana entrante. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_proxima[1]} '\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Hola quiero reservar el próximo miercoles, el miercoles de la semana que viene, \
        el siguiente miercoles, el miercoles de la semana siguiente, el miercoles de la semana entrante. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_proxima[2]} '\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Hola quiero reservar el próximo jueves, el jueves de la semana que viene, \
        el siguiente jueves, el jueves de la semana siguiente, el jueves de la semana entrante. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_proxima[3]} '\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Hola quiero reservar el próximo viernes, el viernes de la semana que viene, \
        el siguiente viernes, el viernes de la semana siguiente, el viernes de la semana entrante. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_proxima[4]} '\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Me gustaría reservar el lunes de la semana que viene no, de la siguiente. El lunes después del próximo,\
        el lunes, en dos semanas, el lunes siguiente al próximo, el lunes que sigue a la próxima semana, para el lunes dentro de dos semanas. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_siguiente[0]}'\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Me gustaría reservar el martes de la semana que viene no, de la siguiente. El martes después del próximo,\
        el martes, en dos semanas, el martes siguiente al próximo, el martes que sigue a la próxima semana, para el martes dentro de dos semanas. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_siguiente[1]}'\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Me gustaría reservar el miercoles de la semana que viene no, de la siguiente. El miercoles después del próximo,\
        el miercoles, en dos semanas, el miercoles siguiente al próximo, el miercoles que sigue a la próxima semana, para el miercoles dentro de dos semanas. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_siguiente[2]}'\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Me gustaría reservar el jueves de la semana que viene no, de la siguiente. El jueves después del próximo,\
        el jueves, en dos semanas, el jueves siguiente al próximo, el jueves que sigue a la próxima semana, para el jueves dentro de dos semanas. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_siguiente[3]}'\n' \
        Otro ejemplo: '\n' \
        Posibles expresiones del usuario: Me gustaría reservar el viernes de la semana que viene no, de la siguiente. El viernes después del próximo,\
        el viernes, en dos semanas, el viernes siguiente al próximo, el viernes que sigue a la próxima semana, para el viernes dentro de dos semanas. '\n' \
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {dias_semana_siguiente[4]}'\n' \
        Otro ejemplo: '\n' \
        Usuario: Tienes un hueco el dia 5. '\n'\
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras})'\n' \
        05 '\n' \
        Otro ejemplo: '\n' \
        Usuario: Quiero reservar el 6 de mayo.\
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) \
        06 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar en febrero.'\n'\
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        00 '\n' \
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar hoy.'\n'\
        (Teniendo en cuenta que hoy estamos a {fecha_actual_con_palabras}) '\n'\
        {hoy_xx}"
    prompt = []
    prompt.append({"role": "system", "content": prompt_instructions})
    prompt.append({"role": "user", "content": texto_usuario})
    try:
        response = openai.ChatCompletion.create( 
            model="gpt-4o-mini", 
            messages=prompt,
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip() 
        print("El dia es: ",reply)
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Obtener el mes de la peticion del usuario
def obtener_mes(texto_usuario: str,dia_usuario):
    mes_actual, fecha_actual_con_palabras = obtener_dia_mes_actual()
    dia_actual = str(date.today().day).zfill(2)
    mes_siguiente = int(mes_actual)+1

    prompt_instructions = f"Extrae el mes del siguiente texto. Te lo puede especificar como formato número o como mes del año (enero, febrero,\
        marzo, abril, mayo, junio, julio, agosto, septiembre, octubre, noviembre, diciembre). Ten en cuenta que hoy es {fecha_actual_con_palabras}. '\n' \
        Devuelveme el número del mes (del 1 al 12). Si el mes se corresponde del 1 al 9, devuelvelo en el formato 01,02,03,04,05,06,07,08 o 09. '\n' \
        Si no se especifica un mes, devuelve el mes actual. '\n' \
        La funcion calculaMes() es la siguiente: '\n'\
        - Si {dia_usuario} es igual a 0, devuelve {mes_actual} \
        - Sólo si no se especifica el mes, y {dia_actual} es menor que {dia_usuario}, devuelve {mes_actual}. '\n'  \
        - Sólo si no se especifica el mes, y {dia_actual} es mayor que {dia_usuario}, devuelve {mes_siguiente}. '\n' \
        Devuelve sólo el número. '\n'\
        Por ejemplo: '\n' \
        Usuario: Hola quiero reservar el próximo martes.'\n'\
        (Teniendo en cuenta que el usuario no ha especificado el mes, realiza la funcion calculaMes())'\n' \
        Devuelve el resultado de calculaMes()'\n' \
        Otro ejemplo: '\n' \
        Usuario: Me gustaría reservar el martes de la semana que viene no, de la siguiente.'\n'\
        (Teniendo en cuenta que el usuario no ha especificado el mes, realiza la funcion calculaMes()) '\n'\
        Devuelve el resultado de calculaMes() '\n'\
        Otro ejemplo: '\n' \
        Usuario: Tienes un hueco para mañana? '\n'\
        (Teniendo en cuenta que el usuario no ha especificado el mes, realiza la funcion calculaMes()) '\n'\
        Devuelve el resultado de calculaMes() '\n'\
        Otro ejemplo: '\n' \
        Usuario: Tienes un hueco para mañana? '\n'\
        (Teniendo en cuenta que el usuario no ha especificado el mes, realiza la funcion calculaMes()) '\n'\
        Devuelve el resultado de calculaMes() '\n'\
        Un ejemplo más: '\n' \
        (Teniendo en cuenta que el usuario no ha especificado el mes, realiza la funcion calculaMes()) '\n'\
        Devuelve el resultado de calculaMes() '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de enero. '\n'\
        01 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de febrero. '\n'\
        02 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de marzo.'\n'\
        03 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de abril.'\n'\
        04 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de mayo.'\n'\
        05 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de junio.'\n'\
        06 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de julio.'\n'\
        07 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de agosto.'\n'\
        08 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de septiembre.'\n'\
        09 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de octubre.'\n'\
        10 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de noviembre.'\n'\
        11 '\n'\
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar para el primer viernes de diciembre.'\n'\
        12 '\n'"
    prompt = []
    prompt.append({"role": "system", "content": prompt_instructions})
    prompt.append({"role": "user", "content": texto_usuario})
    try:
        response = openai.ChatCompletion.create( 
            model=MODELO_OPEN_AI, 
            messages=prompt,
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Obtener el mes de la peticion del usuario
def obtener_ano(texto_usuario: str):
    prompt_instructions = "Devuelve el año actual. Devuelve solo el número. \
        Por ejemplo: '\n' \
        Usuario: Hola quiero reservar el próximo martes.\
        (Teniendo en cuenta que hoy es 28 de enero de 2025) \
        Tú: 2025 \
        Otro ejemplo: '\n' \
        Usuario: Me gustaría reservar el martes de la semana que viene no, de la siguiente.\
        (Teniendo en cuenta que hoy es 28 de enero de 2025) \
        Tú: 2025 \
        Otro ejemplo: '\n' \
        Usuario: Tienes un hueco para mañana?.\
        (Teniendo en cuenta que hoy es 28 de enero de 2025) \
        Tú: 2025 \
        Un ejemplo más: '\n' \
        Usuario: Quiero reservar el 3 de enero de 2026.\
        (Teniendo en cuenta que hoy es 28 de enero de 2025) \
        Tú: 2026"
    prompt = []
    prompt.append({"role": "system", "content": prompt_instructions})
    prompt.append({"role": "user", "content": texto_usuario})
    try:
        response = openai.ChatCompletion.create(
            model=MODELO_OPEN_AI,  
            messages=prompt, 
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip() 
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Comprobar dia, mes y año del usuario 
def comprobar_fecha_cita(texto_usuario):
    global conversation_history

    dia = obtener_dia(texto_usuario)
    mes = obtener_mes(texto_usuario,dia)
    ano = obtener_ano(texto_usuario)

    print(dia,mes,ano)

    print(f"Fecha obtenida: el dia {dia}, {mes}, {ano}")
    return dia,mes,ano

# Obtener hora del usuario
def obtener_hora(texto_usuario):
    prompt_instructions = "Obtén la hora del siguiente texto. Devuelvela en el formato hh:mm. Si no se especifica devuelve 00:00 '\n'\
        Por ejemplo: '\n' \
        Usuario: A las cuatro de la tarde'\n'\
        Tú: 16:00 '\n'\
        Otro ejemplo: '\n' \
        Usuario: A las 14 '\n'\
        Tú: 14:00 '\n' \
        Otro ejemplo: '\n' \
        Usuario: a las cinco y cuarto de la tarde'\n'\
        Tú: 17:15 '\n'\
        Un ejemplo más: '\n' \
        Usuario: a las once menos cuarto de la mañana '\n'\
        Tú: 10:45 '\n'\
        Si no se especifica la hora, devuelve solo 00:00. '\n'"
    prompt = []
    prompt.append({"role": "system", "content": prompt_instructions})
    prompt.append({"role": "user", "content": texto_usuario})
    try:
        response = openai.ChatCompletion.create( 
            model=MODELO_OPEN_AI,  
            messages=prompt, 
            temperature=0.7
        )
        reply = response['choices'][0]['message']['content'].strip()
        return reply
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

# Calcular la hora fin de la cita del cliente
def obtener_hora_fin(hora):
    prompt = []
    prompt_instructions = f"Dime qué hora sería, si son las {hora} y pasa media hora. \
        Devuelve únicamente la hora en el formato hh:mm. Nada mas"
    prompt.append({"role": "system", "content": prompt_instructions})
    try:
        response = openai.ChatCompletion.create( 
            model=MODELO_OPEN_AI,  
            messages=prompt, 
            temperature=0.7
        )
        hora_fin = response['choices'][0]['message']['content'].strip() 
        return hora_fin
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

def hora_en_horario_laboral(hora):
    # Definir el horario laboral
    hora_inicio = datetime.strptime("09:00", "%H:%M").time()
    hora_fin = datetime.strptime("17:35", "%H:%M").time()

    # Convertir la hora de entrada a un objeto time
    hora_usuario = datetime.strptime(hora, "%H:%M").time()

    # Comprobar si la hora está dentro del horario laboral
    if hora_inicio <= hora_usuario <= hora_fin:
        return True
    else:
        return False

def es_dia_entre_semana(dia, mes):
    try:
        dt = datetime(2025, int(mes), int(dia))
        print(dt)
    except ValueError:
        return False
    return dt.weekday() < 5
    # dia = int(dia)
    # mes = int(mes)
    # hoy = datetime.today()
    # sabados_y_domingos = []

    # # Buscar el próximo sábado
    # while hoy.weekday() != 5:
    #     hoy += timedelta(days=1)

    # # Generar los próximos 10 sábados y domingos
    # for _ in range(10):
    #     sabado = hoy
    #     domingo = hoy + timedelta(days=1)

    #     sabados_y_domingos.append((int(sabado.strftime('%d')), int(sabado.strftime('%m'))))
    #     sabados_y_domingos.append((int(domingo.strftime('%d')), int(domingo.strftime('%m'))))

    #     hoy += timedelta(days=7)

    # # Verificar si el día y mes están en la lista
    # return (dia, mes) not in sabados_y_domingos
 
# Comprobar si hay disponibilidad en la franja horaria
def hay_disponibilidad(dia,mes,hora_inicio,hora_fin):
    """
    Verifica si hay un evento en una fecha y hora específica.
    :param dia: Día del evento
    :param mes: Mes del evento
    :param hora_inicio: Hora de inicio en formato HH:MM
    :param duracion: Duración del evento en minutos (default: 30 min)
    :return: True si hay un evento, False si está libre
    """
    # Ruta al archivo JSON de la cuenta de servicio
    SERVICE_ACCOUNT_FILE = "credentials.json"

    # Autenticación con la cuenta de servicio
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/calendar"])

    # ID del calendario
    CALENDAR_ID = "c_15c3fe34d87bf464fb97157917c395b707437b1cbac3b3cefbbf5b3393fa39d5@group.calendar.google.com"
    fecha = f"2025-{mes}-{dia}"
    
    # Convertir la hora de inicio y fin a UTC
    hora_inicio_utc = convertir_a_utc(fecha, hora_inicio)
    
    # Calcular la hora de fin sumando la duración
    dt_inicio = datetime.strptime(hora_inicio_utc, "%Y-%m-%dT%H:%M:%S%z")
    dt_fin = dt_inicio + timedelta(minutes=30)
    hora_fin_utc = dt_fin.isoformat()

    service = build("calendar", "v3", credentials=creds)

    # Buscar eventos en la franja horaria
    events_result = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=hora_inicio_utc,  # Hora de inicio
        timeMax=hora_fin_utc,     # Hora de fin
        singleEvents=True,
        orderBy="startTime"
    ).execute()

    eventos = events_result.get("items", [])

    if eventos:
        print("⛔ La franja horaria NO está disponible. Hay eventos existentes:")
        for event in eventos:
            print(f"- {event['summary']} desde {event['start']['dateTime']} hasta {event['end']['dateTime']}")
        return False  # Hay eventos en ese horario
    else:
        print("✅ La franja horaria está libre.")
        return True # No hay eventos en ese horario

def obtener_horas_disponibles_mismo_dia(dia,mes):
        
    def obtener_horas_disponibles(horas_reservadas, inicio="09:00", fin="18:00"):
        # Lista de horas posibles en intervalos de 30 minutos
        horas_disponibles = []
        
        # Crear el rango de horas
        hora_inicio = datetime.strptime(inicio, "%H:%M")
        hora_fin = datetime.strptime(fin, "%H:%M")
        while hora_inicio < hora_fin:
            horas_disponibles.append(hora_inicio.strftime("%H:%M"))
            hora_inicio += timedelta(minutes=30)

        # Eliminar las horas que están reservadas
        horas_finales = [hora for hora in horas_disponibles if hora not in horas_reservadas]

        return horas_finales
    def obtener_disponibilidad(dia,mes):
        # Ruta al archivo JSON de la cuenta de servicio
        SERVICE_ACCOUNT_FILE = "credentials.json"

        # Autenticación con la cuenta de servicio
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/calendar"])

        # ID del calendario
        CALENDAR_ID = "c_15c3fe34d87bf464fb97157917c395b707437b1cbac3b3cefbbf5b3393fa39d5@group.calendar.google.com"
        fecha = f"2025-{mes}-{dia}"
        
        # Hora de inicio y fin del día
        time_min = convertir_a_utc(fecha, "09:00")  # Comienzo del día
        time_max = convertir_a_utc(fecha, "17:59")  # Fin del día
        
        # # Calcular la hora de fin sumando la duración
        # dt_inicio = datetime.strptime(hora_inicio_utc, "%Y-%m-%dT%H:%M:%S%z")
        # dt_fin = dt_inicio + timedelta(minutes=30)
        # hora_fin_utc = dt_fin.isoformat()

        service = build("calendar", "v3", credentials=creds)

        # Buscar eventos en la franja horaria
        events_result = service.events().list(
            calendarId=CALENDAR_ID,
            timeMin=time_min,  # Hora de inicio
            timeMax=time_max,     # Hora de fin
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        eventos = events_result.get("items", [])

        horas_ya_reservadas = []

        if eventos:
            print("⛔ La franja horaria NO está disponible. Hay eventos existentes:")
            for event in eventos:
                print(f"- {event['summary']} desde {event['start']['dateTime']} hasta {event['end']['dateTime']}")
                
                dt = datetime.fromisoformat(event['start']['dateTime'])
                hora_minuto = dt.strftime("%H:%M")
                horas_ya_reservadas.append(hora_minuto)
        return horas_ya_reservadas
    def pasar_horas_a_texto(horas):
        horas_en_texto = []
        for hora in horas:
            prompt = []
            prompt_instructions = f"Devuelveme qué hora es esta: {hora}. \
                No me la devuelvas en formato numérico, sino en texto escrito. \
                Devuelve solo la informacion de la hora, no añadas ningún texto extra. \
                Emplea y cuarto, y media, menos cuarto; para representar las quince, y treinta, y menos quince, respectivamente. \
                Por ejemplo: si la hora es 15:30, devuelve: tres y media. \
                Si la hora es 12:30, devuelve: doce y cuarto. \
                NO uses son ni nada para decir la hora"
            prompt.append({"role": "system", "content": prompt_instructions})
            try:
                response = openai.ChatCompletion.create( # Generar respuesta utilizando la API de OpenAI
                    model=MODELO_OPEN_AI,  # Modelo que deseas usar
                    messages=prompt, # max_tokens=150,
                    temperature=0.7
                )
                hora_en_texto = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
                horas_en_texto.append(hora_en_texto)
            except Exception as e:
                return f"Hubo un error al generar la respuesta: {str(e)}"
        return horas_en_texto

    horas_ya_reservadas = obtener_disponibilidad(dia,mes)
    horas_disponibles = obtener_horas_disponibles(horas_ya_reservadas)
    horas_en_texto = pasar_horas_a_texto(horas_disponibles)
    return horas_en_texto
 
def pasar_horas_a_texto_v2(hora):
    hora_en_texto = ""
    prompt = []
    prompt_instructions = f"Devuelveme qué hora es esta: {hora}. \
        No me la devuelvas en formato numérico, sino en texto escrito. \
        Devuelve solo la informacion de la hora, no añadas ningún texto extra. \
        Emplea y cuarto, y media, menos cuarto; para representar las quince, y treinta, y menos quince, respectivamente. \
        Por ejemplo: si la hora es 15:30, devuelve: tres y media. \
        Si la hora es 12:30, devuelve: doce y cuarto. \
        NO uses son ni nada para decir la hora. No incluyas informacion adicional. Me da igual si es de la mañana o de la tarde. \
            Por ejemplo: si la hora es 15:30, devuelve: tres y media."
    prompt.append({"role": "system", "content": prompt_instructions})
    try:
        response = openai.ChatCompletion.create( # Generar respuesta utilizando la API de OpenAI
            model=MODELO_OPEN_AI,  # Modelo que deseas usar
            messages=prompt, # max_tokens=150,
            temperature=0.7
        )
        hora_en_texto = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
        return hora_en_texto
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}"

def dia_a_texto_v2(dia):
    dia_en_texto = ""
    prompt = []
    prompt_instructions = f"Devuelveme qué dia es este: {dia}. \
        No me la devuelvas en formato numérico, sino en texto escrito. \
        Devuelve solo la informacion del dia. \
        Por ejemplo: si el dia es 04, devuelve: cuatro. \
        Si el dia es 07, devuelve: siete. \
        Devuelve solo estrictamente el dia, no añadas ningún texto extra."
    prompt.append({"role": "system", "content": prompt_instructions})
    try:
        response = openai.ChatCompletion.create( # Generar respuesta utilizando la API de OpenAI
            model=MODELO_OPEN_AI,  # Modelo que deseas usar
            messages=prompt, # max_tokens=150,
            temperature=0.7
        )
        dia_en_texto = response['choices'][0]['message']['content'].strip() # Extraer la respuesta del asistente
        return dia_en_texto
    except Exception as e:
        return f"Hubo un error al generar la respuesta: {str(e)}" 

# Pasar las horas a un formato correcto para google calendar
def convertir_a_utc(fecha, hora, zona_horaria="Europe/Madrid"):
    """
    Convierte una fecha y hora local a UTC en formato ISO 8601 para Google Calendar.
    :param fecha: Fecha en formato YYYY-MM-DD
    :param hora: Hora en formato HH:MM
    :param zona_horaria: Zona horaria local (Ejemplo: 'Europe/Madrid')
    :return: Fecha y hora en formato ISO 8601 (UTC)
    """
    local_tz = pytz.timezone(zona_horaria)
    dt_local = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M")  # Convierte a datetime
    dt_local = local_tz.localize(dt_local)  # Asigna la zona horaria local
    dt_utc = dt_local.astimezone(pytz.utc)  # Convierte a UTC
    return dt_utc.isoformat()

def reservar_cita(dia, mes, hora_inicio, hora_fin):
    """Reserva una cita en Google Calendar con conversión de hora"""
    # Ruta al archivo JSON de la cuenta de servicio
    SERVICE_ACCOUNT_FILE = "credentials.json"

    # Autenticación con la cuenta de servicio
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/calendar"])

    # ID del calendario
    CALENDAR_ID = "c_15c3fe34d87bf464fb97157917c395b707437b1cbac3b3cefbbf5b3393fa39d5@group.calendar.google.com"
    
    # Convertir fecha a formato correcto
    fecha = f"2025-{mes}-{dia}"
    
    # Convertir las horas a UTC para evitar errores de desfase
    hora_inicio_utc = convertir_a_utc(fecha, hora_inicio)
    hora_fin_utc = convertir_a_utc(fecha, hora_fin)

    service = build("calendar", "v3", credentials=creds)

    event = {
        "summary": "Cita con cliente",
        "location": "Calle Mexico, Alicante",
        "description": "Realizar servicio deseado",
        "start": {
            "dateTime": hora_inicio_utc,  # Formato ISO 8601 en UTC
            "timeZone": "UTC",  # Ahora la API lo interpretará correctamente
        },
        "end": {
            "dateTime": hora_fin_utc,  # Formato ISO 8601 en UTC
            "timeZone": "UTC",
        },
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 24 * 60},
                {"method": "popup", "minutes": 10},
            ],
        },
    }

    event = service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
    print(f"🎉 Evento creado correctamente: {event.get('htmlLink')}")
    #print(event)

def eliminar_cita(dia, mes, hora_inicio, hora_fin):
    """Elimina una cita en Google Calendar dentro de un rango de tiempo especificado."""
    SERVICE_ACCOUNT_FILE = "credentials.json"
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=["https://www.googleapis.com/auth/calendar"])
    CALENDAR_ID = "c_15c3fe34d87bf464fb97157917c395b707437b1cbac3b3cefbbf5b3393fa39d5@group.calendar.google.com"
    
    fecha = f"2025-{mes}-{dia}"
    hora_inicio_utc = convertir_a_utc(fecha, hora_inicio)
    hora_fin_utc = convertir_a_utc(fecha, hora_fin)
    
    service = build("calendar", "v3", credentials=creds)
    
    # Buscar eventos en el rango de tiempo especificado
    eventos = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=hora_inicio_utc,
        timeMax=hora_fin_utc,
        singleEvents=True
    ).execute()
    
    items = eventos.get("items", [])
    
    if not items:
        print("❌ No se encontraron citas en el rango de tiempo especificado.")
        return False
    
    # Eliminar el primer evento encontrado en ese rango
    event_id = items[0]["id"]
    service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
    print("✅ Cita eliminada correctamente.")
    return True
      
def modifica(texto_usuario):
    global conversation_history
    sabado_o_domingo = False
    hora_no_laboral = False
    preguntarDatos= "El usuario quiere modificar una cita que ha hecho con la peluquería. \
        Los datos que te hacen falta para modificar la reserva son la fecha y la hora de la cita que desea cambiar. \
        Sin embargo, si el cliente tiene alguna duda relacionada con el ámbito de la peluquería, respóndesela y pide educadamente los datos. \
        Ofrecele tu ayuda cuando el usuario tenga alguna duda, sino, tu objetivo es preguntar los datos para hacer la nueva reserva. "
    
    dia, mes, ano = comprobar_fecha_cita(texto_usuario)
    hora = obtener_hora(texto_usuario)
    if (hora != "00:00"): hora_fin = obtener_hora_fin(hora)

    if (not(es_dia_entre_semana(dia,mes)) and dia != "00"):
        no_es_entre_semana = f"El usuario ha intentado cambiar una cita que tenía para el {dia}, pero su cita no puede ser ese \
                                    dia ya que es sabado o domingo. Indicale que la peluquería no abre los sábados y domingos.\
                                    Dale la oportunidad de cambiar su cita de nuevo."
        conversation_history.append({"role":"system","content":no_es_entre_semana})
        sabado_o_domingo = True
        dia = "00"     
        print("dia fuera del rango2")

    if (not(hora_en_horario_laboral(hora)) and hora != "00:00"):
        no_es_laboral = f"El usuario ha elegido una hora que no es laboral. \
            Indicale que la peluquería abre de nueve a seis. Dale la oportunidad de cambiar su cita de nuevo."
        conversation_history.append({"role":"system","content":no_es_laboral})
        hora_no_laboral = True        
        hora = "00:00"
        print("hora fuera del rango2")
    

    while (dia == "00" or mes == "00" or hora == "00:00"):
        if (sabado_o_domingo): preguntarDatos += no_es_entre_semana
        if (hora_no_laboral):  preguntarDatos += no_es_laboral
        asistente = generar_texto_respuesta_asistente('system',preguntarDatos)
        conversation_history.append({"role":"system","content":preguntarDatos})

        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        #sintetizar_audio_con_gtts(preguntarDatos)
        grabar_audio()
        texto_usuario = transcribir_audio()
        print(texto_usuario)
        print("El nuevo texto es: ", texto_usuario)

        prompt = f"En base a toda la conversación, y a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
        quiere terminar con el proceso de modificar la cita o se ha arrepentido. Sino devuelve un 0."
        salida = generar_texto_respuesta_asistente('system',prompt)

        if (salida == "-1"):
            conversation_history.append({"role":"system","content":"El usuario se ha arrepentido de modificar la cita."})
            return texto_usuario

        auxDia, auxMes, auxAno = comprobar_fecha_cita(texto_usuario)
        auxHora = obtener_hora(texto_usuario)

        if (auxDia != "00"):
            dia = auxDia
            mes = auxMes
            ano = auxAno
        if (auxHora != "00:00" and auxHora != "00"):
            hora = auxHora
            hora_fin = obtener_hora_fin(hora)

        if (not(es_dia_entre_semana(dia,mes))):
            conversation_history.append({"role":"system","content":f"El usuario ha intentado cambiar una cita que tenía para el {dia}, pero su cita no puede ser ese \
                                    dia ya que es sabado o domingo. Indicale que la peluquería no abre los sábados y domingos.\
                                    Dale la oportunidad de cambiar su cita de nuevo."})
            dia = "00"
            print("dia fuera del rango")
        if (not(hora_en_horario_laboral(hora))):
            conversation_history.append({"role":"system","content":"El usuario ha elegido una hora que no es laboral. \
                Indicale que la peluquería abre de nueve a seis. Dale la oportunidad de cambiar su cita de nuevo."})
            print("hora fuera del rango")
            hora = "00:00" 

        conversation_history.append({"role":"user","content":texto_usuario})

    if (not(hay_disponibilidad(dia,mes,hora,hora_fin))):
        eliminar_cita(dia,mes,hora,hora_fin)
        reserva("")
        conversation_history.append({"role":"system","content":"La cita ha sido modificada correctamente."})
        dia_en_texto = dia_a_texto_v2(dia)
        hora_en_texto = pasar_horas_a_texto_v2(hora)
        prompt = f"Dile al usuario que ya hemos cambiado su cita. Se breve"
        asistente = generar_texto_respuesta_asistente('system',prompt)
        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        #sintetizar_audio_con_gtts(asistente)
        return ""
    else:
        dia_en_texto = dia_a_texto_v2(dia)
        hora_en_texto = pasar_horas_a_texto_v2(hora)
        no_hay_cita = f"Dile al usuario de forma educada que no tenemos ninguna cita disponible para el dia {dia_en_texto} a las {hora_en_texto}. \
            Invitale a revisar su agenda y a que vuelva a preguntar por otra fecha."
        asistente = generar_texto_respuesta_asistente('system',no_hay_cita)
        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        
        grabar_audio()
        texto_usuario = transcribir_audio()
        print(texto_usuario)
        print("El nuevo texto es: ", texto_usuario)

        prompt = f"En base a toda la conversación, y a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
        quiere terminar con el proceso de modificar la cita o se ha arrepentido. Sino devuelve un 0."
        salida = generar_texto_respuesta_asistente('system',prompt)

        if (salida == "-1"):
            conversation_history.append({"role":"system","content":"El usuario se ha arrepentido de modificar la cita."})
            return texto_usuario
        modifica(texto_usuario)
        
def elimina(texto_usuario):
    global conversation_history
    sabado_o_domingo = False
    hora_no_laboral = False
    preguntarDatos= "El usuario quiere eliminar una cita que ha hecho con la peluquería. \
        Los datos que te hacen falta para eliminar la reserva son la fecha y la hora. \
        Sin embargo, si el cliente tiene alguna duda relacionada con el ámbito de la peluquería, respóndesela y pide educadamente los datos. \
        Ofrecele tu ayuda cuando el usuario tenga alguna duda, sino, tu objetivo es preguntar los datos para hacer la reserva. \
        De momento el usuario ha dicho: " + texto_usuario + ". Tenlo en cuenta a la hora de pedir los datos"
    
    dia, mes, ano = comprobar_fecha_cita(texto_usuario)
    hora = obtener_hora(texto_usuario)
    print(dia, mes, ano, hora)
    if (hora != "00:00"): hora_fin = obtener_hora_fin(hora)

    if (not(es_dia_entre_semana(dia,mes)) and dia != "00"):
        no_es_entre_semana = f"El usuario ha intentado borrar una cita para el dia {dia}, pero ha elegido \
                                     un dia que es sabado o domingo. Indicale que la peluquería no abre los sábados y domingos.\
                                    Dale la oportunidad de borrar la cita de nuevo."
        conversation_history.append({"role":"system","content":no_es_entre_semana})
        sabado_o_domingo = True
        dia = "00"     
        print("dia fuera del rango2")

    if (not(hora_en_horario_laboral(hora)) and hora != "00:00"):
        no_es_laboral = f"El usuario ha elegido una hora que no es laboral. \
            Indicale que la peluquería abre de nueve a seis. Dale la oportunidad de borrar la cita de nuevo."
        conversation_history.append({"role":"system","content":no_es_laboral})
        hora_no_laboral = True        
        hora = "00:00"
        print("hora fuera del rango2")

    while (dia == "00" or mes == "00" or hora == "00:00"):
        if (sabado_o_domingo): preguntarDatos += no_es_entre_semana
        if (hora_no_laboral):  preguntarDatos += no_es_laboral
        asistente = generar_texto_respuesta_asistente('system',preguntarDatos)
        conversation_history.append({"role":"system","content":preguntarDatos})

        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        #sintetizar_audio_con_gtts(preguntarDatos)
        grabar_audio()
        texto_usuario = transcribir_audio()
        print(texto_usuario)
        print("El nuevo texto es: ", texto_usuario)

        prompt = f"En base a toda la conversación, y a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
        quiere terminar con el borrado de la cita o se ha arrepentido. Sino devuelve un 0"
        salida = generar_texto_respuesta_asistente('system',prompt)

        if (salida == "-1"):
            conversation_history.append({"role":"system","content":"El usuario se ha arrepentido de borrar la cita."})
            return texto_usuario
        
        if (dia == "00"):
            dia, mes, ano = comprobar_fecha_cita(texto_usuario)
            print(dia, mes, ano, hora)
        if (hora == "00:00"):
            hora = obtener_hora(texto_usuario)
            hora_fin = obtener_hora_fin(hora)

        if (not(es_dia_entre_semana(dia,mes))):
            conversation_history.append({"role":"system","content":"El usuario ha elegido un dia que es sabado o domingo. \
                Indicale que la cita que quiere borra no puede ser los sábados y domingos porque la peluquería no está abierta.\
                Dale la oportunidad de borrar la cita de nuevo."})
            dia = "00"
            print("dia fuera del rango")
        if (not(hora_en_horario_laboral(hora))):
            conversation_history.append({"role":"system","content":"El usuario ha elegido una hora que no es laboral. \
                Indicale que la peluquería abre de nueve a seis. Dale la oportunidad de borrar la cita de nuevo."})
            print("hora fuera del rango")
            hora = "00:00"        
    
        conversation_history.append({"role":"user","content":texto_usuario})

    if(not(hay_disponibilidad(dia,mes,hora,hora_fin))):
        eliminar_cita(dia,mes,hora,hora_fin)
        conversation_history.append({"role":"system","content":"La cita ha sido eliminada correctamente."})
        dia_en_texto = dia_a_texto_v2(dia)
        hora_en_texto = pasar_horas_a_texto_v2(hora)
        prompt = f"Dile al usuario que ha eliminado la cita que tenía para el dia {dia_en_texto} a las {hora_en_texto}. Se breve"
        asistente = generar_texto_respuesta_asistente('system',prompt)
        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        return ""
    else:
        dia_en_texto = dia_a_texto_v2(dia)
        hora_en_texto = pasar_horas_a_texto_v2(hora)
        no_hay_cita = f"Dile al usuario de forma educada que no tenemos ninguna cita ya reservada para el dia {dia_en_texto} a las {hora_en_texto}. \
            Invitale a revisar su agenda y a que vuelva a preguntar por otra fecha."
        asistente = generar_texto_respuesta_asistente('system',no_hay_cita)
        print(asistente)
        sintetizar_audio_con_gtts(asistente)

        grabar_audio()
        texto_usuario = transcribir_audio()
        print(texto_usuario)
        print("El nuevo texto es: ", texto_usuario)

        prompt = f"En base a toda la conversación, y a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
        quiere terminar con el proceso de eliminar la cita o se ha arrepentido. Sino devuelve un 0."
        salida = generar_texto_respuesta_asistente('system',prompt)

        if (salida == "-1"):
            conversation_history.append({"role":"system","content":"El usuario se ha arrepentido de eliminar o cancelar la cita."})
            return texto_usuario
        elimina(texto_usuario)

def reserva(texto_usuario):
    global conversation_history
    sabado_o_domingo = False
    hora_no_laboral = False
    preguntarDatos= "El usuario quiere reservar una cita en la peluquería. \
        Los datos que te hacen falta para hacer la reserva son la fecha y la hora. \
        Sin embargo, si el cliente tiene alguna duda relacionada con el ámbito de la peluquería, respóndesela y pide educadamente los datos. \
        Tu objetivo es preguntar los datos para hacer la reserva de forma amable. \
        Recuerdale muy brevemente al usuario que la peluquería está abierta de nueve a seis. \
        De momento el usuario ha dicho: " + texto_usuario + ". Tenlo en cuenta a la hora de pedir los datos."
        
    dia, mes, ano = comprobar_fecha_cita(texto_usuario)
    hora = obtener_hora(texto_usuario)
    print(dia, mes, ano, hora)
    if (hora != "00:00"): hora_fin = obtener_hora_fin(hora)

    # prompt = f"En base a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
    #     quiere saber los horarios que hay disponibles. Sino devuelve un 0. Por ejemplo: si el usuario dice: \
    #         '¿Cuáles son los horarios disponibles?', ¿Qué horas tienes disponibles para el dia X?, ¿Cuando tienes hueco?: devuelve un -1."
    # salida = generar_texto_respuesta_asistente('system',prompt)

    # if (salida == "-1"):
    # salida = conocer_intencion(texto_usuario)
    # if (salida == "16"):
    #     conversation_history.append({"role":"system","content":"El usuario ha preguntado por los horarios disponibles en la peluquería."})
    #     print("El usuario ha preguntado por los horarios disponibles en la peluquería.")
    #     if (dia != "00"):
    #         print("El dia es: ", dia)
    #         print("El mes es: ", mes)
    #         horas_en_texto = obtener_horas_disponibles_mismo_dia(dia,mes)
    #         conversation_history.append({"role":"system","content":f"El usuario ha preguntado por los horarios disponibles en la \
    #                                 peluquería para el día {dia}. Ofrecele las horas: {horas_en_texto[0]} , \
    #                                     {horas_en_texto[1]} y {horas_en_texto[2]} como disponibles."})
    #     else:
    #         conversation_history.append({"role":"system","content":f"El usuario ha preguntado por los horarios disponibles en la \
    #                                 peluquería para el día {dia}. Sin embargo, no ha especificado el dia. Dile que primero comente un día \
    #                                 para decirle los huecos que hay disponibles"})

    if (not(es_dia_entre_semana(dia,mes)) and dia != "00"):
        no_es_entre_semana = f"El usuario ha intentado reservar el dia {dia} pero ha elegido \
                                     un dia que es sabado o domingo. Indicale que la peluquería no abre los sábados y domingos.\
                                    Dale la oportunidad de reservar de nuevo. Se breve."
        conversation_history.append({"role":"system","content":no_es_entre_semana})
        sabado_o_domingo = True
        dia = "00"     
        print("dia fuera del rango2")

    if (not(hora_en_horario_laboral(hora)) and hora != "00:00"):
        no_es_laboral = f"El usuario ha elegido una hora que no es laboral. \
            Indicale que la peluquería no está abierta. Dale la oportunidad de reservar de nuevo. Se breve."
        conversation_history.append({"role":"system","content":no_es_laboral})
        hora_no_laboral = True        
        hora = "00:00"
        print("hora fuera del rango2")

    while (dia == "00" or mes == "00" or hora == "00:00"):
        if (sabado_o_domingo): preguntarDatos += no_es_entre_semana
        if (hora_no_laboral):  preguntarDatos += no_es_laboral
        asistente = generar_texto_respuesta_asistente('system',preguntarDatos)
        conversation_history.append({"role":"system","content":preguntarDatos})

        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        #sintetizar_audio_con_gtts_con_gtts(asistente)

        grabar_audio()
        texto_usuario = transcribir_audio()
        print(texto_usuario)
        print("El nuevo texto es: ", texto_usuario)

        prompt = f"En base a toda la conversación, y a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
        quiere terminar con el proceso de reserva o se ha arrepentido. Sino devuelve un 0"
        salida = generar_texto_respuesta_asistente('system',prompt)

        if (salida == "-1"):
            conversation_history.append({"role":"system","content":"El usuario se ha arrepentido de reservar la cita."})
            #return True
            return texto_usuario

        auxDia, auxMes, auxAno = comprobar_fecha_cita(texto_usuario)
        auxHora = obtener_hora(texto_usuario)

        if (auxDia != "00"):
            dia = auxDia
            mes = auxMes
            ano = auxAno
        if (auxHora != "00:00" and auxHora != "00"):
            hora = auxHora
            hora_fin = obtener_hora_fin(hora)

        print("Dia antes de checkear: ", dia)
        print("Hora antes de checkear: ", hora)
        
        if (not(es_dia_entre_semana(dia,mes))):
            conversation_history.append({"role":"system","content":"El usuario ha elegido un dia que es sabado o domingo. \
                Indicale que la peluquería no abre los sábados y domingos. Dale la oportunidad de reservar de nuevo. Se breve."})
            dia = "00"
            print("dia fuera del rango")
        if (not(hora_en_horario_laboral(hora))):
            conversation_history.append({"role":"system","content":"El usuario ha elegido una hora que no es laboral. \
                Indicale que la peluquería no está abierta. Dale la oportunidad de reservar de nuevo. Se breve."})
            print("hora fuera del rango")
            hora = "00:00"

        salida = conocer_intencion(texto_usuario)
        print("Salida Intencion: ", salida)
        if (salida == "16"):
            conversation_history.append({"role":"system","content":"El usuario ha preguntado por los horarios disponibles en la peluquería."})
            print("El usuario ha preguntado por los horarios disponibles en la peluquería.")
            if (dia != "00"):
                print("El dia es: ", dia)
                print("El mes es: ", mes)
                horas_en_texto = obtener_horas_disponibles_mismo_dia(dia,mes)
                conversation_history.append({"role":"system","content":f"El usuario ha preguntado por los horarios disponibles en la \
                                        peluquería para el día {dia}. Ofrecele las horas: {horas_en_texto[0]} , \
                                            {horas_en_texto[1]} y {horas_en_texto[2]} como disponibles."})
            else:
                conversation_history.append({"role":"system","content":f"El usuario ha preguntado por los horarios disponibles en la \
                                        peluquería para el día {dia}. Sin embargo, no ha especificado el dia. Dile que primero comente un día \
                                        para decirle los huecos que hay disponibles."})
        
        conversation_history.append({"role":"user","content":texto_usuario})

    if(hay_disponibilidad(dia,mes,hora,hora_fin)):
        reservar_cita(dia,mes,hora,hora_fin)
        conversation_history.append({"role":"system","content":"La cita ha sido reservada correctamente."})
        dia_texto = dia_a_texto_v2(dia)
        hora_texto = pasar_horas_a_texto_v2(hora)
        prompt = f"Dile al usuario que la cita ha sido reservada para el dia {dia_texto} a las {hora_texto}. Se breve"
        asistente = generar_texto_respuesta_asistente('system',prompt)
        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        #sintetizar_audio_con_gtts_con_gtts(asistente)
        #return True
        return ""
    else:
        horas_en_texto = obtener_horas_disponibles_mismo_dia(dia,mes)
        prompt= f"Dile al cliente que no hay disponibilidad para la fecha y hora que ha solicitado. Sin embargo, proponle \
            las siguientes horas: {horas_en_texto[0]} , {horas_en_texto[1]} y {horas_en_texto[2]} para el mismo dia que había pedido. \
            Preguntale si quiere reservar una de esas horas."
        asistente = generar_texto_respuesta_asistente('system',prompt)
        print(asistente)
        sintetizar_audio_con_gtts(asistente)
        #sintetizar_audio_con_gtts_con_gtts(asistente)

        grabar_audio()
        texto_usuario = transcribir_audio()
        print(texto_usuario)
        print("El nuevo texto es: ", texto_usuario)
        conversation_history.append({"role":"user","content":texto_usuario})

        prompt = f"En base a toda la conversación, y a la última respuesta del usuario: {texto_usuario}. Devuelve sólo un -1 si el usuario \
        no quiere reservar una de las horas proporcionadas anteriormente. Si ha dicho una de esas horas a la que le gustaría reservar, \
        considera que quiere seguir con el proceso de reserva y devuelve un 0."
        salida = generar_texto_respuesta_asistente('system',prompt)

        if (salida == "-1"):
            conversation_history.append({"role":"system","content":"El usuario se ha arrepentido de reservar la cita."})
            print("El usuario se ha arrepentido de reservar la cita.")
            #return True, 
            return texto_usuario
        else:
            texto_usuario = f"Quiero reservar la cita para el dia {dia}. La hora que me gustaría la puedes encontrar en mi respuesta: {texto_usuario}.\
                Si he dicho que quiero la primera, me refiero a las {horas_en_texto[0]}. \
                Si he dicho que quiero la segunda, me refiero a las {horas_en_texto[1]}. \
                Si he dicho que quiero la tercera, me refiero a las {horas_en_texto[2]}."
            reserva(texto_usuario)
    
if __name__ == "__main__":
    id_intencion = "1" # incializar un valor de intencion
    textoparaAnalizar = ""
    sintetizar_audio_con_gtts("empiezo")
    #sintetizar_audio_con_gtts(generar_bienvenida()) # saludamos al cliente
    #generar_idiomas_disponibles()

    while id_intencion != "2":
        if textoparaAnalizar == "":
            grabar_audio() 
            texto_usuario = transcribir_audio()
            id_intencion = conocer_intencion(texto_usuario)
        else:
            texto_usuario = textoparaAnalizar
            id_intencion = conocer_intencion(textoparaAnalizar)

        if id_intencion != "2": # Mientras que el usuario no quiera terminar la conversacion       
            if (id_intencion == "3"): # Reservar cita
                textoparaAnalizar = reserva(texto_usuario)

            elif(id_intencion == "4"): # Modificar cita
                textoparaAnalizar = modifica(texto_usuario)             

            elif(id_intencion == "5"): # Eliminar cita
                textoparaAnalizar = elimina(texto_usuario)
            elif id_intencion == "16":
                dia = obtener_dia(texto_usuario)
                mes = obtener_mes(texto_usuario,dia)
                horas_en_texto = obtener_horas_disponibles_mismo_dia(dia,mes)
                print("El dia es: ", dia)
                print("El mes es: ", mes)

                dia_en_texto = dia_a_texto_v2(dia)
                asistente = generar_texto_respuesta_asistente('system',f"El usuario ha preguntado por los horarios disponibles en la \
                                        peluquería para el día {dia_en_texto}. Ofrecele las horas: {horas_en_texto[0]} , \
                                            {horas_en_texto[1]} y {horas_en_texto[2]} como disponibles.")
                print(asistente)
                sintetizar_audio_con_gtts(asistente)
            else:
                respuesta = generar_texto_respuesta_asistente("user",texto_usuario)
                textoparaAnalizar = ""
                sintetizar_audio_con_gtts(respuesta)
            
            if textoparaAnalizar == "" and id_intencion != "16":
                prompt = "Preguntale al usuario si le puedes en ayudar en algo más. Se breve, directo y educado. Hazlo a forma de pregunta."
                conversation_history.append({"role":"system","content":prompt})
                asistente = generar_texto_respuesta_asistente('system',prompt)
                print(asistente)
                sintetizar_audio_con_gtts(asistente)
        else:
            id_intencion = "2"
    sintetizar_audio_con_gtts(generar_despedida())
    guardar_conversacion()
    print("Fin de la conversación")