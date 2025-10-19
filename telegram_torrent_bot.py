#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import shutil
import signal
import subprocess
import argparse
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import NetworkError, TimedOut, BadRequest

# ========================
# Utilidades de logging
# ========================
def setup_logging(quiet: bool, log_file: Optional[str]):
    level = logging.WARNING if quiet else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = []
    if log_file:
        rf = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        rf.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(rf)
    if not quiet:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(ch)
    logging.basicConfig(level=level, handlers=handlers)
    return logging.getLogger("tg-aria2c-simple")

# ========================
# Config default
# ========================
TRACKERS = [
    "udp://tracker.opentrackr.org:1337/announce",
    "udp://open.stealth.si:80/announce",
    "udp://tracker.torrent.eu.org:451/announce",
    "udp://tracker.dler.org:6969/announce",
    "udp://exodus.desync.com:6969",
]

TIMEOUT_SECONDS = 30 * 60  # 30 minutos

RE_PROGRESS = re.compile(r"\[#([0-9a-fA-F]{6})\s+([^\]]+)\]")

@dataclass
class Job:
    chat_id: int
    kind: str          # "pelicula" | "serie"
    info_hash: str     # BTIH
    base_dir: str      # destino raíz
    work_dir: Path     # subcarpeta dedicada a este job (T_<hash12>)

QUEUE: asyncio.Queue["Job"] = asyncio.Queue()
CURRENT: Dict[str, str] = {"hash": "", "line": "", "name": ""}  # estado visible en /status
ENQUEUED: set[str] = set()
app_global: Application
log: logging.Logger
worker_running: bool = True  # Control del estado del worker

# ========================
# Manejo de señales y cierre graceful
# ========================
def signal_handler(signum, frame):
    """Manejador de señales para cierre graceful"""
    global worker_running
    import logging
    logging.info("Recibida señal %d, iniciando cierre graceful...", signum)
    worker_running = False

def cleanup_on_exit():
    """Limpieza al salir del programa"""
    global worker_running
    worker_running = False
    import logging
    logging.info("Ejecutando limpieza final...")

# Registrar manejadores de señales y limpieza
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_on_exit)

# ========================
# Helpers
# ========================
def extract_hashes_from_text(text: str) -> list:
    """
    Extrae hashes de 40 caracteres hexadecimales de cualquier texto.
    Busca hashes en URLs, magnet links y texto plano.
    """
    # Regex para encontrar hashes de 40 caracteres hexadecimales
    # Busca tanto en mayúsculas como minúsculas
    # No usa \b porque puede estar dentro de URLs o magnet links
    hash_pattern = r'[0-9A-Fa-f]{40}'
    
    # Encuentra todos los matches
    matches = re.findall(hash_pattern, text)
    
    # Convierte a mayúsculas y elimina duplicados manteniendo el orden
    unique_hashes = []
    seen = set()
    for match in matches:
        upper_hash = match.upper()
        if upper_hash not in seen:
            unique_hashes.append(upper_hash)
            seen.add(upper_hash)
    
    return unique_hashes

def build_magnet(info_hash: str) -> str:
    base = f"magnet:?xt=urn:btih:{info_hash}&dn={info_hash}"
    tr = "".join(f"&tr={t}" for t in TRACKERS)
    return base + tr

def safe_mkdir(p: Path):
    """Crear directorio de forma segura con verificación de permisos"""
    try:
        p.mkdir(parents=True, exist_ok=True)
        # Verificar que podemos escribir en el directorio
        test_file = p / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        return True
    except (PermissionError, OSError) as e:
        # Usar logging básico si log no está disponible
        import logging
        logging.error("Error creando directorio %s: %s", p, e)
        return False

def human(msg: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", msg).strip()

async def run_refresh_plex(refresh_script: str):
    try:
        if refresh_script and os.path.isfile(refresh_script):
            subprocess.call(["bash", refresh_script])
            log.info("Refrescado Plex ejecutado.")
    except Exception as e:
        log.error("Error ejecutando refresh Plex: %s", e)

async def kill_process_tree(proc: asyncio.subprocess.Process):
    try:
        if proc.returncode is None:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
    except ProcessLookupError:
        pass

def cleanup_incomplete(job: Job):
    """Limpieza robusta de directorio de trabajo incompleto"""
    try:
        if job.work_dir.exists():
            # Intentar eliminar archivos individualmente primero
            for item in job.work_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except (PermissionError, OSError) as e:
                    # Usar logging básico si log no está disponible
                    import logging
                    logging.warning("No se pudo eliminar %s: %s", item, e)
            
            # Intentar eliminar el directorio principal
            try:
                job.work_dir.rmdir()
                import logging
                logging.info("Limpieza completada: %s", job.work_dir)
            except OSError:
                # Si no se puede eliminar, al menos está vacío
                import logging
                logging.warning("Limpieza incompleta: %s (directorio no vacío)", job.work_dir)
                
    except Exception as e:
        import logging
        logging.error("Error limpiando carpeta incompleta: %s", e)

def parse_and_store_progress(line: str):
    line = human(line)
    m = RE_PROGRESS.search(line)
    if m:
        gid = m.group(1)
        payload = m.group(2)
        CURRENT["line"] = f"[#{gid} {payload}]"
        parts = line.split("] ", 1)
        if len(parts) == 2 and parts[1]:
            CURRENT["name"] = parts[1].strip()

def try_extract_name(line: str):
    line = human(line)
    if "Download complete:" in line:
        tail = line.split("Download complete:", 1)[1].strip()
        nm = os.path.basename(tail.rstrip("/"))
        if nm:
            CURRENT["name"] = nm

def which(cmd: str) -> Optional[str]:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / cmd
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None

def has_cmd(cmd: str) -> bool:
    return which(cmd) is not None

# ========================
# Worker
# ========================
async def worker(aria2c_bin: str, summary_interval: int, nice: Optional[int], ionice_class: Optional[int], ionice_prio: Optional[int], refresh_script: str):
    global worker_running
    while worker_running:
        try:
            job = await QUEUE.get()
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                log.info("Event loop cerrado, terminando worker")
                break
            else:
                raise
        except Exception as e:
            log.error("Error inesperado en worker: %s", e)
            break
        key = f"{job.kind}:{job.info_hash}"
        ENQUEUED.discard(key)
        CURRENT.update({"hash": job.info_hash, "line": "", "name": ""})

        log.info("▶️ Iniciando: %s (%s) → %s", job.info_hash, job.kind, job.work_dir)
        
        # Verificar que podemos crear y escribir en el directorio
        if not safe_mkdir(job.work_dir):
            await notify(job.chat_id, f"❌ Error de permisos en {job.work_dir}. Descarga cancelada.")
            log.error("No se puede crear directorio de trabajo: %s", job.work_dir)
            return

        magnet = build_magnet(job.info_hash)

        base_cmd = [
            aria2c_bin,
            "--seed-time=0",
            "--file-allocation=none",
            "--max-connection-per-server=4",
            "--max-concurrent-downloads=1",
            "--summary-interval", str(summary_interval),
            "--dir", str(job.work_dir),
            "--bt-save-metadata=false",  # Evitar problemas con archivos de metadatos
            "--bt-metadata-only=false",
            "--bt-detach-seed-only=true",
            "--bt-remove-unselected-file=true",
            "--continue=true",  # Permitir continuar descargas interrumpidas
            "--retry-wait=5",  # Esperar 5 segundos entre reintentos
            "--max-tries=3",   # Máximo 3 intentos por archivo
            "--timeout=30",    # Timeout de 30 segundos para conexiones
            "--connect-timeout=10",  # Timeout de conexión de 10 segundos
            magnet,
        ]

        # ionice si está disponible (reduce prioridad de IO)
        if ionice_class is not None and has_cmd("ionice"):
            ionice_cmd = ["ionice", f"-c{ionice_class}"]
            if ionice_prio is not None:
                ionice_cmd += [f"-n{ionice_prio}"]
            cmd = ionice_cmd + base_cmd
        else:
            cmd = base_cmd

        # nice (reduce prioridad CPU) solo al subproceso
        def _preexec():
            if nice is not None:
                try:
                    os.nice(nice)
                except Exception:
                    pass

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=_preexec if nice is not None else None,
        )

        async def reader():
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode(errors="ignore").rstrip("\n")
                if RE_PROGRESS.search(line):
                    parse_and_store_progress(line)
                    # Progreso a DEBUG (silencioso con --quiet)
                    log.debug(CURRENT["line"])
                else:
                    # Estas líneas a INFO solo si no estás en quiet
                    if "Download complete:" in line:
                        try_extract_name(line)
                        log.info(human(line))
                    elif "Seeding is over" in line:
                        log.info(human(line))
                    elif "ERROR" in line or "Exception" in line:
                        log.error(human(line))
                        # Detectar errores específicos de archivos de segmento
                        if "Failed to write into the segment file" in line:
                            log.error("Error crítico de aria2c: No se puede escribir archivo de segmento")
                            # Intentar limpiar y reiniciar
                            await kill_process_tree(proc)
                            cleanup_incomplete(job)
                            await notify(job.chat_id, f"❌ Error de escritura en {job.info_hash}. Descarga cancelada.")
                            return
                    elif "Downloading" in line:
                        log.info(human(line))

        reader_task = asyncio.create_task(reader())
        rc: Optional[int] = None

        try:
            await asyncio.wait_for(proc.wait(), timeout=TIMEOUT_SECONDS)
            rc = proc.returncode
        except asyncio.TimeoutError:
            await kill_process_tree(proc)
            cleanup_incomplete(job)
            CURRENT.update({"line": "", "name": ""})
            await notify(job.chat_id, f"⏱ Timeout (30 min): {job.info_hash}. Descarga cancelada y limpiada.")
            log.warning("⏱ Timeout: %s", job.info_hash)
        finally:
            try:
                await asyncio.wait_for(reader_task, timeout=3)
            except asyncio.TimeoutError:
                reader_task.cancel()

        if rc is not None:
            if rc == 0:
                nm = CURRENT["name"] or job.info_hash
                await notify(job.chat_id, f"✅ Descarga completada: {nm}")
                await run_refresh_plex(refresh_script)
                log.info("✅ Completado: %s", nm)
            else:
                cleanup_incomplete(job)
                await notify(job.chat_id, f"❌ Error descargando {job.info_hash}. Carpeta limpiada.")
                log.error("❌ aria2c retornó código %s", rc)

        CURRENT.update({"hash": "", "line": "", "name": ""})
        QUEUE.task_done()

# ========================
# Telegram
# ========================
async def notify(chat_id: int, text: str, max_retries: int = 3):
    """Enviar notificación con reintentos automáticos"""
    for attempt in range(max_retries):
        try:
            await app_global.bot.send_message(chat_id, text)
            return  # Éxito, salir
        except (NetworkError, TimedOut) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Backoff exponencial: 1s, 2s, 4s
                log.warning("Error de red al enviar notificación (intento %d/%d), reintentando en %ds: %s", 
                           attempt + 1, max_retries, wait_time, e)
                await asyncio.sleep(wait_time)
            else:
                log.error("Error de red al enviar notificación después de %d intentos: %s", max_retries, e)
        except BadRequest as e:
            log.warning("Error de solicitud al enviar notificación: %s", e)
            return  # No reintentar errores de solicitud
        except Exception as e:
            log.error("Error inesperado al enviar notificación: %s", e)
            return  # No reintentar errores inesperados

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manejador global de errores para evitar crashes del bot"""
    error = context.error
    
    if isinstance(error, NetworkError):
        log.warning("Error de red de Telegram: %s", error)
        return
    
    if isinstance(error, TimedOut):
        log.warning("Timeout en Telegram: %s", error)
        return
    
    if isinstance(error, BadRequest):
        log.warning("Solicitud incorrecta a Telegram: %s", error)
        return
    
    # Para otros errores, intentar notificar al usuario si es posible
    if update and update.effective_chat:
        try:
            await context.bot.send_message(
                update.effective_chat.id,
                "❌ Ocurrió un error inesperado. Por favor, intenta de nuevo."
            )
        except Exception:
            pass  # No queremos crear un bucle de errores
    
    log.error("Error no manejado: %s", error, exc_info=True)

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Verificar que update.message existe
    if not update.message:
        log.error("update.message es None en cmd_start")
        return
    
    await update.message.reply_text(
        "👋 Hola!\n"
        "Comandos:\n"
        "• /add pelicula <HASH1> <HASH2> ... o URLs/magnet links\n"
        "• /add serie <HASH1> <HASH2> ... o URLs/magnet links\n"
        "• /status (progreso actual + cola)\n"
        "\n"
        "Ejemplos:\n"
        "• /add pelicula 063A8D1602B018CEF86F34FF540D69D29F46CBBA\n"
        "• /add serie https://yts.mx/torrent/download/063A8D1602B018CEF86F34FF540D69D29F46CBBA\n"
        "• /add pelicula magnet:?xt=urn:btih:D1AD4F4CCCC44E6227283BD334487E777EB88EDC\n"
    )

async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE, args_ns):
    # Verificar que update.message existe
    if not update.message:
        log.error("update.message es None en cmd_add")
        return
    
    # Seguridad opcional
    if args_ns.allowed_chat and update.effective_chat.id != args_ns.allowed_chat:
        return

    if len(context.args) < 2:
        await update.message.reply_text("Uso: /add pelicula|serie HASH1 HASH2 ... o URLs/magnet links")
        return

    kind = context.args[0].lower()
    if kind not in ("pelicula", "serie"):
        await update.message.reply_text("Tipo inválido. Usa: pelicula | serie")
        return

    base = args_ns.movie_dir if kind == "pelicula" else args_ns.series_dir
    Path(base).mkdir(parents=True, exist_ok=True)

    # Unir todos los argumentos en un solo texto para extraer hashes
    full_text = " ".join(context.args[1:])
    
    # Extraer hashes usando la nueva función
    extracted_hashes = extract_hashes_from_text(full_text)
    
    if not extracted_hashes:
        await update.message.reply_text("❗ No se encontraron hashes válidos en el mensaje.")
        return

    added = 0
    for h in extracted_hashes:
        key = f"{kind}:{h.upper()}"
        if key in ENQUEUED:
            continue
        ENQUEUED.add(key)
        work_dir = Path(base) / f"T_{h[:12].upper()}"
        await QUEUE.put(Job(
            chat_id=update.effective_chat.id,
            kind=kind,
            info_hash=h.upper(),
            base_dir=base,
            work_dir=work_dir,
        ))
        added += 1

    if added:
        await update.message.reply_text(f"🧾 Agregados {added} item(s) a la cola ({kind}).")
    else:
        await update.message.reply_text("Nada agregado (ya estaban en cola).")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE, args_ns):
    # Verificar que update.message existe
    if not update.message:
        log.error("update.message es None en cmd_status")
        return
    
    # Seguridad opcional
    if args_ns.allowed_chat and update.effective_chat.id != args_ns.allowed_chat:
        return

    lines = []
    if CURRENT["hash"]:
        name = CURRENT["name"] or CURRENT["hash"]
        prog = CURRENT["line"] or "(inicializando…)"
        lines.append(f"📥 Actual: {name}\n{prog}")
    else:
        lines.append("✅ No hay descarga activa.")

    qsize = QUEUE.qsize()
    if qsize:
        lines.append(f"🕒 En cola: {qsize} pendiente(s).")

    await update.message.reply_text("\n".join(lines))

async def test_telegram_connection(app: Application) -> bool:
    """Probar la conectividad con Telegram"""
    try:
        bot_info = await app.bot.get_me()
        log.info("Conexión a Telegram exitosa. Bot: @%s", bot_info.username)
        return True
    except (NetworkError, TimedOut) as e:
        log.error("Error de conectividad con Telegram: %s", e)
        return False
    except Exception as e:
        log.error("Error inesperado al conectar con Telegram: %s", e)
        return False

async def post_init(app: Application, args_ns, aria2c_bin: str):
    global app_global
    app_global = app
    
    # Probar conectividad con Telegram
    if not await test_telegram_connection(app):
        log.error("No se pudo conectar con Telegram. El bot puede no funcionar correctamente.")
    
    # Crear la tarea del worker usando asyncio.create_task en lugar de app.create_task
    # para evitar la advertencia PTBUserWarning
    worker_task = asyncio.create_task(worker(
        aria2c_bin=aria2c_bin,
        summary_interval=args_ns.summary_interval,
        nice=args_ns.nice,
        ionice_class=args_ns.ionice_class,
        ionice_prio=args_ns.ionice_prio,
        refresh_script=args_ns.refresh_plex,
    ))
    log.info("Worker iniciado correctamente")
    
    # Guardar referencia a la tarea para poder cancelarla al cerrar
    app._worker_task = worker_task
    
    log.info("Bot listo. Modo sin RPC, aria2c secuencial; quiet=%s", args_ns.quiet)

async def post_stop(app: Application):
    """Limpieza al detener la aplicación"""
    global worker_running
    worker_running = False
    
    # Cancelar la tarea del worker si existe
    if hasattr(app, '_worker_task'):
        app._worker_task.cancel()
        try:
            await app._worker_task
        except asyncio.CancelledError:
            pass
    
    log.info("Worker detenido correctamente")

def build_app(token: str, args_ns):
    application = (
        Application.builder()
        .token(token)
        .post_init(lambda app: post_init(app, args_ns, args_ns.aria2c))
        .post_stop(lambda app: post_stop(app))
        .read_timeout(30)  # Timeout de lectura de 30 segundos
        .write_timeout(30)  # Timeout de escritura de 30 segundos
        .connect_timeout(30)  # Timeout de conexión de 30 segundos
        .pool_timeout(30)  # Timeout del pool de conexiones
        .build()
    )
    
    # Agregar manejador de errores global
    application.add_error_handler(error_handler)
    
    # Agregar manejadores de comandos
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("add", lambda u, c: cmd_add(u, c, args_ns)))
    application.add_handler(CommandHandler("status", lambda u, c: cmd_status(u, c, args_ns)))
    
    return application

# ========================
# Main / CLI
# ========================
def main():
    parser = argparse.ArgumentParser(description="Bot de Telegram para aria2c (sin RPC), eficiente/silencioso.")
    parser.add_argument("--token", default=os.getenv("TELEGRAM_TOKEN"), help="Token del bot (o env TELEGRAM_TOKEN).")
    parser.add_argument("--movie-dir", default="/DATA/Media/USB/USB_Movies")
    parser.add_argument("--series-dir", default="/DATA/Media/USB/USB_Series")
    parser.add_argument("--aria2c", default="aria2c", help="Ruta a aria2c (si no está en PATH).")
    parser.add_argument("--refresh-plex", default="/home/panchocosil/scripts/refresh-plex.sh")
    parser.add_argument("--summary-interval", type=int, default=10, help="Segundos entre resúmenes de aria2c (más alto = menos I/O).")
    parser.add_argument("--nice", type=int, default=10, help="Niceness del proceso aria2c (>=0). None para desactivar.", nargs="?")
    parser.add_argument("--ionice-class", type=int, choices=[1,2,3], default=2, help="ionice class (1=RT,2=BestEffort,3=Idle).", nargs="?")
    parser.add_argument("--ionice-prio", type=int, choices=range(0,8), default=7, help="ionice priority (0..7).", nargs="?")
    parser.add_argument("--quiet", action="store_true", help="Silenciar consola (solo warnings/errores).")
    parser.add_argument("--log-file", default=None, help="Guardar logs en archivo (rotación).")
    parser.add_argument("--allowed-chat", type=int, default=None, help="Limitar control a este chat_id (opcional).")

    args = parser.parse_args()

    global log
    log = setup_logging(args.quiet, args.log_file)

    if not args.token:
        log.error("Falta --token o variable TELEGRAM_TOKEN")
        raise SystemExit(1)

    # Ajuste automático: si quiet y summary_interval < 10, súbelo para ahorrar CPU/I/O
    if args.quiet and args.summary_interval < 10:
        args.summary_interval = 10

    app = build_app(args.token, args)
    app.run_polling()

if __name__ == "__main__":
    main()
