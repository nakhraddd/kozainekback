import asyncio
import logging
import os
import tempfile
import sys
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class VoiceAssistant:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.process = None
        self.worker_task = None
        self.current_language = "ru-RU" 
        self.language_map = {
            "ENGLISH": "en-US",
            "RUSSIAN": "ru-RU",
            "KAZAKH": "kk-KZ"
        }
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.script_path = None

    def set_language(self, lang_name: str):
        """Sets the TTS language based on the provided name (ENGLISH, RUSSIAN, KAZAKH)."""
        if lang_name in self.language_map:
            new_lang_code = self.language_map[lang_name]
            if self.current_language != new_lang_code:
                self.current_language = new_lang_code
                self._clear_queue() # Clear pending messages in old language
                logger.info(f"Language set to {lang_name} ({self.current_language}) - Queue cleared.")
        else:
            logger.warning(f"Unknown language: {lang_name}, keeping current: {self.current_language}")

    def _clear_queue(self):
        """Clears all pending messages from the queue."""
        size = self.queue.qsize()
        for _ in range(size):
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
        if size > 0:
            logger.info(f"Cleared {size} pending TTS messages.")

    def _start_process_sync(self):
        """Starts the persistent PowerShell TTS process synchronously (for threading)."""
        if self.process and self.process.poll() is None:
            return

        logger.info("Initializing persistent PowerShell process for TTS (Synchronous)...")
        
        ps_script = """
        Add-Type -AssemblyName System.Speech;
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        
        while ($true) {
            $line = [Console]::In.ReadLine();
            if ($line -eq $null -or $line -eq 'EXIT_TTS') { break }
            
            if ($line.Contains('|')) {
                $parts = $line.Split('|', 2)
                $lang = $parts[0]
                $text = $parts[1]
                
                $voice = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq $lang } | Select-Object -First 1
                
                if ($voice) {
                    $synth.SelectVoice($voice.VoiceInfo.Name)
                }
                
                $synth.SpeakAsync($text) | Out-Null
            }
        }
        """
        
        try:
            if not self.script_path or not os.path.exists(self.script_path):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(ps_script)
                    self.script_path = tmp_file.name
                logger.info(f"Created temp TTS script at: {self.script_path}")

            powershell_exe = "powershell"
            if shutil.which("powershell") is None:
                sys_path = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
                if os.path.exists(sys_path):
                    powershell_exe = sys_path
                else:
                    logger.error("PowerShell executable not found.")
                    return

            self.process = subprocess.Popen(
                [powershell_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", self.script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False 
            )
            
            logger.info(f"PowerShell TTS process started (PID: {self.process.pid}).")
            
        except Exception as e:
            logger.error(f"Failed to initialize PowerShell TTS: {type(e).__name__}: {e}")
            self.process = None

    async def _worker(self):
        """Background worker to consume the speech queue."""
        logger.info("TTS Worker started.")
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                text = await self.queue.get()
                if text is None: break 
                
                await loop.run_in_executor(self.executor, self._start_process_sync)
                
                if self.process and self.process.stdin:
                    clean_text = text.replace('|', '').replace('\n', ' ').strip()
                    # Use current_language at the moment of sending to process
                    payload = f"{self.current_language}|{clean_text}\n"
                    
                    try:
                        def write_payload():
                            self.process.stdin.write(payload.encode('utf-8'))
                            self.process.stdin.flush()
                        
                        await loop.run_in_executor(self.executor, write_payload)
                        
                    except (OSError, BrokenPipeError) as e:
                        logger.warning(f"TTS process pipe error: {e}. Restarting...")
                        if self.process:
                            self.process.kill()
                        self.process = None 
                        logger.error(f"Failed to speak: {clean_text}")

                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TTS worker: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def speak(self, text: str):
        """Queues text for speech synthesis."""
        if not text: return
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
        await self.queue.put(text)

    async def shutdown(self):
        """Shuts down the TTS service."""
        logger.info("Shutting down TTS service...")
        if self.worker_task:
            self.worker_task.cancel()
            try: await self.worker_task
            except: pass
        
        if self.process:
            try:
                self.process.terminate()
            except: pass
            
        if self.script_path and os.path.exists(self.script_path):
            try: os.remove(self.script_path)
            except: pass
            
        self.executor.shutdown(wait=False)
        logger.info("TTS service shut down complete.")
