import asyncio
import logging

logger = logging.getLogger(__name__)

class VoiceAssistant:
    """
    Handles text-to-speech synthesis using a persistent PowerShell process
    and an asynchronous queue for optimized, non-blocking performance.
    """
    def __init__(self):
        self.queue = asyncio.Queue()
        self.process = None
        self.worker_task = None

    async def _initialize(self):
        """
        Starts the persistent PowerShell process and the worker task.
        This is called lazily on the first speak request.
        """
        if self.process is not None:
            return

        logger.info("Initializing persistent PowerShell process for TTS...")
        # This PowerShell script loads the speech assembly once, then enters a loop
        # to read lines from stdin and speak them.
        ps_script = """
        Add-Type -AssemblyName System.Speech;
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        while ($line = [Console]::In.ReadLine()) {
            if ($line -eq 'EXIT_TTS') {
                break;
            }
            $synth.Speak($line);
        }
        """
        
        try:
            # Added -NoProfile and -ExecutionPolicy Bypass for robustness
            self.process = await asyncio.create_subprocess_exec(
                'powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send the script to the PowerShell process to execute
            self.process.stdin.write(ps_script.encode('utf-8'))
            await self.process.stdin.drain()
            
            # Start the background worker that will process the queue
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("TTS Worker and PowerShell process started successfully.")

        except Exception as e:
            stderr_output = ""
            if self.process and self.process.stderr:
                try:
                    # Attempt to read any error output from PowerShell's stderr
                    # Use wait_for with a timeout to prevent blocking indefinitely
                    stderr_bytes = await asyncio.wait_for(self.process.stderr.read(), timeout=1.0)
                    stderr_output = stderr_bytes.decode('utf-8', errors='ignore').strip()
                except asyncio.TimeoutError:
                    stderr_output = " (stderr read timed out)"
                except Exception as stderr_e:
                    stderr_output = f" (error reading stderr: {stderr_e})"

            logger.error(f"Failed to initialize PowerShell TTS process: {e}. PowerShell stderr: {stderr_output}")
            self.process = None

    async def _worker(self):
        """The background worker that pulls text from the queue and sends it to PowerShell."""
        while True:
            try:
                text = await self.queue.get()
                if text is None:  # Sentinel value to stop the worker
                    break
                
                if self.process and self.process.stdin and not self.process.stdin.is_closing():
                    # Sanitize text and add a newline to signal 'ReadLine' in PowerShell
                    sanitized_text = text.replace("'", "''")
                    logger.info(f"Sending to PowerShell TTS: {sanitized_text}")
                    self.process.stdin.write((sanitized_text + '\n').encode('utf-8'))
                    await self.process.stdin.drain()
                
                self.queue.task_done()
            except asyncio.CancelledError:
                logger.info("TTS worker task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in TTS worker: {e}")

    async def speak(self, text: str):
        """
        Adds text to the speech queue. The call returns immediately.
        """
        if not text:
            return
        
        # Lazily initialize the process and worker on the first call
        if self.process is None:
            await self._initialize()
        
        if self.process is None:
            logger.error("Cannot speak, TTS process is not running.")
            return

        await self.queue.put(text)

    async def shutdown(self):
        """Gracefully shuts down the worker and the PowerShell process."""
        logger.info("Shutting down TTS service...")
        if self.worker_task:
            # Send sentinel value to stop the worker, or just cancel it
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        if self.process and self.process.stdin and not self.process.stdin.is_closing():
            try:
                # Tell the PowerShell loop to exit
                self.process.stdin.write('EXIT_TTS\n'.encode('utf-8'))
                await self.process.stdin.drain()
            except (ConnectionResetError, BrokenPipeError) as e:
                logger.warning(f"Pipe to PowerShell already closed: {e}")

        if self.process:
            # Give PowerShell a moment to process the EXIT_TTS command
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("PowerShell process did not terminate gracefully, killing it.")
                self.process.kill()
            logger.info("PowerShell TTS process terminated.")
        
        logger.info("TTS service shut down complete.")
