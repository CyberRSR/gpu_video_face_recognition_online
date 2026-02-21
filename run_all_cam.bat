@echo off
echo Launching scripts...

start py recgn.py --queue-mode --queue-poll-seconds 20
timeout /t 1 /nobreak >nul
start py "svcam_recordings_queue.py"
timeout /t 1 /nobreak >nul


echo Done.
