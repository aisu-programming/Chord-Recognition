@ECHO OFF

SET ExecuteCommand=nvidia-smi
SET ExecutePeriod=1

SETLOCAL EnableDelayedExpansion

:loop

  cls

  echo !date! !time!
  echo 每 !ExecutePeriod! 秒執行一次，指令^: !ExecuteCommand!

  echo.

  %ExecuteCommand%
  
  timeout /t %ExecutePeriod% > nul

goto loop