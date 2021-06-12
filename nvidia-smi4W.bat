@ECHO OFF

SET ExecuteCommand=nvidia-smi
SET ExecutePeriod=1

SETLOCAL EnableDelayedExpansion

:loop

  cls

  echo !date! !time!
  echo Execute !ExecuteCommand! per !ExecutePeriod! second(s).

  %ExecuteCommand%
  
  timeout /t %ExecutePeriod% > nul

goto loop