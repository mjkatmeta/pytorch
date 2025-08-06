set ADDITIONAL_OPTIONS=""
set PYTHON_EXEC="python"

:: Free threaded python is like 3.14t, 3.13t, etc
if "%DESIRED_PYTHON:~-1%"=="t" (
    set ADDITIONAL_OPTIONS="Include_freethreaded=1"
    set PYTHON_EXEC="python%DESIRED_PYTHON%" %= @lint-ignore =%
)

if "%DESIRED_PYTHON:~0,4%"=="3.14" (
    echo Python version is set to 3.14 or 3.14t
    set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.14.0/python-3.14.0rc1-arm64.exe"
) else (
    echo Python version is set to %DESIRED_PYTHON%
    set "PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/%DESIRED_PYTHON%.0/python-%DESIRED_PYTHON%.0-amd64.exe" %= @lint-ignore =%
)

del python-amd64.exe
curl --retry 3 -kL "%PYTHON_INSTALLER_URL%" --output python-amd64.exe
if errorlevel 1 exit /b 1

start /wait "" python-amd64.exe /quiet InstallAllUsers=1 PrependPath=0 Include_test=0 %ADDITIONAL_OPTIONS% TargetDir=%CD%\Python
if errorlevel 1 exit /b 1

set "PATH=%CD%\Python\Scripts;%CD%\Python;%PATH%"
%PYTHON_EXEC% -m pip install --upgrade pip setuptools packaging wheel
if errorlevel 1 exit /b 1
