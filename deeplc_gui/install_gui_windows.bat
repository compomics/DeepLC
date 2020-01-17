@echo off

set downloadurl=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
echo downloading miniconda . . .
powershell -Command "Invoke-WebRequest %downloadurl% -OutFile Miniconda3-latest-Windows-x86_64.exe"
echo download of miniconda complete and extracted to the directory.

echo installing miniconda . . .
pushd %~dp0
set script_dir=%CD%
start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%script_dir%\Miniconda3
echo installing miniconda complete . . .
echo installing environment in miniconda . . .
call Miniconda3/Scripts/activate.bat
call conda create --yes -n deeplc_gui python=3.7
call Miniconda3/Scripts/activate.bat deeplc_gui
call conda config --add channels defaults
call conda config --add channels bioconda
call conda config --add channels conda-forge
call conda install --yes deeplc
echo done installing environment in miniconda . . .
pause