$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\..\.venv\Scripts\python.exe" -m unittest discover -s "$PSScriptRoot\..\tests" -v

