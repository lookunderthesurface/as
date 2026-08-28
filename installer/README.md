# Optional packaging

The MVP is intentionally not installing system-level installer tools. A future
release build can use PyInstaller (or Nuitka) in a Windows GitHub Actions runner,
then feed the portable directory into Inno Setup or NSIS to produce:

- `AmbientSecretary-x.y.z-x64-setup.exe`
- `AmbientSecretary-x.y.z-portable.zip`

The package must remain a normal user-space tray application and must not create a
Windows Service or require administrator privileges.

