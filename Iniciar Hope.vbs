' Abre Hope sin ventana de consola (doble clic).
' La primera vez muestra la instalacion en una ventana visible; despues arranca oculta.
' Para cerrarla: boton «Cerrar la aplicacion» en la pagina. El entrenamiento sigue igual.
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = dir
If fso.FileExists(dir & "\.venv\Scripts\pythonw.exe") Then
  sh.Run """" & dir & "\.venv\Scripts\pythonw.exe"" -m hope.app", 0, False
Else
  sh.Run "cmd /c """ & dir & "\Hope.bat""", 1, False
End If
