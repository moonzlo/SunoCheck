; Inno Setup script for SunoCheck-Setup.exe, packing the PyInstaller folder build dist\SunoCheck.
; Build: ISCC /DAppVersion=X.Y.Z packaging\sunocheck.iss  ->  dist\SunoCheck-Setup.exe

#ifndef AppVersion
  #error Pass the version: ISCC /DAppVersion=X.Y.Z packaging\sunocheck.iss
#endif

[Setup]
AppId={{6F3B2A1E-5C4D-4E8F-9A7B-2D1C0E9F8A6B}
AppName=SunoCheck
AppVersion={#AppVersion}
AppVerName=SunoCheck {#AppVersion}
AppPublisher=Mikhail (moonzlo)
AppPublisherURL=https://github.com/moonzlo/SunoCheck
AppSupportURL=https://github.com/moonzlo/SunoCheck/issues
AppUpdatesURL=https://github.com/moonzlo/SunoCheck/releases
VersionInfoVersion={#AppVersion}
; Per-user install without admin rights: the app writes its data folder next to SunoCheck.exe.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\SunoCheck
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; The running app holds this mutex (gui.INSTALLER_MUTEX); setup and uninstall ask to close it first.
AppMutex=SunoCheckRunning
OutputDir=..\dist
OutputBaseFilename=SunoCheck-Setup
SetupIconFile=sunocheck.ico
UninstallDisplayIcon={app}\SunoCheck.exe
UninstallDisplayName=SunoCheck
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An upgrade replaces the bundled libraries wholesale, so files of the previous version never mix with the new ones.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\SunoCheck\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\SunoCheck"; Filename: "{app}\SunoCheck.exe"
Name: "{autodesktop}\SunoCheck"; Filename: "{app}\SunoCheck.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SunoCheck.exe"; Description: "{cm:LaunchProgram,SunoCheck}"; Flags: nowait postinstall skipifsilent

[Code]
// The app writes its own autostart value (Settings -> Start with Windows); a removed app must not stay in Run.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RegDeleteValue(HKEY_CURRENT_USER, 'Software\Microsoft\Windows\CurrentVersion\Run', 'SunoCheck');
end;
