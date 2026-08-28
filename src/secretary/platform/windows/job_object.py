from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class NullJobObject:
    def add_process(self, pid: int) -> bool:
        return True

    def close(self) -> None:
        return None


if os.name == "nt":
    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [("ReadOperationCount", ctypes.c_ulonglong), ("WriteOperationCount", ctypes.c_ulonglong), ("OtherOperationCount", ctypes.c_ulonglong), ("ReadTransferCount", ctypes.c_ulonglong), ("WriteTransferCount", ctypes.c_ulonglong), ("OtherTransferCount", ctypes.c_ulonglong)]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimitInformation), ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    def _configure_job_apis(kernel32) -> None:
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

    class WindowsJobObject:
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

        def __init__(self) -> None:
            kernel32 = ctypes.windll.kernel32
            _configure_job_apis(kernel32)
            self._kernel32 = kernel32
            self._handle = kernel32.CreateJobObjectW(None, None)
            if not self._handle:
                raise OSError("CreateJobObjectW failed")
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            ok = kernel32.SetInformationJobObject(self._handle, self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info))
            if not ok:
                kernel32.CloseHandle(self._handle)
                self._handle = None
                raise OSError("SetInformationJobObject failed")

        def add_process(self, pid: int) -> bool:
            if not self._handle:
                return False
            process = self._kernel32.OpenProcess(0x0200 | 0x0400, False, pid)
            if not process:
                return False
            try:
                return bool(self._kernel32.AssignProcessToJobObject(self._handle, process))
            finally:
                self._kernel32.CloseHandle(process)

        def close(self) -> None:
            if self._handle:
                self._kernel32.CloseHandle(self._handle)
                self._handle = None

    class _ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    def _configure_process_apis(kernel32) -> None:
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD

    def owned_descendant_pids(root_pid: int) -> set[int]:
        """Return only descendants of a PID that Secretary explicitly owns."""
        kernel32 = ctypes.windll.kernel32
        _configure_job_apis(kernel32)
        _configure_process_apis(kernel32)
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot in (0, -1):
            return set()
        rows: list[tuple[int, int]] = []
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                while True:
                    rows.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
                    if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                        break
        finally:
            kernel32.CloseHandle(snapshot)
        children: dict[int, list[int]] = {}
        for pid, parent in rows:
            children.setdefault(parent, []).append(pid)
        descendants: set[int] = set()
        pending = list(children.get(root_pid, []))
        while pending:
            pid = pending.pop()
            if pid in descendants or pid == root_pid:
                continue
            descendants.add(pid)
            pending.extend(children.get(pid, []))
        return descendants

    def terminate_owned_pid(pid: int) -> bool:
        """Terminate one already-verified owned descendant PID; never search by name."""
        kernel32 = ctypes.windll.kernel32
        _configure_job_apis(kernel32)
        _configure_process_apis(kernel32)
        handle = kernel32.OpenProcess(0x0001 | 0x00100000, False, pid)
        if not handle:
            return False
        try:
            if not kernel32.TerminateProcess(handle, 0):
                return False
            kernel32.WaitForSingleObject(handle, 5000)
            return True
        finally:
            kernel32.CloseHandle(handle)
else:
    class WindowsJobObject(NullJobObject):
        pass

    def owned_descendant_pids(root_pid: int) -> set[int]:
        return set()

    def terminate_owned_pid(pid: int) -> bool:
        return False
