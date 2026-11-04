"""Ascend driver admin-compatibility filter for a private AArch64 userns.

Syscall numbers: local /usr/include/asm-generic/unistd.h. The driver tests
capability bits numerically. They remain in a child userns; this filter prevents
using them to replace the read-only mounts, namespaces or device-node boundary.
This is an experiment harness boundary, not a device ioctl security monitor.
"""
from pathlib import Path
import platform,struct,sys

POLICY='ascend-admin-userns-seccomp-v1'
# 来源: /usr/include/asm-generic/unistd.h:103 (__NR_*表逐项核对；AArch64固定UAPI号)。
DENIED={
    'mount':40,'umount2':39,'pivot_root':41,'move_mount':429,'open_tree':428,
    'fsopen':430,'fsconfig':431,'fsmount':432,'fspick':433,'mount_setattr':442,
    'unshare':97,'setns':268,'mknodat':33,'ptrace':117,
    'process_vm_readv':270,'process_vm_writev':271,'process_madvise':440,
    'kexec_load':104,'kexec_file_load':294,'reboot':142,'bpf':280,
    'perf_event_open':241,'syslog':116,'init_module':105,'finit_module':273,
    'delete_module':106,'swapon':224,'swapoff':225,'acct':89,'quotactl':60,
    'fanotify_init':262,'io_uring_setup':425,'open_by_handle_at':265,
    'name_to_handle_at':264,'keyctl':219,'add_key':217,'request_key':218,'chroot':51}
# NEWNS/NEWCGROUP/NEWUTS/NEWIPC/NEWUSER/NEWPID/NEWNET; ordinary threads allowed.
NAMESPACE_FLAGS=0x00020000|0x02000000|0x04000000|0x08000000|0x10000000|0x20000000|0x40000000


def filter_bytes():
    if platform.machine()!='aarch64' or sys.byteorder!='little':raise ValueError('Ascend sandbox filter requires little-endian AArch64')
    # Classic BPF seccomp_data: nr@0, arch@4, args[0]@16.
    code=[(0x20,0,0,4),(0x15,1,0,0xc00000b7),(0x06,0,0,0x80000000),(0x20,0,0,0)]
    for nr in sorted(DENIED.values()):code.extend(((0x15,0,1,nr),(0x06,0,0,0x00050001)))
    # clone3 ENOSYS makes libc fall back to inspectable clone arguments.
    code.extend(((0x15,0,1,435),(0x06,0,0,0x00050026),
                 (0x15,0,3,220),(0x20,0,0,16),(0x45,0,1,NAMESPACE_FLAGS),
                 (0x06,0,0,0x00050001),(0x06,0,0,0x7fff0000)))
    return b''.join(struct.pack('<HBBI',*instruction) for instruction in code)


def validate_filter(identity):
    if identity is None:raise ValueError('Ascend compatibility mode requires a frozen seccomp filter')
    identity.verify()
    if identity.role!='tool' or Path(identity.path).read_bytes()!=filter_bytes():raise ValueError('Ascend seccomp policy differs')
    return True
