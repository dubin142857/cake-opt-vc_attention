"""Real CPU namespace checks for the driver compatibility policy; no NPU nodes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,os,platform,struct,subprocess,sys,tempfile,unittest
from .ascend_sandbox import DENIED,NAMESPACE_FLAGS,filter_bytes,validate_filter
from .runner import AscendLease,Stage,identify,_sandbox_argv
from .runner_test import fixture,sandbox_path


def interpret(program,nr,flags=0,arch=0xc00000b7):
    instructions=list(struct.iter_unpack('<HBBI',program));pc=0;acc=0
    while pc<len(instructions):
        op,jt,jf,k=instructions[pc];pc+=1
        if op==0x20:acc={0:nr,4:arch,16:flags&0xffffffff}[k]
        elif op==0x15:pc+=jt if acc==k else jf
        elif op==0x45:pc+=jt if acc&k else jf
        elif op==0x06:return k
        else:raise AssertionError('Unknown BPF operation')
    raise AssertionError('No result')


@unittest.skipUnless(platform.machine()=='aarch64','fixed Ascend AArch64 policy')
class AscendSandboxTests(unittest.TestCase):
    def test_filter_decisions_arch_namespace_and_regular_threads(self):
        data=filter_bytes()
        for nr in DENIED.values():self.assertEqual(interpret(data,nr),0x00050001)
        self.assertEqual(interpret(data,435),0x00050026)
        for bit in (0x20000,0x2000000,0x4000000,0x8000000,0x10000000,0x20000000,0x40000000):
            self.assertEqual(interpret(data,220,bit|17),0x00050001)
        self.assertEqual(interpret(data,220,0x00000100|0x00010000),0x7fff0000)
        self.assertEqual(interpret(data,63),0x7fff0000)
        self.assertEqual(interpret(data,63,arch=0xc000003e),0x80000000)

    def test_frozen_policy_missing_or_self_consistently_rehashed_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'filter.bpf';p.write_bytes(filter_bytes());self.assertTrue(validate_filter(identify(p,'tool')))
            p.write_bytes(p.read_bytes()[:-1]+b'X')
            with self.assertRaisesRegex(ValueError,'policy differs'):validate_filter(identify(p,'tool'))
            with self.assertRaises(ValueError):validate_filter(None)

    @unittest.skipUnless(sandbox_path().is_file(),'bubblewrap unavailable; mechanism unknown')
    def test_real_admin_compatibility_boundary_without_device_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=fixture(root);work=root/'payload';work.mkdir();witness=root/'outside';witness.write_text('keep')
            filterpath=root/'sandbox.bpf';filterpath.write_bytes(filter_bytes());profile=SimpleNamespace(nodes=(),seccomp=identify(filterpath,'tool'))
            plan=replace(plan,device=AscendLease(identify(witness,'input')),inputs=plan.inputs+(identify(witness,'harness'),))
            code="""import ctypes,errno,json,os,stat,subprocess,sys,threading
libc=ctypes.CDLL(None,use_errno=True)
results={}
def check(name,nr,*args):
 ctypes.set_errno(0);rc=libc.syscall(nr,*args)
 if name=='clone' and rc==0:os._exit(0)
 results[name]={'rc':rc,'errno':ctypes.get_errno()}
check('unshare',97,0x10000000)
check('mknod',33,-100,ctypes.c_char_p(b'illegal-node'),stat.S_IFCHR|0o600,os.makedev(1,3))
check('mount',40,ctypes.c_char_p(b'tmpfs'),ctypes.c_char_p(b'.'),ctypes.c_char_p(b'tmpfs'),0,0)
check('clone',220,0x10000000|17,0,0,0,0)
try:
 open(sys.argv[1],'w').write('changed');results['outside_write']=0
except OSError as error:results['outside_write']=error.errno
results['npu_node_present']=any(name.startswith('davinci') for name in os.listdir('/dev'))
results['userns']=os.readlink('/proc/self/ns/user')
status=open('/proc/self/status').read();results['no_new_privs']=next(s.split(':')[1].strip() for s in status.splitlines() if s.startswith('NoNewPrivs:'))
seen=[];thread=threading.Thread(target=lambda:seen.append(1));thread.start();thread.join();results['thread']=seen
results['subprocess_rc']=subprocess.run([sys.executable,'-c','pass']).returncode
open('own-output','w').write('allowed');print(json.dumps(results),flush=True)
"""
            stage=Stage('boundary','guard',(sys.executable,'-c',code,str(witness)),'unused.json')
            fd=os.open(filterpath,os.O_RDONLY)
            try:
                with patch('core.opgen.ascend.load_profile',return_value=profile):
                    with self.assertRaisesRegex(ValueError,'seccomp fd'):_sandbox_argv(plan,work,stage,'',0,'test')
                    command=_sandbox_argv(plan,work,stage,'',0,'test',fd)
                done=subprocess.run(command,capture_output=True,text=True,env={'PATH':'/usr/bin:/bin'},pass_fds=(fd,),timeout=30)
            finally:os.close(fd)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr);result=json.loads(done.stdout)
            for name in ('unshare','mknod','mount','clone'):self.assertEqual(result[name],{'rc':-1,'errno':1})
            self.assertEqual(result['outside_write'],30);self.assertFalse(result['npu_node_present'])
            self.assertNotEqual(result['userns'],os.readlink('/proc/self/ns/user'));self.assertEqual(result['no_new_privs'],'1')
            self.assertEqual(result['thread'],[1]);self.assertEqual(result['subprocess_rc'],0)
            self.assertEqual(witness.read_text(),'keep');self.assertEqual((work/'own-output').read_text(),'allowed')

if __name__=='__main__':unittest.main()
