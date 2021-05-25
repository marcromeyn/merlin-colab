from subprocess import call, check_output
from datetime import datetime, timedelta
from subprocess import Popen, PIPE
import shlex
import sys, os, shutil

from IPython import get_ipython

from merlin_colab.conda import install_miniconda


def _run_command(cmd):
    with Popen(shlex.split(cmd), stdout=PIPE, bufsize=1, universal_newlines=True) as p:
        while True:
            line = p.stdout.readline()
            if not line:
                break
            print(line)    
        exit_code = p.poll()
    return exit_code


def install_merlin(rapids_version="0.19", cuda_version="11.0", py_version="3.7", nvtabular_version="0.5.1"):
    t0 = datetime.now()
    install_miniconda(restart_kernel=False)
    print("📦 Installing Rapids...")
    _run_command(f"conda install -c rapidsai -c nvidia -c conda-forge rapids={rapids_version} python={py_version} cudatoolkit={cuda_version} -y")

    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

    for so in ['cudf', 'rmm', 'nccl', 'cuml', 'cugraph', 'xgboost', 'cuspatial']:
        fn = 'lib'+so+'.so'
        source_fn = '/usr/local/lib/'+fn
        dest_fn = '/usr/lib/'+fn
        if os.path.exists(source_fn):
            print(f'Copying {source_fn} to {dest_fn}')
            shutil.copyfile(source_fn, dest_fn)
    
    print("📦 Installing Merlin...")
    call(["pip", "install", f"nvtabular=={nvtabular_version}"])

    taken = timedelta(seconds=round((datetime.now() - t0).total_seconds(), 0))
    print(f"⏲ Done in {taken}")

    print("🔁 Restarting kernel...")
    get_ipython().kernel.do_shutdown(True)
