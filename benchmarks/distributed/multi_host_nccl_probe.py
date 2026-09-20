"""Bounded-launcher two-host NCCL probe; use an external process timeout.

TCP bootstraps communicator IDs only. NCCL establishes its own data connections.
Run only between trusted private peers; do not expose the pickle control plane.
"""
import argparse
import json
from pathlib import Path
import socket
import time

import tiga as tg
from tiga.distributed import TCPTransport, NCCLTransport, nccl_unique_id, DeviceBufferSlice


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, choices=(0,1), required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=29650)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result=dict(rank=a.rank, hostname=socket.gethostname(), status='connecting-control', correct=False)
    def record():
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    record()
    connect=TCPTransport.host if a.rank==0 else TCPTransport.join
    control=connect(a.rank,2,host=a.host,port=a.port,timeout=20)
    try:
        if a.rank==0:
            identifier=nccl_unique_id()
            control.send(1,identifier)
        else:
            identifier=control.receive(0)
        result['status']='initializing-nccl'
        record()
        with_scope=NCCLTransport(a.rank,2,identifier,device='cuda:0')
        try:
            result['nccl_version']=with_scope.version
            stream=tg.runtime.Stream('cuda:0')
            source=tg.runtime.Buffer(1<<20,device='cuda:0')
            destination=tg.runtime.Buffer(1<<20,device='cuda:0')
            source.write(bytes([a.rank+17])*(1<<20))
            start=time.perf_counter()
            with_scope.exchange_device(((1-a.rank,DeviceBufferSlice(source,0,1<<20)),),
                                       ((1-a.rank,DeviceBufferSlice(destination,0,1<<20)),),stream=stream).wait()
            result['seconds']=time.perf_counter()-start
            result['correct']=destination.read()==bytes([18-a.rank])*(1<<20)
            result['status']='passed' if result['correct'] else 'incorrect'
            record()
            stream.close(); source.close(); destination.close()
        finally:
            with_scope.close()
    except Exception as error:
        result.update(status='failed',error=str(error))
        record()
        raise
    finally:
        control.close()


if __name__=='__main__':
    main()
