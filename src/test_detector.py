#!/usr/bin/env python3
"""Pure-C decoder tests; no photos or ML files are required."""
import ctypes as ct
import pathlib
import subprocess
import tempfile
root=pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as temp:
    libpath=pathlib.Path(temp)/'libdetector.so'
    subprocess.run(['clang','-shared','-fPIC','-O2','-Wall','-Wextra',str(root/'PGDetector.c'),'-o',str(libpath)],check=True)
    lib=ct.CDLL(str(libpath))
    class Box(ct.Structure):
        _fields_=[('x',ct.c_double),('y',ct.c_double),('width',ct.c_double),('height',ct.c_double),('score',ct.c_double),('class_id',ct.c_int)]
    Reader=ct.CFUNCTYPE(ct.c_double,ct.c_void_p,ct.c_int,ct.c_int)
    arr=(Box*40)()
    vals=[[0.0]*4 for _ in range(22)]
    def add(i,box,k,score):
        for c,v in enumerate(box):vals[c][i]=v
        vals[k+4][i]=score
    add(0,(160,80,80,80),3,.94)
    add(1,(164,80,82,80),3,.88)
    add(2,(130,100,55,55),1,.98)
    add(3,(32,280,20,20),14,.86)
    @Reader
    def raw(ctx,c,p):return vals[c][p]
    parse=lib.pg_decode_yolo
    parse.argtypes=[Reader,ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.c_double,ct.c_double,ct.POINTER(Box),ct.c_int]
    parse.restype=ct.c_int
    assert parse(raw,None,22,4,320,320,.55,arr,40)==2
    assert {arr[i].class_id for i in range(2)}=={3,14}
    assert abs(arr[0].x-.375)<1e-8 and abs(arr[0].y-.625)<1e-8
    assert parse(raw,None,21,4,320,320,.55,arr,40)==0

    # NMS Core ML output: coordinates[N,4], confidence[N,80].
    coords=[[.5,.25,.25,.25],[.5,.5,.2,.2],[.1,.9,.1,.1]]
    confidence=[[0.0]*80 for _ in range(3)]
    confidence[0][3]=.91   # breast: redact
    confidence[1][1]=.95   # face: leave alone
    confidence[2][14]=.84  # male genitalia: redact
    @Reader
    def xy(ctx,c,p):return coords[p][c]
    @Reader
    def sc(ctx,c,p):return confidence[p][c]
    nms=lib.pg_decode_nms
    nms.argtypes=[Reader,ct.c_void_p,Reader,ct.c_void_p,ct.c_int,ct.c_int,ct.c_double,ct.POINTER(Box),ct.c_int]
    nms.restype=ct.c_int
    n=nms(xy,None,sc,None,80,3,.55,arr,40)
    assert n==2 and arr[0].class_id==3 and arr[1].class_id==14
    assert abs(arr[0].x-.375)<1e-8 and abs(arr[0].y-.625)<1e-8
    assert nms(xy,None,sc,None,4,3,.55,arr,40)==0
    assert nms(xy,None,sc,None,80,3,.95,arr,40)==0
    # Out-of-range boxes are clamped to the image; tiny/invalid ones ignored.
    coords[0]=[-.2,.2,.8,.4]
    n=nms(xy,None,sc,None,80,3,.55,arr,40)
    assert n==2 and arr[0].x==0 and 0<arr[0].width<=1
    print('PASS: raw YOLO, NMS pairs, score thresholds, exposed-only labels, NMS, bounding-box clamping')
