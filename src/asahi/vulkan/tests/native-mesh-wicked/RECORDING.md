# Wicked mesh component recording

`host-record.py` compiles the four unchanged corpus mesh modules with the
unchanged `objectAS` task module and small authored fragment consumers of every
per-primitive output. The source manifest pins each original module. The ARM
driver records through the non-executing DRM shim: native routing, task
predecessors, the expanded V188/P124 allocation, and the stream are checked.
Readback must remain CPU poison. This is not GPU execution or an engine run.

`record.c` has dummy descriptor contents and requires the host recording
environment. Never use it as a device workload. The diagnostic consumers are
explicitly separate from the engine's fragment modules. The earlier prepass
adapter at commit `038f348cab6` documented an original fragment interface
mismatch (missing `PerPrimitiveEXT`); this harness does not rewrite that module
or claim that its unchanged original pair has become valid.

The compiler uses ordinary flat varyings on a private provoking vertex for
each primitive. Dynamic first/last provoking mode comes from the normal draw
root. The original per-vertex components, including clip mirrors, are copied
unchanged, and only the provoking corner is reindexed. The declared V+P bound
must fit 256 vertices and the ordinary local allocation limit. This is a
research route and does not establish the native primitive-record ABI.
