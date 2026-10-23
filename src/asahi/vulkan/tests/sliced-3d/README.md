# Apple9 sliced 3D storage views

`VK_EXT_image_sliced_view_of_3d` is enabled on G16 only. Storage coordinates
are relative to `sliceOffset` at the view's single selected mip, and storage
queries and bounds use `sliceCount` (including `VK_REMAINING_3D_SLICES_EXT`).
Sampled and other descriptors ignore the slicing structure, as required by
[VkImageViewSlicedCreateInfoEXT](https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewSlicedCreateInfoEXT.html).

## Descriptor and shader audit

AIL stores Z slices as XY mip trees with a constant physical slice stride.
The storage texture/PBE bases can therefore advance by `sliceOffset *
layer_stride_B` while retaining the selected mip and the physical stride.
The offset is in selected-mip Z slices; it must not be shifted by the mip.

The native texture must retain the original depth. In a Z-major image,
`ail_make_miptree()` includes depth when determining the full mip-tree length,
which the native descriptor infers from its geometry. Reducing native depth
could shorten the inferred stride. Apple9 already keeps its native texture
entry separate from the legacy texture prefix used for software size queries.
Only that prefix gets `sliceCount << first_level`, because `libagx_txs` minifies
its depth. Repacking clears the overlapping linear-stride field. This requires
neither descriptor growth nor new shader operations.

| Path | Representation and bounds |
| --- | --- |
| View/mip normalization | `vk_image_view_init` supplies the selected mip's slice offset and count, resolving remaining slices. |
| Typed storage loads and all 3D stores | Existing global-memory lowering checks the query prefix, then uses PBE base, selected-level offset, tile shape and explicit physical slice stride. |
| Formatless read-only loads | Existing software bounds check precedes a native fetch using original geometry and shifted base. Read/write or volatile paths retain the existing global-memory route. |
| Image atomics | Existing robustness lowering queries the storage texture prefix; global address lowering uses the rebased PBE with its original stride. |
| Uncompressed aliases of block-compressed images | Their single-level rebase cannot encode the original Z stride in a native 3D entry. Disable native storage reads for these views and use the existing explicit-stride fallback. The query prefix uses the alias's actual `first_level` (zero). |
| Sampled/input/render/meta descriptors | Slice offset/count are not applied. Host tests compare unsliced and sliced-view bytes for these uses. |
| Descriptor sets, copies, templates, push descriptors, mutable bindings | All writes share `hk_pack_storage_image`; copies carry the complete 128-byte record. Host tests exercise these entry points. |
| Descriptor buffers | `vkGetDescriptorEXT` uses the same packer and existing 128-byte ABI. Nonuniform indexing changes the record address, not its interpretation. |
| Capture/replay | 3D views allocate no native PBE table entries and add no capture state. Recreated image/view addresses follow the existing replay path. |
| Sparse binding | The rebased image VA and physical stride remain sufficient. 3D sparse **residency** is not advertised and is rejected by format queries; there is no 3D residency table to adjust. |
| Compression/tiling | Storage usage excludes hardware compression. Linear and DRM-modifier 3D images are rejected by format queries. Block-compressed aliases above are distinct from hardware compression. |

Apple8 remains disabled: it uses hardware storage descriptors directly and
does not have Apple9's separation between native geometry and query bounds.

## Host checks

`hk-sliced-3d` runs actual image-view construction, descriptor entry points and
NIR query/address lowering against host-backed descriptors. Its 321 cases cover
odd extents, every mip, explicit/remaining counts, first/last slices, Z-major
volumes, depth-one high mips, mutable formats, sparse-binding flags, capture
flags, block-texel aliases and null descriptors. It checks first and last
addresses against AIL and independently checks native geometry bits.
`hk-image-limits` and `hk-native-image-store` cover neighboring format and
storage eligibility contracts. Both assertion and release builds are required.
These checks do not execute GPU instructions or certify native sampling.

## Device coverage

`probe.c` plus five variants of `probe.comp` check 25 combinations of sliced
views and typed loads, formatless loads, typed stores, formatless stores and
32-bit atomics. They check `imageSize`, negative and one-past-end Z coordinates,
first/last accessible slices, nonzero mips, remaining slices and Z-major mip
trees. Read shaders sample the **same view** through a sampled descriptor and
check the full selected mip. Write/atomic cases read back every pixel to ensure
that prefix/suffix slices and untouched XY positions retain their seed values.
The formatless read shader has no image writes, so it exercises native reads.
The probe enables `robustImageAccess2`, prints renderer/driver identity and
rejects non-M4 renderers. It returns 77 when the extension is unavailable.

The exact CTS lists are pinned in `cts-source.json`:

- `sliced.txt`: 156 `dEQP-VK.pipeline.monolithic.sliced_view_of_3d_image.*` cases.
  This checkout has **no** `dEQP-VK.image.sliced_view_of_3d.*` group.
- `storage.txt`: 300 neighboring 3D cases: 154 typed/formatless load-store,
  78 load-store LOD, 24 image-size, 8 atomic, 32 robustness and 4 block-texel
  alias cases. These include descriptor templates and fragment/compute paths.

The packaged kit includes baseline/candidate libraries, both probe binaries,
SPIR-V, these lists and `run.py`. On the M4, using the installed CI configuration:

```sh
python3 /path/to/kit/run.py --arm candidate --suite all --output /path/to/new-results
python3 /path/to/kit/run.py --arm baseline --suite storage --output /path/to/new-control-results
```

`--prepare-only` writes commands, ICD and suite files without executing them.
`--config` selects an alternative installed Mesa CI config. The runner verifies
the packaged library hash, selects that library through an absolute ICD, leaves
compression at its normal setting, uses one CTS worker and existing CI
expectations, and never installs Mesa or restarts the desktop. It stops on a
probe failure or a CTS runner failure. Inspect raw CTS results separately for
passes, failures, skips, expected failures and flakes. A runner success is not
equivalent to every test passing. Candidate feature absence is a failure;
baseline feature absence is a skip. Device results are pending manager execution.
