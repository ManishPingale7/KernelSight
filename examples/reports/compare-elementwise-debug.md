# KernelSight comparison: elementwise.cu → elementwise.cu -G

**elementwise.cu**
- Input: `elementwise.cu` (cuda), sha256 `434ee685819011ec…`
- NVRTC 12.9 options: `-arch=compute_89`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`

**elementwise.cu -G**
- Input: `elementwise.cu` (cuda), sha256 `434ee685819011ec…`
- NVRTC 12.9 options: `-arch=compute_89 -G`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`

- Module `target`: `sm_89` → `sm_89, debug`
- Functions: 3 changed, 0 with identical PTX but different ptxas results, 0 identical, 0 only in elementwise.cu, 1 only in elementwise.cu -G

## `vector_add`: changed

| Metric | elementwise.cu | elementwise.cu -G | Δ |
|---|---|---|---|
| instructions | 22 | 28 | +6 |
| instructions in loops | 0 | 0 | 0 |
| basic blocks | 3 | 7 | +4 |
| loops | 0 | 0 | 0 |
| virtual registers | 23 | 27 | +4 |
| peak live slots | 10 | 9 | -1 |
| ptxas registers | 12 | 17 | +5 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 0 | 0 | 0 |

**Changed instructions** (count per mnemonic)

| Mnemonic | elementwise.cu | elementwise.cu -G |
|---|---|---|
| `add.s32` | 0 | 1 |
| `bra.uni` | 0 | 2 |
| `cvt.s64.s32` | 0 | 3 |
| `cvta.to.global.u64` | 3 | 0 |
| `ld.f32` | 0 | 2 |
| `ld.global.f32` | 2 | 0 |
| `mad.lo.s32` | 1 | 0 |
| `mul.lo.s32` | 0 | 1 |
| `mul.wide.s32` | 1 | 0 |
| `not.pred` | 0 | 1 |
| `setp.ge.s32` | 1 | 0 |
| `setp.lt.s32` | 0 | 1 |
| `shl.b64` | 0 | 3 |
| `st.f32` | 0 | 1 |
| `st.global.f32` | 1 | 0 |

<details><summary>Normalised instruction diff</summary>

```diff
--- elementwise.cu:vector_add
+++ elementwise.cu -G:vector_add
@@ -1,23 +1,37 @@
+$L1:
     ld.param.u64 %rd1, [vector_add_param_0];
     ld.param.u64 %rd2, [vector_add_param_1];
     ld.param.u64 %rd3, [vector_add_param_2];
     ld.param.u32 %r1, [vector_add_param_3];
+$L2:
     mov.u32 %r2, %ctaid.x;
     mov.u32 %r3, %ntid.x;
-    mov.u32 %r4, %tid.x;
-    mad.lo.s32 %r5, %r2, %r3, %r4;
-    setp.ge.s32 %p1, %r5, %r1;
-    @%p1 bra $L1;
-    cvta.to.global.u64 %rd4, %rd1;
-    mul.wide.s32 %rd5, %r5, 4;
-    add.s64 %rd6, %rd4, %rd5;
-    cvta.to.global.u64 %rd7, %rd2;
-    add.s64 %rd8, %rd7, %rd5;
-    ld.global.f32 %f1, [%rd8];
-    ld.global.f32 %f2, [%rd6];
-    add.f32 %f3, %f2, %f1;
-    cvta.to.global.u64 %rd9, %rd3;
-    add.s64 %rd10, %rd9, %rd5;
-    st.global.f32 [%rd10], %f3;
-$L1:
+    mul.lo.s32 %r4, %r2, %r3;
+    mov.u32 %r5, %tid.x;
+    add.s32 %r6, %r4, %r5;
+$L3:
+    setp.lt.s32 %p1, %r6, %r1;
+    not.pred %p2, %p1;
+    @%p2 bra $L4;
+    bra.uni $L5;
+$L5:
+$L6:
+    cvt.s64.s32 %rd4, %r6;
+    shl.b64 %rd5, %rd4, 2;
+    add.s64 %rd6, %rd1, %rd5;
+    ld.f32 %f1, [%rd6];
+    cvt.s64.s32 %rd7, %r6;
+    shl.b64 %rd8, %rd7, 2;
+    add.s64 %rd9, %rd2, %rd8;
+    ld.f32 %f2, [%rd9];
+    add.f32 %f3, %f1, %f2;
+    cvt.s64.s32 %rd10, %r6;
+    shl.b64 %rd11, %rd10, 2;
+    add.s64 %rd12, %rd3, %rd11;
+    st.f32 [%rd12], %f3;
+    bra.uni $L4;
+$L7:
+$L4:
     ret;
+$L8:
+$L9:
```

</details>

## `saxpy`: changed

| Metric | elementwise.cu | elementwise.cu -G | Δ |
|---|---|---|---|
| instructions | 25 | 38 | +13 |
| instructions in loops | 10 | 26 | +16 |
| basic blocks | 4 | 14 | +10 |
| loops | 1 | 1 | 0 |
| virtual registers | 27 | 34 | +7 |
| peak live slots | 12 | 10 | -2 |
| ptxas registers | 12 | 16 | +4 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 0 | 0 | 0 |

**Changed instructions** (count per mnemonic)

| Mnemonic | elementwise.cu | elementwise.cu -G |
|---|---|---|
| `add.f32` | 0 | 1 |
| `add.s32` | 1 | 2 |
| `add.s64` | 2 | 3 |
| `bra` | 2 | 1 |
| `bra.uni` | 0 | 4 |
| `cvt.s64.s32` | 0 | 3 |
| `cvta.to.global.u64` | 2 | 0 |
| `fma.rn.f32` | 1 | 0 |
| `ld.f32` | 0 | 2 |
| `ld.global.f32` | 1 | 0 |
| `ld.global.nc.f32` | 1 | 0 |
| `mad.lo.s32` | 1 | 0 |
| `mov.u32` | 4 | 8 |
| `mul.f32` | 0 | 1 |
| `mul.lo.s32` | 1 | 2 |
| `mul.wide.s32` | 1 | 0 |
| `not.pred` | 0 | 1 |
| `setp.ge.s32` | 1 | 0 |
| `shl.b64` | 0 | 3 |
| `st.f32` | 0 | 1 |
| `st.global.f32` | 1 | 0 |

<details><summary>Normalised instruction diff</summary>

```diff
--- elementwise.cu:saxpy
+++ elementwise.cu -G:saxpy
@@ -1,27 +1,54 @@
+$L1:
     ld.param.f32 %f1, [saxpy_param_0];
     ld.param.u64 %rd1, [saxpy_param_1];
     ld.param.u64 %rd2, [saxpy_param_2];
     ld.param.u32 %r1, [saxpy_param_3];
-    mov.u32 %r2, %ntid.x;
-    mov.u32 %r3, %ctaid.x;
-    mov.u32 %r4, %tid.x;
-    mad.lo.s32 %r5, %r3, %r2, %r4;
-    setp.ge.s32 %p1, %r5, %r1;
-    @%p1 bra $L1;
-    mov.u32 %r6, %nctaid.x;
-    mul.lo.s32 %r7, %r2, %r6;
-    cvta.to.global.u64 %rd3, %rd1;
-    cvta.to.global.u64 %rd4, %rd2;
 $L2:
-    mul.wide.s32 %rd5, %r5, 4;
-    add.s64 %rd6, %rd3, %rd5;
-    ld.global.nc.f32 %f2, [%rd6];
-    add.s64 %rd7, %rd4, %rd5;
-    ld.global.f32 %f3, [%rd7];
-    fma.rn.f32 %f4, %f2, %f1, %f3;
-    st.global.f32 [%rd7], %f4;
-    add.s32 %r5, %r5, %r7;
-    setp.lt.s32 %p2, %r5, %r1;
-    @%p2 bra $L2;
-$L1:
+    mov.u32 %r2, %ctaid.x;
+    mov.u32 %r3, %ntid.x;
+    mul.lo.s32 %r4, %r2, %r3;
+    mov.u32 %r5, %tid.x;
+    add.s32 %r6, %r4, %r5;
+$L3:
+    mov.u32 %r7, %r6;
+$L4:
+    bra.uni $L5;
+$L5:
+    mov.u32 %r8, %r7;
+$L6:
+    setp.lt.s32 %p1, %r8, %r1;
+    not.pred %p2, %p1;
+    @%p2 bra $L7;
+    bra.uni $L8;
+$L8:
+$L9:
+    cvt.s64.s32 %rd3, %r8;
+    shl.b64 %rd4, %rd3, 2;
+    add.s64 %rd5, %rd1, %rd4;
+    ld.f32 %f2, [%rd5];
+    mul.f32 %f3, %f1, %f2;
+    cvt.s64.s32 %rd6, %r8;
+    shl.b64 %rd7, %rd6, 2;
+    add.s64 %rd8, %rd2, %rd7;
+    ld.f32 %f4, [%rd8];
+    add.f32 %f5, %f3, %f4;
+    cvt.s64.s32 %rd9, %r8;
+    shl.b64 %rd10, %rd9, 2;
+    add.s64 %rd11, %rd2, %rd10;
+    st.f32 [%rd11], %f5;
+$L10:
+    bra.uni $L11;
+$L11:
+    mov.u32 %r9, %ntid.x;
+    mov.u32 %r10, %nctaid.x;
+    mul.lo.s32 %r11, %r9, %r10;
+    add.s32 %r12, %r8, %r11;
+$L12:
+    mov.u32 %r7, %r12;
+$L13:
+    bra.uni $L5;
+$L14:
+$L7:
     ret;
+$L15:
+$L16:
```

</details>

## `scale_vec4`: changed

| Metric | elementwise.cu | elementwise.cu -G | Δ |
|---|---|---|---|
| instructions | 22 | 43 | +21 |
| instructions in loops | 0 | 0 | 0 |
| basic blocks | 3 | 7 | +4 |
| loops | 0 | 0 | 0 |
| virtual registers | 30 | 44 | +14 |
| peak live slots | 9 | 10 | +1 |
| ptxas registers | 14 | 28 | +14 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 16 | +16 |
| ptxas shared mem | 0 | 0 | 0 |

**Changed instructions** (count per mnemonic)

| Mnemonic | elementwise.cu | elementwise.cu -G |
|---|---|---|
| `add.s32` | 0 | 1 |
| `bra.uni` | 0 | 2 |
| `call.uni` | 0 | 1 |
| `cvt.s64.s32` | 0 | 2 |
| `cvta.local.u64` | 0 | 1 |
| `cvta.to.global.u64` | 2 | 0 |
| `ld.f32` | 0 | 4 |
| `ld.global.nc.v4.f32` | 1 | 0 |
| `ld.param.v4.f32` | 0 | 1 |
| `ld.v4.u32` | 0 | 1 |
| `mad.lo.s32` | 1 | 0 |
| `mov.u64` | 0 | 1 |
| `mul.lo.s32` | 0 | 1 |
| `mul.wide.s32` | 1 | 0 |
| `not.pred` | 0 | 1 |
| `setp.ge.s32` | 1 | 0 |
| `setp.lt.s32` | 0 | 1 |
| `shl.b64` | 0 | 2 |
| `st.f32` | 0 | 4 |
| `st.global.v4.f32` | 1 | 0 |
| `st.param.f32` | 0 | 4 |
| `st.v4.u32` | 0 | 1 |

<details><summary>Normalised instruction diff</summary>

```diff
--- elementwise.cu:scale_vec4
+++ elementwise.cu -G:scale_vec4
@@ -1,23 +1,52 @@
+$L1:
+    mov.u64 %SPL, __local_depot2;
+    cvta.local.u64 %SP, %SPL;
     ld.param.u64 %rd1, [scale_vec4_param_0];
     ld.param.u64 %rd2, [scale_vec4_param_1];
     ld.param.f32 %f1, [scale_vec4_param_2];
     ld.param.u32 %r1, [scale_vec4_param_3];
+$L2:
     mov.u32 %r2, %ctaid.x;
     mov.u32 %r3, %ntid.x;
-    mov.u32 %r4, %tid.x;
-    mad.lo.s32 %r5, %r2, %r3, %r4;
-    setp.ge.s32 %p1, %r5, %r1;
-    @%p1 bra $L1;
-    cvta.to.global.u64 %rd3, %rd1;
-    mul.wide.s32 %rd4, %r5, 16;
-    add.s64 %rd5, %rd3, %rd4;
-    ld.global.nc.v4.f32 {%f2, %f3, %f4, %f5}, [%rd5];
-    cvta.to.global.u64 %rd6, %rd2;
-    add.s64 %rd7, %rd6, %rd4;
-    mul.f32 %f6, %f5, %f1;
-    mul.f32 %f7, %f4, %f1;
-    mul.f32 %f8, %f3, %f1;
-    mul.f32 %f9, %f2, %f1;
-    st.global.v4.f32 [%rd7], {%f9, %f8, %f7, %f6};
-$L1:
+    mul.lo.s32 %r4, %r2, %r3;
+    mov.u32 %r5, %tid.x;
+    add.s32 %r6, %r4, %r5;
+$L3:
+    setp.lt.s32 %p1, %r6, %r1;
+    not.pred %p2, %p1;
+    @%p2 bra $L4;
+    bra.uni $L5;
+$L5:
+$L6:
+    cvt.s64.s32 %rd3, %r6;
+    shl.b64 %rd4, %rd3, 4;
+    add.s64 %rd5, %rd1, %rd4;
+    ld.v4.u32 {%r7, %r8, %r9, %r10}, [%rd5];
+    st.v4.u32 [%SP], {%r7, %r8, %r9, %r10};
+    ld.f32 %f2, [%SP];
+    mul.f32 %f3, %f2, %f1;
+    ld.f32 %f4, [%SP+4];
+    mul.f32 %f5, %f4, %f1;
+    ld.f32 %f6, [%SP+8];
+    mul.f32 %f7, %f6, %f1;
+    ld.f32 %f8, [%SP+12];
+    mul.f32 %f9, %f8, %f1;
+    st.param.f32 [param0], %f3;
+    st.param.f32 [param1], %f5;
+    st.param.f32 [param2], %f7;
+    st.param.f32 [param3], %f9;
+    call.uni (retval0), _Z11make_float4ffff, (param0, param1, param2, param3);
+    ld.param.v4.f32 {%f10, %f11, %f12, %f13}, [retval0];
+    cvt.s64.s32 %rd6, %r6;
+    shl.b64 %rd7, %rd6, 4;
+    add.s64 %rd8, %rd2, %rd7;
+    st.f32 [%rd8+12], %f13;
+    st.f32 [%rd8+8], %f12;
+    st.f32 [%rd8+4], %f11;
+    st.f32 [%rd8], %f10;
+    bra.uni $L4;
+$L7:
+$L4:
     ret;
+$L8:
+$L9:
```

</details>

## `_Z11make_float4ffff`: only in B

