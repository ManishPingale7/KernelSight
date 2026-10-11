# KernelSight comparison: math_heavy.cu → math_heavy.cu --use_fast_math

**math_heavy.cu**
- Input: `math_heavy.cu` (cuda), sha256 `b8ddd537bc811604…`
- NVRTC 12.9 options: `-arch=compute_89`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`

**math_heavy.cu --use_fast_math**
- Input: `math_heavy.cu` (cuda), sha256 `b8ddd537bc811604…`
- NVRTC 12.9 options: `-arch=compute_89 --use_fast_math`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`

- Functions: 2 changed, 0 with identical PTX but different ptxas results, 1 identical, 0 only in math_heavy.cu, 0 only in math_heavy.cu --use_fast_math

## `_Z6smoothf`: changed

| Metric | math_heavy.cu | math_heavy.cu --use_fast_math | Δ |
|---|---|---|---|
| instructions | 6 | 6 | 0 |
| instructions in loops | 0 | 0 | 0 |
| basic blocks | 1 | 1 | 0 |
| loops | 0 | 0 | 0 |
| virtual registers | 5 | 5 | 0 |
| peak live slots | 2 | 2 | 0 |
| ptxas registers |  |  |  |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 0 | 0 | 0 |

**Changed instructions** (count per mnemonic)

| Mnemonic | math_heavy.cu | math_heavy.cu --use_fast_math |
|---|---|---|
| `abs.f32` | 1 | 0 |
| `abs.ftz.f32` | 0 | 1 |
| `add.f32` | 1 | 0 |
| `add.ftz.f32` | 0 | 1 |
| `div.approx.ftz.f32` | 0 | 1 |
| `div.rn.f32` | 1 | 0 |

Findings added: none; removed: precise-f32-math

<details><summary>Normalised instruction diff</summary>

```diff
--- math_heavy.cu:_Z6smoothf
+++ math_heavy.cu --use_fast_math:_Z6smoothf
@@ -1,6 +1,6 @@
     ld.param.f32 %f1, [_Z6smoothf_param_0];
-    abs.f32 %f2, %f1;
-    add.f32 %f3, %f2, 0f3F800000;
-    div.rn.f32 %f4, %f1, %f3;
+    abs.ftz.f32 %f2, %f1;
+    add.ftz.f32 %f3, %f2, 0f3F800000;
+    div.approx.ftz.f32 %f4, %f1, %f3;
     st.param.f32 [func_retval0], %f4;
     ret;
```

</details>

## `math_heavy`: changed

| Metric | math_heavy.cu | math_heavy.cu --use_fast_math | Δ |
|---|---|---|---|
| instructions | 199 | 31 | -168 |
| instructions in loops | 9 | 0 | -9 |
| basic blocks | 18 | 3 | -15 |
| loops | 1 | 0 | -1 |
| virtual registers | 214 | 31 | -183 |
| peak live slots | 21 | 8 | -13 |
| ptxas registers | 22 | 14 | -8 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 0 | 0 | 0 |

**Changed instructions** (count per mnemonic)

| Mnemonic | math_heavy.cu | math_heavy.cu --use_fast_math |
|---|---|---|
| `abs.f32` | 1 | 0 |
| `add.f32` | 5 | 0 |
| `add.ftz.f32` | 0 | 2 |
| `add.s32` | 6 | 0 |
| `add.s64` | 6 | 2 |
| `add.u64` | 1 | 0 |
| `and.b32` | 6 | 0 |
| `bfi.b64` | 1 | 0 |
| `bra` | 8 | 1 |
| `bra.uni` | 2 | 0 |
| `cvt.rn.f32.f64` | 1 | 0 |
| `cvt.rn.f32.s32` | 2 | 0 |
| `cvt.rn.f64.s64` | 1 | 0 |
| `cvt.rni.s32.f32` | 1 | 0 |
| `cvt.s64.s32` | 1 | 0 |
| `cvt.sat.f32.f32` | 1 | 0 |
| `cvt.u64.u32` | 2 | 0 |
| `div.approx.ftz.f32` | 0 | 1 |
| `div.rn.f32` | 1 | 0 |
| `fma.rm.f32` | 1 | 0 |
| `fma.rn.f32` | 25 | 0 |
| `fma.rn.ftz.f32` | 0 | 2 |
| `ld.global.nc.u32` | 1 | 0 |
| `ld.local.u32` | 3 | 0 |
| `lg2.approx.ftz.f32` | 0 | 1 |
| `mad.wide.u32` | 1 | 0 |
| `mov.b32` | 5 | 0 |
| `mov.f32` | 27 | 0 |
| `mov.u32` | 8 | 3 |
| `mov.u64` | 4 | 0 |
| `mul.f32` | 5 | 0 |
| `mul.f64` | 1 | 0 |
| `mul.ftz.f32` | 0 | 2 |
| `mul.rn.f32` | 2 | 0 |
| `mul.wide.s32` | 3 | 1 |
| `neg.f32` | 3 | 0 |
| `neg.s32` | 1 | 0 |
| `or.b32` | 2 | 0 |
| `selp.b32` | 3 | 0 |
| `selp.f32` | 7 | 0 |
| `setp.eq.f32` | 2 | 0 |
| `setp.eq.s32` | 5 | 0 |
| `setp.lt.f32` | 1 | 0 |
| `setp.lt.u32` | 1 | 0 |
| `setp.ltu.f32` | 1 | 0 |
| `setp.ne.s32` | 2 | 0 |
| `shl.b32` | 6 | 0 |
| `shl.b64` | 1 | 0 |
| `shr.u32` | 7 | 0 |
| `shr.u64` | 1 | 0 |
| `sin.approx.ftz.f32` | 0 | 1 |
| `sqrt.approx.ftz.f32` | 0 | 1 |
| `sqrt.rn.f32` | 1 | 0 |
| `st.local.u32` | 2 | 0 |
| `sub.s32` | 4 | 0 |
| `xor.b32` | 3 | 0 |

Findings added: none; removed: fp64-arithmetic, local-memory, precise-f32-math

<details><summary>Normalised instruction diff</summary>

```diff
--- math_heavy.cu:math_heavy
+++ math_heavy.cu --use_fast_math:math_heavy
@@ -1,208 +1,32 @@
-    mov.u64 %SPL, __local_depot1;
     ld.param.u64 %rd1, [math_heavy_param_0];
     ld.param.u64 %rd2, [math_heavy_param_1];
     ld.param.u32 %r1, [math_heavy_param_2];
-    add.u64 %rd3, %SPL, 0;
-    mov.u32 %r2, %ntid.x;
-    mov.u32 %r3, %ctaid.x;
+    mov.u32 %r2, %ctaid.x;
+    mov.u32 %r3, %ntid.x;
     mov.u32 %r4, %tid.x;
-    mad.lo.s32 %r5, %r3, %r2, %r4;
+    mad.lo.s32 %r5, %r2, %r3, %r4;
     setp.ge.s32 %p1, %r5, %r1;
     @%p1 bra $L1;
-    cvta.to.global.u64 %rd4, %rd1;
-    cvt.s64.s32 %rd5, %r5;
-    mul.wide.s32 %rd6, %r5, 4;
-    add.s64 %rd7, %rd4, %rd6;
-    ld.global.f32 %f1, [%rd7];
-    fma.rn.f32 %f2, %f1, %f1, 0f3F800000;
-    mul.f32 %f3, %f1, 0f3F22F983;
-    cvt.rni.s32.f32 %r6, %f3;
-    cvt.rn.f32.s32 %f4, %r6;
-    mov.f32 %f5, 0fBFC90FDA;
-    fma.rn.f32 %f6, %f4, %f5, %f1;
-    mov.f32 %f7, 0fB3A22168;
-    fma.rn.f32 %f8, %f4, %f7, %f6;
-    mov.f32 %f9, 0fA7C234C5;
-    fma.rn.f32 %f10, %f4, %f9, %f8;
-    abs.f32 %f11, %f1;
-    setp.ltu.f32 %p2, %f11, 0f47CE4780;
-    @%p2 bra $L2;
-    setp.eq.f32 %p3, %f11, 0f7F800000;
-    @%p3 bra $L3;
-    bra.uni $L4;
-$L3:
-    mov.f32 %f12, 0f00000000;
-    mul.rn.f32 %f10, %f1, %f12;
-    mov.u32 %r6, 0;
-    bra.uni $L2;
-$L4:
-    mov.b32 %r7, %f1;
-    shr.u32 %r8, %r7, 23;
-    and.b32 %r9, %r8, 255;
-    add.s32 %r10, %r9, -128;
-    shl.b32 %r11, %r7, 8;
-    or.b32 %r12, %r11, -2147483648;
-    shr.u32 %r13, %r10, 5;
-    mov.u64 %rd8, 0;
-    mov.u32 %r14, 0;
-    mov.u64 %rd9, __cudart_i2opi_f;
-    mov.u64 %rd10, %rd3;
-$L5:
-    ld.global.nc.u32 %r15, [%rd9];
-    mad.wide.u32 %rd11, %r15, %r12, %rd8;
-    shr.u64 %rd8, %rd11, 32;
-    st.local.u32 [%rd10], %rd11;
-    add.s64 %rd9, %rd9, 4;
-    add.s64 %rd10, %rd10, 4;
-    add.s32 %r14, %r14, 1;
-    setp.ne.s32 %p4, %r14, 6;
-    @%p4 bra $L5;
-    st.local.u32 [%rd3+24], %rd8;
-    mov.u32 %r16, 4;
-    sub.s32 %r17, %r16, %r13;
-    mov.u32 %r18, 6;
-    sub.s32 %r19, %r18, %r13;
-    mul.wide.s32 %rd12, %r19, 4;
-    add.s64 %rd13, %rd3, %rd12;
-    ld.local.u32 %r20, [%rd13];
-    ld.local.u32 %r21, [%rd13+-4];
-    and.b32 %r22, %r10, 31;
-    setp.eq.s32 %p5, %r22, 0;
-    @%p5 bra $L6;
-    mov.u32 %r23, 32;
-    sub.s32 %r24, %r23, %r22;
-    shr.u32 %r25, %r21, %r24;
-    shl.b32 %r26, %r20, %r22;
-    add.s32 %r20, %r25, %r26;
-    mul.wide.s32 %rd14, %r17, 4;
-    add.s64 %rd15, %rd3, %rd14;
-    ld.local.u32 %r27, [%rd15];
-    shr.u32 %r28, %r27, %r24;
-    shl.b32 %r29, %r21, %r22;
-    add.s32 %r21, %r28, %r29;
-$L6:
-    and.b32 %r30, %r7, -2147483648;
-    shr.u32 %r31, %r21, 30;
-    shl.b32 %r32, %r20, 2;
-    or.b32 %r33, %r31, %r32;
-    shr.u32 %r34, %r33, 31;
-    shr.u32 %r35, %r20, 30;
-    add.s32 %r36, %r34, %r35;
-    neg.s32 %r37, %r36;
-    setp.eq.s32 %p6, %r30, 0;
-    selp.b32 %r6, %r36, %r37, %p6;
-    setp.ne.s32 %p7, %r34, 0;
-    xor.b32 %r38, %r30, -2147483648;
-    selp.b32 %r39, %r38, %r30, %p7;
-    selp.b32 %r40, -1, 0, %p7;
-    xor.b32 %r41, %r33, %r40;
-    shl.b32 %r42, %r21, 2;
-    xor.b32 %r43, %r42, %r40;
-    cvt.u64.u32 %rd16, %r41;
-    cvt.u64.u32 %rd17, %r43;
-    bfi.b64 %rd18, %rd16, %rd17, 32, 32;
-    cvt.rn.f64.s64 %fd1, %rd18;
-    mul.f64 %fd2, %fd1, 0d3BF921FB54442D19;
-    cvt.rn.f32.f64 %f13, %fd2;
-    setp.eq.s32 %p8, %r39, 0;
-    neg.f32 %f14, %f13;
-    selp.f32 %f10, %f13, %f14, %p8;
-$L2:
-    and.b32 %r44, %r6, 1;
-    setp.eq.s32 %p9, %r44, 0;
-    mul.rn.f32 %f15, %f10, %f10;
-    mov.f32 %f16, 0fB94D4153;
-    @%p9 bra $L7;
-    mov.f32 %f17, 0fBAB607ED;
-    mov.f32 %f18, 0f37CBAC00;
-    fma.rn.f32 %f16, %f18, %f15, %f17;
-$L7:
-    selp.f32 %f19, %f10, 0f3F800000, %p9;
-    selp.f32 %f20, 0f3C0885E4, 0f3D2AAABB, %p9;
-    fma.rn.f32 %f21, %f16, %f15, %f20;
-    selp.f32 %f22, 0fBE2AAAA8, 0fBEFFFFFF, %p9;
-    fma.rn.f32 %f23, %f21, %f15, %f22;
-    mov.f32 %f24, 0f00000000;
-    fma.rn.f32 %f25, %f15, %f19, %f24;
-    fma.rn.f32 %f26, %f23, %f25, %f19;
-    and.b32 %r45, %r6, 2;
-    setp.eq.s32 %p10, %r45, 0;
-    @%p10 bra $L8;
-    mov.f32 %f27, 0fBF800000;
-    fma.rn.f32 %f26, %f26, %f27, %f24;
-$L8:
-    neg.f32 %f28, %f1;
-    mov.f32 %f29, 0f3F000000;
-    mov.f32 %f30, 0f3BBB989D;
-    fma.rn.f32 %f31, %f28, %f30, %f29;
-    cvt.sat.f32.f32 %f32, %f31;
-    mov.f32 %f33, 0f4B400001;
-    mov.f32 %f34, 0f437C0000;
-    fma.rm.f32 %f35, %f32, %f34, %f33;
-    add.f32 %f36, %f35, 0fCB40007F;
-    neg.f32 %f37, %f36;
-    mov.f32 %f38, 0f3FB8AA3B;
-    fma.rn.f32 %f39, %f28, %f38, %f37;
-    mov.f32 %f40, 0f32A57060;
-    fma.rn.f32 %f41, %f28, %f40, %f39;
-    ex2.approx.ftz.f32 %f42, %f41;
-    mov.b32 %r46, %f35;
-    shl.b32 %r47, %r46, 23;
-    mov.b32 %f43, %r47;
-    mul.f32 %f44, %f42, %f43;
-    mul.f32 %f45, %f44, %f26;
-    mul.f32 %f46, %f2, 0f4B000000;
-    setp.lt.f32 %p11, %f2, 0f00800000;
-    selp.f32 %f47, %f46, %f2, %p11;
-    selp.f32 %f48, 0fC1B80000, 0f00000000, %p11;
-    mov.b32 %r48, %f47;
-    add.s32 %r49, %r48, -1059760811;
-    and.b32 %r50, %r49, -8388608;
-    sub.s32 %r51, %r48, %r50;
-    mov.b32 %f49, %r51;
-    cvt.rn.f32.s32 %f50, %r50;
-    mov.f32 %f51, 0f34000000;
-    fma.rn.f32 %f52, %f50, %f51, %f48;
-    add.f32 %f53, %f49, 0fBF800000;
-    mov.f32 %f54, 0f3E1039F6;
-    mov.f32 %f55, 0fBE055027;
-    fma.rn.f32 %f56, %f55, %f53, %f54;
-    mov.f32 %f57, 0fBDF8CDCC;
-    fma.rn.f32 %f58, %f56, %f53, %f57;
-    mov.f32 %f59, 0f3E0F2955;
-    fma.rn.f32 %f60, %f58, %f53, %f59;
-    mov.f32 %f61, 0fBE2AD8B9;
-    fma.rn.f32 %f62, %f60, %f53, %f61;
-    mov.f32 %f63, 0f3E4CED0B;
-    fma.rn.f32 %f64, %f62, %f53, %f63;
-    mov.f32 %f65, 0fBE7FFF22;
-    fma.rn.f32 %f66, %f64, %f53, %f65;
-    mov.f32 %f67, 0f3EAAAA78;
-    fma.rn.f32 %f68, %f66, %f53, %f67;
-    mov.f32 %f69, 0fBF000000;
-    fma.rn.f32 %f70, %f68, %f53, %f69;
-    mul.f32 %f71, %f53, %f70;
-    fma.rn.f32 %f72, %f71, %f53, %f53;
-    mov.f32 %f73, 0f3F317218;
-    fma.rn.f32 %f74, %f52, %f73, %f72;
-    setp.lt.u32 %p12, %r48, 2139095040;
-    add.f32 %f75, %f1, 0f40000000;
-    sqrt.rn.f32 %f76, %f2;
-    div.rn.f32 %f77, %f76, %f75;
-    @%p12 bra $L9;
-    mov.f32 %f78, 0f7F800000;
-    fma.rn.f32 %f74, %f47, %f78, %f78;
-$L9:
-    setp.eq.f32 %p13, %f47, 0f00000000;
-    selp.f32 %f79, 0fFF800000, %f74, %p13;
-    add.f32 %f80, %f45, %f79;
-    add.f32 %f81, %f77, %f80;
-    st.param.f32 [param0], %f81;
+    cvta.to.global.u64 %rd3, %rd1;
+    mul.wide.s32 %rd4, %r5, 4;
+    add.s64 %rd5, %rd3, %rd4;
+    ld.global.f32 %f1, [%rd5];
+    fma.rn.ftz.f32 %f2, %f1, %f1, 0f3F800000;
+    sqrt.approx.ftz.f32 %f3, %f2;
+    add.ftz.f32 %f4, %f1, 0f40000000;
+    div.approx.ftz.f32 %f5, %f3, %f4;
+    mul.ftz.f32 %f6, %f1, 0fBFB8AA3B;
+    ex2.approx.ftz.f32 %f7, %f6;
+    sin.approx.ftz.f32 %f8, %f1;
+    lg2.approx.ftz.f32 %f9, %f2;
+    mul.ftz.f32 %f10, %f9, 0f3F317218;
+    fma.rn.ftz.f32 %f11, %f7, %f8, %f10;
+    add.ftz.f32 %f12, %f5, %f11;
+    st.param.f32 [param0], %f12;
     call.uni (retval0), _Z6smoothf, (param0);
-    ld.param.f32 %f82, [retval0];
-    cvta.to.global.u64 %rd19, %rd2;
-    shl.b64 %rd20, %rd5, 2;
-    add.s64 %rd21, %rd19, %rd20;
-    st.global.f32 [%rd21], %f82;
+    ld.param.f32 %f13, [retval0];
+    cvta.to.global.u64 %rd6, %rd2;
+    add.s64 %rd7, %rd6, %rd4;
+    st.global.f32 [%rd7], %f13;
 $L1:
     ret;
```

</details>

## `polynomial`: identical

| Metric | math_heavy.cu | math_heavy.cu --use_fast_math | Δ |
|---|---|---|---|
| instructions | 25 | 25 | 0 |
| instructions in loops | 0 | 0 | 0 |
| basic blocks | 3 | 3 | 0 |
| loops | 0 | 0 | 0 |
| virtual registers | 26 | 26 | 0 |
| peak live slots | 8 | 8 | 0 |
| ptxas registers | 10 | 10 | 0 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 0 | 0 | 0 |

