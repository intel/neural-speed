# Usage

Inherit `Xbyak::CodeGenerator` class and make the class method.
```
#include <xbyak/xbyak.h>

struct Code : Xbyak::CodeGenerator {
    Code(int x)
    {
        mov(eax, x);
        ret();
    }
};
```
Or you can pass the instance of CodeGenerator without inheriting.
```
void genCode(Xbyak::CodeGenerator& code, int x) {
    using namespace Xbyak::util;
    code.mov(eax, x);
    code.ret();
}
```

Make an instance of the class and get the function
pointer by calling `getCode()` and call it.
```
Code c(5);
int (*f)() = c.getCode<int (*)()>();
printf("ret=%d\n", f()); // ret = 5
```

## Syntax
Similar to MASM/NASM syntax with parentheses.

```
NASM              Xbyak
mov eax, ebx  --> mov(eax, ebx);
inc ecx           inc(ecx);
ret           --> ret();
```

## Addressing
Use `qword`, `dword`, `word` and `byte` if it is necessary to specify the size of memory,
otherwise use `ptr`.

```
(ptr|qword|dword|word|byte) [base + index * (1|2|4|8) + displacement]
                            [rip + 32bit disp] ; x64 only

NASM                   Xbyak
mov eax, [ebx+ecx] --> mov(eax, ptr [ebx+ecx]);
mov al, [ebx+ecx]  --> mov(al, ptr [ebx + ecx]);
test byte [esp], 4 --> test(byte [esp], 4);
inc qword [rax]    --> inc(qword [rax]);
```
**Note**: `qword`, ... are member variables, then don't use `dword` as unsigned int type.

### How to use Selector (Segment Register)
```
mov eax, [fs:eax] --> putSeg(fs);
                      mov(eax, ptr [eax]);
mov ax, cs        --> mov(ax, cs);
```
**Note**: Segment class is not derived from `Operand`.

## AVX

```
vaddps(xmm1, xmm2, xmm3); // xmm1 <- xmm2 + xmm3
vaddps(xmm2, xmm3, ptr [rax]); // use ptr to access memory
vgatherdpd(xmm1, ptr [ebp + 256 + xmm2*4], xmm3);
```

**Note**:
If `XBYAK_ENABLE_OMITTED_OPERAND` is defined, then you can use two operand version for backward compatibility.
But the newer version will not support it.
```
vaddps(xmm2, xmm3); // xmm2 <- xmm2 + xmm3
```

## AVX-512

```
vaddpd zmm2, zmm5, zmm30                --> vaddpd(zmm2, zmm5, zmm30);
vaddpd xmm30, xmm20, [rax]              --> vaddpd(xmm30, xmm20, ptr [rax]);
vaddps xmm30, xmm20, [rax]              --> vaddps(xmm30, xmm20, ptr [rax]);
vaddpd zmm2{k5}, zmm4, zmm2             --> vaddpd(zmm2 | k5, zmm4, zmm2);
vaddpd zmm2{k5}{z}, zmm4, zmm2          --> vaddpd(zmm2 | k5 | T_z, zmm4, zmm2);
vaddpd zmm2{k5}{z}, zmm4, zmm2,{rd-sae} --> vaddpd(zmm2 | k5 | T_z, zmm4, zmm2 | T_rd_sae);
                                            vaddpd(zmm2 | k5 | T_z | T_rd_sae, zmm4, zmm2); // the position of `|` is arbitrary.
vcmppd k4{k3}, zmm1, zmm2, {sae}, 5     --> vcmppd(k4 | k3, zmm1, zmm2 | T_sae, 5);

vaddpd xmm1, xmm2, [rax+256]            --> vaddpd(xmm1, xmm2, ptr [rax+256]);
vaddpd xmm1, xmm2, [rax+256]{1to2}      --> vaddpd(xmm1, xmm2, ptr_b [rax+256]);
vaddpd ymm1, ymm2, [rax+256]{1to4}      --> vaddpd(ymm1, ymm2, ptr_b [rax+256]);
vaddpd zmm1, zmm2, [rax+256]{1to8}      --> vaddpd(zmm1, zmm2, ptr_b [rax+256]);
vaddps zmm1, zmm2, [rax+rcx*8+8]{1to16} --> vaddps(zmm1, zmm2, ptr_b [rax+rcx*8+8]);
vmovsd [rax]{k1}, xmm4                  --> vmovsd(ptr [rax] | k1, xmm4);

vcvtpd2dq xmm16, oword [eax+33]         --> vcvtpd2dq(xmm16, xword [eax+33]); // use xword for m128 instead of oword
                                            vcvtpd2dq(xmm16, ptr [eax+33]); // default xword
vcvtpd2dq xmm21, [eax+32]{1to2}         --> vcvtpd2dq(xmm21, ptr_b [eax+32]);
vcvtpd2dq xmm0, yword [eax+33]          --> vcvtpd2dq(xmm0, yword [eax+33]); // use yword for m256
vcvtpd2dq xmm19, [eax+32]{1to4}         --> vcvtpd2dq(xmm19, yword_b [eax+32]); // use yword_b to broadcast

vfpclassps k5{k3}, zword [rax+64], 5    --> vfpclassps(k5|k3, zword [rax+64], 5); // specify m512
vfpclasspd k5{k3}, [rax+64]{1to2}, 5    --> vfpclasspd(k5|k3, xword_b [rax+64], 5); // broadcast 64-bit to 128-bit
vfpclassps k5{k3}, [rax+64]{1to4}, 5    --> vfpclassps(k5|k3, yword_b [rax+64], 5); // broadcast 64-bit to 256-bit

vpdpbusd(xm0, xm1, xm2); // default encoding is EVEX
vpdpbusd(xm0, xm1, xm2, EvexEncoding); // same as the above
vpdpbusd(xm0, xm1, xm2, VexEncoding); // VEX encoding
setDefaultEncoding(VexEncoding); // default encoding is VEX
vpdpbusd(xm0, xm1, xm2); // VEX encoding
```

- setDefaultEncoding(PreferredEncoding encoding);
  - Set the default encoding to select EVEX or VEX.
  - The default value is EvexEncoding.
  - This function affects only an instruction that has a PreferredEncoding argument such as vpdpbusd.

### Remark
* `k1`, ..., `k7` are opmask registers.
  - `k0` is dealt as no mask.
  - e.g. `vmovaps(zmm0|k0, ptr[rax]);` and `vmovaps(zmm0|T_z, ptr[rax]);` are same to `vmovaps(zmm0, ptr[rax]);`.
* use `| T_z`, `| T_sae`, `| T_rn_sae`, `| T_rd_sae`, `| T_ru_sae`, `| T_rz_sae` instead of `,{z}`, `,{sae}`, `,{rn-sae}`, `,{rd-sae}`, `,{ru-sae}`, `,{rz-sae}` respectively.
* `k4 | k3` is different from `k3 | k4`.
* use `ptr_b` for broadcast `{1toX}`. X is automatically determined.
* specify `xword`/`yword`/`zword(_b)` for m128/m256/m512 if necessary.

## Label
Two kinds of Label are supported. (String literal and Label class).

### String literal
```
L("L1");
  jmp("L1");

  jmp("L2");
  ...
  a few mnemonics (8-bit displacement jmp)
  ...
L("L2");

  jmp("L3", T_NEAR);
  ...
  a lot of mnemonics (32-bit displacement jmp)
  ...
L("L3");
```

* Call `hasUndefinedLabel()` to verify your code has no undefined label.
* you can use a label for immediate value of mov like as `mov(eax, "L2")`.

### Support `@@`, `@f`, `@b` like MASM

```
L("@@"); // <A>
  jmp("@b"); // jmp to <A>
  jmp("@f"); // jmp to <B>
L("@@"); // <B>
  jmp("@b"); // jmp to <B>
  mov(eax, "@b");
  jmp(eax); // jmp to <B>
```

### Local label

Label symbols beginning with a period between `inLocalLabel()` and `outLocalLabel()`
are treated as a local label.
`inLocalLabel()` and `outLocalLabel()` can be nested.

```
void func1()
{
    inLocalLabel();
  L(".lp"); // <A> ; local label
    ...
    jmp(".lp"); // jmp to <A>
  L("aaa"); // global label <C>
    outLocalLabel();

    inLocalLabel();
  L(".lp"); // <B> ; local label
    func1();
    jmp(".lp"); // jmp to <B>
    inLocalLabel();
    jmp("aaa"); // jmp to <C>
}
```

### short and long jump
Xbyak deals with jump mnemonics of an undefined label as short jump if no type is specified.
So if the size between jmp and label is larger than 127 byte, then xbyak will cause an error.

```
jmp("short-jmp"); // short jmp
// small code
L("short-jmp");

jmp("long-jmp");
// long code
L("long-jmp"); // throw exception
```
Then specify T_NEAR for jmp.
```
jmp("long-jmp", T_NEAR); // long jmp
// long code
L("long-jmp");
```
Or call `setDefaultJmpNEAR(true);` once, then the default type is set to T_NEAR.
```
jmp("long-jmp"); // long jmp
// long code
L("long-jmp");
```

### Label class

`L()` and `jxx()` support Label class.

```
  Xbyak::Label label1, label2;
L(label1);
  ...
  jmp(label1);
  ...
  jmp(label2);
  ...
L(label2);
```

Use `putL` for jmp table
```
    Label labelTbl, L0, L1, L2;
    mov(rax, labelTbl);
    // rdx is an index of jump table
    jmp(ptr [rax + rdx * sizeof(void*)]);
L(labelTbl);
    putL(L0);
    putL(L1);
    putL(L2);
L(L0);
    ....
L(L1);
    ....
```

`assignL(dstLabel, srcLabel)` binds dstLabel with srcLabel.

```
  Label label2;
  Label label1 = L(); // make label1 ; same to Label label1; L(label1);
  ...
  jmp(label2); // label2 is not determined here
  ...
  assignL(label2, label1); // label2 <- label1
```
The `jmp` in the above code jumps to label1 assigned by `assignL`.

**Note**:
* srcLabel must be used in `L()`.
* dstLabel must not be used in `L()`.

`Label::getAddress()` returns the address specified by the label instance and 0 if not specified.
```
// not AutoGrow mode
Label  label;
assert(label.getAddress() == 0);
L(label);
assert(label.getAddress() == getCurr());
```

### Rip ; relative addressing
```
Label label;
mov(eax, ptr [rip + label]); // eax = 4
...

L(label);
dd(4);
```
```
int x;
...
  mov(eax, ptr[rip + &x]); // throw exception if the difference between &x and current position is larger than 2GiB
```

## Far jump

Use `word|dword|qword` instead of `ptr` to specify the address size.

### 32 bit mode
```
jmp(word[eax], T_FAR);  // jmp m16:16(FF /5)
jmp(dword[eax], T_FAR); // jmp m16:32(FF /5)
```

### 64 bit mode
```
jmp(word[rax], T_FAR);  // jmp m16:16(FF /5)
jmp(dword[rax], T_FAR); // jmp m16:32(FF /5)
jmp(qword[rax], T_FAR); // jmp m16:64(REX.W FF /5)
```
The same applies to `call`.

## Code size
The default max code size is 4096 bytes.
Specify the size in constructor of `CodeGenerator()` if necessary.

```
class Quantize : public Xbyak::CodeGenerator {
public:
  Quantize()
    : CodeGenerator(8192)
  {
  }
  ...
};
```

## User allocated memory

You can make jit code on prepared memory.

Call `setProtectModeRE` yourself to change memory mode if using the prepared memory.

```
uint8_t alignas(4096) buf[8192]; // C++11 or later

struct Code : Xbyak::CodeGenerator {
    Code() : Xbyak::CodeGenerator(sizeof(buf), buf)
    {
        mov(rax, 123);
        ret();
    }
};

int main()
{
    Code c;
    c.setProtectModeRE(); // set memory to Read/Exec
    printf("%d\n", c.getCode<int(*)()>()());
}
```

**Note**: See [../sample/test0.cpp](../sample/test0.cpp).

### AutoGrow

The memory region for jit is automatically extended if necessary when `AutoGrow` is specified in a constructor of `CodeGenerator`.

Call `ready()` or `readyRE()` before calling `getCode()` to fix jump address.
```
struct Code : Xbyak::CodeGenerator {
  Code()
    : Xbyak::CodeGenerator(<default memory size>, Xbyak::AutoGrow)
  {
     ...
  }
};
Code c;
// generate code for jit
c.ready(); // mode = Read/Write/Exec
```

**Note**:
* Don't use the address returned by `getCurr()` before calling `ready()` because it may be invalid address.

### Read/Exec mode
Xbyak set Read/Write/Exec mode to memory to run jit code.
If you want to use Read/Exec mode for security, then specify `DontSetProtectRWE` for `CodeGenerator` and
call `setProtectModeRE()` after generating jit code.

```
struct Code : Xbyak::CodeGenerator {
    Code()
        : Xbyak::CodeGenerator(4096, Xbyak::DontSetProtectRWE)
    {
        mov(eax, 123);
        ret();
    }
};

Code c;
c.setProtectModeRE();
...

```
Call `readyRE()` instead of `ready()` when using `AutoGrow` mode.
See [protect-re.cpp](../sample/protect-re.cpp).

## Exception-less mode
If `XBYAK_NO_EXCEPTION` is defined, then gcc/clang can compile xbyak with `-fno-exceptions`.
In stead of throwing an exception, `Xbyak::GetError()` returns non-zero value (e.g. `ERR_BAD_ADDRESSING`) if there is something wrong.
The status will not be changed automatically, then you should reset it by `Xbyak::ClearError()`.
`CodeGenerator::reset()` calls `ClearError()`.

## Macro

* **XBYAK32** is defined on 32bit.
* **XBYAK64** is defined on 64bit.
* **XBYAK64_WIN** is defined on 64bit Windows(VC).
* **XBYAK64_GCC** is defined on 64bit gcc, cygwin.
* define **XBYAK_USE_OP_NAMES** on gcc with `-fno-operator-names` if you want to use `and()`, ....
* define **XBYAK_ENABLE_OMITTED_OPERAND** if you use omitted destination such as `vaddps(xmm2, xmm3);`(deprecated in the future).
* define **XBYAK_UNDEF_JNL** if Bessel function jnl is defined as macro.
* define **XBYAK_NO_EXCEPTION** for a compiler option `-fno-exceptions`.
* define **XBYAK_USE_MEMFD** on Linux then /proc/self/maps shows the area used by xbyak.
* define **XBYAK_OLD_DISP_CHECK** if the old disp check is necessary (deprecated in the future).

## Sample

* [test0.cpp](../sample/test0.cpp) ; tiny sample (x86, x64)
* [quantize.cpp](../sample/quantize.cpp) ; JIT optimized quantization by fast division (x86 only)
* [calc.cpp](../sample/calc.cpp) ; assemble and estimate a given polynomial (x86, x64)
* [bf.cpp](../sample/bf.cpp) ; JIT brainfuck (x86, x64)
