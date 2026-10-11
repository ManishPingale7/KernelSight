import pytest

from kernelsight import toolchain

pytestmark = pytest.mark.toolchain

KERNEL = 'extern "C" __global__ void k(float* x) { x[threadIdx.x] *= 2.0f; }'


def test_compile_cuda_produces_ptx_for_requested_arch():
    result = toolchain.compile_cuda(KERNEL, ["-arch=compute_89"])
    assert ".target sm_89" in result.ptx
    assert ".entry k(" in result.ptx
    assert result.options == ("-arch=compute_89",)


def test_compile_error_includes_compiler_log():
    with pytest.raises(toolchain.ToolchainError, match="undefined_name"):
        toolchain.compile_cuda('extern "C" __global__ void k() { undefined_name(); }')


def test_ptxas_reports_resource_usage():
    ptx = toolchain.compile_cuda(KERNEL, ["-arch=compute_89"]).ptx
    out = toolchain.run_ptxas(ptx, "sm_89")
    assert "Compiling entry function 'k' for 'sm_89'" in out
    assert "registers" in out


def test_ptxas_rejects_invalid_ptx():
    with pytest.raises(toolchain.ToolchainError, match="ptxas rejected"):
        toolchain.run_ptxas(".version 8.5\n.target sm_89\n.entry k() { bogus; }")
