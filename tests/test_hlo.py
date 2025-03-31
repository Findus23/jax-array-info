"""
This is completely unrelated to jax_array_info,
but I want to know when my ugly hack for debugging HLO
breaks in newer jax versions, so I will also test it here.

See https://github.com/jax-ml/jax/issues/19691#issuecomment-2748975994
and
https://github.com/openxla/xla/issues/24186
for context.

Also more about StableHLO and XLA can be found here:
- https://openxla.org/stablehlo/tutorials/jax-export
- https://openxla.org/stablehlo/compatibility

"""
import dataclasses
import os

import jax
import pytest
from jax._src.interpreters import mlir as jax_mlir
from jax.export import export
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import Module
from jaxlib.xla_client import XlaRuntimeError

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
jax.config.update('jax_enable_x64', True)

mesh = Mesh(jax.devices(), ('a',))
sharding = NamedSharding(mesh, PartitionSpec('a'))

# use a no-op function as the input
jitted_f = jax.jit(lambda a: a)
input_shape = (128,)
input_dtype = jax.numpy.float64
input_sharding = sharding
abstract_input = jax.ShapeDtypeStruct(input_shape, input_dtype, sharding=input_sharding)

exported = export(jitted_f)(abstract_input)
lowered = jitted_f.lower(abstract_input)
compiled = lowered.compile()

print(lowered.as_text())


def test_mlir_module_drectly():
    import jaxlib.xla_extension.mlir as mlir

    assert exported.mlir_module_serialized.startswith(b"ML\xefR\rStableHLO_v1.9.1")
    manually_converted_to_stablehlo = mlir.stablehlo_to_mhlo(exported.mlir_module_serialized)
    assert manually_converted_to_stablehlo.startswith(b"ML\xefR\rMLIR21.0.0git")


def test_stablehlo_dialect():
    # these functions are described in https://openxla.org/stablehlo/compatibility
    assert stablehlo.get_current_version() == "1.10.0"
    assert stablehlo.get_minimum_version() == "0.9.0"


def test_lowered():
    assert "return %arg0 : tensor<128xf64>" in lowered.as_text()
    assert lowered.as_text() == """
module @jit__lambda_ attributes {mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128xf64> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> (tensor<128xf64> {jax.result_info = "result"}) {
    return %arg0 : tensor<128xf64>
  }
}
""".lstrip()


def test_reading_stablehlo():
    mlir_module = exported.mlir_module_serialized
    context = jax_mlir.make_ir_context()
    out: Module = stablehlo.deserialize_portable_artifact(context, mlir_module)
    stablehlo_str = out.operation.get_asm(enable_debug_info=False)
    # this is essentially identical to lowered.as_text(), with minor debug info differences
    assert stablehlo_str == """
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128xf64> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> (tensor<128xf64> {jax.result_info = "result"}) {
    return %arg0 : tensor<128xf64>
  }
}
""".lstrip()


def test_replacing_stablehlo_trivial_case():
    """
    replace the lambda a: a function defined above with one that adds 42 to all entries
    by replacing the stablehlo
    """
    custom_stablehlo = """
module @jit__lambda_ attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128xf64> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> (tensor<128xf64> {jax.result_info = "result"}) {
    %some_int_constant = stablehlo.constant dense<42> : tensor<i64>
    %const_as_float = stablehlo.convert %some_int_constant : (tensor<i64>) -> tensor<f64>
    %filled_array = stablehlo.broadcast_in_dim %const_as_float, dims = [] : (tensor<f64>) -> tensor<128xf64>
    // add the full 42-array to the input and return the result
    %output = stablehlo.add %filled_array, %arg0 : tensor<128xf64>

    return %output : tensor<128xf64>
  }
}
"""
    mlir_module_replaced = stablehlo.serialize_portable_artifact_str(custom_stablehlo, stablehlo.get_current_version())
    # this should replace the no-op function with one that adds 42 to all entries
    replaced_function = dataclasses.replace(exported, mlir_module_serialized=mlir_module_replaced)

    # create actual input for the function
    input = jax.numpy.full(input_shape, 1, dtype=input_dtype)
    expected_output = jax.numpy.full(input_shape, 1. + 42., dtype=input_dtype)
    input = jax.device_put(input, input_sharding)
    expected_output = jax.device_put(expected_output, input_sharding)

    output = replaced_function.call(input)
    assert jax.numpy.all(output == expected_output)


def test_replacing_stablehlo_bug():
    """
    doing the same to reproduce the bug in
    https://github.com/jax-ml/jax/issues/19691#issuecomment-2748975994
    """
    custom_stablehlo = """
module @jit_f attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%input_array: tensor<128xf64> {mhlo.sharding = "{devices=[4]<=[4]}"}) -> (tensor<128xf64> {jax.result_info = ""}) {
    %const_scalar = stablehlo.constant dense<1> : tensor<i64>
    %const_scalar2 = stablehlo.constant dense<2> : tensor<i64>
    %slice = stablehlo.dynamic_slice %input_array, %const_scalar, sizes = [1] : (tensor<128xf64>, tensor<i64>) -> tensor<1xf64>
    %output = stablehlo.dynamic_update_slice %input_array, %slice, %const_scalar2 : (tensor<128xf64>, tensor<1xf64>, tensor<i64>) -> tensor<128xf64>
    return %output : tensor<128xf64>
  }
}"""
    mlir_module_replaced = stablehlo.serialize_portable_artifact_str(custom_stablehlo, stablehlo.get_current_version())
    replaced_function = dataclasses.replace(exported, mlir_module_serialized=mlir_module_replaced)

    input = jax.numpy.zeros(input_shape, dtype=input_dtype)
    input = jax.device_put(input, input_sharding)

    with pytest.raises(XlaRuntimeError) as e:
        replaced_function.call(input)
    assert e.value.args[0] == (
        'INVALID_ARGUMENT: during context [hlo verifier]: '
        'Binary op compare with different element types: '
        's64[] and s32[].\n\t, for instruction '
        '%compare.1 = pred[] compare(%constant.4, %multiply), '
        'direction=GE, metadata={source_file="-" source_line=7}\n\n'
        'Failed after spmd-partitioning')
