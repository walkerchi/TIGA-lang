#include <Python.h>

#include "graphforge/Dialect/Control/ControlDialect.h"
#include "graphforge/Dialect/Domain/DomainDialect.h"
#include "graphforge/Dialect/Iter/IterDialect.h"
#include "graphforge/Dialect/Kernel/KernelDialect.h"
#include "graphforge/Dialect/Storage/StorageDialect.h"
#include "graphforge/Dialect/Task/TaskDialect.h"
#include "graphforge/Dialect/Tensor/TensorDialect.h"
#include "graphforge/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <optional>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace mlir;
namespace gf = mlir::graphforge;
namespace gfc = mlir::graphforge::control;
namespace gft = mlir::graphforge::tensor;

struct PyOwned {
  PyObject *value = nullptr;
  explicit PyOwned(PyObject *value = nullptr) : value(value) {}
  ~PyOwned() { Py_XDECREF(value); }
  PyOwned(const PyOwned &) = delete;
  PyOwned &operator=(const PyOwned &) = delete;
  operator bool() const { return value != nullptr; }
};

static PyObject *attribute(PyObject *object, const char *name) {
  return PyObject_GetAttrString(object, name);
}

static FailureOr<int64_t> integer(PyObject *value) {
  long long result = PyLong_AsLongLong(value);
  if (PyErr_Occurred())
    return failure();
  return static_cast<int64_t>(result);
}

static FailureOr<SmallVector<int64_t>> integerSequence(PyObject *value) {
  PyOwned sequence(PySequence_Fast(value, "expected an integer sequence"));
  if (!sequence)
    return failure();
  SmallVector<int64_t> result;
  Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence.value);
  result.reserve(size);
  for (Py_ssize_t index = 0; index < size; ++index) {
    FailureOr<int64_t> item =
        integer(PySequence_Fast_GET_ITEM(sequence.value, index));
    if (failed(item))
      return failure();
    result.push_back(*item);
  }
  return result;
}

static FailureOr<std::string> stringAttribute(PyObject *object,
                                               const char *name) {
  PyOwned value(attribute(object, name));
  if (!value)
    return failure();
  const char *text = PyUnicode_AsUTF8(value.value);
  if (!text)
    return failure();
  return std::string(text);
}

static FailureOr<Type> elementType(MLIRContext &context, PyObject *tensor) {
  PyOwned dtype(attribute(tensor, "dtype"));
  if (!dtype)
    return failure();
  FailureOr<std::string> name = stringAttribute(dtype.value, "name");
  if (failed(name))
    return failure();
  if (*name == "float16") return Float16Type::get(&context);
  if (*name == "float32") return Float32Type::get(&context);
  if (*name == "float64") return Float64Type::get(&context);
  if (*name == "int32") return IntegerType::get(&context, 32);
  if (*name == "int64") return IntegerType::get(&context, 64);
  if (*name == "bool") return IntegerType::get(&context, 1);
  if (*name == "complex64")
    return ComplexType::get(Float32Type::get(&context));
  if (*name == "complex128")
    return ComplexType::get(Float64Type::get(&context));
  PyErr_Format(PyExc_TypeError, "unsupported GraphForge dtype %s", name->c_str());
  return failure();
}

static FailureOr<Type> elementType(MLIRContext &context, StringRef name) {
  if (name == "float16") return Float16Type::get(&context);
  if (name == "float32") return Float32Type::get(&context);
  if (name == "float64") return Float64Type::get(&context);
  if (name == "int32") return IntegerType::get(&context, 32);
  if (name == "int64") return IntegerType::get(&context, 64);
  if (name == "bool") return IntegerType::get(&context, 1);
  if (name == "complex64")
    return ComplexType::get(Float32Type::get(&context));
  if (name == "complex128")
    return ComplexType::get(Float64Type::get(&context));
  PyErr_Format(PyExc_TypeError, "unsupported GraphForge dtype %s",
               name.str().c_str());
  return failure();
}

static FailureOr<Type> valueType(MLIRContext &context, StringRef spelling) {
  if (!spelling.starts_with("vector:"))
    return elementType(context, spelling);
  SmallVector<StringRef> parts;
  spelling.split(parts, ':');
  if (parts.size() != 3) {
    PyErr_Format(PyExc_TypeError, "invalid vector type %s",
                 spelling.str().c_str());
    return failure();
  }
  int64_t width = 0;
  if (parts[1].getAsInteger(10, width) || width <= 0) {
    PyErr_Format(PyExc_TypeError, "invalid vector width in %s",
                 spelling.str().c_str());
    return failure();
  }
  FailureOr<Type> element = elementType(context, parts[2]);
  if (failed(element)) return failure();
  return VectorType::get({width}, *element);
}

static FailureOr<SmallVector<Type>> typeSequence(MLIRContext &context,
                                                  PyObject *value) {
  PyOwned sequence(PySequence_Fast(value, "expected a type sequence"));
  if (!sequence) return failure();
  SmallVector<Type> result;
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(sequence.value);
       ++index) {
    const char *spelling = PyUnicode_AsUTF8(
        PySequence_Fast_GET_ITEM(sequence.value, index));
    if (!spelling) return failure();
    FailureOr<Type> type = valueType(context, spelling);
    if (failed(type)) return failure();
    result.push_back(*type);
  }
  return result;
}

static FailureOr<RankedTensorType> tensorType(MLIRContext &context,
                                               PyObject *tensor) {
  PyOwned shape(attribute(tensor, "shape"));
  if (!shape)
    return failure();
  FailureOr<SmallVector<int64_t>> dimensions = integerSequence(shape.value);
  FailureOr<Type> element = elementType(context, tensor);
  if (failed(dimensions) || failed(element))
    return failure();
  return RankedTensorType::get(*dimensions, *element);
}

static PyObject *identity(PyObject *, PyObject *object) {
  return PyLong_FromVoidPtr(object);
}

struct TensorBuilder {
  MLIRContext &context;
  OpBuilder builder;
  ModuleOp module;
  func::FuncOp function;
  DenseMap<void *, Value> values;
  DenseMap<void *, unsigned> arguments;

  explicit TensorBuilder(MLIRContext &context)
      : context(context), builder(&context),
        module(ModuleOp::create(builder.getUnknownLoc())) {}

  FailureOr<Value> build(PyObject *tensor) {
    auto found = values.find(tensor);
    if (found != values.end())
      return found->second;
    FailureOr<RankedTensorType> type = tensorType(context, tensor);
    if (failed(type))
      return failure();
    PyOwned expression(attribute(tensor, "_expr"));
    if (!expression)
      return failure();
    if (expression.value == Py_None) {
      unsigned index = arguments.size();
      arguments[tensor] = index;
      // Input ops are created after the function signature is known.
      values[tensor] = Value();
      return Value();
    }
    FailureOr<std::string> operation = stringAttribute(expression.value, "op");
    PyOwned operandsObject(attribute(expression.value, "operands"));
    if (failed(operation) || !operandsObject)
      return failure();
    PyOwned operands(PySequence_Fast(operandsObject.value, "expected operands"));
    if (!operands)
      return failure();
    SmallVector<Value> inputs;
    SmallVector<PyObject *> operandObjects;
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(operands.value);
         ++index) {
      PyObject *operand = PySequence_Fast_GET_ITEM(operands.value, index);
      operandObjects.push_back(operand);
      FailureOr<Value> input = build(operand);
      if (failed(input))
        return failure();
      inputs.push_back(*input);
    }
    // Defer actual op construction until leaf arguments have been rebound.
    values[tensor] = Value();
    return Value();
  }

  FailureOr<Value> emit(PyObject *tensor) {
    Value &cached = values[tensor];
    if (cached)
      return cached;
    FailureOr<RankedTensorType> resultType = tensorType(context, tensor);
    if (failed(resultType))
      return failure();
    PyOwned expression(attribute(tensor, "_expr"));
    if (!expression)
      return failure();
    Location location = builder.getUnknownLoc();
    if (expression.value == Py_None) {
      Value argument = function.getArgument(arguments.lookup(tensor));
      PyOwned offsetObject(attribute(tensor, "offset"));
      PyOwned stridesObject(attribute(tensor, "strides"));
      if (!offsetObject || !stridesObject)
        return failure();
      FailureOr<int64_t> offset = integer(offsetObject.value);
      FailureOr<SmallVector<int64_t>> strides = integerSequence(stridesObject.value);
      if (failed(offset) || failed(strides))
        return failure();
      cached = builder.create<gft::InputOp>(
          location, *resultType, argument, builder.getI64IntegerAttr(*offset),
          builder.getDenseI64ArrayAttr(*strides));
      return cached;
    }
    FailureOr<std::string> operation = stringAttribute(expression.value, "op");
    PyOwned operandsObject(attribute(expression.value, "operands"));
    if (failed(operation) || !operandsObject)
      return failure();
    PyOwned operands(PySequence_Fast(operandsObject.value, "expected operands"));
    SmallVector<Value> inputs;
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(operands.value);
         ++index) {
      FailureOr<Value> input =
          emit(PySequence_Fast_GET_ITEM(operands.value, index));
      if (failed(input))
        return failure();
      inputs.push_back(*input);
    }
    if (*operation == "repeat") {
      PyOwned attrs(attribute(expression.value, "attrs"));
      PyOwned regionObject(attribute(expression.value, "region"));
      if (!attrs || !regionObject || regionObject.value == Py_None)
        return failure();
      PyOwned attrsSequence(PySequence_Fast(attrs.value, "expected attrs"));
      PyObject *iterationsObject = nullptr;
      for (Py_ssize_t index = 0;
           index < PySequence_Fast_GET_SIZE(attrsSequence.value); ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(attrsSequence.value, index);
        PyObject *name = PyTuple_GetItem(pair, 0);
        if (name && PyUnicode_CompareWithASCIIString(name, "iterations") == 0)
          iterationsObject = PyTuple_GetItem(pair, 1);
      }
      FailureOr<int64_t> iterations = iterationsObject
          ? integer(iterationsObject) : FailureOr<int64_t>(failure());
      PyOwned argumentsObject(attribute(regionObject.value, "arguments"));
      PyOwned outputObject(attribute(regionObject.value, "output"));
      PyOwned arguments(argumentsObject
          ? PySequence_Fast(argumentsObject.value, "expected region arguments")
          : nullptr);
      if (failed(iterations) || !arguments || !outputObject ||
          PySequence_Fast_GET_SIZE(arguments.value) !=
              static_cast<Py_ssize_t>(inputs.size()))
        return failure();

      OperationState state(location, gfc::RepeatOp::getOperationName());
      state.addOperands(inputs);
      state.addTypes(*resultType);
      state.addAttribute("iterations", builder.getI64IntegerAttr(*iterations));
      state.addRegion();
      auto repeat = cast<gfc::RepeatOp>(builder.create(state));
      Block *body = new Block();
      repeat.getBody().push_back(body);
      for (Value input : inputs)
        body->addArgument(input.getType(), location);

      struct SavedValue {
        void *key;
        Value value;
        bool existed;
      };
      SmallVector<SavedValue> saved;
      saved.reserve(inputs.size());
      for (auto [index, argument] : llvm::enumerate(body->getArguments())) {
        PyObject *object = PySequence_Fast_GET_ITEM(arguments.value, index);
        auto found = values.find(object);
        saved.push_back({object, found == values.end() ? Value() : found->second,
                         found != values.end()});
        values[object] = argument;
      }
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(body);
        FailureOr<Value> bodyResult = emit(outputObject.value);
        if (failed(bodyResult)) return failure();
        builder.create<gfc::ControlYieldOp>(location, *bodyResult);
      }
      for (const SavedValue &item : saved) {
        if (item.existed) values[item.key] = item.value;
        else values.erase(item.key);
      }
      cached = repeat.getResult();
    }
    else if (*operation == "loop_argument") {
      PyErr_SetString(PyExc_RuntimeError,
                      "loop argument escaped gf_control.repeat capture");
      return failure();
    }
    else if (*operation == "add")
      cached = builder.create<gft::AddOp>(location, *resultType, inputs[0], inputs[1]);
    else if (*operation == "mul")
      cached = builder.create<gft::MulOp>(location, *resultType, inputs[0], inputs[1]);
    else if (*operation == "matmul")
      cached = builder.create<gft::MatmulOp>(
          location, *resultType, inputs[0], inputs[1]);
    else if (*operation == "div")
      cached = builder.create<gft::DivOp>(location, *resultType, inputs[0], inputs[1]);
    else if (*operation == "neg")
      cached = builder.create<gft::NegOp>(location, *resultType, inputs[0]);
    else if (*operation == "exp")
      cached = builder.create<gft::ExpOp>(location, *resultType, inputs[0]);
    else if (*operation == "sqrt")
      cached = builder.create<gft::SqrtOp>(location, *resultType, inputs[0]);
    else if (*operation == "cumsum") {
      PyOwned attrs(attribute(expression.value, "attrs"));
      if (!attrs) return failure();
      PyOwned sequence(PySequence_Fast(attrs.value, "expected attrs"));
      if (!sequence) return failure();
      PyObject *axisObject = nullptr;
      PyObject *reverseObject = nullptr;
      for (Py_ssize_t index = 0;
           index < PySequence_Fast_GET_SIZE(sequence.value); ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(sequence.value, index);
        PyObject *name = PyTuple_GetItem(pair, 0);
        if (name && PyUnicode_CompareWithASCIIString(name, "axis") == 0)
          axisObject = PyTuple_GetItem(pair, 1);
        if (name && PyUnicode_CompareWithASCIIString(name, "reverse") == 0)
          reverseObject = PyTuple_GetItem(pair, 1);
      }
      FailureOr<int64_t> axis = integer(axisObject);
      if (failed(axis) || !reverseObject) return failure();
      int reverse = PyObject_IsTrue(reverseObject);
      if (reverse < 0) return failure();
      cached = builder.create<gft::CumsumOp>(
          location, *resultType, inputs[0], builder.getI64IntegerAttr(*axis),
          builder.getBoolAttr(reverse != 0));
    }
    else if (*operation == "conj")
      cached = builder.create<gft::ConjOp>(location, *resultType, inputs[0]);
    else if (*operation == "checkpoint")
      cached = builder.create<gft::CheckpointOp>(
          location, *resultType, inputs[0]);
    else if (*operation == "checkpoint_candidate")
      cached = builder.create<gft::CheckpointCandidateOp>(
          location, *resultType, inputs[0]);
    else if (*operation == "gather")
      cached = builder.create<gft::GatherOp>(
          location, *resultType, inputs[0], inputs[1]);
    else if (*operation == "scatter_rows") {
      PyOwned attrs(attribute(expression.value, "attrs"));
      if (!attrs) return failure();
      PyOwned sequence(PySequence_Fast(attrs.value, "expected attrs"));
      if (!sequence) return failure();
      PyObject *count = nullptr;
      for (Py_ssize_t index = 0;
           index < PySequence_Fast_GET_SIZE(sequence.value); ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(sequence.value, index);
        PyObject *name = PyTuple_GetItem(pair, 0);
        if (name && PyUnicode_CompareWithASCIIString(name, "num_rows") == 0)
          count = PyTuple_GetItem(pair, 1);
      }
      FailureOr<int64_t> rows = count
          ? integer(count) : FailureOr<int64_t>(failure());
      if (failed(rows) || inputs.size() != 3) return failure();
      cached = builder.create<gft::ScatterRowsOp>(
          location, *resultType, inputs[0], inputs[1], inputs[2],
          builder.getI64IntegerAttr(*rows));
    }
    else if (*operation == "csr_euclidean_distance_sum_vjp") {
      PyOwned attrs(attribute(expression.value, "attrs"));
      if (!attrs) return failure();
      PyOwned sequence(PySequence_Fast(attrs.value, "expected attrs"));
      if (!sequence) return failure();
      PyObject *dimensions = nullptr;
      PyObject *periodic = nullptr;
      for (Py_ssize_t index = 0;
           index < PySequence_Fast_GET_SIZE(sequence.value); ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(sequence.value, index);
        PyObject *name = PyTuple_GetItem(pair, 0);
        if (name && PyUnicode_CompareWithASCIIString(name, "dimensions") == 0)
          dimensions = PyTuple_GetItem(pair, 1);
        if (name && PyUnicode_CompareWithASCIIString(name, "periodic") == 0)
          periodic = PyTuple_GetItem(pair, 1);
      }
      FailureOr<int64_t> dimensionValue = dimensions
          ? integer(dimensions) : FailureOr<int64_t>(failure());
      int periodicValue = periodic ? PyObject_IsTrue(periodic) : -1;
      if (failed(dimensionValue) || periodicValue < 0 || inputs.size() != 7)
        return failure();
      cached = builder.create<gft::CSREuclideanDistanceSumVJPOp>(
          location, *resultType, inputs[0], inputs[1], inputs[2], inputs[3],
          inputs[4], inputs[5], inputs[6],
          builder.getI64IntegerAttr(*dimensionValue),
          builder.getBoolAttr(periodicValue));
    }
    else if (*operation == "csr_expand_rows" ||
             *operation == "csr_segment_sum" ||
             *operation == "csr_segment_product" ||
             *operation == "csr_segment_product_vjp" ||
             *operation == "csr_segment_max_stop_gradient") {
      PyOwned attrs(attribute(expression.value, "attrs"));
      if (!attrs) return failure();
      PyOwned sequence(PySequence_Fast(attrs.value, "expected attrs"));
      if (!sequence) return failure();
      const char *key = *operation == "csr_expand_rows" ? "num_edges" : "num_rows";
      PyObject *count = nullptr;
      PyObject *degreeMin = nullptr;
      PyObject *degree = nullptr;
      PyObject *uniform = nullptr;
      for (Py_ssize_t index = 0;
           index < PySequence_Fast_GET_SIZE(sequence.value); ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(sequence.value, index);
        PyObject *name = PyTuple_GetItem(pair, 0);
        if (name && PyUnicode_CompareWithASCIIString(name, key) == 0) {
          count = PyTuple_GetItem(pair, 1);
        }
        if (name && PyUnicode_CompareWithASCIIString(name, "max_degree") == 0)
          degree = PyTuple_GetItem(pair, 1);
        if (name && PyUnicode_CompareWithASCIIString(name, "degree_min") == 0)
          degreeMin = PyTuple_GetItem(pair, 1);
        if (name && PyUnicode_CompareWithASCIIString(name, "degree_max") == 0)
          degree = PyTuple_GetItem(pair, 1);
        if (name && PyUnicode_CompareWithASCIIString(name, "uniform_degree") == 0)
          uniform = PyTuple_GetItem(pair, 1);
      }
      FailureOr<int64_t> value = count
          ? integer(count) : FailureOr<int64_t>(failure());
      if (failed(value)) return failure();
      if (*operation == "csr_expand_rows")
        cached = builder.create<gft::CSRExpandRowsOp>(
            location, *resultType, inputs[0], inputs[1],
            builder.getI64IntegerAttr(*value));
      else if (*operation == "csr_segment_sum") {
        FailureOr<int64_t> minimum = degreeMin
            ? integer(degreeMin) : FailureOr<int64_t>(failure());
        FailureOr<int64_t> maximum = degree
            ? integer(degree) : FailureOr<int64_t>(failure());
        if (failed(minimum) || failed(maximum)) return failure();
        cached = builder.create<gft::CSRSegmentSumOp>(
            location, *resultType, inputs[0], inputs[1],
            builder.getI64IntegerAttr(*value),
            builder.getI64IntegerAttr(*minimum),
            builder.getI64IntegerAttr(*maximum));
      }
      else if (*operation == "csr_segment_product" ||
               *operation == "csr_segment_product_vjp") {
        if (!degree || !uniform) return failure();
        FailureOr<int64_t> maximum = integer(degree);
        if (failed(maximum)) return failure();
        int uniformTruth = PyObject_IsTrue(uniform);
        if (uniformTruth < 0) return failure();
        if (*operation == "csr_segment_product")
          cached = builder.create<gft::CSRSegmentProductOp>(
              location, *resultType, inputs[0], inputs[1], inputs[2],
              builder.getI64IntegerAttr(*value),
              builder.getI64IntegerAttr(*maximum),
              builder.getBoolAttr(uniformTruth));
        else
          cached = builder.create<gft::CSRSegmentProductVJPOp>(
              location, *resultType, inputs[0], inputs[1], inputs[2], inputs[3],
              builder.getI64IntegerAttr(*value),
              builder.getI64IntegerAttr(*maximum),
              builder.getBoolAttr(uniformTruth));
      }
      else
        cached = builder.create<gft::CSRSegmentMaxStopGradientOp>(
            location, *resultType, inputs[0], inputs[1],
            builder.getI64IntegerAttr(*value));
    }
    else if (*operation == "reshape" || *operation == "broadcast" ||
             *operation == "permute" || *operation == "sum" ||
             *operation == "segment_sum") {
      PyOwned attrs(attribute(expression.value, "attrs"));
      if (!attrs)
        return failure();
      auto lookup = [&](const char *key) -> PyOwned {
        PyOwned sequence(PySequence_Fast(attrs.value, "expected attrs"));
        if (!sequence) return PyOwned();
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(sequence.value); ++i) {
          PyObject *pair = PySequence_Fast_GET_ITEM(sequence.value, i);
          PyObject *name = PyTuple_GetItem(pair, 0);
          if (name && PyUnicode_CompareWithASCIIString(name, key) == 0) {
            PyObject *value = PyTuple_GetItem(pair, 1);
            Py_XINCREF(value);
            return PyOwned(value);
          }
        }
        PyErr_Format(PyExc_KeyError, "missing expression attribute %s", key);
        return PyOwned();
      };
      if (*operation == "reshape" || *operation == "broadcast") {
        PyOwned shape = lookup("shape");
        FailureOr<SmallVector<int64_t>> values =
            shape ? integerSequence(shape.value) : FailureOr<SmallVector<int64_t>>(failure());
        if (failed(values)) return failure();
        auto attr = builder.getDenseI64ArrayAttr(*values);
        cached = *operation == "reshape"
                     ? Value(builder.create<gft::ReshapeOp>(location, *resultType, inputs[0], attr))
                     : Value(builder.create<gft::BroadcastOp>(location, *resultType, inputs[0], attr));
      } else if (*operation == "permute") {
        PyOwned axes = lookup("axes");
        FailureOr<SmallVector<int64_t>> values =
            axes ? integerSequence(axes.value) : FailureOr<SmallVector<int64_t>>(failure());
        if (failed(values)) return failure();
        cached = builder.create<gft::PermuteOp>(
            location, *resultType, inputs[0], builder.getDenseI64ArrayAttr(*values));
      } else if (*operation == "sum") {
        PyOwned axes = lookup("axes");
        PyOwned keepdims = lookup("keepdims");
        FailureOr<SmallVector<int64_t>> values =
            axes ? integerSequence(axes.value) : FailureOr<SmallVector<int64_t>>(failure());
        if (failed(values) || !keepdims) return failure();
        int truth = PyObject_IsTrue(keepdims.value);
        if (truth < 0) return failure();
        cached = builder.create<gft::ReduceSumOp>(
            location, *resultType, inputs[0], builder.getDenseI64ArrayAttr(*values),
            builder.getBoolAttr(truth));
      } else {
        PyOwned count = lookup("num_segments");
        FailureOr<int64_t> value =
            count ? integer(count.value) : FailureOr<int64_t>(failure());
        if (failed(value)) return failure();
        cached = builder.create<gft::SegmentSumOp>(
            location, *resultType, inputs[0], inputs[1],
            builder.getI64IntegerAttr(*value));
      }
    } else {
      PyErr_Format(PyExc_NotImplementedError,
                   "native Tensor capture does not support %s", operation->c_str());
      return failure();
    }
    return cached;
  }

  FailureOr<ModuleOp> finish(PyObject *output, StringRef functionName,
                             PyObject *wrt = nullptr,
                             PyObject *cotangent = nullptr) {
    if (failed(build(output)) ||
        (cotangent && failed(build(cotangent))))
      return failure();
    SmallVector<Type> argumentTypes(arguments.size());
    for (auto [object, index] : arguments) {
      FailureOr<RankedTensorType> type = tensorType(context, (PyObject *)object);
      if (failed(type)) return failure();
      argumentTypes[index] = *type;
    }
    FailureOr<RankedTensorType> outputType = tensorType(context, output);
    FailureOr<RankedTensorType> resultType =
        wrt ? tensorType(context, wrt) : tensorType(context, output);
    if (failed(outputType) || failed(resultType)) return failure();
    builder.setInsertionPointToStart(module.getBody());
    function = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), functionName,
        builder.getFunctionType(argumentTypes, TypeRange{*resultType}));
    Block *entry = function.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    FailureOr<Value> result = emit(output);
    if (failed(result)) return failure();
    if (wrt) {
      FailureOr<Value> wrtValue = emit(wrt);
      FailureOr<Value> cotangentValue = emit(cotangent);
      if (failed(wrtValue) || failed(cotangentValue)) return failure();
      result = Value(builder.create<gft::GradOp>(
          builder.getUnknownLoc(), *resultType, *result, *wrtValue,
          *cotangentValue));
    }
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), *result);
    return module;
  }
};

static void initializeContext(MLIRContext &context) {
  DialectRegistry registry;
  registry.insert<gfc::GraphForgeControlDialect,
                  gf::GraphForgeDomainDialect, gf::iter::GraphForgeIterDialect,
                  gf::kernel::GraphForgeKernelDialect,
                  gf::storage::GraphForgeStorageDialect,
                  gf::task::GraphForgeTaskDialect,
                  gft::GraphForgeTensorDialect, arith::ArithDialect,
                  complex::ComplexDialect, func::FuncDialect,
                  math::MathDialect, memref::MemRefDialect,
                  scf::SCFDialect, vector::VectorDialect>();
  registerLLVMDialectTranslation(registry);
  registerBuiltinDialectTranslation(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  registerConvertComplexToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

static std::string printOperation(Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream.flush();
  return text;
}

static PyObject *stringTuple(ArrayAttr values) {
  PyObject *result = PyTuple_New(values.size());
  if (!result) return nullptr;
  for (auto [index, attribute] : llvm::enumerate(values)) {
    auto value = dyn_cast<StringAttr>(attribute);
    if (!value) {
      Py_DECREF(result);
      PyErr_SetString(PyExc_TypeError, "expected a verified string array");
      return nullptr;
    }
    PyObject *item = PyUnicode_FromStringAndSize(
        value.getValue().data(), value.getValue().size());
    if (!item) {
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, index, item);
  }
  return result;
}

static bool setDictionaryItem(PyObject *dictionary, const char *name,
                              PyObject *value) {
  if (!value) return false;
  int status = PyDict_SetItemString(dictionary, name, value);
  Py_DECREF(value);
  return status == 0;
}

static PyObject *kernelSchedules(PyObject *, PyObject *argument) {
  Py_ssize_t size = 0;
  const char *text = PyUnicode_AsUTF8AndSize(argument, &size);
  if (!text) return nullptr;
  MLIRContext context;
  initializeContext(context);
  OwningOpRef<ModuleOp> source =
      parseSourceString<ModuleOp>(StringRef(text, size), &context);
  if (!source || failed(verify(*source))) {
    PyErr_SetString(PyExc_ValueError,
                    "kernel schedule inspection requires verified MLIR");
    return nullptr;
  }
  SmallVector<Operation *> launches;
  source->walk([&](Operation *operation) {
    if (isa<gf::kernel::LaunchOp, gf::kernel::GeneratedLaunchOp,
            gf::kernel::DenseLaunchOp>(operation) &&
        operation->hasAttr("schedule_kind"))
      launches.push_back(operation);
  });
  PyObject *result = PyTuple_New(launches.size());
  if (!result) return nullptr;
  for (auto [index, launch] : llvm::enumerate(launches)) {
    PyObject *entry = PyDict_New();
    if (!entry) {
      Py_DECREF(result);
      return nullptr;
    }
    auto string = [&](StringRef name) {
      return launch->getAttrOfType<StringAttr>(name).getValue();
    };
    auto integer = [&](StringRef name) {
      return launch->getAttrOfType<IntegerAttr>(name).getInt();
    };
    std::string location;
    llvm::raw_string_ostream locationStream(location);
    launch->getLoc().print(locationStream);
    locationStream.flush();
    bool ok =
        setDictionaryItem(entry, "operation", PyUnicode_FromString(
            launch->getName().getStringRef().str().c_str())) &&
        setDictionaryItem(entry, "source_location",
                          PyUnicode_FromString(location.c_str())) &&
        setDictionaryItem(entry, "kind", PyUnicode_FromString(
            string("schedule_kind").str().c_str())) &&
        setDictionaryItem(entry, "block_rows",
                          PyLong_FromLongLong(integer("block_rows"))) &&
        setDictionaryItem(entry, "block_neighbors",
                          PyLong_FromLongLong(integer("block_neighbors"))) &&
        setDictionaryItem(entry, "num_warps",
                          PyLong_FromLongLong(integer("num_warps"))) &&
        setDictionaryItem(entry, "pipeline_stages",
                          PyLong_FromLongLong(integer("pipeline_stages"))) &&
        setDictionaryItem(entry, "target_contract", PyUnicode_FromString(
            string("target_contract").str().c_str())) &&
        setDictionaryItem(entry, "resources", stringTuple(
            launch->getAttrOfType<ArrayAttr>("schedule_resources"))) &&
        setDictionaryItem(entry, "roles", stringTuple(
            launch->getAttrOfType<ArrayAttr>("execution_roles"))) &&
        setDictionaryItem(entry, "handoffs", stringTuple(
            launch->getAttrOfType<ArrayAttr>("schedule_handoffs"))) &&
        setDictionaryItem(entry, "instructions", stringTuple(
            launch->getAttrOfType<ArrayAttr>("instruction_contracts")));
    if (!ok) {
      Py_DECREF(entry);
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(result, index, entry);
  }
  return result;
}

static PyObject *tensorIR(PyObject *, PyObject *args, PyObject *keywords) {
  PyObject *output = nullptr;
  const char *functionName = "tensor_main";
  PyObject *wrt = Py_None;
  PyObject *cotangent = Py_None;
  int vjp = 0;
  static const char *names[] = {"output", "function_name", "wrt",
                                "cotangent", "run_vjp", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|sOOp",
                                   const_cast<char **>(names), &output,
                                   &functionName, &wrt, &cotangent, &vjp))
    return nullptr;
  if ((wrt == Py_None) != (cotangent == Py_None)) {
    PyErr_SetString(PyExc_ValueError,
                    "wrt and cotangent must be supplied together");
    return nullptr;
  }
  MLIRContext context;
  initializeContext(context);
  TensorBuilder capture(context);
  FailureOr<ModuleOp> module = capture.finish(
      output, functionName, wrt == Py_None ? nullptr : wrt,
      cotangent == Py_None ? nullptr : cotangent);
  if (failed(module))
    return nullptr;
  if (failed(verify(*module))) {
    PyErr_SetString(PyExc_RuntimeError, "native GraphForge IR verification failed");
    return nullptr;
  }
  if (vjp) {
    PassManager manager(&context);
    manager.addPass(gf::createGFTensorVJP());
    manager.addPass(createCSEPass());
    if (failed(manager.run(*module))) {
      PyErr_SetString(PyExc_RuntimeError, "GraphForge VJP pipeline failed");
      return nullptr;
    }
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  module->print(stream, OpPrintingFlags().enableDebugInfo(false));
  stream.flush();
  return PyUnicode_FromStringAndSize(text.data(), text.size());
}

static FailureOr<Value> emitReducerExpr(PyObject *expression, Block &block,
                                        OpBuilder &builder) {
  if (PyLong_Check(expression) || PyFloat_Check(expression)) {
    double value = PyFloat_AsDouble(expression);
    if (PyErr_Occurred()) return failure();
    return Value(builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getF32FloatAttr(value)));
  }
  FailureOr<std::string> operation = stringAttribute(expression, "op");
  PyOwned argumentsObject(attribute(expression, "args"));
  if (failed(operation) || !argumentsObject) return failure();
  PyOwned arguments(PySequence_Fast(argumentsObject.value, "expected Expr args"));
  if (!arguments) return failure();
  auto argument = [&](Py_ssize_t index) -> PyObject * {
    return PySequence_Fast_GET_ITEM(arguments.value, index);
  };
  Location location = builder.getUnknownLoc();
  if (*operation == "reducer_arg") {
    FailureOr<int64_t> index = integer(argument(0));
    if (failed(index) || *index < 0 || *index >= block.getNumArguments()) {
      PyErr_SetString(PyExc_IndexError, "reducer block argument is out of range");
      return failure();
    }
    return block.getArgument(*index);
  }
  if (*operation == "typed_constant") {
    double value = PyFloat_AsDouble(argument(0));
    const char *spelling = PyUnicode_AsUTF8(argument(1));
    if (PyErr_Occurred() || !spelling) return failure();
    FailureOr<Type> type = valueType(*builder.getContext(), spelling);
    if (failed(type)) return failure();
    TypedAttr constant;
    if (auto vector = dyn_cast<VectorType>(*type)) {
      auto element = dyn_cast<FloatType>(vector.getElementType());
      if (!element) {
        PyErr_SetString(PyExc_TypeError,
                        "typed reducer vector constants must be floating point");
        return failure();
      }
      constant = DenseElementsAttr::get(
          vector, builder.getFloatAttr(element, value));
    } else if (auto scalar = dyn_cast<FloatType>(*type)) {
      constant = builder.getFloatAttr(scalar, value);
    } else {
      PyErr_SetString(PyExc_TypeError,
                      "typed reducer constants must be floating point");
      return failure();
    }
    return Value(builder.create<arith::ConstantOp>(location, *type, constant));
  }
  if (*operation == "constant")
    return emitReducerExpr(argument(0), block, builder);
  if (*operation == "neg") {
    FailureOr<Value> operand = emitReducerExpr(argument(0), block, builder);
    if (failed(operand)) return failure();
    Value zero = builder.create<arith::ConstantOp>(
        location, builder.getF32FloatAttr(0.0));
    return Value(builder.create<arith::SubFOp>(location, zero, *operand));
  }
  if (*operation == "exp") {
    FailureOr<Value> operand = emitReducerExpr(argument(0), block, builder);
    if (failed(operand)) return failure();
    return Value(builder.create<math::ExpOp>(location, *operand));
  }
  if (*operation == "cast") {
    FailureOr<Value> operand = emitReducerExpr(argument(0), block, builder);
    const char *spelling = PyUnicode_AsUTF8(argument(1));
    FailureOr<Type> target = spelling
                                 ? valueType(*builder.getContext(), spelling)
                                 : FailureOr<Type>(failure());
    if (failed(operand) || failed(target)) return failure();
    auto sourceElement = getElementTypeOrSelf((*operand).getType());
    auto targetElement = getElementTypeOrSelf(*target);
    auto sourceFloat = dyn_cast<FloatType>(sourceElement);
    auto targetFloat = dyn_cast<FloatType>(targetElement);
    if (!sourceFloat || !targetFloat) {
      PyErr_SetString(PyExc_TypeError, "reducer cast requires floating types");
      return failure();
    }
    if (sourceFloat.getWidth() < targetFloat.getWidth())
      return Value(builder.create<arith::ExtFOp>(location, *target, *operand));
    if (sourceFloat.getWidth() > targetFloat.getWidth())
      return Value(builder.create<arith::TruncFOp>(location, *target, *operand));
    return *operand;
  }
  if (*operation == "broadcast") {
    FailureOr<Value> operand = emitReducerExpr(argument(0), block, builder);
    const char *spelling = PyUnicode_AsUTF8(argument(1));
    FailureOr<Type> target = spelling
                                 ? valueType(*builder.getContext(), spelling)
                                 : FailureOr<Type>(failure());
    auto vector = succeeded(target) ? dyn_cast<VectorType>(*target) : VectorType();
    if (failed(operand) || !vector) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError,
                        "reducer broadcast target must be a vector");
      return failure();
    }
    return Value(builder.create<vector::BroadcastOp>(location, vector, *operand));
  }
  FailureOr<Value> left = emitReducerExpr(argument(0), block, builder);
  FailureOr<Value> right = emitReducerExpr(argument(1), block, builder);
  if (failed(left) || failed(right)) return failure();
  if (*operation == "add")
    return Value(builder.create<arith::AddFOp>(location, *left, *right));
  if (*operation == "sub")
    return Value(builder.create<arith::SubFOp>(location, *left, *right));
  if (*operation == "mul")
    return Value(builder.create<arith::MulFOp>(location, *left, *right));
  if (*operation == "div")
    return Value(builder.create<arith::DivFOp>(location, *left, *right));
  if (*operation == "maximum")
    return Value(builder.create<arith::MaximumFOp>(location, *left, *right));
  PyErr_Format(PyExc_NotImplementedError,
               "native reducer capture does not support %s", operation->c_str());
  return failure();
}

static LogicalResult fillReducerRegion(Region &region, PyObject *descriptor,
                                       ArrayRef<Type> argumentTypes,
                                       ArrayRef<Type> resultTypes,
                                       OpBuilder &builder) {
  PyOwned argumentCountObject(attribute(descriptor, "arguments"));
  PyOwned resultsObject(attribute(descriptor, "results"));
  if (!argumentCountObject || !resultsObject) return failure();
  FailureOr<int64_t> argumentCount = integer(argumentCountObject.value);
  PyOwned results(PySequence_Fast(resultsObject.value, "expected reducer results"));
  if (failed(argumentCount) || !results) return failure();
  if (PySequence_Fast_GET_SIZE(results.value) !=
      static_cast<Py_ssize_t>(resultTypes.size())) {
    PyErr_SetString(PyExc_TypeError,
                    "reducer region result count/type mismatch");
    return failure();
  }
  auto block = std::make_unique<Block>();
  if (*argumentCount != static_cast<int64_t>(argumentTypes.size())) {
    PyErr_SetString(PyExc_TypeError,
                    "reducer descriptor argument count/type mismatch");
    return failure();
  }
  for (Type type : argumentTypes)
    block->addArgument(type, builder.getUnknownLoc());
  Block *body = block.get();
  region.push_back(block.release());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  SmallVector<Value> values;
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(results.value);
       ++index) {
    FailureOr<Value> value = emitReducerExpr(
        PySequence_Fast_GET_ITEM(results.value, index), *body, builder);
    if (failed(value)) return failure();
    Type expected = resultTypes[index];
    if ((*value).getType() != expected) {
      auto vector = dyn_cast<VectorType>(expected);
      if (!vector || (*value).getType() != vector.getElementType()) {
        PyErr_SetString(PyExc_TypeError,
                        "reducer region result does not match its declared type");
        return failure();
      }
      value = Value(builder.create<vector::BroadcastOp>(
          builder.getUnknownLoc(), vector, *value));
    }
    values.push_back(*value);
  }
  builder.create<gf::ReducerYieldOp>(builder.getUnknownLoc(), values);
  return success();
}

static FailureOr<gf::ReducerOp> buildReducer(PyObject *descriptor,
                                             OpBuilder &builder) {
  FailureOr<std::string> symbol = stringAttribute(descriptor, "symbol");
  FailureOr<std::string> name = stringAttribute(descriptor, "name");
  FailureOr<std::string> kind = stringAttribute(descriptor, "kind");
  PyOwned associativeObject(attribute(descriptor, "associative"));
  PyOwned commutativeObject(attribute(descriptor, "commutative"));
  PyOwned identity(attribute(descriptor, "identity"));
  PyOwned lift(attribute(descriptor, "lift"));
  PyOwned combine(attribute(descriptor, "combine"));
  PyOwned finalize(attribute(descriptor, "finalize"));
  PyOwned messageDtypes(attribute(descriptor, "message_dtypes"));
  PyOwned stateDtypes(attribute(descriptor, "state_dtypes"));
  PyOwned resultDtypes(attribute(descriptor, "result_dtypes"));
  PyOwned blockPruneThreshold(attribute(descriptor, "block_prune_threshold"));
  if (failed(symbol) || failed(name) || failed(kind) || !associativeObject ||
      !commutativeObject || !identity || !lift || !combine || !finalize)
    return failure();
  if (!messageDtypes || !stateDtypes || !resultDtypes) return failure();
  int associative = PyObject_IsTrue(associativeObject.value);
  int commutative = PyObject_IsTrue(commutativeObject.value);
  if (associative < 0 || commutative < 0) return failure();
  FailureOr<SmallVector<Type>> messageTypes =
      typeSequence(*builder.getContext(), messageDtypes.value);
  FailureOr<SmallVector<Type>> stateTypes =
      typeSequence(*builder.getContext(), stateDtypes.value);
  FailureOr<SmallVector<Type>> resultTypes =
      typeSequence(*builder.getContext(), resultDtypes.value);
  if (failed(messageTypes) || failed(stateTypes) || failed(resultTypes) ||
      messageTypes->empty() || stateTypes->empty() || resultTypes->empty())
    return failure();
  auto types = [&](ArrayRef<Type> sequence) {
    SmallVector<Attribute> values;
    for (Type type : sequence) values.push_back(TypeAttr::get(type));
    return builder.getArrayAttr(values);
  };
  gf::ReducerOp operation = builder.create<gf::ReducerOp>(
      builder.getUnknownLoc(), *symbol, *kind, types(*messageTypes),
      types(*stateTypes), types(*resultTypes), associative != 0,
      commutative != 0);
  if (blockPruneThreshold && blockPruneThreshold.value != Py_None) {
    double threshold = PyFloat_AsDouble(blockPruneThreshold.value);
    if (PyErr_Occurred()) return failure();
    operation->setAttr("block_prune_threshold",
                       builder.getF64FloatAttr(threshold));
  }
  SmallVector<Type> combineTypes(*stateTypes);
  llvm::append_range(combineTypes, *stateTypes);
  if (failed(fillReducerRegion(operation.getIdentity(), identity.value, {},
                               *stateTypes,
                               builder)) ||
      failed(fillReducerRegion(operation.getLift(), lift.value, *messageTypes,
                               *stateTypes,
                               builder)) ||
      failed(fillReducerRegion(operation.getCombine(), combine.value,
                               combineTypes, *stateTypes, builder)) ||
      failed(fillReducerRegion(operation.getFinalize(), finalize.value,
                               *stateTypes, *resultTypes, builder)))
    return failure();
  return operation;
}

static PyObject *reducerIR(PyObject *, PyObject *descriptor) {
  MLIRContext context;
  initializeContext(context);
  OpBuilder builder(&context);
  Location location = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());
  if (failed(buildReducer(descriptor, builder))) return nullptr;
  if (failed(verify(module))) {
    PyErr_SetString(PyExc_RuntimeError, "native reducer IR verification failed");
    return nullptr;
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return PyUnicode_FromStringAndSize(text.data(), text.size());
}

static FailureOr<Value> emitDomainExpr(
    PyObject *expression, Block &block,
    ArrayRef<std::pair<std::string, std::string>> bindings,
    OpBuilder &builder) {
  FailureOr<std::string> operation = stringAttribute(expression, "op");
  PyOwned argumentsObject(attribute(expression, "args"));
  if (failed(operation) || !argumentsObject) return failure();
  PyOwned arguments(PySequence_Fast(argumentsObject.value, "expected Expr args"));
  if (!arguments) return failure();
  auto argument = [&](Py_ssize_t index) -> PyObject * {
    return PySequence_Fast_GET_ITEM(arguments.value, index);
  };
  Location location = builder.getUnknownLoc();
  if (*operation == "field" || *operation == "param") {
    const char *role = *operation == "param" ? "param"
                                               : PyUnicode_AsUTF8(argument(0));
    const char *name = PyUnicode_AsUTF8(
        *operation == "param" ? argument(0) : argument(1));
    if (!role || !name) return failure();
    auto found = llvm::find(bindings, std::make_pair(std::string(role),
                                                    std::string(name)));
    if (found == bindings.end()) {
      PyErr_Format(PyExc_KeyError, "unbound captured field %s.%s", role, name);
      return failure();
    }
    return block.getArgument(std::distance(bindings.begin(), found));
  }
  if (*operation == "aggregate") {
    auto found = llvm::find(
        bindings, std::make_pair(std::string("aggregate"), std::string("0")));
    if (found == bindings.end()) {
      PyErr_SetString(PyExc_KeyError,
                      "aggregate is only available in a node region");
      return failure();
    }
    return block.getArgument(std::distance(bindings.begin(), found));
  }
  if (*operation == "constant") {
    double value = PyFloat_AsDouble(argument(0));
    if (PyErr_Occurred()) return failure();
    return Value(builder.create<arith::ConstantOp>(
        location, builder.getF32FloatAttr(value)));
  }
  if (*operation == "neg") {
    FailureOr<Value> value = emitDomainExpr(argument(0), block, bindings, builder);
    if (failed(value)) return failure();
    Value zero = builder.create<arith::ConstantOp>(
        location, builder.getF32FloatAttr(0.0));
    return Value(builder.create<arith::SubFOp>(location, zero, *value));
  }
  if (*operation == "exp") {
    FailureOr<Value> value = emitDomainExpr(argument(0), block, bindings, builder);
    if (failed(value)) return failure();
    return Value(builder.create<math::ExpOp>(location, *value));
  }
  if (*operation == "sum") {
    FailureOr<Value> value = emitDomainExpr(argument(0), block, bindings, builder);
    if (failed(value)) return failure();
    auto vectorType = dyn_cast<VectorType>((*value).getType());
    if (!vectorType || vectorType.getRank() != 1) {
      PyErr_SetString(PyExc_NotImplementedError,
                      "native Domain sum currently reduces one vector");
      return failure();
    }
    Value reduced = builder.create<vector::ReductionOp>(
        location, vector::CombiningKind::ADD, *value);
    auto element = dyn_cast<FloatType>(vectorType.getElementType());
    if (element && element.getWidth() < 32)
      return Value(builder.create<arith::ExtFOp>(
          location, builder.getF32Type(), reduced));
    return reduced;
  }
  FailureOr<Value> left = emitDomainExpr(argument(0), block, bindings, builder);
  FailureOr<Value> right = emitDomainExpr(argument(1), block, bindings, builder);
  if (failed(left) || failed(right)) return failure();
  auto promote = [&](Value &scalar, Value other) -> LogicalResult {
    auto vector = dyn_cast<VectorType>(other.getType());
    if (!vector || scalar.getType() != vector.getElementType())
      return failure();
    scalar = builder.create<vector::BroadcastOp>(location, vector, scalar);
    return success();
  };
  if ((*left).getType() != (*right).getType()) {
    if (failed(promote(*left, *right)) && failed(promote(*right, *left))) {
      PyErr_SetString(PyExc_TypeError,
                      "captured binary expression has incompatible types");
      return failure();
    }
  }
  if (*operation == "add")
    return Value(builder.create<arith::AddFOp>(location, *left, *right));
  if (*operation == "sub")
    return Value(builder.create<arith::SubFOp>(location, *left, *right));
  if (*operation == "mul")
    return Value(builder.create<arith::MulFOp>(location, *left, *right));
  if (*operation == "div")
    return Value(builder.create<arith::DivFOp>(location, *left, *right));
  if (*operation == "maximum")
    return Value(builder.create<arith::MaximumFOp>(location, *left, *right));
  PyErr_Format(PyExc_NotImplementedError,
               "native Domain capture does not support %s", operation->c_str());
  return failure();
}

static PyObject *domainIR(PyObject *, PyObject *descriptor) {
  MLIRContext context;
  initializeContext(context);
  OpBuilder builder(&context);
  Location location = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());

  PyOwned reducer(attribute(descriptor, "reducer"));
  PyOwned graph(attribute(descriptor, "graph"));
  PyOwned fieldsObject(attribute(descriptor, "fields"));
  PyOwned implicitFieldsObject(attribute(descriptor, "implicit_fields"));
  PyOwned paramsObject(attribute(descriptor, "params"));
  PyOwned messagesObject(attribute(descriptor, "messages"));
  PyOwned nodeExpression(attribute(descriptor, "node_expression"));
  PyOwned nodeInputIndicesObject(attribute(descriptor, "node_input_indices"));
  PyOwned lanesObject(attribute(descriptor, "iteration_lanes"));
  FailureOr<std::string> symbol = stringAttribute(descriptor, "symbol");
  if (!reducer || !graph || !fieldsObject || !implicitFieldsObject ||
      !paramsObject || !messagesObject || !nodeExpression ||
      !nodeInputIndicesObject || !lanesObject || failed(symbol))
    return nullptr;
  FailureOr<gf::ReducerOp> reducerOp = buildReducer(reducer.value, builder);
  if (failed(reducerOp)) return nullptr;
  FailureOr<std::string> reducerSymbol = stringAttribute(reducer.value, "symbol");
  FailureOr<std::string> realization = stringAttribute(graph.value, "realization");
  FailureOr<std::string> denseBoundary =
      stringAttribute(graph.value, "dense_boundary");
  PyOwned numSrcObject(attribute(graph.value, "num_src"));
  PyOwned numDstObject(attribute(graph.value, "num_dst"));
  if (failed(reducerSymbol) || failed(realization) || failed(denseBoundary) ||
      !numSrcObject || !numDstObject)
    return nullptr;
  FailureOr<int64_t> numSrc = integer(numSrcObject.value);
  FailureOr<int64_t> numDst = integer(numDstObject.value);
  if (failed(numSrc) || failed(numDst)) return nullptr;

  PyOwned fields(PySequence_Fast(fieldsObject.value, "expected Domain fields"));
  PyOwned implicitFields(PySequence_Fast(
      implicitFieldsObject.value, "expected implicit Domain fields"));
  PyOwned params(PySequence_Fast(paramsObject.value, "expected Domain params"));
  PyOwned messages(PySequence_Fast(messagesObject.value, "expected messages"));
  FailureOr<SmallVector<int64_t>> iterationLanes =
      integerSequence(lanesObject.value);
  FailureOr<SmallVector<int64_t>> nodeInputIndices =
      integerSequence(nodeInputIndicesObject.value);
  if (!fields || !implicitFields || !params || !messages ||
      failed(iterationLanes) || failed(nodeInputIndices))
    return nullptr;
  SmallVector<Type> argumentTypes;
  SmallVector<Type> localTypes;
  SmallVector<std::pair<std::string, std::string>> bindings;
  Type indexElement;
  if (*realization == "materialized_csr") {
    FailureOr<std::string> indexName = stringAttribute(graph.value, "index_dtype");
    FailureOr<Type> converted = failed(indexName)
                                    ? FailureOr<Type>(failure())
                                    : elementType(context, *indexName);
    if (failed(converted)) return nullptr;
    indexElement = *converted;
    Type indexTensor = RankedTensorType::get({ShapedType::kDynamic}, indexElement);
    argumentTypes.append({indexTensor, indexTensor});
  } else if (*realization == "generated_radius") {
    PyOwned dimensionsObject(attribute(graph.value, "dimensions"));
    PyOwned neighborsObject(attribute(graph.value, "neighbor_count"));
    if (!dimensionsObject || !neighborsObject) return nullptr;
    FailureOr<int64_t> dimensions = integer(dimensionsObject.value);
    FailureOr<int64_t> neighbors = integer(neighborsObject.value);
    if (failed(dimensions) || failed(neighbors)) return nullptr;
    Type f32 = builder.getF32Type();
    Type i64 = builder.getI64Type();
    argumentTypes.append({
        RankedTensorType::get({ShapedType::kDynamic, *dimensions}, f32),
        RankedTensorType::get({ShapedType::kDynamic}, i64),
        RankedTensorType::get({ShapedType::kDynamic}, i64),
        RankedTensorType::get({ShapedType::kDynamic, *dimensions}, i64),
        RankedTensorType::get({*dimensions}, i64),
        RankedTensorType::get({*dimensions}, i64),
        RankedTensorType::get({*neighbors, *dimensions}, i64),
        RankedTensorType::get({*dimensions, *dimensions}, f32),
        RankedTensorType::get({*dimensions, *dimensions}, f32),
    });
  } else if (*realization == "procedural_knn") {
    PyOwned dimensionsObject(attribute(graph.value, "dimensions"));
    if (!dimensionsObject) return nullptr;
    FailureOr<int64_t> dimensions = integer(dimensionsObject.value);
    if (failed(dimensions) || *dimensions <= 0) return nullptr;
    Type positions = RankedTensorType::get(
        {ShapedType::kDynamic, *dimensions}, builder.getF32Type());
    argumentTypes.append({positions, positions});
  } else if (*realization != "implicit_dense") {
    PyErr_SetString(PyExc_NotImplementedError,
                    "native Domain builder does not support this relation realization");
    return nullptr;
  }
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(fields.value); ++index) {
    PyObject *field = PySequence_Fast_GET_ITEM(fields.value, index);
    FailureOr<std::string> role = stringAttribute(field, "role");
    FailureOr<std::string> name = stringAttribute(field, "name");
    FailureOr<std::string> dtype = stringAttribute(field, "dtype");
    PyOwned shapeObject(attribute(field, "shape"));
    FailureOr<SmallVector<int64_t>> shape =
        shapeObject ? integerSequence(shapeObject.value)
                    : FailureOr<SmallVector<int64_t>>(failure());
    if (failed(role) || failed(name) || failed(dtype) || failed(shape) ||
        shape->empty())
      return nullptr;
    FailureOr<Type> element = elementType(context, *dtype);
    if (failed(element)) return nullptr;
    SmallVector<int64_t> tensorShape(*shape);
    tensorShape.front() = ShapedType::kDynamic;
    argumentTypes.push_back(RankedTensorType::get(tensorShape, *element));
    size_t projectedRank = 1 + iterationLanes->size();
    if (shape->size() < projectedRank) {
      PyErr_SetString(PyExc_TypeError,
                      "field rank is smaller than its projected iteration rank");
      return nullptr;
    }
    ArrayRef<int64_t> valueShape = ArrayRef<int64_t>(*shape).drop_front(projectedRank);
    localTypes.push_back(
        valueShape.empty() ||
                (valueShape.size() == 1 && valueShape.front() == 1)
            ? *element
            : Type(VectorType::get(valueShape, *element)));
    bindings.emplace_back(*role, *name);
  }
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(params.value);
       ++index) {
    PyObject *parameter = PySequence_Fast_GET_ITEM(params.value, index);
    FailureOr<std::string> name = stringAttribute(parameter, "name");
    FailureOr<std::string> dtype = stringAttribute(parameter, "dtype");
    FailureOr<Type> type = failed(dtype)
                               ? FailureOr<Type>(failure())
                               : elementType(context, *dtype);
    if (failed(name) || failed(type)) return nullptr;
    argumentTypes.push_back(*type);
    localTypes.push_back(*type);
    bindings.emplace_back("param", *name);
  }
  ArrayAttr reducerResults = reducerOp->getResultTypes();
  if (reducerResults.size() != 1) {
    PyErr_SetString(PyExc_NotImplementedError,
                    "native Domain builder currently returns one reducer result");
    return nullptr;
  }
  Type reducedType = cast<TypeAttr>(reducerResults[0]).getValue();
  SmallVector<int64_t> outputShape{ShapedType::kDynamic};
  llvm::append_range(outputShape, *iterationLanes);
  Type outputElement = reducedType;
  if (auto vector = dyn_cast<VectorType>(reducedType)) {
    llvm::append_range(outputShape, vector.getShape());
    outputElement = vector.getElementType();
  }
  Type outputType = RankedTensorType::get(outputShape, outputElement);
  builder.setInsertionPointToEnd(module.getBody());
  func::FuncOp function = builder.create<func::FuncOp>(
      location, *symbol,
      builder.getFunctionType(argumentTypes, TypeRange{outputType}));
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value relation;
  unsigned fieldOffset = 0;
  if (*realization == "materialized_csr") {
    OperationState relationState(location, gf::RelationOp::getOperationName());
    relationState.addOperands({entry->getArgument(0), entry->getArgument(1)});
    relationState.addTypes(gf::RelationType::get(&context));
    relationState.addAttribute("origin", builder.getStringAttr("external"));
    relationState.addAttribute("lifecycle", builder.getStringAttr("frozen"));
    relationState.addAttribute("realization", builder.getStringAttr("materialized"));
    relationState.addAttribute("relation_id", builder.getStringAttr("graph"));
    relationState.addAttribute("version", builder.getI64IntegerAttr(0));
    relationState.addAttribute("num_src", builder.getI64IntegerAttr(*numSrc));
    relationState.addAttribute("num_dst", builder.getI64IntegerAttr(*numDst));
    for (const char *name : {"degree_min", "degree_max", "degree_sum"}) {
      PyOwned value(attribute(graph.value, name));
      if (!value) return nullptr;
      if (value.value != Py_None) {
        FailureOr<int64_t> number = integer(value.value);
        if (failed(number)) return nullptr;
        relationState.addAttribute(name, builder.getI64IntegerAttr(*number));
      }
    }
    PyOwned histogram(attribute(graph.value, "degree_histogram"));
    if (!histogram) return nullptr;
    if (histogram.value != Py_None) {
      FailureOr<SmallVector<int64_t>> counts = integerSequence(histogram.value);
      if (failed(counts)) return nullptr;
      relationState.addAttribute("degree_histogram",
                                 builder.getDenseI64ArrayAttr(*counts));
    }
    PyOwned sourceSpan(attribute(graph.value, "source_index_span_ratio"));
    if (!sourceSpan) return nullptr;
    if (sourceSpan.value != Py_None) {
      double ratio = PyFloat_AsDouble(sourceSpan.value);
      if (PyErr_Occurred()) return nullptr;
      relationState.addAttribute("source_index_span_ratio",
                                 builder.getF64FloatAttr(ratio));
    }
    PyOwned meshShape(attribute(graph.value, "mesh_shape"));
    PyOwned meshAxis(attribute(graph.value, "mesh_axis"));
    PyOwned haloDepth(attribute(graph.value, "halo_depth"));
    PyOwned balance(attribute(graph.value, "partition_balance"));
    if (!meshShape || !meshAxis || !haloDepth || !balance) return nullptr;
    if (meshShape.value != Py_None) {
      FailureOr<SmallVector<int64_t>> mesh = integerSequence(meshShape.value);
      FailureOr<int64_t> axis = integer(meshAxis.value);
      FailureOr<int64_t> depth = integer(haloDepth.value);
      const char *policy = PyUnicode_AsUTF8(balance.value);
      if (failed(mesh) || failed(axis) || failed(depth) || !policy)
        return nullptr;
      relationState.addAttribute("mesh_shape",
                                 builder.getDenseI64ArrayAttr(*mesh));
      relationState.addAttribute("mesh_axis", builder.getI64IntegerAttr(*axis));
      relationState.addAttribute("halo_depth", builder.getI64IntegerAttr(*depth));
      relationState.addAttribute("partition_balance",
                                 builder.getStringAttr(policy));
    }
    relation = builder.create(relationState)->getResult(0);
    fieldOffset = 2;
  } else if (*realization == "implicit_dense") {
    auto cartesian = builder.create<gf::CartesianOp>(
        location, gf::RelationType::get(&context),
        static_cast<uint64_t>(*numSrc), static_cast<uint64_t>(*numDst), 0);
    cartesian->setAttr("boundary", builder.getStringAttr(*denseBoundary));
    relation = cartesian.getResult();
  } else if (*realization == "generated_radius") {
    PyOwned cutoffObject(attribute(graph.value, "cutoff"));
    PyOwned dimensionsObject(attribute(graph.value, "dimensions"));
    PyOwned periodicObject(attribute(graph.value, "periodic"));
    if (!cutoffObject || !dimensionsObject || !periodicObject) return nullptr;
    double cutoff = PyFloat_AsDouble(cutoffObject.value);
    FailureOr<int64_t> dimensions = integer(dimensionsObject.value);
    if (PyErr_Occurred() || failed(dimensions)) return nullptr;
    OperationState relationState(location,
                                 gf::GeneratedRadiusOp::getOperationName());
    relationState.addOperands(entry->getArguments().take_front(9));
    relationState.addTypes(gf::RelationType::get(&context));
    relationState.addAttribute("cutoff", builder.getF64FloatAttr(cutoff));
    relationState.addAttribute("dimensions",
                               builder.getI64IntegerAttr(*dimensions));
    relationState.addAttribute(
        "periodic", builder.getBoolAttr(PyObject_IsTrue(periodicObject.value)));
    relationState.addAttribute("num_entities",
                               builder.getI64IntegerAttr(*numDst));
    relationState.addAttribute("version", builder.getI64IntegerAttr(0));
    relation = builder.create(relationState)->getResult(0);
    fieldOffset = 9;
  } else {
    PyOwned dimensionsObject(attribute(graph.value, "dimensions"));
    PyOwned kObject(attribute(graph.value, "k"));
    PyOwned metricObject(attribute(graph.value, "metric"));
    PyOwned selectionObject(attribute(graph.value, "selection"));
    PyOwned tieBreakObject(attribute(graph.value, "tie_break"));
    PyOwned excludeSelfObject(attribute(graph.value, "exclude_self"));
    PyOwned sameDomainObject(attribute(graph.value, "same_entity_domain"));
    PyOwned exactObject(attribute(graph.value, "exact"));
    if (!dimensionsObject || !kObject || !metricObject || !selectionObject ||
        !tieBreakObject || !excludeSelfObject || !sameDomainObject ||
        !exactObject)
      return nullptr;
    FailureOr<int64_t> dimensions = integer(dimensionsObject.value);
    FailureOr<int64_t> k = integer(kObject.value);
    const char *metric = PyUnicode_AsUTF8(metricObject.value);
    const char *selection = PyUnicode_AsUTF8(selectionObject.value);
    const char *tieBreak = PyUnicode_AsUTF8(tieBreakObject.value);
    int excludeSelf = PyObject_IsTrue(excludeSelfObject.value);
    int sameDomain = PyObject_IsTrue(sameDomainObject.value);
    int exact = PyObject_IsTrue(exactObject.value);
    if (failed(dimensions) || failed(k) || !metric || !selection || !tieBreak ||
        excludeSelf < 0 || sameDomain < 0 || exact < 0)
      return nullptr;
    OperationState relationState(location,
                                 gf::RankedRelationOp::getOperationName());
    Value query = entry->getArgument(0);
    Value candidate = sameDomain ? query : entry->getArgument(1);
    relationState.addOperands({query, candidate});
    relationState.addTypes(gf::RelationType::get(&context));
    relationState.addAttribute("num_queries", builder.getI64IntegerAttr(*numDst));
    relationState.addAttribute("num_candidates",
                               builder.getI64IntegerAttr(*numSrc));
    relationState.addAttribute("dimensions",
                               builder.getI64IntegerAttr(*dimensions));
    relationState.addAttribute("k", builder.getI64IntegerAttr(*k));
    relationState.addAttribute("metric", builder.getStringAttr(metric));
    relationState.addAttribute("selection", builder.getStringAttr(selection));
    relationState.addAttribute("tie_break", builder.getStringAttr(tieBreak));
    relationState.addAttribute("exclude_self", builder.getBoolAttr(excludeSelf));
    relationState.addAttribute("same_entity_domain",
                               builder.getBoolAttr(sameDomain));
    relationState.addAttribute("exact", builder.getBoolAttr(exact));
    relationState.addAttribute("version", builder.getI64IntegerAttr(0));
    relation = builder.create(relationState)->getResult(0);
    fieldOffset = 2;
  }
  SmallVector<Value> fieldValues;
  SmallVector<Attribute> roles;
  SmallVector<Attribute> names;
  for (unsigned index = 0; index < bindings.size(); ++index) {
    fieldValues.push_back(entry->getArgument(fieldOffset + index));
    roles.push_back(builder.getStringAttr(bindings[index].first));
    names.push_back(builder.getStringAttr(bindings[index].second));
  }
  for (Py_ssize_t index = 0;
       index < PySequence_Fast_GET_SIZE(implicitFields.value); ++index) {
    PyObject *field = PySequence_Fast_GET_ITEM(implicitFields.value, index);
    FailureOr<std::string> role = stringAttribute(field, "role");
    FailureOr<std::string> name = stringAttribute(field, "name");
    FailureOr<std::string> dtype = stringAttribute(field, "dtype");
    if (failed(role) || failed(name) || failed(dtype) || *dtype != "float32") {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_TypeError,
                        "implicit Domain fields currently require float32");
      return nullptr;
    }
    bindings.emplace_back(*role, *name);
  }
  OperationState applyState(location, gf::ApplyOp::getOperationName());
  applyState.addOperands(relation);
  applyState.addOperands(fieldValues);
  applyState.addTypes(outputType);
  applyState.addAttribute(
      "reducers", builder.getArrayAttr({FlatSymbolRefAttr::get(&context, *reducerSymbol)}));
  bool hasNode = nodeExpression.value != Py_None;
  applyState.addAttribute(
      "region_kinds",
      builder.getDenseI64ArrayAttr(hasNode ? ArrayRef<int64_t>({0, 1})
                                           : ArrayRef<int64_t>({0})));
  applyState.addAttribute(
      "input_segment_sizes",
      builder.getDenseI64ArrayAttr({static_cast<int64_t>(fieldValues.size())}));
  applyState.addAttribute(
      "snapshot_versions",
      builder.getDenseI64ArrayAttr(SmallVector<int64_t>(fieldValues.size(), 0)));
  applyState.addAttribute(
      "effects", builder.getArrayAttr(
                     SmallVector<Attribute>(fieldValues.size(),
                                            builder.getStringAttr("read"))));
  applyState.addAttribute("deterministic", builder.getBoolAttr(false));
  applyState.addAttribute("input_roles", builder.getArrayAttr(roles));
  applyState.addAttribute("input_names", builder.getArrayAttr(names));
  applyState.addAttribute("node_input_indices",
                          builder.getDenseI64ArrayAttr(*nodeInputIndices));
  applyState.addAttribute(
      "node_input_segment_sizes",
      builder.getDenseI64ArrayAttr(
          {hasNode ? static_cast<int64_t>(nodeInputIndices->size()) : 0}));
  if (!iterationLanes->empty())
    applyState.addAttribute("iteration_lanes",
                            builder.getDenseI64ArrayAttr(*iterationLanes));
  applyState.addRegion();
  if (hasNode) applyState.addRegion();
  auto apply = cast<gf::ApplyOp>(builder.create(applyState));
  auto body = std::make_unique<Block>();
  for (Type type : localTypes) body->addArgument(type, location);
  for (Py_ssize_t index = 0;
       index < PySequence_Fast_GET_SIZE(implicitFields.value); ++index)
    body->addArgument(builder.getF32Type(), location);
  Block *edgeBody = body.get();
  apply.getRegions().front().push_back(body.release());
  builder.setInsertionPointToStart(edgeBody);
  SmallVector<Value> yieldedMessages;
  for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(messages.value);
       ++index) {
    FailureOr<Value> message = emitDomainExpr(
        PySequence_Fast_GET_ITEM(messages.value, index), *edgeBody, bindings,
        builder);
    if (failed(message)) return nullptr;
    yieldedMessages.push_back(*message);
  }
  builder.create<gf::YieldOp>(location, yieldedMessages);
  if (hasNode) {
    auto nodeBody = std::make_unique<Block>();
    SmallVector<std::pair<std::string, std::string>> nodeBindings;
    for (int64_t index : *nodeInputIndices) {
      if (index < 0 || index >= static_cast<int64_t>(localTypes.size()) ||
          index >= static_cast<int64_t>(bindings.size())) {
        PyErr_SetString(PyExc_IndexError,
                        "node input index is outside captured fields");
        return nullptr;
      }
      nodeBody->addArgument(localTypes[index], location);
      nodeBindings.push_back(bindings[index]);
    }
    nodeBody->addArgument(reducedType, location);
    Block *nodeBlock = nodeBody.get();
    apply.getRegions()[1].push_back(nodeBody.release());
    nodeBindings.emplace_back("aggregate", "0");
    builder.setInsertionPointToStart(nodeBlock);
    FailureOr<Value> updated = emitDomainExpr(
        nodeExpression.value, *nodeBlock, nodeBindings, builder);
    if (failed(updated)) return nullptr;
    if ((*updated).getType() != reducedType) {
      PyErr_SetString(
          PyExc_TypeError,
          "node() result type must match the reducer finalized result type");
      return nullptr;
    }
    builder.create<gf::YieldOp>(location, ValueRange{*updated});
  }
  builder.setInsertionPointAfter(apply);
  builder.create<func::ReturnOp>(location, apply.getResults());
  if (failed(verify(module))) {
    PyErr_SetString(PyExc_RuntimeError, "native Domain IR verification failed");
    return nullptr;
  }
  std::string domain = printOperation(module);

  PassManager toIteration(&context);
  toIteration.addPass(gf::createGFLowerDomainToIter());
  if (failed(toIteration.run(module))) {
    PyErr_SetString(PyExc_RuntimeError,
                    "native Domain to Iter lowering failed");
    return nullptr;
  }
  std::string iteration = printOperation(module);

  PassManager toKernel(&context);
  toKernel.addPass(gf::createGFLowerIterToKernel());
  toKernel.addPass(gf::createGFSelectKernelSchedule());
  if (failed(toKernel.run(module))) {
    PyErr_SetString(PyExc_RuntimeError,
                    "native Iter to Kernel lowering failed");
    return nullptr;
  }
  std::string kernel = printOperation(module);

  PassManager toTask(&context);
  toTask.addPass(gf::createGFPlanDegreeBuckets());
  toTask.addPass(gf::createGFPlanSplitRows());
  toTask.addPass(gf::createGFDecomposeDegreeWorklists());
  toTask.addPass(gf::createGFPlanDistributedTasks());
  if (failed(toTask.run(module))) {
    PyErr_SetString(PyExc_RuntimeError,
                    "native distributed task planning failed");
    return nullptr;
  }
  std::string task = printOperation(module);
  PyObject *result = PyTuple_New(4);
  if (!result) return nullptr;
  PyTuple_SET_ITEM(result, 0,
                   PyUnicode_FromStringAndSize(domain.data(), domain.size()));
  PyTuple_SET_ITEM(
      result, 1,
      PyUnicode_FromStringAndSize(iteration.data(), iteration.size()));
  PyTuple_SET_ITEM(result, 2,
                   PyUnicode_FromStringAndSize(kernel.data(), kernel.size()));
  PyTuple_SET_ITEM(result, 3,
                   PyUnicode_FromStringAndSize(task.data(), task.size()));
  return result;
}

/// Compose independently captured native Domain modules into one straight-line
/// SSA program.  Python supplies only operand ownership: negative integers are
/// canonical external argument slots, non-negative integers reference the
/// single result of an earlier apply.  All operation construction/copying,
/// relation CSE, verification, fusion and lowering remain inside MLIR.
static PyObject *programIR(PyObject *, PyObject *args) {
  PyObject *modulesObject = nullptr;
  PyObject *bindingsObject = nullptr;
  PyObject *outputsObject = nullptr;
  if (!PyArg_ParseTuple(args, "OOO", &modulesObject, &bindingsObject,
                        &outputsObject))
    return nullptr;
  PyOwned modulesSequence(
      PySequence_Fast(modulesObject, "expected Domain module sequence"));
  PyOwned bindingsSequence(
      PySequence_Fast(bindingsObject, "expected program binding sequence"));
  FailureOr<SmallVector<int64_t>> outputIndices =
      integerSequence(outputsObject);
  if (!modulesSequence || !bindingsSequence || failed(outputIndices) ||
      PySequence_Fast_GET_SIZE(modulesSequence.value) == 0 ||
      PySequence_Fast_GET_SIZE(modulesSequence.value) !=
          PySequence_Fast_GET_SIZE(bindingsSequence.value))
    return nullptr;

  MLIRContext context;
  initializeContext(context);
  SmallVector<OwningOpRef<ModuleOp>> sources;
  SmallVector<SmallVector<int64_t>> bindings;
  llvm::DenseMap<int64_t, Type> externalTypes;
  int64_t maximumExternal = -1;
  for (Py_ssize_t index = 0;
       index < PySequence_Fast_GET_SIZE(modulesSequence.value); ++index) {
    const char *text = PyUnicode_AsUTF8(
        PySequence_Fast_GET_ITEM(modulesSequence.value, index));
    if (!text) return nullptr;
    OwningOpRef<ModuleOp> source = parseSourceString<ModuleOp>(text, &context);
    if (!source || failed(verify(*source))) {
      PyErr_Format(PyExc_ValueError,
                   "program input %zd is not verified GraphForge Domain IR",
                   index);
      return nullptr;
    }
    SmallVector<func::FuncOp> functions;
    source->walk([&](func::FuncOp function) { functions.push_back(function); });
    if (functions.size() != 1 || !llvm::hasSingleElement(functions[0].getBody())) {
      PyErr_SetString(PyExc_ValueError,
                      "each program input must contain one straight-line function");
      return nullptr;
    }
    FailureOr<SmallVector<int64_t>> current = integerSequence(
        PySequence_Fast_GET_ITEM(bindingsSequence.value, index));
    if (failed(current) || current->size() != functions[0].getNumArguments()) {
      if (!PyErr_Occurred())
        PyErr_Format(PyExc_ValueError,
                     "program binding %zd does not match function arguments",
                     index);
      return nullptr;
    }
    for (auto [argument, binding] :
         llvm::zip(functions[0].getArgumentTypes(), *current)) {
      if (binding >= 0) continue;
      int64_t slot = -binding - 1;
      maximumExternal = std::max(maximumExternal, slot);
      auto [found, inserted] = externalTypes.try_emplace(slot, argument);
      if (!inserted && found->second != argument) {
        PyErr_SetString(PyExc_TypeError,
                        "one program external slot has conflicting types");
        return nullptr;
      }
    }
    bindings.push_back(std::move(*current));
    sources.push_back(std::move(source));
  }
  if (maximumExternal + 1 != static_cast<int64_t>(externalTypes.size())) {
    PyErr_SetString(PyExc_ValueError,
                    "program external slots must be contiguous from zero");
    return nullptr;
  }

  OpBuilder builder(&context);
  Location location = builder.getUnknownLoc();
  ModuleOp destination = ModuleOp::create(location);
  SmallVector<Type> argumentTypes;
  for (int64_t slot = 0; slot <= maximumExternal; ++slot)
    argumentTypes.push_back(externalTypes.lookup(slot));
  builder.setInsertionPointToStart(destination.getBody());
  func::FuncOp function = builder.create<func::FuncOp>(
      location, "graph_program",
      builder.getFunctionType(argumentTypes, TypeRange{}));
  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  SmallVector<Value> produced;
  llvm::DenseMap<std::pair<Value, Value>, Value> csrRelations;
  for (auto [programIndex, source] : llvm::enumerate(sources)) {
    func::FuncOp sourceFunction;
    source->walk([&](func::FuncOp candidate) { sourceFunction = candidate; });
    SmallVector<gf::ReducerOp> sourceReducers;
    source->walk([&](gf::ReducerOp reducer) { sourceReducers.push_back(reducer); });
    if (sourceReducers.size() != 1) {
      PyErr_SetString(PyExc_ValueError,
                      "each program apply currently requires one reducer symbol");
      return nullptr;
    }
    std::string reducerName =
        (Twine(sourceReducers[0].getSymName()) + "_p" +
         Twine(programIndex)).str();
    Operation *reducerClone = sourceReducers[0]->clone();
    reducerClone->setAttr(SymbolTable::getSymbolAttrName(),
                          builder.getStringAttr(reducerName));
    destination.getBody()->getOperations().insert(
        Block::iterator(function), reducerClone);

    IRMapping mapping;
    for (auto [argument, binding] : llvm::zip(
             sourceFunction.getArguments(), bindings[programIndex])) {
      Value replacement;
      if (binding < 0) {
        replacement = entry->getArgument(-binding - 1);
      } else {
        if (binding >= static_cast<int64_t>(produced.size())) {
          PyErr_SetString(PyExc_ValueError,
                          "program dependency must reference an earlier apply");
          return nullptr;
        }
        replacement = produced[binding];
      }
      if (replacement.getType() != argument.getType()) {
        PyErr_SetString(PyExc_TypeError,
                        "program dependency type does not match consumer input");
        return nullptr;
      }
      mapping.map(argument, replacement);
    }

    gf::ApplyOp clonedApply;
    for (Operation &operation : sourceFunction.getBody().front()) {
      if (isa<func::ReturnOp>(operation)) continue;
      if (auto relation = dyn_cast<gf::RelationOp>(operation)) {
        Value row = mapping.lookup(relation.getRowPtr());
        Value col = mapping.lookup(relation.getColIdx());
        auto found = csrRelations.find({row, col});
        if (found != csrRelations.end() &&
            relation.getVersionAttr().getInt() == 0) {
          mapping.map(relation.getResult(), found->second);
          continue;
        }
      }
      Operation *clone = builder.clone(operation, mapping);
      if (auto relation = dyn_cast<gf::RelationOp>(clone))
        csrRelations[{relation.getRowPtr(), relation.getColIdx()}] =
            relation.getResult();
      if (auto apply = dyn_cast<gf::ApplyOp>(clone)) {
        apply->setAttr(
            "reducers",
            builder.getArrayAttr({FlatSymbolRefAttr::get(&context, reducerName)}));
        clonedApply = apply;
      }
    }
    if (!clonedApply || clonedApply.getNumResults() != 1) {
      PyErr_SetString(PyExc_ValueError,
                      "program input must contain one single-result gf.apply");
      return nullptr;
    }
    produced.push_back(clonedApply.getResult(0));
  }

  SmallVector<Value> returned;
  SmallVector<Type> resultTypes;
  for (int64_t index : *outputIndices) {
    if (index < 0 || index >= static_cast<int64_t>(produced.size())) {
      PyErr_SetString(PyExc_IndexError,
                      "program output does not reference an apply result");
      return nullptr;
    }
    returned.push_back(produced[index]);
    resultTypes.push_back(produced[index].getType());
  }
  function.setFunctionType(builder.getFunctionType(argumentTypes, resultTypes));
  builder.setInsertionPointToEnd(entry);
  builder.create<func::ReturnOp>(location, returned);
  if (failed(verify(destination))) {
    PyErr_SetString(PyExc_RuntimeError,
                    "native GraphProgram verification failed");
    return nullptr;
  }
  std::string domain = printOperation(destination);

  PassManager fusion(&context);
  fusion.addPass(gf::createGFFormApplyFusionGroups());
  fusion.addPass(gf::createGFFuseCompatibleApplies());
  if (failed(fusion.run(destination))) {
    PyErr_SetString(PyExc_RuntimeError, "GraphProgram fusion failed");
    return nullptr;
  }
  std::string fused = printOperation(destination);
  PassManager iterationPasses(&context);
  iterationPasses.addPass(gf::createGFLowerDomainToIter());
  if (failed(iterationPasses.run(destination))) {
    PyErr_SetString(PyExc_RuntimeError, "GraphProgram Iter lowering failed");
    return nullptr;
  }
  std::string iteration = printOperation(destination);
  PassManager kernelPasses(&context);
  kernelPasses.addPass(gf::createGFLowerIterToKernel());
  kernelPasses.addPass(gf::createGFSelectKernelSchedule());
  if (failed(kernelPasses.run(destination))) {
    PyErr_SetString(PyExc_RuntimeError, "GraphProgram Kernel lowering failed");
    return nullptr;
  }
  std::string kernel = printOperation(destination);
  PyObject *result = PyTuple_New(4);
  PyTuple_SET_ITEM(result, 0,
                   PyUnicode_FromStringAndSize(domain.data(), domain.size()));
  PyTuple_SET_ITEM(result, 1,
                   PyUnicode_FromStringAndSize(fused.data(), fused.size()));
  PyTuple_SET_ITEM(result, 2,
                   PyUnicode_FromStringAndSize(iteration.data(), iteration.size()));
  PyTuple_SET_ITEM(result, 3,
                   PyUnicode_FromStringAndSize(kernel.data(), kernel.size()));
  return result;
}

struct CPUJob {
  SmallVector<void *> descriptorArguments;
  SmallVector<void *> packed;
  int64_t begin = 0;
  int64_t end = 0;
};

struct CPUState {
  using PackedFunction = void (*)(void **);

  CPUState(std::unique_ptr<ExecutionEngine> engine, PackedFunction function,
           int64_t vectorWidth, unsigned threadCount, bool serialControl)
      : engine(std::move(engine)), function(function),
        vectorWidth(std::max<int64_t>(1, vectorWidth)),
        threadCount(std::max(1u, threadCount)), serialControl(serialControl) {}

  ~CPUState() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      ++generation;
    }
    workReady.notify_all();
    for (std::thread &worker : workers)
      if (worker.joinable()) worker.join();
  }

  void launch(std::vector<CPUJob> nextJobs) {
    std::lock_guard<std::mutex> launchGuard(launchMutex);
    // Most compiled expressions are tiny and execute on the caller.  Create
    // persistent workers only once an executable actually has parallel work;
    // otherwise the executable cache would multiply idle OS threads by the
    // number of compiled Tensor DAGs.
    if (nextJobs.size() > 1 && workers.empty()) {
      workers.reserve(threadCount - 1);
      for (unsigned worker = 1; worker < threadCount; ++worker)
        workers.emplace_back([this, worker] { workerLoop(worker); });
    }
    {
      std::lock_guard<std::mutex> lock(mutex);
      jobs = std::move(nextJobs);
      completed = 0;
      ++generation;
    }
    workReady.notify_all();
    function(jobs[0].packed.data());
    if (jobs.size() == 1) return;
    std::unique_lock<std::mutex> lock(mutex);
    workDone.wait(lock, [&] { return completed == jobs.size() - 1; });
  }

  std::unique_ptr<ExecutionEngine> engine;
  PackedFunction function;
  int64_t vectorWidth;
  unsigned threadCount;
  bool serialControl;

private:
  void workerLoop(unsigned worker) {
    size_t seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      workReady.wait(lock, [&] { return stopping || generation != seen; });
      if (stopping) return;
      seen = generation;
      if (worker >= jobs.size()) continue;
      CPUJob *job = &jobs[worker];
      lock.unlock();
      function(job->packed.data());
      lock.lock();
      ++completed;
      if (completed == jobs.size() - 1) workDone.notify_one();
    }
  }

  std::mutex launchMutex;
  std::mutex mutex;
  std::condition_variable workReady;
  std::condition_variable workDone;
  std::vector<std::thread> workers;
  std::vector<CPUJob> jobs;
  size_t generation = 0;
  size_t completed = 0;
  bool stopping = false;
};

static unsigned cpuThreadCount() {
  unsigned fallback = std::min(
      32u, std::max(1u, std::thread::hardware_concurrency()));
  const char *raw = std::getenv("GRAPHFORGE_CPU_THREADS");
  if (!raw || !*raw) return fallback;
  char *end = nullptr;
  unsigned long parsed = std::strtoul(raw, &end, 10);
  if (!end || *end != '\0' || parsed == 0)
    return fallback;
  return static_cast<unsigned>(std::min<unsigned long>(parsed, 256));
}

static PyObject *planTensorCheckpoints(PyObject *, PyObject *args,
                                       PyObject *keywords) {
  PyObject *output = nullptr;
  long long memoryBudgetBytes = -1;
  long long spillBudgetBytes = 0;
  static const char *names[] = {"output", "memory_budget_bytes",
                                "spill_budget_bytes", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|LL",
                                   const_cast<char **>(names), &output,
                                   &memoryBudgetBytes, &spillBudgetBytes))
    return nullptr;
  if (memoryBudgetBytes < -1 || spillBudgetBytes < -1) {
    PyErr_SetString(PyExc_ValueError,
                    "checkpoint budgets must be -1 or non-negative");
    return nullptr;
  }

  MLIRContext context;
  initializeContext(context);
  TensorBuilder capture(context);
  FailureOr<ModuleOp> module =
      capture.finish(output, "tensor_checkpoint_plan");
  if (failed(module)) return nullptr;

  gf::GFPlanTensorCheckpointsOptions options;
  options.memoryBudgetBytes = memoryBudgetBytes;
  options.spillBudgetBytes = spillBudgetBytes;
  PassManager manager(&context);
  manager.addPass(gf::createGFPlanTensorCheckpoints(options));
  if (failed(manager.run(*module))) {
    PyErr_SetString(PyExc_RuntimeError,
                    "native tensor checkpoint planning failed");
    return nullptr;
  }

  auto decisions = (*module)->getAttrOfType<DenseI64ArrayAttr>(
      "gf_tensor.checkpoint_decisions");
  auto saved = (*module)->getAttrOfType<IntegerAttr>(
      "gf_tensor.checkpoint_saved_bytes");
  auto tiers = (*module)->getAttrOfType<ArrayAttr>(
      "gf_tensor.checkpoint_tiers");
  auto costs = (*module)->getAttrOfType<DenseI64ArrayAttr>(
      "gf_tensor.checkpoint_recompute_costs");
  auto intervals = (*module)->getAttrOfType<DenseI64ArrayAttr>(
      "gf_tensor.checkpoint_live_intervals");
  auto spilled = (*module)->getAttrOfType<IntegerAttr>(
      "gf_tensor.checkpoint_spilled_bytes");
  auto peakDevice = (*module)->getAttrOfType<IntegerAttr>(
      "gf_tensor.checkpoint_peak_live_bytes");
  auto peakSpill = (*module)->getAttrOfType<IntegerAttr>(
      "gf_tensor.checkpoint_peak_spill_bytes");
  if (!decisions || !saved || !tiers || !costs || !intervals || !spilled ||
      !peakDevice || !peakSpill) {
    PyErr_SetString(PyExc_RuntimeError,
                    "checkpoint planner did not emit diagnostics");
    return nullptr;
  }
  PyObject *pythonDecisions = PyTuple_New(decisions.size());
  if (!pythonDecisions) return nullptr;
  for (auto [index, decision] : llvm::enumerate(decisions.asArrayRef()))
    PyTuple_SET_ITEM(pythonDecisions, index,
                     PyBool_FromLong(decision != 0));
  std::string planned = printOperation(*module);
  PyOwned details(PyDict_New());
  PyOwned pythonTiers(PyTuple_New(tiers.size()));
  PyOwned pythonCosts(PyTuple_New(costs.size()));
  PyOwned pythonIntervals(PyTuple_New(costs.size()));
  if (!details || !pythonTiers || !pythonCosts || !pythonIntervals) {
    Py_DECREF(pythonDecisions);
    return nullptr;
  }
  for (auto [index, tier] : llvm::enumerate(tiers)) {
    auto text = dyn_cast<StringAttr>(tier);
    if (!text) {
      PyErr_SetString(PyExc_RuntimeError, "invalid checkpoint tier diagnostic");
      Py_DECREF(pythonDecisions);
      return nullptr;
    }
    PyTuple_SET_ITEM(
        pythonTiers.value, index,
        PyUnicode_FromStringAndSize(text.getValue().data(), text.size()));
  }
  ArrayRef<int64_t> intervalValues = intervals.asArrayRef();
  for (auto [index, cost] : llvm::enumerate(costs.asArrayRef())) {
    PyTuple_SET_ITEM(pythonCosts.value, index, PyLong_FromLongLong(cost));
    PyObject *interval = PyTuple_New(2);
    PyTuple_SET_ITEM(interval, 0,
                     PyLong_FromLongLong(intervalValues[2 * index]));
    PyTuple_SET_ITEM(interval, 1,
                     PyLong_FromLongLong(intervalValues[2 * index + 1]));
    PyTuple_SET_ITEM(pythonIntervals.value, index, interval);
  }
  PyDict_SetItemString(details.value, "tiers", pythonTiers.value);
  PyDict_SetItemString(details.value, "recompute_costs", pythonCosts.value);
  PyDict_SetItemString(details.value, "live_intervals", pythonIntervals.value);
  PyOwned spilledObject(PyLong_FromLongLong(spilled.getInt()));
  PyOwned peakDeviceObject(PyLong_FromLongLong(peakDevice.getInt()));
  PyOwned peakSpillObject(PyLong_FromLongLong(peakSpill.getInt()));
  PyDict_SetItemString(details.value, "spilled_bytes", spilledObject.value);
  PyDict_SetItemString(details.value, "peak_live_bytes", peakDeviceObject.value);
  PyDict_SetItemString(details.value, "peak_spill_bytes", peakSpillObject.value);
  PyObject *result = PyTuple_New(4);
  if (!result) {
    Py_DECREF(pythonDecisions);
    return nullptr;
  }
  PyTuple_SET_ITEM(result, 0, pythonDecisions);
  PyTuple_SET_ITEM(result, 1, PyLong_FromLongLong(saved.getInt()));
  PyTuple_SET_ITEM(result, 2,
                   PyUnicode_FromStringAndSize(planned.data(), planned.size()));
  PyTuple_SET_ITEM(result, 3, details.value);
  details.value = nullptr;
  return result;
}

static void destroyCPUState(PyObject *capsule) {
  void *pointer = PyCapsule_GetPointer(capsule, "graphforge.cpu.executable");
  if (pointer) delete static_cast<CPUState *>(pointer);
  else PyErr_Clear();
}

static PyObject *compileCPU(PyObject *, PyObject *output) {
  static bool targetInitialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)targetInitialized;
  MLIRContext context;
  initializeContext(context);
  TensorBuilder capture(context);
  FailureOr<ModuleOp> module = capture.finish(output, "graphforge_run");
  if (failed(module)) return nullptr;
  std::string semantic = printOperation(*module);

  gf::GFPlanTensorCheckpointsOptions checkpointOptions;
  checkpointOptions.memoryBudgetBytes = 0;
  PassManager planning(&context);
  planning.addPass(gf::createGFPlanTensorCheckpoints(checkpointOptions));
  if (failed(planning.run(*module))) {
    PyErr_SetString(PyExc_RuntimeError,
                    "gf_tensor CPU checkpoint planning failed");
    return nullptr;
  }

  PassManager lowering(&context);
  lowering.addPass(gf::createGFLowerTensorToCPU());
  if (failed(lowering.run(*module))) {
    PyErr_SetString(PyExc_RuntimeError, "gf_tensor CPU loop lowering failed");
    return nullptr;
  }
  std::string loops = printOperation(*module);
  int64_t vectorWidth = 1;
  bool serialControl = false;
  module->walk([&](func::FuncOp function) {
    if (function.getName() == "graphforge_run") {
      if (auto width = function->getAttrOfType<IntegerAttr>(
              "graphforge.cpu.vector_width"))
        vectorWidth = width.getInt();
      serialControl = function->hasAttr("graphforge.cpu.serial_control");
    }
  });

  PassManager llvmLowering(&context);
  llvmLowering.addPass(createSCFToControlFlowPass());
  llvmLowering.addPass(createConvertMathToLLVMPass());
  llvmLowering.addPass(createConvertVectorToLLVMPass());
  llvmLowering.addPass(createConvertToLLVMPass());
  llvmLowering.addPass(createReconcileUnrealizedCastsPass());
  if (failed(llvmLowering.run(*module))) {
    PyErr_SetString(PyExc_RuntimeError, "GraphForge CPU LLVM lowering failed");
    return nullptr;
  }
  std::string llvmDialect = printOperation(*module);
  auto expected = ExecutionEngine::create(module->getOperation());
  if (!expected) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    stream << expected.takeError();
    stream << "\n--- LLVM dialect module ---\n" << llvmDialect;
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    return nullptr;
  }
  auto packed = (*expected)->lookupPacked("_mlir_ciface_graphforge_run");
  if (!packed) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    stream << packed.takeError();
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    return nullptr;
  }
  auto *state = new CPUState(
      std::move(*expected), *packed, vectorWidth, cpuThreadCount(),
      serialControl);
  PyObject *capsule = PyCapsule_New(
      state, "graphforge.cpu.executable", destroyCPUState);
  if (!capsule) {
    delete state;
    return nullptr;
  }
  PyObject *result = PyTuple_New(4);
  PyTuple_SET_ITEM(result, 0, capsule);
  PyTuple_SET_ITEM(result, 1, PyUnicode_FromStringAndSize(
                                  semantic.data(), semantic.size()));
  PyTuple_SET_ITEM(result, 2,
                   PyUnicode_FromStringAndSize(loops.data(), loops.size()));
  PyTuple_SET_ITEM(result, 3, PyUnicode_FromStringAndSize(
                                  llvmDialect.data(), llvmDialect.size()));
  return result;
}

struct MemRef1D {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t size;
  int64_t stride;
};

static PyObject *launchCPU(PyObject *, PyObject *args) {
  PyObject *capsule = nullptr;
  PyObject *addressesObject = nullptr;
  PyObject *sizesObject = nullptr;
  unsigned long long outputAddress = 0;
  long long outputElements = 0;
  if (!PyArg_ParseTuple(args, "OOOKL", &capsule, &addressesObject, &sizesObject,
                        &outputAddress, &outputElements))
    return nullptr;
  auto *state = static_cast<CPUState *>(PyCapsule_GetPointer(
      capsule, "graphforge.cpu.executable"));
  if (!state) return nullptr;
  PyOwned addresses(PySequence_Fast(addressesObject, "expected input addresses"));
  PyOwned sizes(PySequence_Fast(sizesObject, "expected input element counts"));
  if (!addresses || !sizes ||
      PySequence_Fast_GET_SIZE(addresses.value) !=
          PySequence_Fast_GET_SIZE(sizes.value)) {
    PyErr_SetString(PyExc_ValueError, "addresses and sizes must have equal length");
    return nullptr;
  }
  SmallVector<MemRef1D> descriptors;
  Py_ssize_t count = PySequence_Fast_GET_SIZE(addresses.value);
  descriptors.reserve(count + 1);
  for (Py_ssize_t index = 0; index < count; ++index) {
    void *address = PyLong_AsVoidPtr(
        PySequence_Fast_GET_ITEM(addresses.value, index));
    FailureOr<int64_t> size = integer(
        PySequence_Fast_GET_ITEM(sizes.value, index));
    if (PyErr_Occurred() || failed(size)) return nullptr;
    descriptors.push_back({address, address, 0, *size, 1});
  }
  void *outputPointer = reinterpret_cast<void *>(outputAddress);
  descriptors.push_back(
      {outputPointer, outputPointer, 0, outputElements, 1});
  SmallVector<void *> descriptorPointers;
  descriptorPointers.reserve(descriptors.size());
  for (MemRef1D &descriptor : descriptors)
    descriptorPointers.push_back(&descriptor);

  unsigned requestedThreads = !state->serialControl && outputElements >= 4096
      ? state->threadCount : 1;
  int64_t alignment = state->vectorWidth;
  int64_t rawChunk = (outputElements + requestedThreads - 1) /
      requestedThreads;
  int64_t chunk = ((rawChunk + alignment - 1) / alignment) * alignment;
  std::vector<CPUJob> jobs;
  jobs.reserve(requestedThreads);
  for (unsigned worker = 0; worker < requestedThreads; ++worker) {
    int64_t begin = std::min<int64_t>(outputElements, worker * chunk);
    int64_t end = std::min<int64_t>(outputElements, begin + chunk);
    if (begin >= end) break;
    jobs.emplace_back();
    CPUJob &job = jobs.back();
    job.begin = begin;
    job.end = end;
    job.descriptorArguments.assign(descriptorPointers.begin(),
                                   descriptorPointers.end());
    job.packed.reserve(job.descriptorArguments.size() + 2);
    for (void *&descriptorPointer : job.descriptorArguments)
      job.packed.push_back(&descriptorPointer);
    job.packed.push_back(&job.begin);
    job.packed.push_back(&job.end);
  }
  if (jobs.empty()) Py_RETURN_NONE;
  Py_BEGIN_ALLOW_THREADS
  state->launch(std::move(jobs));
  Py_END_ALLOW_THREADS
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"tensor_ir", _PyCFunction_CAST(tensorIR),
     METH_VARARGS | METH_KEYWORDS,
     "Build and verify canonical gf_tensor IR in-process."},
    {"reducer_ir", reducerIR, METH_O,
     "Build and verify one UDF gf.reducer in-process."},
    {"domain_ir", domainIR, METH_O,
     "Build and verify one coarse MessagePassing module in-process."},
    {"program_ir", programIR, METH_VARARGS,
     "Compose native Domain captures into one verified straight-line SSA program."},
    {"kernel_schedules", kernelSchedules, METH_O,
     "Inspect verified compiler-generated gf_kernel machine schedules."},
    {"plan_checkpoints", _PyCFunction_CAST(planTensorCheckpoints),
     METH_VARARGS | METH_KEYWORDS,
     "Plan tensor checkpoints in MLIR under a byte budget."},
    {"compile_cpu", compileCPU, METH_O,
     "Lower gf_tensor through SCF/MemRef/LLVM and create an ExecutionEngine."},
    {"launch_cpu", launchCPU, METH_VARARGS,
     "Launch an MLIR ExecutionEngine CPU executable."},
    {"object_identity", identity, METH_O, "Return a stable in-process object id."},
    {nullptr, nullptr, 0, nullptr}};

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_graphforge_compiler",
                             "GraphForge native compiler binding", -1, methods};
} // namespace

PyMODINIT_FUNC PyInit__graphforge_compiler() { return PyModule_Create(&module); }
