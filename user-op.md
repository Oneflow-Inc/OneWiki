# How to Add User Operators in OneFlow

This article introduces how to add operators in OneFlow with the case of developing a leaky_relu_yzh op.

View the pull request corresponding to this article from [here](https://github.com/Oneflow-Inc/oneflow/pull/8350).


- [Background](#background)
  - [Op and Kernel](#op-and-kernel)
  - [System op and user op in OneFlow](#system-op-and-user-op-in-oneflow)
  - [ODS and TableGen](#ods-and-tablegen)
- [Develop op](#develop-op)
  - [Define op](#define-op)
  - [Implement Kernel logic](#implement-kernel-logic)
  - [Export functional interface](#export-functional-interface)
  - [Implement the backward logic used in differentiation](#implement-the-backward-logic-used-in-differentiation)
- [Tests and docs](#tests-and-docs)


## Background

### Op and kernel


In the content above, we have mentioned two steps: define op and implement kernel computational logic. Actually, the op and kernel mentioned here are two correlative concepts.


Op is a logical operator, which contains some essential information required by OneFlow Compiling Runtime when it constructs the computation graph, including the input shape, the output shape, tensors that need to be automatically differentiated. With the information, OneFlow Compiling Runtime can construct the computation graph and conduct other operations like resource applying and constructing according to the computation graph it constructs (for example, it will request memory according to the input/output size of the tensor), but the op doesn't contain the logic for processing data.

When processing data, OneFlow Executing Runtime will launch the kernel to compute, so it is the kernel that contains the logic for processing data. For a logical operator, OneFlow Executing Runtime will launch different kernels according to the specific data types and hardware devices (CPU or CUDA, etc.).

### System op and user op in OneFlow

OneFlow contains two classes of ops: system op and user op.

The definitions of system ops have been included in the directory named [oneflow/core/operator/](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/core/operator), and the implementation of their corresponding kernels can be found in the directory named [oneflow/core/kernel](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/core/kernel). System ops are operators that are crucial to system performance, such as graph construction and pipelines.

System ops only take over a small proportion, the majority of operators in OneFlow are user ops that relate to the business logic of user models. The definitions of user ops have been included in the directory named [oneflow/user/ops](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/user/ops), and the implementation of their kernel has been listed under the directory named [oneflow/user/kernels](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/user/kernels).

At present, OneFlow has an abundant operator repository. However, if the existing operators in the repository can't fulfill your need in constructing your model, you can add a **new user operator** by yourself. 

### ODS and TableGen

[TableGen](https://llvm.org/docs/TableGen/index.html) is a code generator tool that reads and parses a `.td` file (whose grammar is similar to C++ template) and then transfers the file into [TableGen Backends](https://llvm.org/docs/TableGen/BackEnds.html) to generate another language.

Based on TableGen, MLIR has established a set of operator definition specifications [ODS](https://mlir.llvm.org/docs/OpDefinitions/) and its corresponding backend [OpDefinitionsGen](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp). 

On the basis of ODS, OneFlow has developed the [TableGen OneFlow Backends](https://github.com/Oneflow-Inc/oneflow/tree/master/tools/oneflow-tblgen) to define 
OneFlow user ops.

Therefore, the definitions of OneFlow user ops have been included in the file named [OneFlowUserOps.td](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/include/OneFlow/OneFlowUserOps.td). 

## Develop op

In OneFlow, developing a new user op mainly contains 4 steps:

1. Define op
2. Implement kernel computational logic
3. Export functional interface 
4. Implement the backward logic used in differentiation

### Define op

Defining an operator entails specifying its name, input and output data types, and attributes. In line with MLIR's [ODS (Operation Definition Specification)](https://mlir.llvm.org/docs/OpDefinitions/), OneFlow implements its own MLIR OneFlow Dialect. In terms of operator definition, the purpose of doing this is to delegate all kinds of inference functions and interfaces for serialization/deserialization to ODS, thus lowering the error rate caused by handwriting and making later optimization, format conversion, and other processes more flexible. 


In OneFlow, defining an user op mainly contains 5 parts:

- Op class
- Input
- Output
- Attributes
- Export and implement the inference interface

#### Op class

You may view the source code of op definition in [oneflow/ir/include/OneFlow/OneFlowUserOps.td](https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/ir/include/OneFlow/OneFlowUserOps.td#L8418-L8451).

Defining an op begins with the `def` keyword, and this op is inherited from `OneFlow_BaseOp`. Meanwhile, specify the template parameters of `OneFlow_BaseOp`, including the op type name and [Trait](https://mlir.llvm.org/docs/Traits/) list.


```c++
def OneFlow_LeakyReluYZHOp : OneFlow_BaseOp<"leaky_relu_yzh", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
//...
}
```

The `"leaky_relu_yzh"` is the specified op type name, and each op needs to be specified with a globally unique op type name as its global identifier.

The second template parameter is list (`[…]`), and each item represents a single Trait. Commonly-used Traits in OneFlow include:

- `NoSideEffect` indicates that the op has no side effect (i.e. it won't change the system state of the memory, network, pipeline, disk, and others), and this trait can instruct to optimize operations.
- `NoGrad` indicates that the op has no gradient in mathematics (i.e. it is non-differentiable).
- `CpuOnly` indicates that the op can only be executed on CPUs.
- `SupportNonContiguous` indicates whether the op supports Non-Contiguous tensor(for the concept of Contiguous Tensor, please refer to [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) ).


#### Input and output

Define the input of the op by overriding the `input` domain, for example:

```c++
// an input: x
let input = (ins
  OneFlow_Tensor:$x
);
```

An input tensor `x` is defined above, and its input format is `input type:$name`.


The input types contain:

- `OneFlow_Tensor`
- `Variadic<OneFlow_Tensor>` indicates the tensor is variable, such as concat op, which supports concatenating a variable number of tensors.
- `Optional<OneFlow_Tensor>` indicates the tensor is an optional one, which is dispensable, such as add_output of conv op.


An op can also define multiple input tensors, for example:

```c++
  // two inputs: a, b
  let input = (ins
    OneFlow_Tensor:$a,
    OneFlow_Tensor:$b
  );
```

Define the output of the op by overriding the `output` domain, for example:

```c++
let output = (outs
  OneFlow_Tensor:$out0,
  OneFlow_Tensor:$out1
);
```

Two input tensors are defined above.


#### Attributes 

By overriding the `attrs` domain, you can define the attributes of the op. For example, you can define the attribute `rate` of [dropout](https://oneflow.readthedocs.io/en/master/functional.html#oneflow.nn.functional.dropout) as follows:

```c++
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$rate
  );
```

The code above represents that the data type of `$rate` is `F32Attr`, and its default value is `0.`. But, we can also not specify its default value as follows:

```c++
  let attrs = (ins
    F32Attr:$rate
  );
```

View the definitions of I32Attr, F32Attr, BoolAttr, StrAttr, I32ArrayAttr, and other common basic data types in [OpBase.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td#L1077-L1086).


View the definitions of ShapeAttr, DTArrayAttr, and other OneFlow's customized data types in [OneFlowBase.td](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/include/OneFlow/OneFlowBase.td#L27-L35). 


#### Export and implement the inference interface

There are some other domains used to specify whether there is a need to generate corresponding interfaces, and these interfaces are usually the ones for inference in the process of constructing the computation graph, such as shape inference, data type inference, SBP inference, and other inferences. 

OneFlow-TableGen is designed to generate interfaces for these functions, so developers need to implement these interfaces in automatically-generated cpp files. By default, no interfaces will be generated, and developers need to explicitly specify which interfaces need to be generated. 



```c++
  let has_check_fn = 1;                         // generating attribute checking interface
  let has_logical_tensor_desc_infer_fn = 1;     // generating logical shape inference interface
  let has_physical_tensor_desc_infer_fn = 1;    // generating physical shape inference interface
  let has_get_sbp_fn = 1;                       // generating get sbp interface
  let has_sbp_signature_infer_fn = 1;           // generating sbp signature interface, which will be removed later, and has_nd_sbp_infer_fn is suggested
  let has_data_type_infer_fn = 1;               // generating data type inference interface
  let has_device_and_stream_infer_fn = 1;       // generating device inference interface
  let has_input_arg_modify_fn = 1;              // generating the input modify interface, such as specify is_mutable、requires_grad (for Lazy)
  let has_output_arg_modify_fn = 1;             // generating the output modify interface, such as specify is_mutable、requires_grad (for Lazy)
  let has_output_blob_time_shape_infer_fn = 1;  // generating the output time shape inference interface
  let has_nd_sbp_infer_fn = 1;                  // generating the nd sbp inference interface
```

Some commonly-used interfaces are as follow:

```c++
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_data_type_infer_fn = 1;
  let has_get_sbp_fn = 1;
```


After getting familiar with the concepts and instructions listed above, you can begin to modify the [oneflow/ir/include/OneFlow/OneFlowUserOps.td](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/include/OneFlow/OneFlowUserOps.td) file.



View the complete definition of the leaky_relu_yzh op [here](https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/ir/include/OneFlow/OneFlowUserOps.td#L8418-L8433).


After adding the op definition in `OneFlowUserOps.td`, the make operation will automatically create some files in the `oneflow/core/framework/` directory under the **build** directory:

- `op_generated.h`：the op C++ class generated by parsing the `.td` file;
- `op_generated.cpp`：the op registration code generated by parsing the `.td` file (which includes the code to call `REGISTER_USER_OP` macro);


Next, a cpp file needs to be created under the [oneflow/user/ops](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/user/ops) directory, which is an interface to implement the operator. 

For example, the file corresponding to leaky_relu_yzh is [oneflow/user/ops/leaky_relu_yzh_op.cpp](https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/user/ops/leaky_relu_yzh_op.cpp#L21-L79), which has implemented interfaces to infer logic tensor, physical tensor, SBP information, and the output data type. 



### Implement Kernel logic

Operators can be executed on different devices spanning CPU, GPU, DCU, and others, so it's necessary to implement different computational logic.

Related code:

- [Leaky ReLU CPU Kernel](https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/user/kernels/leaky_relu_yzh_kernel.cpp)
- [Leaky ReLU GPU Kernel](https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/user/kernels/leaky_relu_yzh_kernel.cu)


#### Computational logic of CPU

```cpp
template<typename T>
class CpuLeakyReluYZHKernel final : public user_op::OpKernel {
 public:
  CpuLeakyReluYZHKernel() = default;
  ~CpuLeakyReluYZHKernel() = default;
 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    const auto alpha = ctx->Attr<float>("alpha");
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha * x_ptr[i]; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
```

To implement the kernel in OneFlow, it's essential to define a class inherited from `oneflow::user_op::OpKernel` and override the virtual functions.


In the code above, two virtual functions: `Compute` and `AlwaysComputeWhenAllOutputsEmpty` have been overridden, and the reasons are listed as follows:

- `Compute` must be overridden to implement specific computational logic;
- `AlwaysComputeWhenAllOutputsEmpty` must be overridden. For most ops, it's fine to directly return false. For the tiny minority of operators that need to maintain state internally and call kernel to compute even though the output is empty, it should return true;


The `Compute` method obtains specific data about the input tensor, the output tensor, and attributes by calling interfaces in `user_op::KernelComputeContext* ctx` then processes the data according to the op's computational logic.


The processing logic of `CpuLeakyReluKernel::Compute` is explained as follows:

- Getting 2 tensors- `x` and `y`. Note that the string passed into `Tensor4ArgNameAndIndex` should be the same as the name defined in `OneFlowUserOps.td`;
- Acquiring the element number of `x`, which will be used for computation in the `for` loop;
- Acquiring the attribute `alpha`;
- Entering the `for` loop, whose number is `elem_cnt`, and writing the result into `y`;


#### Register kernels

After implementing a kernel class, we need to call `REGISTER_USER_KERNEL` to register it.

```cpp
#define REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("leaky_relu_yzh")                              \
      .SetCreateFn<CpuLeakyReluYZHKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));
```


The macro `REGISTER_USER_KERNEL` being called here contains the following information:

1. op type name: For which op is this kernel registered?
2. `SetCreateFn<T>()`: The template parameter `T` in this template method is the kernel class that is implemented. It will be used to create kernel objects by OneFlow Runtime.
3. `SetIsMatchedHob`: Since there might be multiple kernels for one op, we call `SetIsMatchedHob` to choose the right kernel for computation based on the needs of various physical devices and data formats. This method takes one expression as the input. If the expression equals `true`, OneFlow will call that kernel to perform the computation. The matching logic of the above code goes: If the hardware device is `cpu` and the data type of `y` is the same as `dtype`, OneFlow will call the kernel class that is registered (`CpuLeakyReluYZHKernel<dtype>`).


#### Computational logic of GPU

To get started on CUDA programming, you may click the following videos:

- [Video: The Rise of CUDA](https://www.bilibili.com/video/BV1Mb4y1p7BG)
- [Video: Start CUDA With A Simple Program](https://www.bilibili.com/video/BV1bF411s76k)
- [Video: CUDA Thread Hierarchies](https://www.bilibili.com/video/BV1MZ4y127Sq)


But, above all, we recommend the official tutorial from Nvidia: [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

With some basic knowledge of CUDA, you will find it easy to understand how leaky_relu is implemented in CUDA.

The first step is to define the CUDA kernel that performs leaky_relu forward computation.

```cpp
template<typename T>
__global__ void LeakyReluForwardGpu(const int n, const float alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : x[i] * alpha; }
}
```


In the above code, the macro [CUDA_1D_KERNEL_LOOP](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/device/cuda_util.h#L91-L94) is called for computation.

In the Compute function, the macro `RUN_CUDA_KERNEL` (also defined in the document `cuda_util.h`) is called to start the kernel.

For the implementation of the corresponding GPU kernel classes, please check:
https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/user/kernels/leaky_relu_yzh_kernel.cu#L32-L49


The aforementioned macro `RUN_CUDA_KERNEL` is used here. It is defined as follows:

```cpp
#define RUN_CUDA_KERNEL(func, device_ctx_ptr, thread_num, ...)           \
  func<<<SMBlocksNum4ThreadsNum(thread_num), kCudaThreadsNumPerBlock, 0, \
         (device_ctx_ptr)->cuda_stream()>>>(__VA_ARGS__)
```

1. The first parameter `func` refers to the name of the kernel.
2. The second parameter `device_ctx_ptr` means device context. It is used later to get the corresponding cuda_stream.
3. The third parameter `thread_num` is the number of threads that are to be started, which determines the number of the blocks needed.


Since leaky_relu is an elementwise computation with no mutual influence between the elements, we start ` elem_cnt ` threads. 


The subsequent registration is the same as that in the CPU version so you may just refer to the following code:
https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/user/kernels/leaky_relu_yzh_kernel.cu#L51-L62


> As can be seen, the `Compute` functions of various devices share a large proportion of codes. A better way to organize the code is to use a `.cpp` document to include the kernel and the registration logic, a `.cu` document to include codes for GPU kernel functions and GPU template specialization, and a `.h` document for defining and registering macros. You may check the code in [dim_gather_kernel_\*](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/user/kernels) for reference.  


> OneFlow provides Primitive components to fit for various kinds of devices. Check [Primitive PR](https://github.com/Oneflow-Inc/oneflow/pull/6234) for details.


### Export functional interface


For more information about the functional interface layer, please click [here](https://github.com/Oneflow-Inc/oneflow/wiki/Functional-Interface).

In simple terms, the functional layer is the connection between the Python and C++ layers:

```text

   ┌─────────────┐
   │   Module    │
   │  (Python)   │
   ├─────────────┤
   │             │
   │ Functional  │
   ├─────────────┤
   │             │
   │ Op/Kernels  │
   │   (C++)     │
   └─────────────┘
```

Therefore, after defining the op and registering the kernels, we need to export the functional interface for the op so that users can call that op using Python code.



It takes the following two steps to export the functional interface: 

1. Implement the corresponding functor and register it.
2. Add description of the interface in [oneflow/core/functional/functional_api.yaml](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/functional/functional_api.yaml).



#### Implement the corresponding functor and register it


For a leaky_relu_yzh op, define it in [activation_functor.cpp](https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/core/functional/impl/activation_functor.cpp#L391-L421) as follows:

```cpp
class LeakyReluYZHFunctor {
 public:
  LeakyReluYZHFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("leaky_relu_yzh").Input("x").Output("y").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& alpha) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<float>("alpha", alpha));
    return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};
```


- Construct the op `leaky_relu_yzh` in the constructor.
- Implement the overloaded operator `operator()`, call the constructed op via `Dispatch`, and pass the input and attributes.


Similarly, we export the functional interface for `LeakyReluGrad` to facilitate the subsequent writing of the differentiation logic. 

Last, we need to register the functor in the Functional Library:
https://github.com/Oneflow-Inc/oneflow/blob/7ab4b0f08c86a6f8af08b44daa510725942288fb/oneflow/core/functional/impl/activation_functor.cpp#L610-L611


```cpp
m.add_functor<impl::LeakyReluYZHFunctor>("LeakyReluYZH"); // Please note that the name in the string will be used in the `functional_api.yaml` later.

After being registered by `m.add_functor`, the functor can be used in the C++ layer. For example, you can call `LeakyReluFunctor` using `functional::LeakyRelu`.



#### Add description of the interface in functional_api.yaml


The functional interface is automatically generated in the `build` process after the `yaml` configuration file is parsed.


Write the configuration in the [functional_api.yaml](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/functional/functional_api.yaml).

https://github.com/Oneflow-Inc/oneflow/pull/8350/files#diff-4b35c1dcdbae81b75439ba570bc149554ca85b83757430613fcb612ae25972afR572-R579

```text
- name: "leaky_relu_yzh"
  signature: "Tensor (Tensor x, Float alpha) => LeakyReluYZH"
  bind_python: True
```


- `name` refers to the name of the function after it is exported to the Python interface. For example, if the function is exported to be used in Python, the usage of it should be: 


```python
flow._C.leaky_relu_yzh(...)
```


- `signature` is used to describe the relationship between the interface prototype and the C++ code. On the left of `=>` is the interface prototype and on the right of it is the corresponding name of the function in the Functional Library. Here the `LeakyRelu` is the same as the designated name in the export process before.

- `bind_python` means if this interface needs to be bound to a Python interface. For example, `leaky_relu_grad` would be used in the C++ layer for differentiation but would not be used in the Python layer, so it would be set to `False`.


After the above steps, the newly added op can now support forward computation. You can test it after the codes are compiled: 

```python
import oneflow as flow 
import numpy as np


x_tensor = flow.Tensor(np.random.randn(3, 3))
out = flow._C.leaky_relu_yzh(x_tensor, alpha=0.2)
```


However, another separate registration is needed for the op to support back propagation. Still, we need to export the `LeakyReluGrad` required in back propagation as a functional interface.

```cpp
- name: "leaky_relu_yzh_grad"
  signature: "Tensor (Tensor x, Tensor dy, Float alpha) => LeakyReluYZHGrad"
  bind_python: False
```



### Implement the backward logic used in differentiation


Back propagation is, in essence, equivalent to the Chain Rule in advanced mathematics, only that it is made more modular and easier to reuse by `Autodiff`.


To learn some basic knowledge about `Autodiff`, please read [CSC321 Lecture 10: Automatic Differentiation
](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf).


Logically speaking, the following information is necessary for differentiating an op in backward computation.

- the input and output in forward computation;
- the attributes in forward computation;
- the forward output gradient from the previous layer in backward computation (the following layer in forward computation).

Separate registrations are needed for the time being, but we plan to merge the backward logic of the Graph mode and that of the Eager mode.

#### Register backward computation for the Eager mode

The differentiation is completed in [oneflow/core/autograd/gradient_funcs/activation.cpp](https://github.com/Oneflow-Inc/oneflow/pull/8350/files#diff-36aeebf7fd5a8b88bd5af87774e7b0b4f76fed42cfb75044d6eebdfe65628347R213-R253), which contains the followings:


- LeakyReluYZHCaptureState: the structure to store data

It is a simple structure inherited from `AutoGradCaptureState`. It is used to store the attributes of the op for subsequent differentiation. 


```cpp
struct LeakyReluYZHCaptureState  : public AutoGradCaptureState {
  bool requires_grad; // If input x requires gradients
  float alpha=0.0; // The input parameter (alpha)
};
```


- LeakyReluYZH class: inherited from `OpExprGradFunction`. Three functions need to be overriden: `Init`, `Capture`, and `Apply`.

```cpp
class LeakyReluYZH : public OpExprGradFunction<LeakyReluYZHCaptureState> {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    //...
  }

  Maybe<void> Capture(LeakyReluYZHCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    //...
  }

  Maybe<void> Apply(const LeakyReluYZHCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    //...
  }
};
```



- Init: performs initialization, such as initializing the attributes based on the configuration of the forward op. 

```cpp
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
    return Maybe<void>::Ok();
  }
```



- Capture: captures the tensor and its attributes in the forward computation for later use in the differentiation.

Take LeakyReluYZH as an example, we need: a) the input tensor (if the tensor > 0, the gradient is 1; if the tensor < 0, the gradient is alpha); b) the value of alpha.

```cpp
  Maybe<void> Capture(LeakyReluYZHCaptureState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs, const AttrMap& attrs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 1);                      // To determine if the number of input is 1
    ctx->requires_grad = inputs.at(0)->requires_grad();        // To determine if the input requires gradients
    if (!ctx->requires_grad) { return Maybe<void>::Ok(); }     // If the input does not require gradients, that means there is no need for differentiation, so the next step is to directly return `Maybe<void>::Ok()`.
    ComposedAttrMap composed_attrs(attrs, base_attrs_);
    ctx->alpha = JUST(composed_attrs.GetAttr<float>("alpha")); // To get alpha and store it in `ctx->alpha`.
    ctx->SaveTensorForBackward(inputs.at(0));                  // To call the `SaveTensorForBackward` method and save the input tensor.
    return Maybe<void>::Ok();
  }
```


- Apply: the function that calculates the gradients. We can call ` LeakyReluGrad ` (registered under the functional interface) to calculate the gradients for the previous tensor and then return the gradients.


```cpp
  Maybe<void> Apply(const LeakyReluYZHCaptureState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);  //Check if the number of tensors is 1
    in_grads->resize(1);                      // Resize(1) since there is only 1 input
    if (ctx->requires_grad) {
      const auto& x = ctx->SavedTensors().at(0); // Call the `SavedTensors` interface and retrieve the tensor that is saved through ` SaveTensorForBackward ` earlier.
      in_grads->at(0) = JUST(functional::LeakyReluYZHGrad(x, out_grads.at(0), ctx->alpha)); // To get x, dy, alpha, pass them to `LeakyReluYZHGrad` for computation，and return the gradients to `in_grads->at(0)`.
    }
    return Maybe<void>::Ok();
  }
```


The final step is registration. The first parameter is `op type name`, and the second parameter a class inherited from `OpExprGradFunction`.


```cpp
REGISTER_OP_EXPR_GRAD_FUNCTION("leaky_relu_yzh", LeakyReluYZH); //The second parameter is the name of the class used for differentiation.
```


#### Register backward computation for the Graph mode


The backward code of registering leaky_relu_yzh op for the Graph mode can be found [here](https://github.com/Oneflow-Inc/oneflow/pull/8350/files#diff-ef94ddb8f5c25689f2c6fab7a9675f16c95a22018a8c01647b4398ce2fb85a8bR81-R970.


```c++
REGISTER_USER_OP_GRAD("leaky_relu_yzh")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      // To form a leaky_relu_yzh_grad_op_name (leaky_relu_yzh_grad)  based on the op type name in the forward computation.
      const std::string leaky_relu_yzh_grad_op_name = ctx->FwOp().op_name() + "_grad";
      
      ctx->DefineOp(leaky_relu_yzh_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        // To construct an op (the op type name of it is leaky_relu_yzh_grad)
        // The gradient of y (output of the forward computation) is used as the input (dy) of leaky_relu_yzh_grad. 
        // The x in the forward computation is used as the input (x) of leaky_relu_yzh_grad.
        // Output as dx
        // attr alpha is the same as that in the forward computation        
        return builder.OpTypeName("leaky_relu_yzh_grad")
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .InputBind("x", ctx->FwOp().input("x", 0))
            .Attr("alpha", ctx->FwOp().attr<float>("alpha"))
            .Output("dx")
            .Build();
      });
      // To bind the output (dx) of the leaky_relu_yzh_grad_op_name op to the backward gradient of the input (x) of the forward computation, which means
      //  the gradient of the input (x) of leaky_relu_yzh = the output (dx) of leaky_relu_yzh_grad
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                                [&ctx, &leaky_relu_yzh_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(leaky_relu_yzh_grad_op_name).output("dx", 0);
                                });
      return Maybe<void>::Ok();
    });
```



## Tests and docs

After completing all the steps introduced in this article, the op is only made "available" in OneFlow. The job is not done yet (i.e. test user op and write API docs). For further information, please refer to our articles:

- [How to Test User Operator](./howto_test_user_op.md)
- [How to Write API Docs](./howto_write_api_docs.md)
