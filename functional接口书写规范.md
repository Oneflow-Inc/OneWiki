functional接口与pytorch的nn.functional接口是对应的，其设计理念是尽可能无状态，调用方式简单高效，且不需要额外的上下文（指的是不需要先构造op）。其存在以下一些优点，

- 高效的参数解析过程

  functional接口直接从C++导出到Python，其参数会自动完成从Python的object对象到functional接口指定的C++类型的转换，整个过程都在C++中完成。相比之前op执行前需要在Python中查询得到属性类型，然后将属性转换到对应类型之后再构造cfg::AttrVal的方式，新的方式理论上更加高效。

- 全局静态算子

  functional接口被设计为尽可能无状态的，所以我们可以使用静态算子，配合动态属性来支持不同情况下的计算，避免算子重复构建。

- 参数更明确

  functional接口的参数都是具名的，也指定了参数类型，相比之前op调用的方式，输入参数非常明确，参数类型也不会传递出错。

- C++和Python可复用

  functional接口可以通过pybind导出到Python，也可以直接在C++中使用，比如在gradient function中就可以直接使用，而不再需要创建OpExpr了。

下文主要介绍如何新增一个functional接口，主要分成两个步骤。

- 首先我们需要为接口增加一个函数执行体。
- 在functional_api.yaml文件中增加自动生成接口的配置信息。

所有为functional接口增加的函数执行体目前都放在目录`oneflow/core/functional/impl`中。函数执行体被设计为类或结构体，持有一个或多个op，在构造函数中应该把所有需要的op都构建好，通常只需要声明op的输入和输出key和count，属性则可以省略。同时函数执行体需要实现`operator()`接口，在这个接口中调用op完成计算。

```c++
class ScalarAddFunctor {
 public:
  ScalarAddFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scalar_add").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const Scalar& scalar) const {
    MutableAttrMap attrs;
    if (scalar.IsFloatingPoint()) {
      JUST(attrs.SetAttr<double>("float_operand", JUST(scalar.As<double>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", true));
      JUST(attrs.SetAttr<bool>("has_int_operand", false));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    } else if (scalar.IsIntegral()) {
      JUST(attrs.SetAttr<int64_t>("int_operand", JUST(scalar.As<int64_t>())));
      JUST(attrs.SetAttr<bool>("has_float_operand", false));
      JUST(attrs.SetAttr<bool>("has_int_operand", true));
      return OpInterpUtil::Dispatch<Tensor>(*op_, {x}, attrs);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
};
```

对于`operator()`接口有几点注意事项，

- 接口中的输入参数目前只支持`Tensor`、`Shape`、`DataType`、`Scalar`、`Generator`、`TensorIndex`、`Device`、`Placement`、`Sbp`、`SbpList`以及大部分标准的基础数据类型（比如`float`、`double`、`int`、`int32`、`int64`、`bool`、`string`等），以及基础数据类型的向量（比如`std::vector<float>`、`std::vector<int32>`等）。

- 部分op的参数存在float和integer都需要支持的情况（比如`scalar_add`），可以使用`Scalar`来代替，调用时可以传任意的浮点或整型值，在`operator()`接口中根据类型来转换。比如下面都是合法的，

  ```python
  y = F.scalar_add(x, 1)
  y = F.scalar_add(x, 1.0)
  ```

函数执行体定义完成后，需要将其注册到Function Library。

```c++
ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ScalarAddFunctor>("ScalarAdd"); // 注意这里的注册name，需要和自动生成接口的配置文件中一致
}
```

之后就开始修改functional_api.yaml文件，增加一条接口配置信息。每一个接口信息都由三个字段组成，

```yaml
- name: "xxx"
  signature: "R(Args...) => Func"
  bind_python: True or False
```

- name指定当前接口的name，导出到Python中的接口用的是这个name。

- signature指定接口函数的签名，签名需要和对应的函数执行体保持一致，并且这里的`Func`作为signature的函数名，需要和前面注册的函数执行体的name一致，C++接口用的是这个name。这里的参数类型做了一些简化，比如现在支持的输入参数类型有，

  ```c++
  "Tensor", "TensorTuple", "Scalar", "Int", "Int32", "Int64", "Float", "Double", "String", "Bool",
  "ScalarList", "IntList", "Int32List", "Int64List", "FloatList", "DoubleList", "StringList",
  "BoolList", "DataType", "Shape"
  ```

  输出参数类型主要有

  ```c++
  "Tensor", "TensorTuple"，"Void", "Int", "Int32", "Int64", "Float", "Double", "String", "Bool"
  ```

- bind_python指定是否需要为当前接口生成Python接口。

一个完整的例子如下，

```yaml
- name: "add_scalar"
  signature: "Tensor ScalarAdd(Tensor x, Scalar alpha)"
  bind_python: True
```

上面的接口自动导出到python中后，可以在python中这么使用，
```python
F.add_scalar(x, 1)
F.add_scalar(x, alpha=1)
F.add_scalar(x=x, alpha=1)
```

signature在书写时也有几点注意事项：

- 参数类型必须和函数执行体的`operator()`接口中的类型保持一致，比如接口中的`std::shared_ptr<Tensor>`对应这里的`Tensor`，输出的`Maybe<Tensor>`也对应这里的`Tensor`(我们会自动替换成Maybe的版本)。

- 参数可以设置默认值，比如可以给上面的alpha设置一个默认值，

  ```yaml
  signature: "Tensor ScalarAdd(Tensor x, Scalar alpha=1)"
  ```

- 参数列表中间可以引入符号`*`，表示在Python中调用时，符号`*`之后的参数必须以key-word方式传参数。比如，

  ```yaml
  signature: "Tensor ScalarAdd(Tensor x, *, Scalar alpha=1)"
  ```
  则合法的使用方式如下，

  ```python
  y = F.add_scalar(x)  # alpha is 1 by default
  y = F.add_scalar(x, alpha=1)
  ```

同时目前也支持多个signatures（支持函数重载）的情况，比如
```yaml
- name: "xxx"
  signature: [
    "R(Args...) => Func1",
    "R(Args...) => Func2",
  ]
  bind_python: True or False
```
需要注意的是，signatures之间必须是正交的，否则会出现signature被掩盖的情况。