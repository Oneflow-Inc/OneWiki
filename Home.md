Welcome to the OneFlow wiki!


# OneFlow roadmap



## May 2022

version:   OneFlow v0.6.0



### OneFlow 1.0 interface project

A **new**, easy-to-use Python user interface, fully compatible with PyTorch, support OOP modeling and **distributed Eager training** !

- oneflow.tensor
- oneflow.autograd
- oneflow.nn.Module
- high level API (oneflow.Model) support one code switch between Lazy and Eager
- oneflow.ComputeGraph
- oneflow.autograd.Function
- OpBuilder
- Multi-Client



### 2-D SBP

Support hierarchical N-D SBP Parallel for super large scale distributed training with data/model parallel at the same time.

- parallel distribution
- hierarchical boxing sub graph builder 
- hierarchical parallel cast
- GPT-3 model release and benchmark



### Device

Support more general computing chips and AI computing chips.

- extend device type
- support Cambrian



