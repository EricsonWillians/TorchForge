# TorchForge Formal Language Specification

## Extended Backus-Naur Form (EBNF)

### Core Program Structure
```ebnf
Program         = { Statement }+ ;

Statement       = ModelDef
                | TrainStmt
                | DataPipeline
                | ImportDecl
                | PythonBlock
                | ConfigBlock
                | Comment ;
```

---

### Model Definitions
```ebnf
ModelDef        = "model" Identifier [Inheritance] "{" Layer+ "}" ;
Inheritance     = ":" Identifier [ "," Identifier ]* ;
Layer           = LayerType [ShapeComment]
                | ControlFlow ;
                
LayerType       = Identifier "(" [ ParameterList ] ")" ;
ShapeComment    = "→" TensorShape ;  <!-- Compiler-enforced shape tracking -->
```

**Example:**
```rust
model ResNet50 : ImageModel, Quantizable {
  Input(shape=[3,224,224]) → [3,224,224]
  Conv2d(filters=64, kernel=7, stride=2) → [64,112,112]
  MaxPool2d(pool_size=3, stride=2)
}
```

---

### Layer Specifications
```ebnf
ParameterList   = Parameter { "," Parameter }* ;
Parameter       = Identifier "=" Value 
                | StringLiteral
                | NumericLiteral
                | BooleanLiteral ;

Value           = ScalarValue 
                | TensorShape
                | ListLiteral
                | MathExpr ;

MathExpr        = "(" MathOp ")" ;
MathOp          = Value BinOp Value ;
BinOp           = "+" | "-" | "*" | "/" | "@" ;
```

**Example:**
```rust
Conv3d(
  filters=(prev_channels * 2),  <!-- Mathematical expression -->
  kernel=[3,3,3],
  padding='same'
)
```

---

### Training Statements
```ebnf
TrainStmt       = "train" Identifier "on" DataRef [DistributedClause] ":" 
                  TrainConfig ;
                  
DistributedClause = "with" DistributionStrategy ;
DistributionStrategy = "strategy=" StrategyType 
                      | "devices=" IntLiteral ;

StrategyType    = "ddp" | "fsdp" | "deepspeed" | "horovod" ;

TrainConfig     = { ConfigPair }+ ;
ConfigPair      = Identifier "=" (Value | PythonLambda) ;
PythonLambda    = "λ" "(" ParamList ")" ":" PythonExpr ;
```

**Example:**
```rust
train GAN on celebahq:
  strategy=ddp
  devices=8
  generator_opt = Adam(lr=0.0002, betas=(0.5, 0.999))
  discriminator_opt = RMSprop(lr=0.00005)
  loss = λ(real, fake): (fake - real + 1).clamp(0, None).mean()
  metrics = ["fid", "is"]
```

---

### Dataset Pipelines
```ebnf
DataPipeline    = Identifier "=" DataSource PipeOperator+ ;
DataSource      = BuiltInDataset | CustomLoader | HFDataset ;

PipeOperator    = "|>" TransformFn ;
TransformFn     = Identifier "(" [ ParameterList ] ")" 
                | AugmentationBlock ;

AugmentationBlock = "augment" "{" { AugmentOp }+ "}" ;
AugmentOp       = Identifier Probability? ;
Probability     = "@" FloatLiteral ;
```

**Example:**
```rust
dataset = hf.load_dataset("coco", split="train")
          |> augment {
            RandomHorizontalFlip @0.5
            ColorJitter(brightness=0.2)
          }
          |> tokenize(model="gpt2")
          |> batch(128, dynamic_padding=True)
```

---

### Type System
```ebnf
TypeDecl        = TensorType | ScalarType | StructType ;
TensorType      = "Tensor" "[" Shape "," DType "]" ;
Shape           = "[" Dim { "," Dim }* "]" ;
Dim             = IntLiteral | Identifier | "?" ;
DType           = "float32" | "bfloat16" | "int64" | "bool" ;

ScalarType      = "Int" | "Float" | "String" | "Boolean" ;
StructType      = "Tuple" "(" TypeDecl { "," TypeDecl }* ")"
                | "Dict" "[" KeyType ":" ValueType "]" ;
```

**Type Rules:**
1. **Dimensionality Preservation**
   ```
   Γ ⊢ input:Tensor[[B,C,H,W], float32]
   Γ ⊢ conv:Conv2d(in=C, out=K)
   ---------------------------- [T-Conv]
   Γ ⊢ conv(input):Tensor[[B,K,H',W'], float32]
   ```
   
2. **Operator Propagation**
   ```
   Γ ⊢ x:Tensor[S1, T], y:Tensor[S2, T]
   Γ ⊢ op:Operator(T, T) → T
   S1 ≡ S2
   -------------------------- [T-BinOp]
   Γ ⊢ x op y:Tensor[S1, T]
   ```

---

### Python Interoperability
```ebnf
PythonBlock     = "python" "{" { PythonStmt }+ "}" ;
PythonStmt      = PythonCode ;  <!-- Arbitrary Python code -->
```

**Example:**
```rust
python {
  class CustomAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        
    def forward(self, x):
        return self.query(x)
}
```

---

### Preprocessor Directives
```ebnf
ImportDecl      = "import" ImportPath [ "as" Identifier ] ;
ImportPath      = Identifier { "." Identifier }* 
                | StringLiteral ;

ConfigBlock     = "config" "{" { ConfigPair }+ "}" ;
```

**Example:**
```rust
import torchvision.transforms as T
import "utils/augment.py"

config {
  precision = bfloat16
  log_dir = "./logs"
  auto_checkpoint = true
}
```

---

### Terminal Symbols
```ebnf
Identifier      = Letter { Letter | Digit | "_" }* ;
StringLiteral   = '"' { PrintableChar }* '"' ;
NumericLiteral  = IntLiteral | FloatLiteral ;
IntLiteral      = Digit { Digit }* ;
FloatLiteral    = Digit+ "." Digit+ [ Exponent ] ;
Exponent        = ("e" | "E") ["+" | "-"] Digit+ ;

Comment         = "//" { PrintableChar }* Newline
                | "/*" { PrintableChar | Newline }* "*/" ;
```

---

This EBNF specification formally defines TorchForge's syntax while maintaining several key properties:

1. **Context-Sensitive Constraints**
   - Shape compatibility between consecutive layers
   - Device placement consistency (CPU/GPU/TPU)
   - Gradient flow validation through computational graph

2. **Static Verification**
   ```rust
   model Invalid {
     Input(shape=[256])
     LSTM(units=128)        → [128, 256]
     Conv1d(filters=64)     // Error: Expecting 3D input
   }
   ```

3. **Target-Specific Code Generation**
   ```ebnf
   CodeGenTarget = "pytorch" 
                 | "onnx" 
                 | "tensorrt" 
                 | "torchscript" ;
   ```

This document serves as the authoritative reference for TorchForge's syntax and static semantics. Runtime semantics are defined by the PyTorch execution model.