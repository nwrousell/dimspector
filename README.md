# Dimspector
An in-development tool to statically infer tensor shapes in PyTorch code. 

## Usage

### Running
```
cargo run -- path/to/file.py
```

### Running tests
```
cargo insta test
```

## Directory structure

- `src/`
    - `main.rs` - CLI
    - `ast/` - mostly just wraps [rustpython parser](https://github.com/RustPython/Parser)
    - `ir/` - types and lowering code for intermediate representation. 
    - `analysis/`
        - `mod.rs` - main analysis code
        - `models.rs` - model definitions
- `tests/programs/` - test files

## Roadmap
- [ ] more models
    - [x] nn.functional (softmax, mean, sum)
    - [x] passthrough
    - [x] ones, ones_like and friends
    - [x] change broadcast to model, use for torch.add, etc.
    - [ ] unsqueeze / squeeze / expand_dims
    - [x] reshape/transpose
    - [ ] flatten/permute

- [ ] DimVar folding / flow from .shape/.size
    - [x] .shape - X.shape - special-cased in Path lookup
    - [ ] .size() - special-cased in method lookup
    - [x] binops between dimvars
    - [x] tuples - Tuple(Vec<Variable>)
    - [x] Dim Exprs
- [ ] torch method calls
- [ ] finish signature model
    - [x] handle concrete args
    - [ ] enforce existence of param with singleton named DimVar for each symbolic DimVar present in signature
    - [ ] Generalize effect of function (optional/tuple/dimvar return type, maybe mutation?)
- [ ] ellipsis

- [ ] Usability improvements
    - [x] miette for code context
        - [ ] more provenance info 
    - [ ] .pyi stubs
    - [ ] LSP
    - [ ] multiple files
    - [ ] object orientation

- [ ] method calls / side effects

- [ ] Flesh out IR
    - [x] Tuples
    - [ ] `break`/`continue`
    - [x] Lists?
- [ ] Import resolution
- [ ] if user has annotated var, check against our inference

### Refactoring
- [x] Make broadcast_resolve a Model
- [ ] Set up interning and stop cloning everything everywhere
- [ ] indexical over locations
- [ ] modularize into smaller functions
- [ ] switch to jaxtyping annotations 
- [ ] doc comments on structs and functions
