- [ ] more models
    - [x] nn.functional (softmax, mean, sum)
    - [x] passthrough
    - [ ] ones, ones_like and friends
    - [x] change broadcast to model, use for torch.add, etc.
    - [ ] unsqueeze / squeeze / expand_dims
    - [ ] reshape/flatten/transpose/permute

- [ ] DimVar folding / flow from .shape/.size
    - [x] .shape - X.shape - special-cased in Path lookup
    - [ ] .size() - special-cased in method lookup
    - [x] binops between dimvars
    - [x] tuples - Tuple(Vec<Variable>)
    - [x] Dim Exprs
- [ ] torch method calls
- [ ] finish signature model
    - [ ] handle concrete args
    - [ ] enforce existence of param with singleton named DimVar for each symbolic DimVar present in signature
    - [ ] Generalize effect of function (optional/tuple/dimvar return type, maybe mutation?)
- [ ] ellipsis

- [ ] Usability improvements
    - [ ] miette-like diagnostics (more provenance info?)
    - [ ] .pyi stubs
    - [ ] LSP
    - [ ] multiple files
    - [ ] object orientation

- [ ] method calls / side effects

- [ ] Flesh out IR
    - [x] Tuples
    - [ ] `break`/`continue`
    - [ ] Lists?
- [ ] Import resolution


# Refactor
- [x] Make broadcast_resolve a Model
- [ ] Set up interning and stop cloning everything everywhere
- [ ] indexical over locations
- [ ] modularize into smaller functions
- [ ] switch to jaxtyping annotations 
- [ ] doc comments on structs and functions
