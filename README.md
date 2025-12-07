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
    - [ ] Dim Exprs

- [ ] method calls / side effects

- [ ] Flesh out IR
    - [x] Tuples
    - [ ] `break`/`continue`
    - [ ] Lists?
- [ ] Import resolution


# Refactor
- [x] Make broadcast_resolve a Model
- [ ] Set up interning and stop cloning everything everywhere
- [ ] proper error structs / messages (might want to retain more provenance info)
- [ ] indexical over locations
- [ ] modularize into smaller functions
- [ ] switch to jaxtyping annotations 
