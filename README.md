- [ ] more models
    - [ ] nn.functional (softmax, mean, sum)
    - [ ] passthrough
    - [ ] reshape/flatten
- [ ] DimVar folding / flow from .shape/.size
    - .shape - X.shape - special-cased in Path lookup
    - .size() - special-cased in method lookup
    - binops between dimvars
    - tuples? - Tuple(Vec<Variable>)

- [ ] method calls / side effects

- [ ] Flesh out IR
    - [ ] Tuples / Lists
    - [ ] `break`/`continue`
- [ ] Import resolution