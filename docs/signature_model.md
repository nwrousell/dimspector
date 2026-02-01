# Signature Model
The core of our analysis is the Signature Model, which allows calls to be checked against annotated function and method signatures, and to infer the return variables.


We start with an annotated function and a callsite:
```python
def foo(a: Float[Tensor, "d1"], b: Float[Tensor, "d2"], c: Float[Tensor, "d1+d2"]) -> Float[Tensor, "d1+d2"]:
    return torch.cat((a, b)) + c

def main():
    a = torch.ones(10)
    b = torch.ones(20)
    c = torch.ones(30)
    ab = foo(a, b, c)
```

Importantly, **each symbolic dimvar must appear as either an integer param or a singleton `DimVar::Named` axis of a tensor annotation**. This makes construction of the association between param and arg dimvars simple (and is best practice while annotating anyhow). 

At the callsite of `foo`, the signature model takes in the signature (parameter and return annotations, represented by `Variable`s) and the `Variable`s of the arguments. 

The Signature Model has two jobs:
1. Check that the arguments meet any constraints defined in the annotation, and
2. Infer the return `Variable` via substitution

## Algorithm
1. Match up each param `Variable` with the corresponding argument `Variable`
2. Do an initial pass to build substitution map:
    1. If param is integer, treat as `Named` dimvar, adding arg dvar to map (erroring if existing value doesn't match)
    2. If param is Tensor, check that rank matches arg and then for each dimension in param:
        1. `Named` dimvar, add arg dimvar to substitution map (erroring if inconsistent with existing value)
        2. else: skip this param (will be handled in second pass)
    3. If param is tuple, recur above procedure for each element
3. Do a second pass over (param, arg) pairs to check that arguments conform to param annotation
    1. if param is Tensor, for each dimension in param:
        1. `Concrete` dimvar: check that arg matches
        2. `Named` dimvar/dim expr: `substitute` param `DimVar` with substitution map, check for equality with argument `DimVar`
    2. If param is tuple, recur above procedure for each element
4. Infer return `Variable` by substituting over annotated return with constructed substitution map


For the example above, this would look like the following:
1. (param=`Tensor[d1]`, arg=`Tensor[10]`), (param=`Tensor[d2]`, arg=`Tensor[20]`)
2. will construct substitution map of `{d1 -> 10, d2 -> 20}`
3. 10=10, 20=20, 10+20=30
4. Infer return as 10+20=30

## Handling Classes
With classes, dimvars in params can also be substituted with dimvars from the class's `__init__` params. These extra substitutions are added to the `SignatureModel` instance in `ClassInstance::create_method_signature`. 