use core::panic;
use std::cmp::max;
use std::num;
use std::path::Path;
use std::{collections::HashMap, sync::LazyLock};

use anyhow::{Result, anyhow};
use itertools::EitherOrBoth::{self, Both, Left, Right};
use itertools::Itertools;
use miette::SourceSpan;

use crate::analysis::errors::ShapeError;
use crate::analysis::{DimKind, DimVar, Shape, Variable};
use crate::ir::{Function, Parameter};

macro_rules! get_args {
    ($args:expr, $model_name:ident, $( $param:ident : $method:ident => $type_name:expr ),+ $(,)?) => {
        {
            $(
                let $param = $args
                    .get(stringify!($param))
                    .ok_or_else(|| anyhow!("param '{}' wasn't supplied to {}", stringify!($param), stringify!($model_name)))?
                    .$method()
                    .ok_or_else(|| anyhow!("param '{}' supplied to {} not a {} or has unknown shape", stringify!($param), stringify!($model_name), $type_name))?;
            )+
            Ok::<_, anyhow::Error>(($( $param ),+))
        }
    };
}

fn constraint_equal(dim1: &DimVar, dim2: &DimVar, span: SourceSpan) -> Result<()> {
    if dim1 != dim2 {
        let err = ShapeError::mismatched(dim1, dim2, span);

        Err(anyhow!(err))
    } else {
        Ok(())
    }
}

pub struct ModelContext {
    pub torch: TorchModels,
    pub user: UserModels,
}

impl ModelContext {
    pub fn new(funcs: &Vec<Function>) -> Self {
        Self {
            torch: TorchModels::default(),
            user: UserModels::new(funcs),
        }
    }
}

pub struct UserModels {
    pub funcs: HashMap<String, Box<dyn Model>>,
}

impl UserModels {
    fn new(funcs: &Vec<Function>) -> Self {
        let map: HashMap<String, Box<dyn Model>> = funcs
            .iter()
            .map(|f| {
                (
                    f.identifier.to_string(),
                    Box::new(SignatureModel::new(f)) as Box<dyn Model>,
                )
            })
            .collect();

        Self { funcs: map }
    }
}

pub struct TorchModels {
    pub matmul: MatmulModel,
    pub passthrough: PassthroughModel,
    pub rdx: RdxModel,
    pub broadcast: BroadcastModel,
    pub concat: ConcatModel,
    pub reshape: ReshapeModel,
}

impl Default for TorchModels {
    fn default() -> Self {
        Self {
            matmul: MatmulModel,
            passthrough: PassthroughModel,
            rdx: RdxModel,
            broadcast: BroadcastModel,
            concat: ConcatModel,
            reshape: ReshapeModel,
        }
    }
}

pub trait Model {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        span: SourceSpan,
    ) -> Result<Shape>;
}

impl ModelContext {
    pub fn resolve_torch_model(&self, path: &str) -> Option<&dyn Model> {
        match path {
            "torch.matmul" => Some(&self.torch.matmul),
            "torch.add" | "torch.sub" | "torch.subtract" | "torch.mul" | "torch.multiply"
            | "torch.div" | "torch.divide" | "torch.true_divide" | "torch.floor_divide"
            | "torch.remainder" | "torch.fmod" | "torch.pow" => Some(&self.torch.broadcast),
            "torch.zeros_like" | "torch.ones_like" | "torch.full_like" | "torch.empty_like"
            | "torch.rand_like" | "torch.randn_like" => Some(&self.torch.passthrough),
            "torch.abs"
            | "torch.acos"
            | "torch.acosh"
            | "torch.asin"
            | "torch.asinh"
            | "torch.atan"
            | "torch.atanh"
            | "torch.ceil"
            | "torch.cos"
            | "torch.cosh"
            | "torch.erf"
            | "torch.erfc"
            | "torch.exp"
            | "torch.expm1"
            | "torch.floor"
            | "torch.frac"
            | "torch.isfinite"
            | "torch.isinf"
            | "torch.isnan"
            | "torch.log"
            | "torch.log10"
            | "torch.log1p"
            | "torch.log2"
            | "torch.neg"
            | "torch.reciprocal"
            | "torch.round"
            | "torch.rsqrt"
            | "torch.sigmoid"
            | "torch.sign"
            | "torch.sin"
            | "torch.sinh"
            | "torch.sqrt"
            | "torch.square"
            | "torch.tan"
            | "torch.tanh"
            | "torch.trunc"
            | "torch.nn.functional.relu"
            | "torch.nn.functional.relu6"
            | "torch.nn.functional.leaky_relu"
            | "torch.nn.functional.elu"
            | "torch.nn.functional.celu"
            | "torch.nn.functional.gelu"
            | "torch.nn.functional.silu"
            | "torch.nn.functional.hardtanh"
            | "torch.nn.functional.hardshrink"
            | "torch.nn.functional.softshrink"
            | "torch.nn.functional.mish"
            | "torch.special.expit"
            | "torch.special.erf"
            | "torch.special.erfc"
            | "torch.special.ndtr"
            | "torch.special.ndtri"
            | "torch.special.logit"
            | "torch.special.digamma" => Some(&self.torch.passthrough),
            "torch.sum" => Some(&self.torch.rdx),
            "torch.concat" => Some(&self.torch.concat),
            "torch.reshape" => Some(&self.torch.reshape),
            _ => None,
        }
    }

    pub fn resolve_user_model(&self, path: &str) -> Option<&dyn Model> {
        self.user.funcs.get(path).map(|boxed| boxed.as_ref())
    }

    pub fn resolve(&self, path: &str) -> Option<&dyn Model> {
        self.resolve_torch_model(path)
            .or_else(|| self.resolve_user_model(path))
    }
}

pub fn resolve_args(
    args: Vec<&Variable>,
    kwargs: HashMap<String, &Variable>,
    signature: impl Iterator<Item = (String, Option<Variable>)>,
) -> HashMap<String, Variable> {
    let mut mapping = HashMap::new();

    for (i, (name, default)) in signature.into_iter().enumerate() {
        let arg = if let Some(pos_arg) = args.get(i) {
            pos_arg
        } else if let Some(arg) = kwargs.get(&name) {
            arg
        } else if let Some(default_arg) = &default {
            default_arg
        } else {
            panic!("arg not found for function");
        };

        mapping.insert(name, arg.clone());
    }

    mapping
}

type Signature = Vec<(String, Option<Variable>)>;

pub struct BroadcastModel;

impl Model for BroadcastModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        span: SourceSpan,
    ) -> Result<Shape> {
        let args = resolve_args(args, kwargs, INPUT_OTHER_SIGNATURE.iter().cloned());
        let (l_shape, r_shape) = get_args!(args, Matmul,
            input: as_shape_dims => "Tensor",
            other: as_shape_dims => "Tensor",
        )?;

        let mut out_shape = Vec::new();
        for pair in l_shape.iter().rev().zip_longest(r_shape.iter().rev()) {
            let next_dim = match pair {
                Both(l_dim, r_dim) => {
                    if l_dim.is_one() {
                        r_dim.clone()
                    } else if r_dim.is_one() {
                        l_dim.clone()
                    } else {
                        constraint_equal(l_dim, r_dim, span)?;
                        l_dim.clone()
                    }
                }
                Left(dim) | Right(dim) => dim.clone(),
            };

            out_shape.push(next_dim);
        }
        out_shape.reverse();
        Ok(Shape(out_shape))
    }
}

pub struct MatmulModel;

static INPUT_OTHER_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| vec![("input".to_string(), None), ("other".to_string(), None)]);

impl Model for MatmulModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        span: SourceSpan,
    ) -> Result<Shape> {
        // TODO: also deal with out (mutates)
        let args = resolve_args(args, kwargs, INPUT_OTHER_SIGNATURE.iter().cloned());
        let (input_shape, other_shape) = get_args!(args, Matmul,
            input: as_shape_dims => "Tensor",
            other: as_shape_dims => "Tensor",
        )?;

        match (input_shape.len(), other_shape.len()) {
            (0, _) | (_, 0) => {
                panic!("matmul with a scalar is not allowed!")
            }

            // dot product
            (1, 1) => {
                constraint_equal(&input_shape[0], &other_shape[0], span)?;
                Ok(Shape(vec![])) // Scalar result
            }

            // matrix-matrix
            (2, 2) => {
                constraint_equal(&input_shape[1], &other_shape[0], span)?;
                Ok(Shape(vec![input_shape[0].clone(), other_shape[1].clone()]))
            }

            // prepend 1, multiply, remove prepended dim
            (1, 2) => {
                constraint_equal(&input_shape[0], &other_shape[0], span)?;
                Ok(Shape(vec![other_shape[1].clone()]))
            }

            // matrix-vector product
            (2, 1) => {
                constraint_equal(&input_shape[1], &other_shape[0], span)?;
                Ok(Shape(vec![input_shape[0].clone()]))
            }

            // batched matrix multiply
            (input_ndim, other_ndim)
                if input_ndim >= 1 && other_ndim >= 1 && (input_ndim > 2 || other_ndim > 2) =>
            {
                // Handle 1D inputs by prepending/appending 1
                let (input_batch, input_matrix) = if input_ndim == 1 {
                    (
                        vec![],
                        (vec![input_shape[0].clone()], vec![input_shape[0].clone()]),
                    )
                } else {
                    let split_idx = input_ndim - 2;
                    (
                        input_shape[..split_idx].to_vec(),
                        (
                            vec![input_shape[split_idx].clone()],
                            vec![input_shape[split_idx + 1].clone()],
                        ),
                    )
                };

                let (other_batch, other_matrix) = if other_ndim == 1 {
                    (
                        vec![],
                        (vec![other_shape[0].clone()], vec![other_shape[0].clone()]),
                    )
                } else {
                    let split_idx = other_ndim - 2;
                    (
                        other_shape[..split_idx].to_vec(),
                        (
                            vec![other_shape[split_idx].clone()],
                            vec![other_shape[split_idx + 1].clone()],
                        ),
                    )
                };

                // Check matrix dimension constraint: input[-1] == other[-2]
                constraint_equal(&input_matrix.1[0], &other_matrix.0[0], span)?;

                // Broadcast batch dimensions (for now, just take the longer one)
                // In a full implementation, we'd need proper broadcasting logic
                let batch_dims = if input_batch.len() >= other_batch.len() {
                    input_batch
                } else {
                    other_batch
                };

                // Build result shape: broadcast_batch + [input[-2], other[-1]]
                let mut result_dims = batch_dims;
                result_dims.push(input_matrix.0[0].clone());
                result_dims.push(other_matrix.1[0].clone());

                // Remove prepended/appended dimensions if original inputs were 1D
                if input_ndim == 1 {
                    result_dims.remove(0);
                }
                if other_ndim == 1 {
                    result_dims.pop();
                }

                Ok(Shape(result_dims))
            }

            _ => unreachable!("above cases are exhaustive"),
        }
    }
}

pub struct PassthroughModel;
static SINGLE_TENSOR_INPUT_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| vec![("input".to_string(), None)]);

// The base model for functions that do an element wise operation, preserving shape
// This should be fine for most activation like functions
impl Model for PassthroughModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        _span: SourceSpan,
    ) -> Result<Shape> {
        let args = resolve_args(args, kwargs, SINGLE_TENSOR_INPUT_SIGNATURE.iter().cloned());
        let input_shape = get_args!(args, Eltwise,
            input: as_shape => "Tensor",
        )?;

        Ok(input_shape)
    }
}

pub struct RdxModel;
static RDX_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    vec![
        ("input".to_string(), None),
        ("dim".to_string(), Some(Variable::None)),
        ("keepdim".to_string(), Some(Variable::None)), // TODO: handle this, default = False
    ]
});

impl Model for RdxModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        _span: SourceSpan,
    ) -> Result<Shape> {
        let args = resolve_args(args, kwargs, RDX_SIGNATURE.iter().cloned());
        let input_shape = get_args!(args, Matmul,
            input: as_shape_dims => "Tensor",
        )?;
        let rdx_dims = args.get("dim").unwrap(); // bad?

        let result_dims = match rdx_dims {
            Variable::DimVar(DimVar {
                kind: DimKind::Concrete(dim),
            }) => match dim {
                -1 => input_shape[..input_shape.len() - 1].to_vec(),
                dim if 0 <= *dim && *dim < input_shape.len() as i64 => {
                    let mut res = input_shape.clone();
                    res.remove(*dim as usize);
                    res
                }
                _ => todo!(),
            },
            Variable::Tuple(vars) => {
                let vars_conc: Vec<i64> = vars
                    .iter()
                    .map(|var| {
                        let Variable::DimVar(DimVar {
                            kind: DimKind::Concrete(v),
                        }) = var
                        else {
                            unreachable!()
                        };
                        *v
                    })
                    .collect();

                input_shape
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, x)| (!vars_conc.contains(&(i as i64))).then_some(x))
                    .collect()
            }
            _ => todo!(),
        };

        Ok(Shape(result_dims))
    }
}

pub struct TensorFromShapeModel;

// we'll have to handle *args in the signature

static TENSOR_FROM_SHAPE_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    vec![
        ("input".to_string(), None),
        ("dim".to_string(), Some(Variable::None)),
        ("keepdim".to_string(), Some(Variable::None)), // TODO: handle this, default = False
    ]
});

pub struct ConcatModel;

static CONCAT_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    vec![
        ("tensors".to_string(), None),
        (
            "dim".to_string(),
            Some(Variable::DimVar(DimVar {
                kind: DimKind::Concrete(0),
            })),
        ),
    ]
});

impl Model for ConcatModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        span: SourceSpan,
    ) -> Result<Shape> {
        let args = resolve_args(args, kwargs, CONCAT_SIGNATURE.iter().cloned());
        let (tensors, dim) = get_args!(args, Concat,
            tensors: as_tuple => "Tuple",
            dim: as_concrete_dimvar => "Int",
        )?;

        let tensors = tensors
            .iter()
            .map(|v| {
                let Some(Shape(s)) = v.as_shape() else {
                    return Err(anyhow!("Argument must be list of Tensor"));
                };
                Ok(s)
            })
            .collect::<Result<Vec<_>>>()?;

        let rank = tensors[0].len() as i64;

        if dim < -rank || dim >= rank {
            return Err(anyhow!(ShapeError::DimOutRange {
                dim_ref: dim,
                rank: rank as usize
            }));
        }

        let dim = if dim < 0 {
            (tensors.len() as i64 + dim) as usize
        } else {
            dim as usize
        };

        // TODO: this needs cleaning to prevent so many clones
        let res: Vec<DimVar> = tensors[1..]
            .iter()
            .try_fold(tensors[0].clone(), |cur_s, s| {
                cur_s
                    .iter()
                    .zip(s.iter())
                    .enumerate()
                    .map(|(i, (c_dv, dv))| {
                        if i == dim {
                            Ok(c_dv.clone() + dv.clone())
                        } else if c_dv != dv {
                            Err(anyhow!(ShapeError::mismatched(c_dv, dv, span)))
                        } else {
                            Ok(c_dv.clone())
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

        Ok(Shape(res))
    }
}

pub struct ReshapeModel;

static RESHAPE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| vec![("input".to_string(), None), ("shape".to_string(), None)]);

impl Model for ReshapeModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        _span: SourceSpan,
    ) -> Result<Shape> {
        let args = resolve_args(args, kwargs, RESHAPE_SIGNATURE.iter().cloned());
        let (src_shape, tgt_shape) = get_args!(args, Concat,
            input: as_shape_dims => "Tensor",
            shape: as_shape_dims => "Tuple",
        )?;

        let num_unspecified = tgt_shape.iter().fold(0, |acc, dv| {
            acc + if *dv == DimVar::from(-1) { 1 } else { 0 }
        });
        if num_unspecified > 1 {
            return Err(anyhow!(
                "Cannot have multiple unspecified dims in torch.reshape"
            ));
        }

        // validate shape is preserved
        let src_shape_prod = src_shape
            .iter()
            .fold(DimVar::from(1), |acc, dv| acc * dv.clone());

        let tgt_shape_prod = tgt_shape.iter().fold(DimVar::from(1), |acc, dv| {
            if let DimKind::Concrete(c) = dv.kind()
                && c == -1
            {
                acc
            } else {
                acc * dv.clone()
            }
        });

        let tgt_shape = tgt_shape
            .iter()
            .map(|dv| -> Result<DimVar> {
                if *dv == DimVar::from(-1) {
                    src_shape_prod.div(&tgt_shape_prod)
                } else {
                    Ok(dv.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let tgt_shape_prod = tgt_shape
            .iter()
            .fold(DimVar::from(1), |acc, dv| acc * dv.clone());

        if tgt_shape_prod != src_shape_prod {
            return Err(anyhow!(ShapeError::BadReshape {
                src: Shape(src_shape),
                tgt: Shape(tgt_shape)
            }));
        }

        Ok(Shape(tgt_shape))
    }
}

struct SignatureModel {
    params: Vec<Parameter>,
    // TODO: in the future with the possibility of mutations,
    // doesn't necc need to have return annotation
    returns: Option<Vec<Variable>>,
}

impl SignatureModel {
    fn new(func: &Function) -> Self {
        // construct signature from function sig
        SignatureModel {
            params: func.params.clone(),
            returns: func.returns.clone(),
        }
    }
}
impl Model for SignatureModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<String, &Variable>,
        span: SourceSpan,
    ) -> Result<Shape> {
        let mut callee_to_caller: HashMap<String, DimVar> = HashMap::new();
        for argv in args.iter().zip_longest(self.params.iter()) {
            let (caller_v, callee_v) = match argv {
                EitherOrBoth::Both(arg_v, param) => {
                    let Some(param_v) = &param.1 else {
                        // param doesn't have tensor annotation, skip
                        continue;
                    };
                    (arg_v, param_v)
                }
                EitherOrBoth::Right(param) => {
                    // do a lookup into kwargs
                    let Some(param_v) = &param.1 else {
                        continue;
                    };
                    let Some(arg_v) = kwargs.get(&param.0.to_string()) else {
                        continue;
                    };
                    (arg_v, param_v)
                }
                EitherOrBoth::Left(_) => unreachable!("args should not be longer than params"),
            };
            match (caller_v, callee_v) {
                (Variable::Tensor(Shape(caller_dims)), Variable::Tensor(Shape(callee_dims))) => {
                    if caller_dims.len() != callee_dims.len() {
                        // TODO: in the future we should handle ellipsis
                        return Err(anyhow!(ShapeError::UnequalRank {
                            rank_1: caller_dims.len(),
                            rank_2: callee_dims.len()
                        }));
                    }

                    let mut eq_constraints: Vec<(&DimVar, &DimVar)> = Vec::new();

                    for (caller_dv, callee_dv) in caller_dims.iter().zip(callee_dims.iter()) {
                        match callee_dv.kind() {
                            DimKind::Named(name) => {
                                if let Some(prev_caller_dv) = callee_to_caller.get(&name) {
                                    // TODO: we see a caller side mismatch here, do something with it
                                    if prev_caller_dv != caller_dv {
                                        return Err(anyhow!(ShapeError::mismatched(
                                            prev_caller_dv,
                                            caller_dv,
                                            span
                                        )));
                                    }
                                } else {
                                    callee_to_caller.insert(name, caller_dv.clone());
                                }
                            }

                            _ => eq_constraints.push((caller_dv, callee_dv)),
                        }
                    }

                    // TODO
                    // (a: T[x-1], b: T[x-1])
                    // need to disallow this ^ by enforcing a singleton DimVar::Named for each symbolic dimvar  in the signature

                    // TODO: handle concrete dimvars for constraints

                    // check constraints are good
                    for (caller_dv, callee_dv) in eq_constraints {
                        if callee_dv.substitute(&callee_to_caller)? != *caller_dv {
                            return Err(anyhow!(ShapeError::mismatched(
                                caller_dv, callee_dv, span
                            )));
                        }
                    }
                }
                _ => continue,
            }
        }
        // TODO: models currently only return one shape, should capture tuple returns in the future
        // for now, just assuming this is the case
        let ret_shape = {
            let Some(returns) = &self.returns else {
                // TODO: this shouldn't be an error. We need to generalize the effect of a function/method call to allow for no return
                let err = ShapeError::UninferrableCall {};
                return Err(anyhow!(err));
            };

            match &returns[0] {
                Variable::Tensor(Shape(shape)) => shape,
                // TODO: handle returning dimvars
                // return of uninferrablecall should result in Variable::Top return from whatever's
                // calling this atm
                _ => return Err(anyhow!(ShapeError::UninferrableCall {})),
            }
        };
        let ret_shape = ret_shape
            .iter()
            .map(|dv| dv.substitute(&callee_to_caller))
            .collect::<Result<Vec<_>>>()?;
        Ok(Shape(ret_shape))
    }
}
