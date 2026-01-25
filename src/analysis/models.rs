use core::panic;
use std::{
    collections::{BTreeMap, HashMap},
    sync::LazyLock,
};

use anyhow::{Result, anyhow};
use itertools::EitherOrBoth::{self, Both, Left, Right};
use itertools::Itertools;
use miette::SourceSpan;

use crate::analysis::errors::ShapeError;
use crate::analysis::types::Collection;
use crate::analysis::{ClassAnalysis, DimKind, DimVar, Shape, Variable};
use crate::ir::{Function, Identifier, Parameter, intern, resolve};

/// Holds spans for positional and keyword arguments
#[derive(Debug, Clone, Default)]
pub struct ArgSpans {
    pub positional: Vec<SourceSpan>,
    pub keyword: HashMap<Identifier, SourceSpan>,
}

impl ArgSpans {
    pub fn new(positional: Vec<SourceSpan>, keyword: HashMap<Identifier, SourceSpan>) -> Self {
        Self {
            positional,
            keyword,
        }
    }

    /// Get span for a positional argument by index
    pub fn get_positional(&self, idx: usize) -> Option<SourceSpan> {
        self.positional.get(idx).copied()
    }

    /// Get span for a keyword argument by name
    pub fn get_keyword(&self, name: Identifier) -> Option<SourceSpan> {
        self.keyword.get(&name).copied()
    }
}

macro_rules! get_args {
    ($args:expr, $model_name:ident, $( $param:ident : $method:ident => $type_name:expr ),+ $(,)?) => {
        {
            $(
                let $param = $args
                    .get(&intern(stringify!($param)))
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

#[derive(Debug)]
pub struct ModelContext {
    pub torch: TorchModels,
    pub user: UserModels,
}

impl ModelContext {
    pub fn new(funcs: &Vec<Function>, symbol_table: &crate::parse::SymbolTable) -> Self {
        Self {
            torch: TorchModels::default(),
            user: UserModels::new(funcs, symbol_table),
        }
    }
}

#[derive(Debug)]
pub struct UserModels {
    pub funcs: HashMap<String, Box<dyn Model>>,
}

impl UserModels {
    fn new(funcs: &Vec<Function>, symbol_table: &crate::parse::SymbolTable) -> Self {
        let map: HashMap<String, Box<dyn Model>> = funcs
            .iter()
            .map(|f| {
                let local_name = resolve(f.identifier);
                let canonical = symbol_table
                    .resolve(&f.file_path, &local_name)
                    .cloned()
                    .unwrap_or(local_name);
                (
                    canonical,
                    Box::new(SignatureModel::new(f)) as Box<dyn Model>,
                )
            })
            .collect();

        Self { funcs: map }
    }
}

#[derive(Debug)]
pub struct TorchModels {
    pub matmul: MatmulModel,
    pub passthrough: PassthroughModel,
    pub rdx: RdxModel,
    pub broadcast: BroadcastModel,
    pub concat: ConcatModel,
    pub reshape: ReshapeModel,
    pub tensor_reshape: TensorReshapeModel,
    pub tranpose: TransposeModel,
    pub tensor_from_size: TensorFromSizeModel,
    pub randint: RandIntModel,
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
            tensor_reshape: TensorReshapeModel,
            tranpose: TransposeModel,
            tensor_from_size: TensorFromSizeModel,
            randint: RandIntModel,
        }
    }
}

pub trait Model: std::fmt::Debug + Send + Sync {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        arg_spans: &ArgSpans,
    ) -> Result<Variable>;
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
            | "torch.nn.functional.softmax"
            | "torch.nn.functional.log_softmax"
            | "torch.softmax"
            | "torch.special.expit"
            | "torch.special.erf"
            | "torch.special.erfc"
            | "torch.special.ndtr"
            | "torch.special.ndtri"
            | "torch.special.logit"
            | "torch.special.digamma" => Some(&self.torch.passthrough),
            "torch.sum" | "torch.mean" | "torch.prod" | "torch.amax" | "torch.amin"
            | "torch.std" | "torch.var" | "torch.logsumexp" | "torch.all" | "torch.any"
            | "torch.nansum" | "torch.nanmean" => Some(&self.torch.rdx),
            "torch.concat" | "torch.cat" => Some(&self.torch.concat),
            "torch.reshape" | "torch.view" => Some(&self.torch.reshape),
            "torch.transpose" => Some(&self.torch.tranpose),
            "torch.zeros" | "torch.ones" | "torch.empty" | "torch.rand" | "torch.randn"
            | "torch.full" => Some(&self.torch.tensor_from_size),
            "torch.randint" => Some(&self.torch.randint),

            // torch.Tensor methods route to existing models where signatures are compatible
            "torch.Tensor.abs"
            | "torch.Tensor.acos"
            | "torch.Tensor.acosh"
            | "torch.Tensor.asin"
            | "torch.Tensor.asinh"
            | "torch.Tensor.atan"
            | "torch.Tensor.atanh"
            | "torch.Tensor.ceil"
            | "torch.Tensor.cos"
            | "torch.Tensor.cosh"
            | "torch.Tensor.erf"
            | "torch.Tensor.erfc"
            | "torch.Tensor.exp"
            | "torch.Tensor.expm1"
            | "torch.Tensor.floor"
            | "torch.Tensor.frac"
            | "torch.Tensor.isfinite"
            | "torch.Tensor.isinf"
            | "torch.Tensor.isnan"
            | "torch.Tensor.log"
            | "torch.Tensor.log10"
            | "torch.Tensor.log1p"
            | "torch.Tensor.log2"
            | "torch.Tensor.neg"
            | "torch.Tensor.reciprocal"
            | "torch.Tensor.round"
            | "torch.Tensor.rsqrt"
            | "torch.Tensor.sigmoid"
            | "torch.Tensor.sign"
            | "torch.Tensor.sin"
            | "torch.Tensor.sinh"
            | "torch.Tensor.sqrt"
            | "torch.Tensor.square"
            | "torch.Tensor.tan"
            | "torch.Tensor.tanh"
            | "torch.Tensor.trunc" => Some(&self.torch.passthrough),
            "torch.Tensor.sum"
            | "torch.Tensor.mean"
            | "torch.Tensor.prod"
            | "torch.Tensor.amax"
            | "torch.Tensor.amin"
            | "torch.Tensor.std"
            | "torch.Tensor.var"
            | "torch.Tensor.logsumexp"
            | "torch.Tensor.all"
            | "torch.Tensor.any" => Some(&self.torch.rdx),
            "torch.Tensor.transpose" => Some(&self.torch.tranpose),
            "torch.Tensor.reshape" | "torch.Tensor.view" => Some(&self.torch.tensor_reshape),
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
    kwargs: HashMap<Identifier, &Variable>,
    signature: &Signature,
) -> HashMap<Identifier, Variable> {
    let mut mapping = HashMap::new();
    match signature {
        Signature::Variadic { kwargs_defaults } => {
            mapping.insert(
                intern("variadic"),
                Collection::tup_from_elts(args.into_iter().map(|v| v.clone()).collect()),
            );

            for (name, default) in kwargs_defaults.into_iter() {
                if let Some(arg) = kwargs.get(name) {
                    mapping.insert(*name, (*arg).clone());
                } else {
                    mapping.insert(*name, default.clone());
                }
            }
        }
        Signature::FixedArity(params) => {
            for (i, (name, default)) in params.iter().enumerate() {
                let arg = if let Some(pos_arg) = args.get(i) {
                    pos_arg
                } else if let Some(arg) = kwargs.get(name) {
                    arg
                } else if let Some(default_arg) = default {
                    default_arg
                } else {
                    panic!("arg not found for function");
                };

                mapping.insert(*name, arg.clone());
            }
        }
    };

    mapping
}

#[derive(Debug)]
pub enum Signature {
    Variadic {
        kwargs_defaults: Vec<(Identifier, Variable)>,
    },
    FixedArity(Vec<(Identifier, Option<Variable>)>),
}

#[derive(Debug)]
pub struct TensorFromSizeModel;

static VARIADIC_SIZE_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| Signature::Variadic {
    kwargs_defaults: Vec::new(),
});

impl Model for TensorFromSizeModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        _span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &VARIADIC_SIZE_SIGNATURE);
        let size_tuple = args
            .get(&intern("variadic"))
            .expect("variadic signature always has variadic tuple")
            .as_vec()
            .expect("ditto");

        let dimvars = size_tuple
            .iter()
            .map(|v| {
                if let Variable::DimVar(d) = v {
                    Ok(d.clone())
                } else {
                    Err(anyhow!("non-dimvar being used as a dimvar"))
                }
            })
            .collect::<Result<Vec<DimVar>>>()?;

        Ok(Variable::Tensor(Shape(dimvars)))
    }
}

#[derive(Debug)]
pub struct RandIntModel;

static RANDINT_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::FixedArity(vec![
        (intern("low"), None),
        (intern("high"), None),
        (intern("size"), None),
    ])
});

impl Model for RandIntModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        _span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &RANDINT_SIGNATURE);
        let size = get_args!(args, RandInt,
            size: as_shape_dims => "Tuple",
        )?;

        Ok(Variable::Tensor(Shape(size)))
    }
}

#[derive(Debug)]
pub struct BroadcastModel;

impl Model for BroadcastModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &INPUT_OTHER_SIGNATURE);
        let (l_shape, r_shape) = get_args!(args, Matmul,
            input: as_shape_dims => "Tensor",
            other: as_shape_dims => "Tensor",
        )?;

        let left_shape = Shape(l_shape.clone());
        let right_shape = Shape(r_shape.clone());

        let mut out_shape = Vec::new();
        for (dim_position, pair) in l_shape
            .iter()
            .rev()
            .zip_longest(r_shape.iter().rev())
            .enumerate()
        {
            let next_dim = match pair {
                Both(l_dim, r_dim) => {
                    if l_dim.is_one() {
                        r_dim.clone()
                    } else if r_dim.is_one() {
                        l_dim.clone()
                    } else if l_dim != r_dim {
                        return Err(anyhow!(ShapeError::BroadcastMismatch {
                            left_shape: left_shape.clone(),
                            right_shape: right_shape.clone(),
                            dim_position: dim_position + 1, // 1-indexed from the right
                            left_dim: l_dim.clone(),
                            right_dim: r_dim.clone(),
                            span,
                        }));
                    } else {
                        l_dim.clone()
                    }
                }
                Left(dim) | Right(dim) => dim.clone(),
            };

            out_shape.push(next_dim);
        }
        out_shape.reverse();
        Ok(Variable::Tensor(Shape(out_shape)))
    }
}

#[derive(Debug)]
pub struct MatmulModel;

static INPUT_OTHER_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::FixedArity(vec![(intern("input"), None), (intern("other"), None)]));

impl Model for MatmulModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        // TODO: also deal with out (mutates)
        let args = resolve_args(args, kwargs, &INPUT_OTHER_SIGNATURE);
        let (input_shape, other_shape) = get_args!(args, Matmul,
            input: as_shape_dims => "Tensor",
            other: as_shape_dims => "Tensor",
        )?;

        let left_shape = Shape(input_shape.clone());
        let right_shape = Shape(other_shape.clone());

        // Helper to check matmul inner dimension constraint
        let check_matmul_dims = |left_dim: &DimVar, right_dim: &DimVar| -> Result<()> {
            if left_dim != right_dim {
                return Err(anyhow!(ShapeError::MatmulMismatch {
                    left_shape: left_shape.clone(),
                    right_shape: right_shape.clone(),
                    left_dim: left_dim.clone(),
                    right_dim: right_dim.clone(),
                    span,
                }));
            }
            Ok(())
        };

        match (input_shape.len(), other_shape.len()) {
            (0, _) | (_, 0) => {
                panic!("matmul with a scalar is not allowed!")
            }

            // dot product
            (1, 1) => {
                check_matmul_dims(&input_shape[0], &other_shape[0])?;
                Ok(Variable::Tensor(Shape(vec![]))) // Scalar result
            }

            // matrix-matrix
            (2, 2) => {
                check_matmul_dims(&input_shape[1], &other_shape[0])?;
                Ok(Variable::Tensor(Shape(vec![
                    input_shape[0].clone(),
                    other_shape[1].clone(),
                ])))
            }

            // prepend 1, multiply, remove prepended dim
            (1, 2) => {
                check_matmul_dims(&input_shape[0], &other_shape[0])?;
                Ok(Variable::Tensor(Shape(vec![other_shape[1].clone()])))
            }

            // matrix-vector product
            (2, 1) => {
                check_matmul_dims(&input_shape[1], &other_shape[0])?;
                Ok(Variable::Tensor(Shape(vec![input_shape[0].clone()])))
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
                check_matmul_dims(&input_matrix.1[0], &other_matrix.0[0])?;

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

                Ok(Variable::Tensor(Shape(result_dims)))
            }

            _ => unreachable!("above cases are exhaustive"),
        }
    }
}

#[derive(Debug)]
pub struct PassthroughModel;
static SINGLE_TENSOR_INPUT_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::FixedArity(vec![(intern("input"), None)]));

// The base model for functions that do an element wise operation, preserving shape
// This should be fine for most activation like functions
impl Model for PassthroughModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        _span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &SINGLE_TENSOR_INPUT_SIGNATURE);
        let input_shape = get_args!(args, Eltwise,
            input: as_shape => "Tensor",
        )?;

        Ok(Variable::Tensor(input_shape))
    }
}

#[derive(Debug)]
pub struct RdxModel;
static RDX_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::FixedArity(vec![
        (intern("input"), None),
        (intern("dim"), Some(Variable::None)),
        (intern("keepdim"), Some(Variable::None)), // TODO: handle this, default = False
    ])
});

impl Model for RdxModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        _span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &RDX_SIGNATURE);
        let input_shape = get_args!(args, Matmul,
            input: as_shape_dims => "Tensor",
        )?;
        let rdx_dims = args.get(&intern("dim")).unwrap(); // bad?

        let result_dims = match rdx_dims {
            Variable::None => {
                // No dim specified - reduce across all dimensions to singleton
                vec![DimVar::from(1)]
            }
            Variable::DimVar(DimVar {
                kind: DimKind::Concrete(dim),
            }) => match dim {
                dim if 0 <= *dim && *dim < input_shape.len() as i64 => {
                    let mut res = input_shape.clone();
                    res.remove(*dim as usize);
                    res
                }
                dim if *dim < 0 && *dim >= -(input_shape.len() as i64) => {
                    // Handle negative indices: -1 is last, -2 is second-to-last, etc.
                    let positive_idx = (input_shape.len() as i64 + dim) as usize;
                    let mut res = input_shape.clone();
                    res.remove(positive_idx);
                    res
                }
                _ => todo!(),
            },
            Variable::Collection(Collection::Tuple(vars) | Collection::List(vars)) => {
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

        Ok(Variable::Tensor(Shape(result_dims)))
    }
}

#[derive(Debug)]
pub struct ConcatModel;

static CONCAT_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::FixedArity(vec![
        (intern("tensors"), None),
        (
            intern("dim"),
            Some(Variable::DimVar(DimVar {
                kind: DimKind::Concrete(0),
            })),
        ),
    ])
});

impl Model for ConcatModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &CONCAT_SIGNATURE);
        let (tensors, dim) = get_args!(args, Concat,
            tensors: as_vec => "Tuple",
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

        Ok(Variable::Tensor(Shape(res)))
    }
}

#[derive(Debug)]
pub struct ReshapeModel;

static RESHAPE_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| Signature::FixedArity(vec![(intern("input"), None), (intern("shape"), None)]));

impl Model for ReshapeModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &RESHAPE_SIGNATURE);
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
                tgt: Shape(tgt_shape.clone()),
                span,
            }));
        }

        Ok(Variable::Tensor(Shape(tgt_shape)))
    }
}

#[derive(Debug)]
pub struct TensorReshapeModel;

static TENSOR_RESHAPE_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| Signature::Variadic {
    kwargs_defaults: Vec::new(),
});

impl Model for TensorReshapeModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let resolved_args = resolve_args(args, kwargs, &TENSOR_RESHAPE_SIGNATURE);
        let variadic_tuple = resolved_args
            .get(&intern("variadic"))
            .expect("variadic signature always has variadic tuple")
            .as_vec()
            .expect("variadic should be tuple");

        let input = variadic_tuple.first().ok_or_else(|| {
            anyhow!("TensorReshapeModel requires at least one argument (the tensor)")
        })?;

        let shape_tuple = if variadic_tuple.len() == 2 {
            if let Some(Variable::Collection(Collection::Tuple(_))) = variadic_tuple.get(1) {
                variadic_tuple[1].clone()
            } else {
                Variable::Collection(Collection::Tuple(variadic_tuple[1..].to_vec()))
            }
        } else {
            Variable::Collection(Collection::Tuple(variadic_tuple[1..].to_vec()))
        };

        let reshape_model = ReshapeModel;
        reshape_model.infer(vec![input, &shape_tuple], HashMap::new(), span, arg_spans)
    }
}

#[derive(Debug)]
pub struct TransposeModel;

static TRANSPOSE_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    Signature::FixedArity(vec![
        (intern("input"), None),
        (intern("dim0"), None),
        (intern("dim1"), None),
    ])
});

impl Model for TransposeModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        _span: SourceSpan,
        _arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        let args = resolve_args(args, kwargs, &TRANSPOSE_SIGNATURE);
        let (mut input_dims, dim0, dim1) = get_args!(args, Transpose,
            input: as_shape_dims => "Tensor",
            dim0: as_concrete_dimvar => "Int",
            dim1: as_concrete_dimvar => "Int",
        )?;

        let rank = input_dims.len() as i64;

        // Normalize negative indices
        let dim0 = if dim0 < 0 {
            (rank + dim0) as usize
        } else {
            dim0 as usize
        };
        let dim1 = if dim1 < 0 {
            (rank + dim1) as usize
        } else {
            dim1 as usize
        };

        input_dims.swap(dim0, dim1);

        Ok(Variable::Tensor(Shape(input_dims)))
    }
}

#[derive(Debug, Clone)]
pub struct SignatureModel {
    pub name: String,
    pub params: Vec<Parameter>,
    // TODO: in the future with the possibility of mutations,
    // doesn't necc need to have return annotation
    pub returns: Option<Vec<Variable>>,
}

impl SignatureModel {
    pub fn new(func: &Function) -> Self {
        SignatureModel {
            name: resolve(func.identifier),
            params: func.params.clone(),
            returns: func.returns.clone(),
        }
    }
}
impl Model for SignatureModel {
    fn infer(
        &self,
        args: Vec<&Variable>,
        kwargs: HashMap<Identifier, &Variable>,
        span: SourceSpan,
        arg_spans: &ArgSpans,
    ) -> Result<Variable> {
        // Track dimvar bindings with the span, shape, and param name where they were first bound
        // (DimVar, SourceSpan, Option<Shape>, param_name) - shape is None for single DimVar args
        let mut param_to_arg: HashMap<String, (DimVar, SourceSpan, Option<Shape>, String)> =
            HashMap::new();

        // Track deferred constraints for expression dimvars (like k-1) that need substitution
        struct DeferredConstraint<'a> {
            arg_dv: &'a DimVar,
            param_dv: &'a DimVar,
            param_name: String,
            // None for single DimVar params, Some for Tensor params
            arg_shape: Option<Shape>,
            param_shape: Option<Shape>,
            arg_span: SourceSpan,
        }
        let mut deferred_constraints: Vec<DeferredConstraint> = Vec::new();

        for (arg_idx, argv) in args.iter().zip_longest(self.params.iter()).enumerate() {
            let (arg_v, param, arg_span) = match argv {
                EitherOrBoth::Both(arg_v, param) => {
                    let Some(param_v) = &param.1 else {
                        // param doesn't have tensor annotation, skip
                        continue;
                    };
                    let arg_span = arg_spans.get_positional(arg_idx).unwrap_or(span);
                    (arg_v, (param, param_v), arg_span)
                }
                EitherOrBoth::Right(param) => {
                    // do a lookup into kwargs
                    let Some(param_v) = &param.1 else {
                        continue;
                    };
                    let Some(arg_v) = kwargs.get(&param.0) else {
                        continue;
                    };
                    let arg_span = arg_spans.get_keyword(param.0).unwrap_or(span);
                    (arg_v, (param, param_v), arg_span)
                }
                EitherOrBoth::Left(_) => unreachable!("args should not be longer than params"),
            };

            let (param_info, param_v) = param;
            let param_name = resolve(param_info.0);

            match (arg_v, param_v) {
                (Variable::DimVar(arg_dv), Variable::DimVar(param_dv)) => {
                    match param_dv.kind() {
                        DimKind::Named(name) => {
                            if let Some((prev_arg_dv, first_span, first_shape, first_param_name)) =
                                param_to_arg.get(&name)
                            {
                                if prev_arg_dv != arg_dv {
                                    return Err(anyhow!(ShapeError::InconsistentDimVars {
                                        func_name: self.name.clone(),
                                        dimvar_name: name.clone(),
                                        first_param_name: first_param_name.clone(),
                                        second_param_name: param_name.clone(),
                                        first_resolved: prev_arg_dv.clone(),
                                        second_resolved: arg_dv.clone(),
                                        first_shape: first_shape.clone(),
                                        second_shape: None,
                                        first_span: *first_span,
                                        second_span: arg_span,
                                    }));
                                }
                            } else {
                                param_to_arg.insert(
                                    name,
                                    (arg_dv.clone(), arg_span, None, param_name.clone()),
                                );
                            }
                        }
                        DimKind::Concrete(_) => {
                            // Concrete dims can be checked immediately
                            if arg_dv != param_dv {
                                return Err(anyhow!(ShapeError::MismatchedDims {
                                    dim1: arg_dv.clone(),
                                    dim2: param_dv.clone(),
                                    span: arg_span,
                                }));
                            }
                        }
                        // Expression dimvars (Add, Mul) need deferred checking
                        _ => {
                            deferred_constraints.push(DeferredConstraint {
                                arg_dv,
                                param_dv,
                                param_name: param_name.clone(),
                                arg_shape: None,
                                param_shape: None,
                                arg_span,
                            });
                        }
                    }
                }
                (Variable::Tensor(arg_shape), Variable::Tensor(param_shape)) => {
                    let arg_dims = &arg_shape.0;
                    let param_dims = &param_shape.0;
                    if arg_dims.len() != param_dims.len() {
                        return Err(anyhow!(ShapeError::UnequalRank {
                            tensor_1: arg_shape.clone(),
                            tensor_2: param_shape.clone(),
                            rank_1: arg_dims.len(),
                            rank_2: param_dims.len(),
                            span: arg_span,
                        }));
                    }

                    for (arg_dv, param_dv) in arg_dims.iter().zip(param_dims.iter()) {
                        match param_dv.kind() {
                            DimKind::Named(name) => {
                                if let Some((
                                    prev_arg_dv,
                                    first_span,
                                    first_shape,
                                    first_param_name,
                                )) = param_to_arg.get(&name)
                                {
                                    if prev_arg_dv != arg_dv {
                                        return Err(anyhow!(ShapeError::InconsistentDimVars {
                                            func_name: self.name.clone(),
                                            dimvar_name: name.clone(),
                                            first_param_name: first_param_name.clone(),
                                            second_param_name: param_name.clone(),
                                            first_resolved: prev_arg_dv.clone(),
                                            second_resolved: arg_dv.clone(),
                                            first_shape: first_shape.clone(),
                                            second_shape: Some(arg_shape.clone()),
                                            first_span: *first_span,
                                            second_span: arg_span,
                                        }));
                                    }
                                } else {
                                    param_to_arg.insert(
                                        name,
                                        (
                                            arg_dv.clone(),
                                            arg_span,
                                            Some(arg_shape.clone()),
                                            param_name.clone(),
                                        ),
                                    );
                                }
                            }
                            // Expression dimvars need deferred checking after all named ones are bound
                            _ => {
                                deferred_constraints.push(DeferredConstraint {
                                    arg_dv,
                                    param_dv,
                                    param_name: param_name.clone(),
                                    arg_shape: Some(arg_shape.clone()),
                                    param_shape: Some(param_shape.clone()),
                                    arg_span,
                                });
                            }
                        }
                    }
                }
                _ => continue,
            }
        }

        // Build substitution map (without spans/shapes) for constraint checking
        let substitution_map: HashMap<String, DimVar> = param_to_arg
            .iter()
            .map(|(k, (dv, _, _, _))| (k.clone(), dv.clone()))
            .collect();

        // Check deferred constraints (expression dimvars like k-1)
        for constraint in deferred_constraints {
            let expected_dv = constraint.param_dv.substitute(&substitution_map)?;
            if expected_dv != *constraint.arg_dv {
                // Use SignatureParamMismatch for tensor shapes, MismatchedDims for single dimvars
                match (constraint.param_shape, constraint.arg_shape) {
                    (Some(param_shape), Some(arg_shape)) => {
                        return Err(anyhow!(ShapeError::SignatureParamMismatch {
                            func_name: self.name.clone(),
                            param_name: constraint.param_name,
                            expected: param_shape,
                            actual: arg_shape,
                            span: constraint.arg_span,
                        }));
                    }
                    _ => {
                        return Err(anyhow!(ShapeError::MismatchedDims {
                            dim1: constraint.arg_dv.clone(),
                            dim2: expected_dv,
                            span: constraint.arg_span,
                        }));
                    }
                }
            }
        }

        let Some(ret_var) = &self.returns else {
            // TODO: we need to be able to use our analysis inferred return shape here
            return Ok(Variable::Top);
        };

        let ret_var = ret_var
            .first()
            .ok_or_else(|| anyhow!(ShapeError::UninferrableCall {}))?;

        let substituted_ret = ret_var.substitute(&substitution_map)?;

        Ok(substituted_ret)
    }
}
