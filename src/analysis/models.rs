use core::panic;
use std::{collections::HashMap, sync::LazyLock};

use anyhow::{Result, anyhow};
use itertools::Itertools;
use num_traits::sign;

use crate::analysis::{DimKind, DimVar, Shape, Variable};

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

fn constraint_equal(_dim1: DimVar, _dim2: DimVar) -> Result<()> {
    // can have this add constraint to some structure to be continually checked
    // or can eagerly check, erroring on cases where a = b (forcing user to annotate them both as 'a')
    // ? Are there other cases where eagerly checking loses info?
    Ok(())
}

pub struct ModelContext {}

pub trait Model {
    fn infer(&self, args: Vec<&Variable>, kwargs: HashMap<String, &Variable>) -> Result<Shape>;
}

impl ModelContext {
    pub fn resolve_torch_model(&self, path: &str) -> Option<Box<dyn Model>> {
        match path {
            "torch.matmul" => Some(Box::new(MatmulModel)),
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
            | "torch.special.digamma" => Some(Box::new(EltwiseModel)),
            "torch.sum" => Some(Box::new(RdxModel)),
            _ => None,
        }
    }

    pub fn resolve(&self, path: &str) -> Option<Box<dyn Model>> {
        self.resolve_torch_model(path)

        // can handle user models here later
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
            pos_arg.clone()
        } else if let Some(arg) = kwargs.get(&name) {
            arg.clone()
        } else if let Some(default_arg) = &default {
            default_arg
        } else {
            panic!("arg not found for function");
        };

        mapping.insert(name, arg.clone());
    }

    mapping
}

#[derive(Clone)]
struct DefaultNone;
type Signature = Vec<(String, Option<Variable>)>;

struct MatmulModel;

static MATMUL_SIGNATURE: LazyLock<Signature> =
    LazyLock::new(|| vec![("input".to_string(), None), ("other".to_string(), None)]);

impl Model for MatmulModel {
    fn infer(&self, args: Vec<&Variable>, kwargs: HashMap<String, &Variable>) -> Result<Shape> {
        // TODO: also deal with out (mutates)
        let args = resolve_args(args, kwargs, MATMUL_SIGNATURE.iter().cloned());
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
                constraint_equal(input_shape[0].clone(), other_shape[0].clone())?;
                Ok(Shape(vec![])) // Scalar result
            }

            // matrix-matrix
            (2, 2) => {
                constraint_equal(input_shape[1].clone(), other_shape[0].clone())?;
                Ok(Shape(vec![input_shape[0].clone(), other_shape[1].clone()]))
            }

            // prepend 1, multiply, remove prepended dim
            (1, 2) => {
                constraint_equal(input_shape[0].clone(), other_shape[0].clone())?;
                Ok(Shape(vec![other_shape[1].clone()]))
            }

            // matrix-vector product
            (2, 1) => {
                constraint_equal(input_shape[1].clone(), other_shape[0].clone())?;
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
                constraint_equal(input_matrix.1[0].clone(), other_matrix.0[0].clone())?;

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

struct EltwiseModel;
static ELTWISE_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| vec![("input".to_string(), None)]);

// The base model for functions that do an element wise operation, preserving shape
// This should be fine for most activation like functions
impl Model for EltwiseModel {
    fn infer(&self, args: Vec<&Variable>, kwargs: HashMap<String, &Variable>) -> Result<Shape> {
        let args = resolve_args(args, kwargs, ELTWISE_SIGNATURE.iter().cloned());
        let input_shape = get_args!(args, Eltwise,
            input: as_shape => "Tensor",
        )?;

        Ok(input_shape)
    }
}

struct RdxModel;
static RDX_SIGNATURE: LazyLock<Signature> = LazyLock::new(|| {
    vec![
        ("input".to_string(), None),
        ("dim".to_string(), Some(Variable::None)),
        ("keepdim".to_string(), Some(Variable::None)), // TODO: handle this, default = False
    ]
});

impl Model for RdxModel {
    fn infer(&self, args: Vec<&Variable>, kwargs: HashMap<String, &Variable>) -> Result<Shape> {
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
                let mut vars_conc = vars.iter().map(|var| {
                    let Variable::DimVar(DimVar {
                        kind: DimKind::Concrete(v),
                    }) = var
                    else {
                        unreachable!("rdx dims should be concrete")
                    };
                    v
                });
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
