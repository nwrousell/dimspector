use std::collections::HashMap;

use anyhow::{Result, anyhow};

use crate::analysis::{DimVar, Shape, Variable};

/// Macro to extract and validate arguments from a HashMap.
///
/// Usage:
/// ```rust
/// let (a, b, dims) = get_args!(args, TensorDot,
///     a: as_shape => "Tensor",
///     b: as_shape_dims => "Tensor",
///     dims: as_dimvar => "DimVar"
/// )?;
/// ```
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

trait Model {
    fn infer(args: HashMap<&str, Variable>) -> Result<Shape>;
}

impl ModelContext {
    pub fn resolve_torch_model(&self, path: &str) -> Option<impl Model> {
        match path {
            path if path == "torch.matmul" => Some(MatmulModel),
            _ => None,
        }
    }
    pub fn resolve(&self, path: &str) -> Option<impl Model> {
        self.resolve_torch_model(path)

        // can handle user models here later
    }
}

struct MatmulModel;
impl Model for MatmulModel {
    fn infer(args: HashMap<&str, Variable>) -> Result<Shape> {
        // TODO: also deal with out (mutates)

        let (input_shape, other_shape) = get_args!(args, Matmul,
            input_shape: as_shape_dims => "Tensor",
            other_shape: as_shape_dims => "Tensor",
        )?;

        match (input_shape.len(), other_shape.len()) {
            // dot product
            (1, 1) => {
                constraint_equal(input_shape[0].clone(), other_shape[0].clone())?;
                Ok(Shape::Known(vec![])) // Scalar result
            }

            // matrix-matrix
            (2, 2) => {
                constraint_equal(input_shape[1].clone(), other_shape[0].clone())?;
                Ok(Shape::Known(vec![
                    input_shape[0].clone(),
                    other_shape[1].clone(),
                ]))
            }

            // prepend 1, multiply, remove prepended dim
            (1, 2) => {
                constraint_equal(input_shape[0].clone(), other_shape[0].clone())?;
                Ok(Shape::Known(vec![other_shape[1].clone()]))
            }

            // matrix-vector product
            (2, 1) => {
                constraint_equal(input_shape[1].clone(), other_shape[0].clone())?;
                Ok(Shape::Known(vec![input_shape[0].clone()]))
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

                Ok(Shape::Known(result_dims))
            }

            // Fallback for unknown cases
            _ => Ok(Shape::Unknown),
        }
    }
}
