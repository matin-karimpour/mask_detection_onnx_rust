use tract_onnx::prelude::*;
use std:: io;

fn main() -> TractResult<()> {
    let model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>> =init_model()?;

     loop {
    let mut image_path = String::new();
    io::stdin().read_line(&mut image_path)?;
    let image_path = String::from(image_path.trim_end());

    if image_path == String::from("n"){
        break
    }
    // open image, resize it and make a Tensor out of it
    let image = preprocess(image_path);
    
    // run the model on the input
    let best = predict(&model, image)?;
    if best == 3 {
        println!("with mask :)");
    }else {
        println!("without mask :(");
    }}
    //println!("result: {best:?}");
    Ok(())
}

fn init_model() -> Result<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>, tract_onnx::tract_core::anyhow::Error>{
    let model: Result<SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>, tract_onnx::tract_core::anyhow::Error>= tract_onnx::onnx()
        // load the model
        .model_for_path("Face_Mask_Classification.onnx")?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable();
    model
}

fn preprocess(image_path: String) -> Tensor {
    let image = image::open(image_path).unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, 480, 480, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 480, 480), |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    }).into();
    image

}

fn predict(model:&SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>
     , image: Tensor ) -> TractResult<i32> {
        
    

    let result = model.run(tvec!(image.into()))?;

    // find and display the max value with its index
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(2..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();

        Ok(best.1)
     }