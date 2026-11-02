
use ndarray::Array1;

// --------------------------------------------------
// 1. Forward pass: calculate predictions
// --------------------------------------------------

fn predict(
    x: &Array1<f64>,
    weight: f64,
    bias: f64,
) -> Array1<f64> {
    x.mapv(|value| weight * value + bias)
}

// --------------------------------------------------
// 2. Calculate Mean Squared Error
// --------------------------------------------------

fn compute_loss(
    predictions: &Array1<f64>,
    targets: &Array1<f64>,
) -> f64 {
    let errors = predictions - targets;

    errors.mapv(|error| error.powi(2))
        .mean()
        .expect("Cannot calculate loss for empty data")
}

// --------------------------------------------------
// 3. Calculate gradients
// --------------------------------------------------

fn compute_gradients(
    x: &Array1<f64>,
    predictions: &Array1<f64>,
    targets: &Array1<f64>,
) -> (f64, f64) {

    let errors = predictions - targets;
    let n = targets.len() as f64;

    let grad_weight =
        2.0 * (&errors * x).sum() / n;

    let grad_bias =
        2.0 * errors.sum() / n;

    (grad_weight, grad_bias)
}

// --------------------------------------------------
// 4. Update model parameters
// --------------------------------------------------

fn update_parameters(
    weight: &mut f64,
    bias: &mut f64,
    grad_weight: f64,
    grad_bias: f64,
    learning_rate: f64,
) {
    *weight -= learning_rate * grad_weight;
    *bias -= learning_rate * grad_bias;
}

// --------------------------------------------------
// 5. Train the linear regression model
// --------------------------------------------------

fn train(
    x: &Array1<f64>,
    y: &Array1<f64>,
    learning_rate: f64,
    epochs: usize,
) -> (f64, f64) {

    assert!(!x.is_empty(), "Training data cannot be empty");
    assert_eq!(x.len(), y.len(), "Input and target lengths must match");

    let mut weight = 0.0;
    let mut bias = 0.0;

    for epoch in 0..epochs {

        // Forward pass
        let predictions = predict(x, weight, bias);

        // Calculate loss
        let loss = compute_loss(&predictions, y);

        // Calculate gradients
        let (grad_weight, grad_bias) =
            compute_gradients(x, &predictions, y);

        // Update parameters
        update_parameters(
            &mut weight,
            &mut bias,
            grad_weight,
            grad_bias,
            learning_rate,
        );

        // Monitor training
        if epoch % 1000 == 0 {
            println!(
                "Epoch: {:5} | Loss: {:.6} | Weight: {:.6} | Bias: {:.6}",
                epoch, loss, weight, bias
            );
        }
    }

    (weight, bias)
}

// --------------------------------------------------
// 6. Main function
// --------------------------------------------------

fn main() {

    // Training data
    let x_data = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let y_data = Array1::from(vec![2.0, 4.0, 6.0, 8.0]);

    let learning_rate = 0.117647;
    let epochs = 10_000;

    // Train model
    let (weight, bias) = train(
        &x_data,
        &y_data,
        learning_rate,
        epochs,
    );

    // Final results
    println!("\nTRAINING RESULTS");
    println!("Trained weight: {:.6}", weight);
    println!("Trained bias:   {:.6}", bias);

    // Make a prediction
    let new_input = 5.0;
    let prediction = weight * new_input + bias;

    println!(
        "Prediction for x = {}: {:.6}",
        new_input, prediction
    );
}
