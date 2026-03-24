package com.vyas.ai.TensorFlow_Java_Tutorial;

import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Result;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Sigmoid;
import org.tensorflow.op.nn.Relu;
import org.tensorflow.types.TFloat32;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;

public class BasicNNTest {
	
	static Function<Double, Double> heNormalWeight = (fin) -> Math.random() * ( (Math.sqrt(2 / fin)) + 1); // He Normal initialization
	
	@Test
	public void FourInput2Layes1OutputTest() {

		try (Graph graph = new Graph(); Session session = new Session(graph)) {
            Ops tf = Ops.create(graph);

            // 1. Define network parameters (using placeholder values for weights and biases)
            int numInputs = 4;
            int hiddenLayer1Neurons = 8; // Example neuron count for hidden layers
            int hiddenLayer2Neurons = 4; 
            int numOutputs = 1;

            // Define placeholders for inputs and outputs
            Placeholder<TFloat32> inputs = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(numInputs, 1)));

            // Define dense layers (weights and biases would be actual Variables initialized randomly in a real scenario)
            // The following are conceptual definitions of operations, not trainable layers themselves:

            // --- Hidden Layer 1 (4 inputs -> 8 outputs, using ReLU activation) ---
            // A real implementation needs proper weight initialization and variable management
            // This is a simplified representation of applying operations:
            // For a production system, manage variables properly using tf.variable()
            Constant<TFloat32> weights1 = tf.constant(createDummyWeights(numInputs, hiddenLayer1Neurons));
            Constant<TFloat32> biases1 = tf.constant(createDummyBiases(hiddenLayer1Neurons));
            // Operation: (inputs * weights1) + biases1
            Add<TFloat32> linear1 = tf.math.add(tf.linalg.matMul(weights1, inputs), biases1);
            // Apply activation function
            Relu<TFloat32> hidden1Output = tf.nn.relu(linear1);

            // --- Hidden Layer 2 (8 inputs -> 4 outputs, using ReLU activation) ---
            Constant<TFloat32> weights2 = tf.constant(createDummyWeights(hiddenLayer1Neurons, hiddenLayer2Neurons));
            Constant<TFloat32> biases2 = tf.constant(createDummyBiases(hiddenLayer2Neurons));
            Add<TFloat32> linear2 = tf.math.add(tf.linalg.matMul(weights2, hidden1Output), biases2);
            Relu<TFloat32> hidden2Output = tf.nn.relu(linear2);

            // --- Output Layer (4 inputs -> 1 output, using Sigmoid activation for binary classification) ---
            Constant<TFloat32> weightsOut = tf.constant(createDummyWeights(hiddenLayer2Neurons, numOutputs));
            Constant<TFloat32> biasesOut = tf.constant(createDummyBiases(numOutputs));
            Add<TFloat32> linearOut = tf.math.add(tf.linalg.matMul(weightsOut, hidden2Output), biasesOut);
            Sigmoid<TFloat32> prediction = tf.math.sigmoid(linearOut);

            // Now, you would need to run this graph in a session to get an output
            // Example of how to feed input data (assuming a single 4-element input vector)
            float[][] inputData = {{0.5f}, {0.2f}, {0.8f}, {0.1f}};
            try (TFloat32 inputTensor = TFloat32.vectorOf(inputData[0])) {
                // You would need to manage the execution and retrieval of results here
                session.runner().feed(inputs, inputTensor).fetch(prediction).run();
                System.out.println("Network structure defined. Training and execution would follow.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
	
	private static float[][] createDummyWeights(int in, int out) {
		
		float[][] weights = new float[out][in];
		
		for(int i = 0; i < out; i++) {
			for(int j = 0; j < in; j++) {
				weights[i][j] =  heNormalWeight.apply((double) in).floatValue();
			}
		}
        return weights;
    }
	
	private static float[] createDummyBiases(int size) {
		float[] biases = new float[size];
		 for(int i = 0; i < size; i++) {
			 biases[i] = Double.valueOf(Math.random()).floatValue(); // Biases can be initialized to zero or small random values
		 }
		 return biases;
    }
	
	@Test
	public void ThreeInputs2Layers1OutputTest() {
		
		int numInputs = 3;
        int hiddenLayer1Neurons = 4; // Example neuron count for hidden layers
        int hiddenLayer2Neurons = 4; 
        int numOutputs = 1;
        
		// Use a Graph to define the computation
        try (Graph g = new Graph(); Session s = new Session(g)) {
            Ops tf = Ops.create(g);

            // 1. Input Layer: 3 inputs (Batch size -1 for flexibility)
            Placeholder<TFloat32> input = tf.withName("Input").placeholder(TFloat32.class, 
                Placeholder.shape(Shape.of(-1, 3)));
            Placeholder<TFloat32> target = tf.withName("Target").placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, 1)));

            // 2. Hidden Layer 1: 4 Neurons
            Variable<TFloat32> w1 = tf.withName("Layer1_Weights").variable(tf.random.randomUniform(tf.constant(Shape.of(3, 4)), TFloat32.class));
            //Variable<TFloat32> b1 = tf.variable(tf.random.randomStandardNormal(tf.constant(Shape.of(4)), TFloat32.class));
            Variable<TFloat32> b1 = tf.withName("Layer1_Bias").variable(tf.random.randomUniform(tf.constant(Shape.of(4)), TFloat32.class));
            Operand<TFloat32> layer1 = tf.nn.relu(tf.math.add(tf.linalg.matMul(input, w1), b1));

            // 3. Hidden Layer 2: 4 Neurons
            Variable<TFloat32> w2 = tf.withName("Layer2_Weights").variable(tf.random.randomUniform(tf.constant(Shape.of(4, 4)), TFloat32.class));
            Variable<TFloat32> b2 = tf.withName("Layer2_Bias").variable(tf.random.randomUniform(tf.constant(Shape.of(4)), TFloat32.class));
            Operand<TFloat32> layer2 = tf.nn.relu(tf.math.add(tf.linalg.matMul(layer1, w2), b2));

            // 4. Output Layer: 1 Neuron
            Variable<TFloat32> wOut = tf.withName("Output_Weight").variable(tf.random.randomUniform(tf.constant(Shape.of(4, 1)), TFloat32.class));
            Variable<TFloat32> bOut = tf.withName("Output_Bias").variable(tf.zeros(tf.constant(Shape.of(1)), TFloat32.class));
            Operand<TFloat32> prediction = tf.math.add(tf.linalg.matMul(layer2, wOut), bOut);
            
            // 2. Initialize Variables (Crucial for execution)
            //s.run(tf.));
            
            Operand<TFloat32> diff = tf.withName("Diffrence").math.sub(prediction, target);
            Operand<TFloat32> loss = tf.withName("Loss").math.mean(tf.math.square(diff), tf.constant(0));
            
         // 3. Prepare Input Data: [2.0, 3.0, -1.0]
            // We create a 2D tensor with shape (1, 3)
            try(TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][]{{2.0f, 3.0f, -1.0f}}));
            	TFloat32 targetTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][]{{10.0f}}))) {
                
                // 4. Run the Session
                try (Result result = s.runner()
                        .feed(input.asOutput(), inputTensor)
                        .feed(target.asOutput(), targetTensor)
                        .fetch(prediction)
                        .fetch(loss)
                        .run()) {
                    
                    TFloat32 outputTensor = (TFloat32) result.get(0);
                    TFloat32 lossVal = (TFloat32) result.get(1);
                    System.out.println("Input: [2.0, 3.0, -1.0]");
                    System.out.println("Model Output: " + outputTensor.getFloat(0, 0));
                    System.out.println("Model Loss(MSE): " + lossVal.getFloat());
                    
                    result.iterator().forEachRemaining(
                    		tensor -> 
                    		System.out.println( "Key: " + tensor.getKey() + " - " +  tensor.getValue() )
                    	); // Ensure all tensors are closed
                }
            }
            
			
			  g.toGraphDef().getNodeList().forEach(node ->
			  System.out.println(node.getName() + " - " + node.getOp() + " - " +
			  node.getName() + " - " + node.getInputList() + " - " +
			  node.getAttrMap().toString()));
			  
			  System.out.println("Model initialized using TensorFlow Java 1.1.0.");
			 
        }
	}
	
	@Test
	public void TwoInputs2Layers1OutputTest() {
		// Define network architecture using the low-level Java API
        // This is a basic illustration; for Keras-style API look into
        // the tensorflow-framework library.
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // 1. Define Input (2 inputs) and Output Placeholders
            // Use placeholders if you need to feed different data during training
            // For simplicity, we use constants here.
            Constant<TFloat32> inputs = tf.constant(new float[][]{
                {0f, 0f}, {0f, 1f}, {1f, 0f}, {1f, 1f} // Example input data
            });
            Constant<TFloat32> labels = tf.constant(new float[][]{
                {0f}, {1f}, {1f}, {2f} // Example target data (e.g. summation)
            });

            // 2. Define Weights and Biases for layers
            // Hidden Layer 1 (e.g., 4 neurons)
            Constant<TFloat32> weights1 = tf.constant(new float[][]{
                {0.1f, 0.2f, 0.3f, 0.4f},
                {0.4f, 0.3f, 0.2f, 0.1f}
            });
            Constant<TFloat32> biases1 = tf.constant(new float[]{0.1f, 0.1f, 0.1f, 0.1f});

            // Hidden Layer 2 (e.g., 3 neurons)
            Constant<TFloat32> weights2 = tf.constant(new float[][]{
                {0.1f, 0.2f, 0.3f},
                {0.3f, 0.2f, 0.1f},
                {0.1f, 0.2f, 0.3f},
                {0.3f, 0.2f, 0.1f}
            });
            Constant<TFloat32> biases2 = tf.constant(new float[]{0.1f, 0.1f, 0.1f});

            // Output Layer (1 output)
            Constant<TFloat32> weights3 = tf.constant(new float[][]{
                {0.5f}, {0.5f}, {0.5f}
            });
            Constant<TFloat32> biases3 = tf.constant(new float[]{0.1f});

            // 3. Build the Network Graph (Forward pass)
            // Layer 1
            Add hidden1_pre = tf.math.add(tf.linalg.matMul(inputs, weights1), biases1);
            // Apply activation (e.g., ReLU)
            Relu<TFloat32> hidden1_out = tf.nn.relu(hidden1_pre);

            // Layer 2
            Add hidden2_pre = tf.math.add(tf.linalg.matMul(hidden1_out, weights2), biases2);
             // Apply activation (e.g., ReLU)
            Relu<TFloat32> hidden2_out = tf.nn.relu(hidden2_pre);


            // Output Layer
            Add output_pre = tf.math.add(tf.linalg.matMul(hidden2_out, weights3), biases3);
            // No activation applied for regression (linear output)

            // This is a simple forward pass; full training requires an optimizer, loss function, etc.
            // For a more complete example with training loop and Keras API, refer to the official 
            // TensorFlow Java documentation.

            // 4. Run the graph (inference example)
            try (Session session = new Session(graph)) {
                // Initialize variables (if using tf.Variable instead of tf.constant)
                // session.run(tf.init().op()); 

                // Run the output operation to get prediction
                try (Tensor result = session.runner().fetch(output_pre.asOutput()).run().get(0)) {
                    System.out.println("Predictions:");
                    TFloat32 t = (TFloat32) result;
                    long rows = t.shape().size(0);
                    long cols = t.shape().size() > 1 ? t.shape().size(1) : 1;
                    for (int i = 0; i < rows; i++) {
                        float[] row = new float[(int) cols];
                        for (int j = 0; j < cols; j++) {
                            row[j] = t.getFloat(i, j);
                        }
                        System.out.println(Arrays.toString(row));
                    }
                }

            }

        } catch (Exception e) {
            e.printStackTrace();
        }
	}
}
