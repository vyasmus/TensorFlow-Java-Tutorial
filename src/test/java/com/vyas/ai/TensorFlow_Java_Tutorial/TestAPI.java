package com.vyas.ai.TensorFlow_Java_Tutorial;

import org.tensorflow.Graph;
import org.tensorflow.Result;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.OpScope;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Shape;
import org.tensorflow.types.TInt32;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class TestAPI {

	@Test
	void testFormulaCalculationWithSession() {
		
		try (Graph graph = new Graph()) {

            // Create constants a = 5 and b = 6
            Tensor a = TInt32.scalarOf(5);
            Tensor b = TInt32.scalarOf(6);       
            
            OpScope scope = new OpScope(graph);

            // Build the computation graph: c = a * b
            graph.opBuilder("Const", "a", scope).setAttr("dtype", a.dataType()).setAttr("value", a).build();
            graph.opBuilder("Const", "b", scope).setAttr("dtype", b.dataType()).setAttr("value", b).build();
            graph.opBuilder("Mul", "c", scope).addInput(graph.operation("a").output(0))
                                       .addInput(graph.operation("b").output(0))
                                       .build();

            
            // Execute the graph
            try (Session session = new Session(graph)) {
                TInt32 result = (TInt32) session.runner().fetch("c").run().get(0);
                
                System.out.println(result.getInt(null)); // Should print 30
                System.out.println("Graph: " + graph.toGraphDef().getNodeList().toString());
            }
        }

	}
	
	@Test
	public void TestSimpleOperation() {

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // a = 5, b = 6
            var a = tf.constant(5);
            var b = tf.constant(6);

            // c = a * b
            var c = tf.math.mul(a, b);

            try (Session session = new Session(graph)) {
            	TInt32 result = (TInt32) session.runner().fetch(c).run().get(0);
                System.out.println(result.getInt(null));
            }
        }
    }


}
