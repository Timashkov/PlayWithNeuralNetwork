package nn

import kotlin.math.exp

typealias Matrix<T> = Array<Array<T>>

fun Matrix<Double>.T(): Matrix<Double> {
    val res = Array(this[0].size, { Array(this.size, { 0.0 }) })
    for (i in 0 until this.size)
        for (j in 0 until this[0].size) {
            res[j][i] = this[i][j]
        }
    return res
}

fun Matrix<Double>.dot(b: Matrix<Double>): Matrix<Double> {
    var minSize = this[0].size
    if (this[0].size > b.size) {
        minSize = b.size
    }
    val res = Array(this.size, { Array(b[0].size, { 0.0 }) })

    for (ai in 0 until this.size) {
        for (bj in 0 until b[0].size) {
            for (k in 0 until minSize) {
                val m = this[ai][k] * b[k][bj]
                res[ai][bj] += m
            }
        }
    }
    return res
}

fun Matrix<Double>.minus(b: Matrix<Double>): Matrix<Double> {
    val minRow = if (this.size > b.size) b.size else this.size
    val minCol = if (this[0].size > b[0].size) b[0].size else this[0].size
    val res = Array(minRow, { Array(minCol, { 0.0 }) })

    for (i in 0 until minRow)
        for (j in 0 until minCol) {
            res[i][j] = this[i][j] - b[i][j]
        }
    return res
}

fun Matrix<Double>.plus(b: Matrix<Double>): Matrix<Double> {
    val minRow = if (this.size > b.size) b.size else this.size
    val minCol = if (this[0].size > b[0].size) b[0].size else this[0].size
    val res = Array(minRow, { Array(minCol, { 0.0 }) })

    for (i in 0 until minRow)
        for (j in 0 until minCol) {
            res[i][j] = this[i][j] + b[i][j]
        }
    return res
}

fun Matrix<Double>.printVal() {
    for (i in 0 until this.size) {
        for (j in 0 until this[0].size) {
            print(" ${this[i][j]} ")
        }
        println()
    }
}

class SampleNN {

    private var mSynapticWeights = arrayOf(arrayOf(0.5), arrayOf(-0.5), arrayOf(0.5))

    /** The Sigmoid function, which describes an S shaped curve.
    We pass the weighted sum of the inputs through this function to
    normalise them between 0 and 1.*/
    fun sigmoid(x: Matrix<Double>): Matrix<Double> {
        val res = Array(x.size, { Array(x[0].size, { 0.0 }) })
        for (row in 0 until res.size)
            for (col in 0 until res[row].size)
                res[row][col] = 1 / (1 + exp(-x[row][col]))

        return res
    }


    /** The derivative of the Sigmoid function.
    This is the gradient of the Sigmoid curve.
    It indicates how confident we are about the existing weight.*/
    fun sigmoid_derivative(x: Matrix<Double>): Matrix<Double> {
        val res = Array(x.size, { Array(x[0].size, { 0.0 }) })
        for (row in 0 until res.size)
            for (col in 0 until res[row].size)
                res[row][col] = x[row][col] * (1 - x[row][col])
        return res
    }

    /** We train the neural network through a process of trial and error.
    Adjusting the synaptic weights each time.*/
    fun train(training_set_inputs: Matrix<Double>, training_set_outputs: Matrix<Double>, number_of_training_iterations: Int) {
        for (i in 0 until number_of_training_iterations) {
            // Pass the training set through our neural network (a single neuron).
            val output = think(training_set_inputs)
            // Calculate the error(The difference between the desired output
            // and the predicted output).
            val error = training_set_outputs.minus(output)

            // Multiply the error by the input and again by the gradient of the Sigmoid curve.
            // This means less confident weights are adjusted more .
            // This means inputs, which are zero, do not cause changes to the weights.
            val adjustment = training_set_inputs.T().dot(error.dot(sigmoid_derivative(output)))

            // Adjust the weights.
            mSynapticWeights = mSynapticWeights.plus(adjustment)
        }
    }

    /** The neural network thinks.
     * Pass inputs through our neural network (our single neuron).
     */
    fun think(inputs: Matrix<Double>) = sigmoid(inputs.dot(mSynapticWeights))

    fun run() {

        // The training set. We have 4 examples, each consisting of 3 input values
        // and 1 output value.
        val trainingSetInputs = arrayOf(arrayOf(0.0, 0.0, 1.0), arrayOf(1.0, 1.0, 1.0), arrayOf(1.0, 0.0, 1.0), arrayOf(0.0, 1.0, 1.0))
        val trainingSetOutputs = arrayOf(arrayOf(0.0, 1.0, 1.0, 0.0)).T()

        // Train the neural network using a training set.
        // Do it 10,000 times and make small adjustments each time.
        train(trainingSetInputs, trainingSetOutputs, 10000)

        println("New synaptic weights after training: ")
        mSynapticWeights.printVal()

        // Test the neural network with a new situation.
        println("Considering new situation [1, 0, 0] -> ?: ")
        think(arrayOf(arrayOf(1.0, 0.0, 0.0))).printVal()


    }


    fun Array<Int>.dot(i: Int) = forEach { it * i }

    override fun toString(): String {
        return "Sample NN Class"
    }
}
/*->>>
[[-0.16595599]
 [ 0.44064899]
 [-0.99977125]]
-->>
[[-0.99977125]
 [-0.72507825]
 [-1.16572724]
 [-0.55912226]]
[[-0.85063774]
 [-0.22582172]
 [-0.73038368]
 [-0.34607578]]
[[-0.72857311]
 [ 0.21837265]
 [-0.32294271]
 [-0.18725774]]
[[-0.66099969]
 [ 0.5350736 ]
 [-0.00398705]
 [-0.12193904]]
*/

/*from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print 1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights))))
*/


/*class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 0] -> ?: "
    print neural_network.think(array([1, 0, 0]))*/