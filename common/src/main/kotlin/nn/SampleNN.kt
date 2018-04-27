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

    override fun toString(): String {
        return "Sample NN Class"
    }
}