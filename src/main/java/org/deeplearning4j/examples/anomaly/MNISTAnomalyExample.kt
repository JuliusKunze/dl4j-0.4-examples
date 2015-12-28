package org.deeplearning4j.examples.anomaly

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.awt.GridLayout
import java.awt.Image
import java.awt.image.BufferedImage
import java.util.*
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JPanel

/**Example: Anomaly Detection on MNIST using simple auto-encoder without pre-training
 * The goal is to identify outliers digits, i.e., those digits that are unusual or
 * not like the typical digits.
 * This is accomplished in this example by using reconstruction error: stereotypical
 * examples should have low reconstruction error, whereas outliers should have high
 * reconstruction error

 * @author Alex Black
 */
object MNISTAnomalyExample {
    @JvmStatic fun main(args: Array<String>) {
        //Set up network. 784 in/out (as MNIST images are 28x28).
        //784 -> 250 -> 10 -> 250 -> 784
        val conf = NeuralNetConfiguration.Builder()
                .seed(12345)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.05)
                .l2(0.001)
                .list(4)
                .layer(0, DenseLayer.Builder()
                        .nIn(784).nOut(250)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.ADAGRAD)
                        .activation("relu")
                        .build())
                .layer(1, DenseLayer.Builder()
                        .nIn(250).nOut(10)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.ADAGRAD)
                        .activation("relu")
                        .build())
                .layer(2, DenseLayer.Builder()
                        .nIn(10).nOut(250)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.ADAGRAD)
                        .activation("relu")
                        .build())
                .layer(3, OutputLayer.Builder()
                        .nIn(250).nOut(784)
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.ADAGRAD)
                        .activation("relu")
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build()

        val net = MultiLayerNetwork(conf)
        net.listeners = Arrays.asList(ScoreIterationListener(1) as IterationListener)

        //Load data and split into training and testing sets. 40000 train, 10000 test
        val iterator = MnistDataSetIterator(100, 50000, false)

        val featuresTrain = ArrayList<INDArray>()
        val featuresTest = ArrayList<INDArray>()
        val labelsTest = ArrayList<INDArray>()

        val r = Random(12345)
        while (iterator.hasNext()) {
            val ds = iterator.next()
            val split = ds.splitTestAndTrain(80, r)  //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.train.featureMatrix)
            val dsTest = split.test
            featuresTest.add(dsTest.featureMatrix)
            val indexes = Nd4j.argMax(dsTest.labels, 1) //Convert from one-hot representation -> index
            labelsTest.add(indexes)
        }

        //Train model:
        val nEpochs = 30
        for (epoch in 0..nEpochs - 1) {
            for (data in featuresTrain) {
                net.fit(data, data)
            }
            println("Epoch $epoch complete")
        }

        class TestResult(val score: Double, val image: INDArray)

        //Evaluate the model on test data
        val lists = (0..9).map { ArrayList<TestResult>() }

        for (i in featuresTest.indices) {
            val testData = featuresTest[i]
            val labels = labelsTest[i]
            for (j in 0..testData.rows() - 1) {
                val image = testData.getRow(j)
                val label = labels.getDouble(j).toInt()
                val score = net.score(DataSet(image, image))
                lists[label].add(TestResult(score = score, image = image))
            }
        }

        val sortedLists = lists.map { it.sortedBy { it.score } }
        val best = sortedLists.flatMap { it.map { it.image }.take(5) }
        val worst = sortedLists.flatMap { it.map { it.image }.takeLast(5) }

        MNISTVisualizer(digits = best, title = "Best (Low Rec. Error)")()
        MNISTVisualizer(digits = worst, title = "Worst (High Rec. Error)")()
    }

    private class MNISTVisualizer(
            private val digits: List<INDArray>, //Digits (as row vectors), one per INDArray
            private val title: String,
            private val imageScale: Double = 2.0
    ) {
        operator fun invoke() {
            JFrame().apply {
                this.title = title
                defaultCloseOperation = JFrame.EXIT_ON_CLOSE
                add(JPanel().apply {
                    layout = GridLayout(0, 5)

                    components.forEach { add(it) }
                })
                isVisible = true
                pack()
            }
        }

        private val components: List<JLabel> get() = digits.map {
            val bi = BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
            for (i in 0..767) {
                bi.raster.setSample(i % 28, i / 28, 0, (255 * it.getDouble(i)).toInt())
            }
            JLabel(ImageIcon(ImageIcon(bi).image.getScaledInstance((imageScale * 28).toInt(), (imageScale * 28).toInt(), Image.SCALE_REPLICATE)))
        }
    }
}
