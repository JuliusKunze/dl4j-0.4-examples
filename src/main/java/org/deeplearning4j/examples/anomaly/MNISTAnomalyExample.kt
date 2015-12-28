package org.deeplearning4j.examples.anomaly

import org.apache.commons.lang3.tuple.ImmutableTriple
import org.apache.commons.lang3.tuple.Triple
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

        //Evaluate the model on test data
        //Score each digit/example in test set separately
        //Then add triple (score, digit, and INDArray data) to lists and sort by score
        //This allows us to get best N and worst N digits for each type
        val lists = (0..9).map {ArrayList<Triple<Double, Int, INDArray>>()}

        var count = 0
        for (i in featuresTest.indices) {
            val testData = featuresTest[i]
            val labels = labelsTest[i]
            for (j in 0..testData.rows() - 1) {
                val example = testData.getRow(j)
                val label = labels.getDouble(j).toInt()
                val score = net.score(DataSet(example, example))
                lists[label].add(ImmutableTriple(score, count++, example))
            }
        }

        //Sort data by score, separately for each digit
        val c = Comparator<org.apache.commons.lang3.tuple.Triple<kotlin.Double, kotlin.Int, org.nd4j.linalg.api.ndarray.INDArray>> { o1, o2 -> java.lang.Double.compare(o1.left, o2.left) }

        for (list in lists) {
            Collections.sort(list, c)
        }

        //Select the 5 best and 5 worst numbers (by reconstruction error) for each digit
        val best = lists.flatMap { it.map { it.right }.take(5) }
        val worst = lists.flatMap { it.map { it.right }.reversed().take(5) }

        //Visualize the best and worst digits
        val bestVisualizer = MNISTVisualizer(2.0, best, "Best (Low Rec. Error)")
        bestVisualizer.visualize()

        val worstVisualizer = MNISTVisualizer(2.0, worst, "Worst (High Rec. Error)")
        worstVisualizer.visualize()
    }

    private class MNISTVisualizer(
            private val imageScale: Double,
            private val digits: List<INDArray>, //Digits (as row vectors), one per INDArray
            private val title: String) {

        fun visualize() {
            val frame = JFrame()
            frame.title = title
            frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE

            val panel = JPanel()
            panel.layout = GridLayout(0, 5)

            val list = components
            for (image in list) {
                panel.add(image)
            }

            frame.add(panel)
            frame.isVisible = true
            frame.pack()
        }

        private val components: List<JLabel>
            get() {
                val images = ArrayList<JLabel>()
                for (arr in digits) {
                    val bi = BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
                    for (i in 0..767) {
                        bi.raster.setSample(i % 28, i / 28, 0, (255 * arr.getDouble(i)).toInt())
                    }
                    val orig = ImageIcon(bi)
                    val imageScaled = orig.image.getScaledInstance((imageScale * 28).toInt(), (imageScale * 28).toInt(), Image.SCALE_REPLICATE)
                    val scaled = ImageIcon(imageScaled)
                    images.add(JLabel(scaled))
                }
                return images
            }
    }
}
