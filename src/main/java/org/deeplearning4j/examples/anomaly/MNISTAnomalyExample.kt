package org.deeplearning4j.examples.anomaly

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
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

/**Anomaly Detection on MNIST using simple auto-encoder without pre-training
 * The goal is to identify outliers digits, i.e., those digits that are unusual or
 * not like the typical digits.
 * This is accomplished in this example by using reconstruction error: stereotypical
 * examples should have low reconstruction error, whereas outliers should have high
 * reconstruction error.
 * @author Alex Black
 */
object MNISTAnomalyExample {
    val imageWidth = 28
    val pixelsPerImage = imageWidth * imageWidth
    val layer2Count = 250
    val layer3Count = 50

    @JvmStatic fun main(args: Array<String>) {
        val net = MultiLayerNetwork(configuration()).apply { listeners = listOf(ScoreIterationListener(1)) }
        val batches = loadDataAsBatchesOf80Training20Test()
        net.trainModel(trainingFeatureMatrices = batches.map { it.train.featureMatrix }.toArrayList())
        visualize(imageRows = net.imagesSortedTypicalToAnomalousGroupedByDigit(batches))
    }

    fun configuration() = NeuralNetConfiguration.Builder()
            .seed(12345)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.05)
            .l2(0.001)
            .list(4)
            .layers(pixelsPerImage, layer2Count, layer3Count, layer2Count, pixelsPerImage)
            .pretrain(false)
            .backprop(true)
            .build()

    fun visualize(imageRows: List<List<INDArray>>, columnCount: Int = 5) {
        Visualizer(images = imageRows.flatMap { it.take(columnCount) }, title = "Typical, best reconstruction", columnCount = columnCount)()
        Visualizer(images = imageRows.flatMap { it.takeLast(columnCount) }, title = "Unusual, worst reconstruction", columnCount = columnCount)()
    }

    fun MultiLayerNetwork.imagesSortedTypicalToAnomalousGroupedByDigit(batches: List<SplitTestAndTrain>): List<List<INDArray>> {
        class ScoredLabeledImage(val image: INDArray, val score: Double, val label: Int)

        val scoredImages = batches.map { it.test }.flatMap {
            val featureMatrix = it.featureMatrix
            val exampleIndices = 0..featureMatrix.rows() - 1
            val images = exampleIndices.map { featureMatrix.getRow(it) }
            val labelMatrix = Nd4j.argMax(it.labels, 1)
            val labels = exampleIndices.map { labelMatrix.getDouble(it).toInt() }
            exampleIndices.map {
                val image = images[it]
                val label = labels[it]
                ScoredLabeledImage(
                        image = image,
                        score = score(DataSet(image, image)),
                        label = label
                )
            }
        }

        return scoredImages.groupBy { it.label }.entries.sortedBy { it.key }.map { it.value.sortedBy { it.score }.map { it.image } }
    }

    fun MultiLayerNetwork.trainModel(trainingFeatureMatrices: ArrayList<INDArray>, epochCount: Int = 1) {
        for (epochNumber in 1..epochCount) {
            trainingFeatureMatrices.forEach { fit(it, it) }
            println("Epoch $epochNumber complete")
        }
    }

    fun loadDataAsBatchesOf80Training20Test() =
            MnistDataSetIterator(100, 50000, false).asSequence().
                    map { it.splitTestAndTrain(80, Random(12345)) }.toArrayList()

    fun NeuralNetConfiguration.ListBuilder.layers(vararg layerConfiguration: Int) =
            layerConfiguration.withIndex().fold(initial = this) { builder, layer ->
                if (layer.index == layerConfiguration.lastIndex) builder
                else builder.layer(layer.index,
                        (if (layer.index + 1 == layerConfiguration.lastIndex)
                            OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        else
                            DenseLayer.Builder())
                                .nIn(layer.value).nOut(layerConfiguration[layer.index + 1])
                                .weightInit(WeightInit.XAVIER)
                                .updater(Updater.ADAGRAD)
                                .activation("relu")
                                .build())
            }

    class Visualizer(
            private val images: List<INDArray>,
            private val title: String,
            private val columnCount: Int,
            private val imageScale: Double = 2.0
    ) {
        private val scaledImageWidth = (imageScale * imageWidth).toInt()

        private fun jLabel(pixels: INDArray) =
                JLabel(ImageIcon(ImageIcon(bufferedImage(pixels)).image.getScaledInstance(scaledImageWidth, scaledImageWidth, Image.SCALE_REPLICATE)))

        private fun bufferedImage(pixels: INDArray) =
                BufferedImage(imageWidth, imageWidth, BufferedImage.TYPE_BYTE_GRAY).apply {
                    (0..pixelsPerImage - 1).forEach { raster.setSample(it % imageWidth, it / imageWidth, 0, (255 * pixels.getDouble(it)).toInt()) }
                }

        operator fun invoke() {
            JFrame().apply {
                this.title = title
                defaultCloseOperation = JFrame.EXIT_ON_CLOSE
                add(JPanel().apply {
                    layout = GridLayout(0, columnCount)

                    images.forEach { add(jLabel(pixels = it)) }
                })
                isVisible = true
                pack()
            }
        }
    }
}