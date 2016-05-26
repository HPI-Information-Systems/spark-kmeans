package de.hpi.isg.mmds.spark

import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

/**
  * This is a simple Spark implementation of the k-means algorithm.
  *
  * @param inputUrl URL to a input CSV file
  * @param k        number of clusters to be created
  */
class KMeans(inputUrl: String, k: Int, numIterations: Int) {

  /**
    * The [[SparkContext]] used by this instance.
    */
  val sc = {
    // If we use spark-submit, the SparkContext will be configured for us.
    val conf = new SparkConf(true)
    conf.setIfMissing("spark.master", "local[2]") // Run locally by default.
    conf.setAppName(s"k-means ($inputUrl, k=$k)")
    new SparkContext(conf)
  }

  def run() = {
    // Read and parse the points from an input file. Cache them because we reuse them.
    val pointsRDD = sc.textFile(inputUrl)
      .map { line =>
        val fields = line.split(",")
        Point(fields(0).toDouble, fields(1).toDouble)
      }
    pointsRDD.cache()

    // Generate random initial centroids.
    val initialCentroids = createRandomCentroids(k)

    // Loop: Iteratively do the two k-means phases.
    var centroids = initialCentroids
    for (iteration <- 1 to numIterations) {
      println(s"Starting iteration $iteration...")

      // Make the centroids available in the Spark workers.
      val centroidsBc = sc.broadcast(centroids)

      // Find the nearest centroid for each point.
      val nearestCentroidRDD = pointsRDD.map { point =>
        var minDistance = Double.PositiveInfinity
        var nearestCentroidId = -1
        for (centroid <- centroidsBc.value) {
          val distance = point.distanceTo(centroid)
          if (distance < minDistance) {
            minDistance = distance
            nearestCentroidId = centroid.centroidId
          }
        }
        new TaggedPointCounter(point, nearestCentroidId)
      }

      // Calculate the new centroids from the assignment.
      val newCentroidsRDD = nearestCentroidRDD.keyBy(_.centroidId).reduceByKey(_ + _).map(_._2.average)

      // Collect the new centroids and clean up.
      centroids = newCentroidsRDD.collect().toIndexedSeq
      centroids = centroids ++ createRandomCentroids(k - centroids.size) // We might have <k centroids.
      centroidsBc.unpersist(false)
    }

    println("Results:")
    centroids.foreach(println _)

    sc.stop()
  }

  /**
    * Creates random centroids.
    *
    * @param n      the number of centroids to create
    * @param random used to draw random coordinates
    * @return the centroids
    */
  def createRandomCentroids(n: Int, random: Random = new Random()) =
    for (i <- 1 to n) yield TaggedPoint(random.nextDouble(), random.nextDouble(), i)


}

/**
  * Companion object for [[KMeans]].
  */
object KMeans {

  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("Usage: scala <main class> <input URL> <k> <#iterations>")
      sys.exit(1)
    }

    val startTime = java.lang.System.currentTimeMillis
    new KMeans(args(0), args(1).toInt, args(2).toInt).run()
    val endTime = java.lang.System.currentTimeMillis

    println(f"Finished in ${endTime - startTime}%,d ms.")  }

}

/**
  * Represents objects with an x and a y coordinate.
  */
sealed trait PointLike {

  /**
    * @return the x coordinate
    */
  def x: Double

  /**
    * @return the y coordinate
    */
  def y: Double

}

/**
  * Represents a two-dimensional point.
  *
  * @param x the x coordinate
  * @param y the y coordinate
  */
case class Point(x: Double, y: Double) extends PointLike {

  /**
    * Calculates the Euclidean distance to another [[Point]].
    *
    * @param that the other [[PointLike]]
    * @return the Euclidean distance
    */
  def distanceTo(that: PointLike) = {
    val dx = this.x - that.x
    val dy = this.y - that.y
    math.sqrt(dx * dx + dy * dy)
  }

}

/**
  * Represents a two-dimensional point with a centroid ID attached.
  */
case class TaggedPoint(x: Double, y: Double, centroidId: Int) extends PointLike

/**
  * Represents a two-dimensional point with a centroid ID and a counter attached.
  */
case class TaggedPointCounter(x: Double, y: Double, centroidId: Int, val count: Int = 1) extends PointLike {

  def this(point: PointLike, centroidId: Int, count: Int = 1) = this(point.x, point.y, centroidId, count)

  /**
    * Adds coordinates and counts of two instances.
    *
    * @param that the other instance
    * @return the sum
    */
  def +(that: TaggedPointCounter) = TaggedPointCounter(this.x + that.x, this.y + that.y, this.centroidId, this.count + that.count)

  /**
    * Calculates the average of all added instances.
    *
    * @return a [[TaggedPoint]] reflecting the average
    */
  def average = TaggedPoint(x / count, y / count, centroidId)

}
