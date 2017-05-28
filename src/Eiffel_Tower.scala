import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by israelcvidal on 28/05/17.
  */
object Eiffel_Tower {

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")

        val stopWords = sc.textFile("Dataset/stop-word-list.csv").flatMap(line => line.split(", ")).collect()

        val textFile = sc.textFile("Dataset/eiffel-tower-reviews.json")

        val counts = textFile.flatMap(line => line.split(" "))
            .map(word => word.toLowerCase)
            .filter(word => !stopWords.contains(word))
            .map(word => (word, 1))
            .reduceByKey(_ + _)

        val top20 = counts
          .sortBy(word => word._2, ascending = false)
          .take(20)

        top20.foreach(println)
    }
}
